"""Core agent loop — orchestrates LLM calls, tool execution, and streaming."""

import json
import logging
import sqlite3
from typing import Generator

from google.genai import types

from src.services import llm_service
from src.models import settings, chat as chat_model
from src.agent.tools import TOOL_DECLARATIONS, execute_tool
from src.agent.state import create_initial_state, state_to_context, apply_state_update

logger = logging.getLogger(__name__)

# Maximum tool call iterations per user message
MAX_TOOL_ITERATIONS = 10


def run_agent_turn(
    chat_id: str,
    user_message: str,
    attachments: list | None = None,
    attachment_urls: list[str] | None = None,
) -> Generator[dict, None, None]:
    """Run one agent turn: process user message, stream response.

    Args:
        chat_id: Chat session ID.
        user_message: Text content from the user.
        attachments: Optional list of dicts with keys
            'filename', 'content_type', 'data' (raw bytes).
        attachment_urls: Optional list of persistent URLs for the attachments.

    Yields event dicts:
        {"type": "token", "text": "..."}
        {"type": "status", "message": "...", "tool": "..."}
        {"type": "tool_result", "tool": "...", "summary": "..."}
        {"type": "error", "message": "..."}
        {"type": "done", "message_id": int}
    """
    try:
        # Load settings
        model_agent = settings.get_setting("model_agent") or "gemini-3-pro-preview"
        system_prompt = settings.get_setting("system_prompt") or ""

        # Load or create agent state
        agent_state = chat_model.get_agent_state(chat_id)
        if agent_state is None:
            agent_state = create_initial_state()
            chat_model.save_agent_state(chat_id, agent_state)

        # Save user message with attachment metadata if present
        msg_metadata = None
        if attachment_urls:
            msg_metadata = {"attachment_urls": attachment_urls}
        user_msg = chat_model.add_message(chat_id, "user", user_message, metadata=msg_metadata)
        yield {"type": "user_saved", "message_id": user_msg["id"]}

        # Build system prompt with agent state context appended
        full_system_prompt = system_prompt + state_to_context(agent_state)

        # Build conversation history (clean — no state injected into user messages)
        messages = _build_message_history(chat_id)

        # If attachments were provided, enhance the last user message with
        # inline image parts so the LLM can see the images.
        if attachments:
            _inject_attachments(messages, attachments)

        # Run agent loop (may involve multiple tool calls)
        full_response = ""
        iteration = 0
        tool_call_log = []  # Collect all tool calls for introspection

        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1

            try:
                response_stream = llm_service.generate_stream(
                    model=model_agent,
                    messages=messages,
                    system_prompt=full_system_prompt,
                    tools=TOOL_DECLARATIONS,
                )
            except Exception as e:
                yield {"type": "error", "message": f"LLM error: {str(e)}"}
                return

            # Process the stream
            current_text = ""
            tool_calls = []
            function_call_parts = []

            try:
                for chunk in response_stream:
                    if not chunk.candidates:
                        continue

                    candidate = chunk.candidates[0]
                    if not candidate.content or not candidate.content.parts:
                        continue

                    for part in candidate.content.parts:
                        if part.text:
                            current_text += part.text
                            yield {"type": "token", "text": part.text}

                        if part.function_call:
                            tool_calls.append(part.function_call)
                            function_call_parts.append(part)

            except Exception as e:
                yield {"type": "error", "message": f"Streaming error: {str(e)}"}
                return

            # If there were tool calls, execute them and continue the loop
            if tool_calls:
                # Append the model's response (with function calls) to messages
                messages.append({
                    "role": "model",
                    "parts": function_call_parts + (
                        [types.Part.from_text(text=current_text)] if current_text else []
                    ),
                })

                # Execute each tool call
                for fc in tool_calls:
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    yield {
                        "type": "status",
                        "message": f"Calling {tool_name}...",
                        "tool": tool_name,
                    }

                    result = execute_tool(tool_name, tool_args)

                    # Record for introspection (ephemeral, not persisted)
                    tool_call_log.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result,
                    })

                    # Handle update_state specially: apply updates to agent state
                    if tool_name == "update_state":
                        agent_state = apply_state_update(agent_state, tool_args)
                        try:
                            chat_model.save_agent_state(chat_id, agent_state)
                        except sqlite3.IntegrityError:
                            logger.warning("Chat %s was deleted mid-turn; stopping agent loop.", chat_id)
                            return
                        # Refresh the system prompt with updated state
                        full_system_prompt = system_prompt + state_to_context(agent_state)
                        logger.info(f"Agent state updated for chat {chat_id}: phase={agent_state.get('phase')}")

                        yield {
                            "type": "tool_result",
                            "tool": tool_name,
                            "summary": "State updated",
                        }
                    else:
                        yield {
                            "type": "tool_result",
                            "tool": tool_name,
                            "summary": f"Found {result.get('count', 0)} results" if 'count' in result else "Done",
                        }

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "parts": [types.Part.from_function_response(
                            name=tool_name,
                            response=result,
                        )],
                    })

                full_response += current_text
                # Continue the loop to let the LLM process tool results
                continue

            else:
                # No tool calls — this is the final response
                full_response += current_text
                break

        # Save the assistant response and final agent state.
        # The chat may have been deleted while the agent was streaming
        # (e.g. user closed the tab); guard against the FK violation.
        try:
            msg = chat_model.add_message(chat_id, "assistant", full_response)
            chat_model.save_agent_state(chat_id, agent_state)
        except sqlite3.IntegrityError:
            logger.warning("Chat %s was deleted before response could be saved; discarding.", chat_id)
            return

        # Generate chat title BEFORE yielding done, so the client gets
        # the updated title when it refreshes the chat list.
        _maybe_summarize_chat(chat_id)

        # Emit tool call introspection summary (ephemeral — not persisted)
        if tool_call_log:
            yield {"type": "tool_calls", "calls": tool_call_log}

        yield {"type": "done", "message_id": msg["id"]}

    except Exception as e:
        logger.error(f"Agent loop error: {e}", exc_info=True)
        yield {"type": "error", "message": f"Internal error: {str(e)}"}


def _inject_attachments(messages: list[dict], attachments: list[dict]):
    """Inject image attachments into the last user message as inline data parts.

    Modifies *messages* in place.  The last user message is converted from a
    simple ``{"role": "user", "content": "..."}`` dict into one that uses
    ``"parts"`` (a list of ``types.Part``) so the Gemini API receives both text
    and images in a single turn.
    """
    if not attachments:
        return

    # Find the last user message (should be the most recent one)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            text_content = messages[i].get("content", "")
            parts = [types.Part.from_text(text=text_content)]

            for att in attachments:
                mime = att.get("content_type", "image/png")
                raw = att.get("data", b"")
                if raw:
                    parts.append(types.Part.from_bytes(
                        data=raw,
                        mime_type=mime,
                    ))
                    logger.debug(
                        "Attached image %s (%s, %d bytes) to user message",
                        att.get("filename", "unknown"),
                        mime,
                        len(raw),
                    )

            # Replace the simple content dict with a parts-based dict
            messages[i] = {"role": "user", "parts": parts}
            break


def _build_message_history(chat_id: str) -> list[dict]:
    """Build the messages array for the LLM from chat history.

    Messages are kept clean — agent state is provided via the system prompt,
    not injected into user messages.
    """
    db_messages = chat_model.get_messages(chat_id)

    messages = []
    for msg in db_messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "model", "content": content})

    return messages


def _maybe_summarize_chat(chat_id: str):
    """Summarize chat title after first assistant response."""
    try:
        chat = chat_model.get_chat(chat_id)
        if not chat:
            return

        # Only summarize if title is still "New Chat"
        if chat["title"] != "New Chat":
            return

        message_count = chat_model.count_messages(chat_id)
        if message_count < 2:
            return

        model_summary = settings.get_setting("model_summary") or "gemini-2.5-flash-lite"
        messages = chat_model.get_messages(chat_id)

        title = llm_service.summarize_chat(model_summary, messages)
        if title and title != "New Chat":
            chat_model.update_chat_title(chat_id, title)
            logger.info(f"Chat {chat_id} titled: {title}")
        else:
            logger.warning(f"Summarization returned empty/unchanged title for chat {chat_id}")

    except Exception as e:
        logger.error(f"Error summarizing chat: {e}")
