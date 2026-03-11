"""Core agent loop — orchestrates LLM calls, tool execution, and streaming."""

import json
import logging
import threading
import time
from typing import Generator

import sqlalchemy.exc

from google.genai import types

from src.services import llm_service, clustering_service
from src.services.cache_service import cache_manager
from src.models import settings, chat as chat_model
from src.models import tool_calls as tool_calls_model
from src.models import generation as gen_model
from src.agent.tools import TOOL_DECLARATIONS, execute_tool, summarize_tool_result
from src.agent.state import create_initial_state, state_to_context, apply_state_update

logger = logging.getLogger(__name__)

# Maximum tool call iterations per user message
MAX_TOOL_ITERATIONS = 10


def run_agent_turn(
    chat_id: str,
    user_message: str,
    attachments: list | None = None,
    attachment_urls: list[str] | None = None,
    cancel_event: threading.Event | None = None,
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

        # System prompt stays static (no agent state) to enable Gemini caching
        full_system_prompt = system_prompt

        # Build conversation history with agent state as first message pair
        history = _build_message_history(chat_id, agent_state)
        state_prefix = [
            {"role": "user", "content": state_to_context(agent_state)},
            {"role": "model", "content": "Acknowledged."},
        ]

        # Inject dataset overview as droppable messages for early turns.
        # Once the agent has accumulated state, the overview is omitted —
        # the agent can use query_dataset_map to re-query if needed.
        overview_prefix = []
        if not _state_has_content(agent_state):
            try:
                overview = clustering_service.get_dataset_overview()
                overview_json = json.dumps(overview, separators=(",", ":"))
                overview_prefix = [
                    {"role": "user", "content": (
                        "Dataset overview (pre-loaded context \u2014 do not call "
                        "get_dataset_overview, this data is already available):\n"
                        + overview_json
                    )},
                    {"role": "model", "content": (
                        "I have the dataset overview with folder structure, "
                        "cross-folder themes, and statistics. I'll call "
                        "get_folder_themes when I need intra-folder details "
                        "for a specific folder. Ready to help."
                    )},
                ]
                logger.info("Injected dataset overview into messages (%d chars)",
                            len(overview_json))
            except Exception as e:
                logger.warning("Failed to load dataset overview, agent will use "
                               "query_dataset_map: %s", e)

        messages = state_prefix + overview_prefix + history

        # If attachments were provided, enhance the last user message with
        # inline image parts so the LLM can see the images.
        if attachments:
            _inject_attachments(messages, attachments)

        # Try to create/reuse an explicit Gemini cache for static content
        cache_name = None
        try:
            client = llm_service.get_client()
            if client:
                cache_name = cache_manager.get_or_create(
                    client, model_agent, full_system_prompt,
                    TOOL_DECLARATIONS,
                )
        except Exception as e:
            logger.warning("Cache creation failed, falling back to uncached: %s", e)
            cache_name = None

        # Run agent loop (may involve multiple tool calls)
        full_response = ""
        iteration = 0
        tool_call_log = []  # Collect all tool calls for introspection
        generation_job_ids = []  # Track generate_image job IDs for message_id backfill
        last_chunk = None  # Track last streaming chunk for finish_reason inspection

        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1

            # Emit context size to frontend before each LLM call
            context_chars = sum(
                len(m.get("content", ""))
                + sum(len(str(p)) for p in m.get("parts", []))
                for m in messages
            )
            yield {
                "type": "context_status",
                "context_chars": context_chars,
                "iteration": iteration,
                "max_iterations": MAX_TOOL_ITERATIONS,
            }

            # Check for cancellation before each LLM call
            if cancel_event and cancel_event.is_set():
                logger.info("Agent run cancelled for chat %s (before LLM call)", chat_id)
                if full_response:
                    try:
                        chat_model.add_message(chat_id, "assistant", full_response,
                                               metadata={"is_partial": True})
                    except Exception:
                        pass
                yield {"type": "done", "message_id": None}
                return

            try:
                if cache_name:
                    response_stream = llm_service.generate_stream(
                        model=model_agent,
                        messages=messages,
                        cached_content=cache_name,
                    )
                else:
                    response_stream = llm_service.generate_stream(
                        model=model_agent,
                        messages=messages,
                        system_prompt=full_system_prompt,
                        tools=TOOL_DECLARATIONS,
                    )
            except Exception as e:
                error_text = f"LLM error: {str(e)}"
                try:
                    chat_model.add_message(chat_id, "assistant", error_text,
                                           metadata={"is_error": True})
                except Exception:
                    pass
                yield {"type": "error", "message": error_text}
                return

            # Process the stream
            current_text = ""
            tool_calls = []
            function_call_parts = []
            chunk_count = 0
            stream_start = time.monotonic()

            try:
                for chunk in response_stream:
                    chunk_count += 1
                    last_chunk = chunk
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
                stream_duration = time.monotonic() - stream_start
                logger.warning("Chat %s stream error after %d chunks, %.1fs: %s",
                               chat_id, chunk_count, stream_duration, e)
                error_text = f"Streaming error: {str(e)}"
                try:
                    chat_model.add_message(chat_id, "assistant", error_text,
                                           metadata={"is_error": True})
                except Exception:
                    pass
                yield {"type": "error", "message": error_text}
                return

            stream_duration = time.monotonic() - stream_start
            logger.info("Chat %s stream stats: chunks=%d, duration=%.1fs, text_len=%d, tool_calls=%d",
                        chat_id, chunk_count, stream_duration, len(current_text), len(tool_calls))

            # If there were tool calls, execute them and continue the loop
            if tool_calls:
                # Append the model's response (with function calls) to messages
                messages.append({
                    "role": "model",
                    "parts": function_call_parts + (
                        [types.Part.from_text(text=current_text)] if current_text else []
                    ),
                })

                # Execute each tool call (check cancellation before each)
                for fc in tool_calls:
                    if cancel_event and cancel_event.is_set():
                        logger.info("Agent run cancelled for chat %s (before tool exec)", chat_id)
                        if full_response or current_text:
                            try:
                                chat_model.add_message(chat_id, "assistant",
                                                       full_response + current_text,
                                                       metadata={"is_partial": True})
                            except Exception:
                                pass
                        yield {"type": "done", "message_id": None}
                        return
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    yield {
                        "type": "status",
                        "message": f"Calling {tool_name}...",
                        "tool": tool_name,
                    }

                    result = execute_tool(tool_name, tool_args,
                                          context={"chat_id": chat_id})

                    # Record for introspection and persistence
                    tool_call_log.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result,
                        "iteration": iteration,
                    })

                    # Handle update_state specially: apply updates to agent state
                    if tool_name == "update_state":
                        agent_state = apply_state_update(agent_state, tool_args)
                        try:
                            chat_model.save_agent_state(chat_id, agent_state)
                        except sqlalchemy.exc.IntegrityError:
                            logger.warning("Chat %s was deleted mid-turn; stopping agent loop.", chat_id)
                            return
                        # Refresh the state prefix in messages (first two entries)
                        messages[0] = {"role": "user", "content": state_to_context(agent_state)}
                        messages[1] = {"role": "model", "content": "Acknowledged."}
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

                    # If generate_image succeeded, yield event for frontend bubble
                    if tool_name == "generate_image" and result.get("status") == "submitted":
                        job_id = result["job_id"]
                        generation_job_ids.append(job_id)
                        yield {
                            "type": "generation_submitted",
                            "job": {
                                "id": job_id,
                                "chat_id": chat_id,
                                "message_id": None,
                                "status": "pending",
                                "settings": result.get("settings_used", {}),
                            },
                        }

                    # Add tool result to messages (with metadata for truncation)
                    messages.append({
                        "role": "tool",
                        "parts": [types.Part.from_function_response(
                            name=tool_name,
                            response=result,
                        )],
                        "_iteration": iteration,
                        "_tool_name": tool_name,
                        "_tool_args": tool_args,
                        "_tool_result": result,
                    })

                full_response += current_text
                # Summarize stale tool results from older iterations
                _summarize_stale_tool_results(messages, iteration)
                # Continue the loop to let the LLM process tool results
                continue

            else:
                # No tool calls — this is the final response
                full_response += current_text
                break

        # Log finish reason for every completed response
        finish_reason = _get_finish_reason(last_chunk)
        logger.info("Chat %s response complete: finish_reason=%s, length=%d chars",
                     chat_id, finish_reason, len(full_response))

        # Detect empty response (entire turn produced nothing)
        if not full_response and not tool_call_log:
            if finish_reason and finish_reason not in ("STOP", "FinishReason.STOP", "None"):
                error_text = f"Response blocked ({finish_reason}). Try rephrasing your message."
            else:
                error_text = "Model returned an empty response. This can happen during high load. Please try again."
            logger.warning("Empty response for chat %s (finish_reason=%s)", chat_id, finish_reason)
            try:
                chat_model.add_message(chat_id, "assistant", error_text,
                                       metadata={"is_error": True})
            except Exception:
                pass
            yield {"type": "error", "message": error_text}
            return

        # Warn about partial responses with non-STOP finish_reason
        finish_reason_str = str(finish_reason) if finish_reason else None
        if full_response and finish_reason_str and finish_reason_str not in ("STOP", "FinishReason.STOP", "None"):
            if "MAX_TOKENS" in finish_reason_str:
                note = "\n\n*(Response truncated due to length limit.)*"
            elif "SAFETY" in finish_reason_str:
                note = "\n\n*(Response cut short by content filter.)*"
            else:
                note = f"\n\n*(Response incomplete: {finish_reason_str})*"
            full_response += note
            yield {"type": "token", "text": note}
            logger.warning("Chat %s partial response: finish_reason=%s", chat_id, finish_reason_str)

        # Save the assistant response and final agent state.
        # The chat may have been deleted while the agent was streaming
        # (e.g. user closed the tab); guard against the FK violation.
        try:
            msg = chat_model.add_message(chat_id, "assistant", full_response)
            chat_model.save_agent_state(chat_id, agent_state)
        except sqlalchemy.exc.IntegrityError:
            logger.warning("Chat %s was deleted before response could be saved; discarding.", chat_id)
            return

        # Backfill message_id on generation jobs created during this turn
        if generation_job_ids:
            for job_id in generation_job_ids:
                try:
                    gen_model.update_job_message_id(job_id, msg["id"])
                except Exception as e:
                    logger.warning("Failed to backfill message_id for job %s: %s", job_id, e)

        # Persist tool calls to DB for reload
        if tool_call_log:
            try:
                tool_calls_model.save_tool_calls(msg["id"], tool_call_log)
            except Exception as e:
                logger.warning("Failed to persist tool calls for message %s: %s", msg["id"], e)

        # Emit tool call introspection summary
        if tool_call_log:
            yield {"type": "tool_calls", "calls": tool_call_log}

        # Summarize chat title before yielding done — code after the
        # final yield in a generator may never execute if the client
        # closes the connection or Flask garbage-collects the generator.
        _maybe_summarize_chat(chat_id)

        yield {"type": "done", "message_id": msg["id"]}

    except Exception as e:
        logger.error(f"Agent loop error: {e}", exc_info=True)
        error_text = f"Internal error: {str(e)}"
        try:
            chat_model.add_message(chat_id, "assistant", error_text,
                                   metadata={"is_error": True})
        except Exception:
            pass
        yield {"type": "error", "message": error_text}


def _get_finish_reason(chunk) -> str | None:
    """Extract finish_reason string from the last streaming chunk."""
    try:
        if chunk and chunk.candidates:
            reason = chunk.candidates[0].finish_reason
            return str(reason) if reason else None
    except Exception:
        pass
    return None


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


def _build_message_history(chat_id: str, agent_state: dict | None = None) -> list[dict]:
    """Build the messages array for the LLM from chat history.

    Messages are kept clean — agent state is provided via the system prompt,
    not injected into user messages.

    When *agent_state* has meaningful content (context or prompt_requirements),
    older DB history is trimmed to a sliding window — the state already
    captures that knowledge.
    """
    db_messages = chat_model.get_messages(chat_id)

    messages = []
    for msg in db_messages:
        role = msg["role"]
        content = msg["content"]

        # Skip persisted error messages — they shouldn't be sent to the LLM
        if role == "assistant" and isinstance(msg.get("metadata"), dict) and msg["metadata"].get("is_error"):
            continue

        # Skip empty assistant messages (from old empty-response bug)
        if role == "assistant" and not content.strip():
            continue

        # Skip partial/cancelled responses (incomplete and confuse the model)
        if role == "assistant" and isinstance(msg.get("metadata"), dict) and msg["metadata"].get("is_partial"):
            continue

        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "model", "content": content})

    # Sliding window: if agent state has captured context from prior turns,
    # older DB history is redundant — the state carries that knowledge.
    if agent_state and _state_has_content(agent_state):
        max_messages = int(settings.get_setting("context_history_pairs") or 3) * 2
        if len(messages) > max_messages:
            trimmed = len(messages) - max_messages
            messages = messages[-max_messages:]
            logger.info("Trimmed %d old history messages (keeping %d)", trimmed, max_messages)

    return messages


def _state_has_content(agent_state: dict) -> bool:
    """Check if agent state has meaningful content worth preserving over history."""
    return bool(
        agent_state.get("context")
        or agent_state.get("prompt_requirements")
    )


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


def _summarize_stale_tool_results(messages: list[dict], current_iteration: int):
    """Summarize tool results from iterations that are stale.

    Modifies messages in-place.  Keeps the most recent N iterations' results
    full (N read from the ``context_stale_iterations`` setting).  Only touches
    messages with ``_iteration`` metadata (in-memory tool results, not DB
    history).
    """
    keep_iterations = int(settings.get_setting("context_stale_iterations") or 2)
    stale_cutoff = current_iteration - keep_iterations + 1
    if stale_cutoff < 1:
        return  # nothing can be stale yet

    summarized = 0
    for i, msg in enumerate(messages):
        if (msg.get("role") == "tool"
                and not msg.get("_summarized")
                and msg.get("_iteration") is not None
                and msg["_iteration"] < stale_cutoff):
            messages[i] = _summarize_tool_message(msg)
            summarized += 1

    if summarized:
        logger.info("Summarized %d stale tool results (iterations < %d)",
                     summarized, stale_cutoff)


def _summarize_tool_message(msg: dict) -> dict:
    """Replace a tool result message with a compact summary."""
    tool_name = msg.get("_tool_name", "unknown")
    summary = summarize_tool_result(
        tool_name, msg.get("_tool_args", {}), msg.get("_tool_result", {}),
    )
    return {
        "role": "tool",
        "parts": [types.Part.from_function_response(name=tool_name, response=summary)],
        "_iteration": msg.get("_iteration"),
        "_tool_name": tool_name,
        "_summarized": True,
    }
