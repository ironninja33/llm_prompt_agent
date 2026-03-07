"""Chat controller — business logic for chat operations."""

import logging
from src.models import chat as chat_model
from src.models import tool_calls as tool_calls_model
from src.agent import runner

logger = logging.getLogger(__name__)


def get_all_chats() -> list[dict]:
    """Get all chats sorted by most recent."""
    return chat_model.get_all_chats()


def create_chat() -> dict:
    """Create a new chat session."""
    return chat_model.create_chat()


def get_chat(chat_id: str) -> dict | None:
    """Get a single chat."""
    return chat_model.get_chat(chat_id)


def delete_chat(chat_id: str) -> bool:
    """Delete a chat and all its messages."""
    runner.cancel_run(chat_id)
    return chat_model.delete_chat(chat_id)


def get_messages(chat_id: str) -> list[dict]:
    """Get all messages for a chat, with attachment_urls and tool_calls extracted."""
    messages = chat_model.get_messages(chat_id)
    for msg in messages:
        # Extract attachment_urls from metadata to top-level for frontend
        if msg.get("metadata") and isinstance(msg["metadata"], dict):
            urls = msg["metadata"].get("attachment_urls")
            if urls:
                msg["attachment_urls"] = urls
            if msg["metadata"].get("is_error"):
                msg["is_error"] = True

        # Attach persisted tool calls for assistant messages
        if msg.get("role") == "assistant" and msg.get("id"):
            try:
                calls = tool_calls_model.get_tool_calls(msg["id"])
                if calls:
                    msg["tool_calls"] = calls
            except Exception:
                pass  # Graceful fallback for old chats
    return messages


def send_message(
    chat_id: str,
    content: str,
    attachments: list | None = None,
    attachment_urls: list[str] | None = None,
) -> runner.AgentRun:
    """Start a background agent run for the user message.

    Returns the AgentRun whose events queue the SSE endpoint reads from.
    """
    # Verify chat exists
    chat = chat_model.get_chat(chat_id)
    if not chat:
        return None

    return runner.start_run(
        chat_id, content,
        attachments=attachments,
        attachment_urls=attachment_urls,
    )


def edit_and_resubmit(chat_id: str, message_id: int, new_content: str) -> runner.AgentRun:
    """Edit a user message and resubmit.

    Deletes the original message and all messages after it,
    then starts a new agent run with the new content.
    """
    # Delete the message and everything after it
    chat_model.delete_messages_after(chat_id, message_id)

    # Start new agent run
    return runner.start_run(chat_id, new_content)
