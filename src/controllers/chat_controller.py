"""Chat controller — business logic for chat operations."""

import logging
from src.models import chat as chat_model
from src.models import tool_calls as tool_calls_model
from src.agent.loop import run_agent_turn

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
):
    """Process a user message through the agent loop.

    Args:
        chat_id: The chat session ID.
        content: Text content of the user message.
        attachments: Optional list of attachment dicts with keys
            'filename', 'content_type', 'data' (raw bytes).
        attachment_urls: Optional list of persistent URLs for the attachments.

    Returns a generator that yields SSE event dicts.
    """
    # Verify chat exists
    chat = chat_model.get_chat(chat_id)
    if not chat:
        yield {"type": "error", "message": "Chat not found"}
        return

    # Run agent turn
    yield from run_agent_turn(
        chat_id, content,
        attachments=attachments,
        attachment_urls=attachment_urls,
    )


def edit_and_resubmit(chat_id: str, message_id: int, new_content: str):
    """Edit a user message and resubmit.

    Deletes the original message and all messages after it,
    then resubmits with new content.
    """
    # Delete the message and everything after it
    chat_model.delete_messages_after(chat_id, message_id)

    # Resubmit
    yield from run_agent_turn(chat_id, new_content)
