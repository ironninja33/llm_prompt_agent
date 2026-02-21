"""Chat controller — business logic for chat operations."""

import logging
from src.models import chat as chat_model
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
    """Get all messages for a chat."""
    return chat_model.get_messages(chat_id)


def send_message(chat_id: str, content: str):
    """Process a user message through the agent loop.

    Returns a generator that yields SSE event dicts.
    """
    # Verify chat exists
    chat = chat_model.get_chat(chat_id)
    if not chat:
        yield {"type": "error", "message": "Chat not found"}
        return

    # Run agent turn
    yield from run_agent_turn(chat_id, content)


def edit_and_resubmit(chat_id: str, message_id: int, new_content: str):
    """Edit a user message and resubmit.

    Deletes the original message and all messages after it,
    then resubmits with new content.
    """
    # Delete the message and everything after it
    chat_model.delete_messages_after(chat_id, message_id)

    # Resubmit
    yield from run_agent_turn(chat_id, new_content)
