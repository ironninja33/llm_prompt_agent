"""Chat and message models for conversation history stored in SQLite."""

import json
import uuid
import logging
from sqlalchemy import text
from src.models.database import get_db, row_to_dict

logger = logging.getLogger(__name__)


# Chat operations

def create_chat() -> dict:
    """Create a new chat session. Returns the created chat."""
    chat_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            text("INSERT INTO chats (id) VALUES (:id)"),
            {"id": chat_id},
        )
    return {
        "id": chat_id,
        "title": "New Chat",
        "summary": None,
        "created_at": None,
        "updated_at": None,
    }


def get_all_chats() -> list[dict]:
    """Get all chats ordered by most recently updated."""
    with get_db() as conn:
        result = conn.execute(
            text("SELECT id, title, summary, created_at, updated_at FROM chats ORDER BY updated_at DESC")
        )
        return [row_to_dict(row) for row in result.fetchall()]


def get_chat(chat_id: str) -> dict | None:
    """Get a single chat by ID."""
    with get_db() as conn:
        result = conn.execute(
            text("SELECT id, title, summary, created_at, updated_at FROM chats WHERE id = :id"),
            {"id": chat_id},
        )
        row = result.fetchone()
        return row_to_dict(row) if row else None


def update_chat_title(chat_id: str, title: str):
    """Update a chat's title/summary."""
    with get_db() as conn:
        conn.execute(
            text("UPDATE chats SET title = :title, updated_at = CURRENT_TIMESTAMP WHERE id = :id"),
            {"title": title, "id": chat_id},
        )


def delete_chat(chat_id: str) -> bool:
    """Delete a chat and all its messages. Returns True if found and deleted."""
    with get_db() as conn:
        # CASCADE should handle messages and agent_state
        result = conn.execute(
            text("DELETE FROM chats WHERE id = :id"),
            {"id": chat_id},
        )
        return result.rowcount > 0


# Message operations

def add_message(chat_id: str, role: str, content: str, metadata: dict = None) -> dict:
    """Add a message to a chat. Returns the created message."""
    metadata_json = json.dumps(metadata) if metadata else None
    with get_db() as conn:
        result = conn.execute(
            text("""INSERT INTO messages (chat_id, role, content, metadata)
               VALUES (:chat_id, :role, :content, :metadata)"""),
            {"chat_id": chat_id, "role": role, "content": content, "metadata": metadata_json},
        )
        conn.execute(
            text("UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = :id"),
            {"id": chat_id},
        )
        return {
            "id": result.lastrowid,
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "metadata": metadata,
        }


def get_messages(chat_id: str) -> list[dict]:
    """Get all messages for a chat in chronological order."""
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT id, chat_id, role, content, metadata, created_at
               FROM messages WHERE chat_id = :chat_id ORDER BY created_at ASC"""),
            {"chat_id": chat_id},
        )
        messages = []
        for row in result.fetchall():
            msg = row_to_dict(row)
            if msg["metadata"]:
                msg["metadata"] = json.loads(msg["metadata"])
            messages.append(msg)
        return messages


def get_message(message_id: int) -> dict | None:
    """Get a single message by ID."""
    with get_db() as conn:
        result = conn.execute(
            text("SELECT id, chat_id, role, content, metadata, created_at FROM messages WHERE id = :id"),
            {"id": message_id},
        )
        row = result.fetchone()
        if row:
            msg = row_to_dict(row)
            if msg["metadata"]:
                msg["metadata"] = json.loads(msg["metadata"])
            return msg
        return None


def delete_messages_after(chat_id: str, message_id: int):
    """Delete all messages in a chat after (and including) the given message ID.
    Used when the user edits and resubmits a message."""
    with get_db() as conn:
        conn.execute(
            text("DELETE FROM messages WHERE chat_id = :chat_id AND id >= :message_id"),
            {"chat_id": chat_id, "message_id": message_id},
        )


def get_last_message_by_role(chat_id: str, role: str) -> dict | None:
    """Get the most recent message of a given role in a chat."""
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT id, chat_id, role, content, metadata, created_at
               FROM messages WHERE chat_id = :chat_id AND role = :role
               ORDER BY created_at DESC LIMIT 1"""),
            {"chat_id": chat_id, "role": role},
        )
        row = result.fetchone()
        if row:
            msg = row_to_dict(row)
            if msg["metadata"]:
                msg["metadata"] = json.loads(msg["metadata"])
            return msg
        return None


def count_messages(chat_id: str) -> int:
    """Count messages in a chat."""
    with get_db() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) as count FROM messages WHERE chat_id = :chat_id"),
            {"chat_id": chat_id},
        )
        return result.fetchone()._mapping["count"]


# Agent state operations

def get_agent_state(chat_id: str) -> dict | None:
    """Get the agent state for a chat."""
    with get_db() as conn:
        result = conn.execute(
            text("SELECT state_json FROM agent_state WHERE chat_id = :chat_id"),
            {"chat_id": chat_id},
        )
        row = result.fetchone()
        if row:
            return json.loads(row._mapping["state_json"])
        return None


def save_agent_state(chat_id: str, state: dict):
    """Save or update the agent state for a chat."""
    state_json = json.dumps(state)
    with get_db() as conn:
        conn.execute(
            text("""INSERT INTO agent_state (chat_id, state_json, updated_at)
               VALUES (:chat_id, :state_json, CURRENT_TIMESTAMP)
               ON CONFLICT(chat_id) DO UPDATE SET
               state_json = excluded.state_json,
               updated_at = CURRENT_TIMESTAMP"""),
            {"chat_id": chat_id, "state_json": state_json},
        )
