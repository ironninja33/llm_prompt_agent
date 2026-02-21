"""Chat and message models for conversation history stored in SQLite."""

import json
import uuid
import logging
from src.models.database import get_db

logger = logging.getLogger(__name__)


# Chat operations

def create_chat() -> dict:
    """Create a new chat session. Returns the created chat."""
    chat_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chats (id) VALUES (?)",
            (chat_id,)
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
        cursor = conn.execute(
            "SELECT id, title, summary, created_at, updated_at FROM chats ORDER BY updated_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]


def get_chat(chat_id: str) -> dict | None:
    """Get a single chat by ID."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT id, title, summary, created_at, updated_at FROM chats WHERE id = ?",
            (chat_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def update_chat_title(chat_id: str, title: str):
    """Update a chat's title/summary."""
    with get_db() as conn:
        conn.execute(
            "UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (title, chat_id)
        )


def delete_chat(chat_id: str) -> bool:
    """Delete a chat and all its messages. Returns True if found and deleted."""
    with get_db() as conn:
        # CASCADE should handle messages and agent_state
        cursor = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        return cursor.rowcount > 0


# Message operations

def add_message(chat_id: str, role: str, content: str, metadata: dict = None) -> dict:
    """Add a message to a chat. Returns the created message."""
    metadata_json = json.dumps(metadata) if metadata else None
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO messages (chat_id, role, content, metadata) 
               VALUES (?, ?, ?, ?)""",
            (chat_id, role, content, metadata_json)
        )
        conn.execute(
            "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (chat_id,)
        )
        return {
            "id": cursor.lastrowid,
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "metadata": metadata,
        }


def get_messages(chat_id: str) -> list[dict]:
    """Get all messages for a chat in chronological order."""
    with get_db() as conn:
        cursor = conn.execute(
            """SELECT id, chat_id, role, content, metadata, created_at 
               FROM messages WHERE chat_id = ? ORDER BY created_at ASC""",
            (chat_id,)
        )
        messages = []
        for row in cursor.fetchall():
            msg = dict(row)
            if msg["metadata"]:
                msg["metadata"] = json.loads(msg["metadata"])
            messages.append(msg)
        return messages


def get_message(message_id: int) -> dict | None:
    """Get a single message by ID."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT id, chat_id, role, content, metadata, created_at FROM messages WHERE id = ?",
            (message_id,)
        )
        row = cursor.fetchone()
        if row:
            msg = dict(row)
            if msg["metadata"]:
                msg["metadata"] = json.loads(msg["metadata"])
            return msg
        return None


def delete_messages_after(chat_id: str, message_id: int):
    """Delete all messages in a chat after (and including) the given message ID.
    Used when the user edits and resubmits a message."""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM messages WHERE chat_id = ? AND id >= ?",
            (chat_id, message_id)
        )


def get_last_message_by_role(chat_id: str, role: str) -> dict | None:
    """Get the most recent message of a given role in a chat."""
    with get_db() as conn:
        cursor = conn.execute(
            """SELECT id, chat_id, role, content, metadata, created_at 
               FROM messages WHERE chat_id = ? AND role = ? 
               ORDER BY created_at DESC LIMIT 1""",
            (chat_id, role)
        )
        row = cursor.fetchone()
        if row:
            msg = dict(row)
            if msg["metadata"]:
                msg["metadata"] = json.loads(msg["metadata"])
            return msg
        return None


def count_messages(chat_id: str) -> int:
    """Count messages in a chat."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM messages WHERE chat_id = ?",
            (chat_id,)
        )
        return cursor.fetchone()["count"]


# Agent state operations

def get_agent_state(chat_id: str) -> dict | None:
    """Get the agent state for a chat."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT state_json FROM agent_state WHERE chat_id = ?",
            (chat_id,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row["state_json"])
        return None


def save_agent_state(chat_id: str, state: dict):
    """Save or update the agent state for a chat."""
    state_json = json.dumps(state)
    with get_db() as conn:
        conn.execute(
            """INSERT INTO agent_state (chat_id, state_json, updated_at) 
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(chat_id) DO UPDATE SET 
               state_json = excluded.state_json, 
               updated_at = CURRENT_TIMESTAMP""",
            (chat_id, state_json)
        )
