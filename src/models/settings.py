"""Settings model for application configuration stored in SQLite."""

import logging
from src.models.database import get_db

logger = logging.getLogger(__name__)


def get_setting(key: str) -> str | None:
    """Get a single setting value by key."""
    with get_db() as conn:
        cursor = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else None


def get_all_settings() -> dict:
    """Get all settings as a dictionary."""
    with get_db() as conn:
        cursor = conn.execute("SELECT key, value FROM settings")
        return {row["key"]: row["value"] for row in cursor.fetchall()}


def update_setting(key: str, value: str):
    """Update a setting value, creating it if it doesn't exist."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO settings (key, value, updated_at) 
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(key) DO UPDATE SET 
               value = excluded.value, 
               updated_at = CURRENT_TIMESTAMP""",
            (key, value)
        )


def update_settings(settings: dict):
    """Update multiple settings at once."""
    with get_db() as conn:
        for key, value in settings.items():
            conn.execute(
                """INSERT INTO settings (key, value, updated_at) 
                   VALUES (?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(key) DO UPDATE SET 
                   value = excluded.value, 
                   updated_at = CURRENT_TIMESTAMP""",
                (key, value)
            )


# Data directory operations

def get_data_directory(dir_id: int) -> dict | None:
    """Get a single data directory by ID."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT id, path, dir_type, active FROM data_directories WHERE id = ?",
            (dir_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_data_directories(active_only: bool = True) -> list[dict]:
    """Get all data directories."""
    with get_db() as conn:
        if active_only:
            cursor = conn.execute(
                "SELECT id, path, dir_type, active FROM data_directories WHERE active = 1"
            )
        else:
            cursor = conn.execute(
                "SELECT id, path, dir_type, active FROM data_directories"
            )
        return [dict(row) for row in cursor.fetchall()]


def add_data_directory(path: str, dir_type: str) -> dict:
    """Add a new data directory. Returns the created record."""
    if dir_type not in ("training", "output"):
        raise ValueError(f"dir_type must be 'training' or 'output', got '{dir_type}'")

    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO data_directories (path, dir_type) VALUES (?, ?)",
            (path, dir_type)
        )
        return {
            "id": cursor.lastrowid,
            "path": path,
            "dir_type": dir_type,
            "active": 1
        }


def update_data_directory(dir_id: int, path: str = None, dir_type: str = None, active: bool = None) -> bool:
    """Update a data directory. Returns True if found and updated."""
    updates = []
    params = []

    if path is not None:
        updates.append("path = ?")
        params.append(path)
    if dir_type is not None:
        if dir_type not in ("training", "output"):
            raise ValueError(f"dir_type must be 'training' or 'output', got '{dir_type}'")
        updates.append("dir_type = ?")
        params.append(dir_type)
    if active is not None:
        updates.append("active = ?")
        params.append(1 if active else 0)

    if not updates:
        return False

    params.append(dir_id)
    with get_db() as conn:
        cursor = conn.execute(
            f"UPDATE data_directories SET {', '.join(updates)} WHERE id = ?",
            params
        )
        return cursor.rowcount > 0


def delete_data_directory(dir_id: int) -> bool:
    """Delete a data directory. Returns True if found and deleted."""
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM data_directories WHERE id = ?",
            (dir_id,)
        )
        return cursor.rowcount > 0
