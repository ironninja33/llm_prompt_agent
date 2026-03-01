"""Settings model for application configuration stored in SQLite."""

import logging
from sqlalchemy import text
from src.models.database import get_db, row_to_dict

logger = logging.getLogger(__name__)


def get_setting(key: str) -> str | None:
    """Get a single setting value by key."""
    with get_db() as conn:
        result = conn.execute(
            text("SELECT value FROM settings WHERE key = :key"),
            {"key": key},
        )
        row = result.fetchone()
        return row._mapping["value"] if row else None


def get_all_settings() -> dict:
    """Get all settings as a dictionary."""
    with get_db() as conn:
        result = conn.execute(text("SELECT key, value FROM settings"))
        return {row._mapping["key"]: row._mapping["value"] for row in result.fetchall()}


def update_setting(key: str, value: str):
    """Update a setting value, creating it if it doesn't exist."""
    with get_db() as conn:
        conn.execute(
            text("""INSERT INTO settings (key, value, updated_at)
               VALUES (:key, :value, CURRENT_TIMESTAMP)
               ON CONFLICT(key) DO UPDATE SET
               value = excluded.value,
               updated_at = CURRENT_TIMESTAMP"""),
            {"key": key, "value": value},
        )


def update_settings(settings: dict):
    """Update multiple settings at once."""
    with get_db() as conn:
        for key, value in settings.items():
            conn.execute(
                text("""INSERT INTO settings (key, value, updated_at)
                   VALUES (:key, :value, CURRENT_TIMESTAMP)
                   ON CONFLICT(key) DO UPDATE SET
                   value = excluded.value,
                   updated_at = CURRENT_TIMESTAMP"""),
                {"key": key, "value": value},
            )


# Data directory operations

def get_data_directory(dir_id: int) -> dict | None:
    """Get a single data directory by ID."""
    with get_db() as conn:
        result = conn.execute(
            text("SELECT id, path, dir_type, active FROM data_directories WHERE id = :id"),
            {"id": dir_id},
        )
        row = result.fetchone()
        return row_to_dict(row) if row else None


def get_data_directories(active_only: bool = True) -> list[dict]:
    """Get all data directories."""
    with get_db() as conn:
        if active_only:
            result = conn.execute(
                text("SELECT id, path, dir_type, active FROM data_directories WHERE active = 1")
            )
        else:
            result = conn.execute(
                text("SELECT id, path, dir_type, active FROM data_directories")
            )
        return [row_to_dict(row) for row in result.fetchall()]


def add_data_directory(path: str, dir_type: str) -> dict:
    """Add a new data directory. Returns the created record."""
    if dir_type not in ("training", "output"):
        raise ValueError(f"dir_type must be 'training' or 'output', got '{dir_type}'")

    with get_db() as conn:
        result = conn.execute(
            text("INSERT INTO data_directories (path, dir_type) VALUES (:path, :dir_type)"),
            {"path": path, "dir_type": dir_type},
        )
        return {
            "id": result.lastrowid,
            "path": path,
            "dir_type": dir_type,
            "active": 1
        }


def update_data_directory(dir_id: int, path: str = None, dir_type: str = None, active: bool = None) -> bool:
    """Update a data directory. Returns True if found and updated."""
    updates = []
    params = {"id": dir_id}

    if path is not None:
        updates.append("path = :path")
        params["path"] = path
    if dir_type is not None:
        if dir_type not in ("training", "output"):
            raise ValueError(f"dir_type must be 'training' or 'output', got '{dir_type}'")
        updates.append("dir_type = :dir_type")
        params["dir_type"] = dir_type
    if active is not None:
        updates.append("active = :active")
        params["active"] = 1 if active else 0

    if not updates:
        return False

    with get_db() as conn:
        result = conn.execute(
            text(f"UPDATE data_directories SET {', '.join(updates)} WHERE id = :id"),
            params,
        )
        return result.rowcount > 0


def delete_data_directory(dir_id: int) -> bool:
    """Delete a data directory. Returns True if found and deleted."""
    with get_db() as conn:
        result = conn.execute(
            text("DELETE FROM data_directories WHERE id = :id"),
            {"id": dir_id},
        )
        return result.rowcount > 0
