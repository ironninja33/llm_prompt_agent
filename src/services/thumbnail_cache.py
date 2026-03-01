"""Thumbnail cache — generates and caches JPEG thumbnails in SQLite."""

import io
import os
import logging

from PIL import Image
from sqlalchemy import text

from src.models.database import get_db

logger = logging.getLogger(__name__)


def get_cached_thumbnail(file_path: str, max_size: int = 256) -> bytes | None:
    """Return a thumbnail for the given file, using cache when possible.

    Checks source_mtime for staleness and regenerates if the source file
    has been modified since the cached version was created.

    Returns JPEG bytes or None if the source file cannot be read.
    """
    norm_path = os.path.normpath(file_path)

    # Check current file mtime
    try:
        current_mtime = os.path.getmtime(norm_path)
    except OSError:
        return None

    # Check cache
    with get_db() as conn:
        result = conn.execute(
            text("SELECT thumbnail, source_mtime FROM thumbnail_cache WHERE file_path = :file_path"),
            {"file_path": norm_path},
        )
        row = result.fetchone()
        if row and row._mapping["source_mtime"] == current_mtime:
            return row._mapping["thumbnail"]

    # Generate thumbnail
    try:
        img = Image.open(norm_path)
        img.thumbnail((max_size, max_size), Image.LANCZOS)

        # Convert to RGB if necessary (handles RGBA, P, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        thumb_bytes = buf.getvalue()
        img.close()
    except Exception as e:
        logger.warning("Failed to generate thumbnail for %s: %s", norm_path, e)
        return None

    # Store in cache
    try:
        with get_db() as conn:
            conn.execute(
                text("""INSERT OR REPLACE INTO thumbnail_cache
                   (file_path, thumbnail, width, height, source_mtime)
                   VALUES (:file_path, :thumbnail, :width, :height, :source_mtime)"""),
                {"file_path": norm_path, "thumbnail": thumb_bytes,
                 "width": max_size, "height": max_size, "source_mtime": current_mtime},
            )
    except Exception as e:
        logger.warning("Failed to cache thumbnail for %s: %s", norm_path, e)

    return thumb_bytes


def invalidate(file_path: str):
    """Remove a cached thumbnail."""
    norm_path = os.path.normpath(file_path)
    try:
        with get_db() as conn:
            conn.execute(
                text("DELETE FROM thumbnail_cache WHERE file_path = :file_path"),
                {"file_path": norm_path},
            )
    except Exception as e:
        logger.warning("Failed to invalidate thumbnail cache for %s: %s", norm_path, e)
