"""Scoring model — CRUD for image quality scores, batch tracking, and keep flags."""

import logging

from sqlalchemy import text

from src.models.database import get_db, row_to_dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality scores
# ---------------------------------------------------------------------------

def upsert_quality_score(image_id: int, overall: float, character: float,
                         composition: float, artifacts: float, theme: float,
                         detail: float, expression: float, notes: str | None,
                         model_used: str):
    """Insert or replace a quality score for an image."""
    with get_db() as conn:
        conn.execute(
            text("""INSERT INTO image_quality_scores
               (image_id, overall, character, composition, artifacts, theme,
                detail, expression, notes, model_used)
               VALUES (:image_id, :overall, :character, :composition, :artifacts,
                :theme, :detail, :expression, :notes, :model_used)
               ON CONFLICT(image_id) DO UPDATE SET
                overall = :overall, character = :character, composition = :composition,
                artifacts = :artifacts, theme = :theme, detail = :detail,
                expression = :expression, notes = :notes, model_used = :model_used,
                scored_at = CURRENT_TIMESTAMP"""),
            {"image_id": image_id, "overall": overall, "character": character,
             "composition": composition, "artifacts": artifacts, "theme": theme,
             "detail": detail, "expression": expression, "notes": notes,
             "model_used": model_used},
        )


def get_quality_scores(image_ids: list[int]) -> dict[int, dict]:
    """Get quality scores for a list of image IDs. Returns {image_id: score_dict}."""
    if not image_ids:
        return {}
    placeholders = ",".join([f":p{i}" for i in range(len(image_ids))])
    params = {f"p{i}": v for i, v in enumerate(image_ids)}
    with get_db() as conn:
        result = conn.execute(
            text(f"""SELECT * FROM image_quality_scores
               WHERE image_id IN ({placeholders})"""),
            params,
        )
        return {row._mapping["image_id"]: row_to_dict(row) for row in result.fetchall()}


def get_all_quality_scores() -> dict[int, dict]:
    """Get all quality scores. Returns {image_id: score_dict}."""
    with get_db() as conn:
        result = conn.execute(text("SELECT * FROM image_quality_scores"))
        return {row._mapping["image_id"]: row_to_dict(row) for row in result.fetchall()}


# ---------------------------------------------------------------------------
# Scoring batches
# ---------------------------------------------------------------------------

def create_scoring_batch(batch_id: str, total_images: int) -> int:
    """Create a new scoring batch record. Returns the DB row ID."""
    with get_db() as conn:
        result = conn.execute(
            text("""INSERT INTO scoring_batches (batch_id, total_images)
               VALUES (:batch_id, :total_images)"""),
            {"batch_id": batch_id, "total_images": total_images},
        )
        return result.lastrowid


def update_batch_status(db_id: int, status: str, scored_count: int | None = None):
    """Update a scoring batch status."""
    with get_db() as conn:
        if scored_count is not None:
            conn.execute(
                text("""UPDATE scoring_batches
                   SET status = :status, scored_count = :scored_count,
                       completed_at = CASE WHEN :status IN ('completed', 'failed')
                                      THEN CURRENT_TIMESTAMP ELSE completed_at END
                   WHERE id = :id"""),
                {"status": status, "scored_count": scored_count, "id": db_id},
            )
        else:
            conn.execute(
                text("""UPDATE scoring_batches
                   SET status = :status,
                       completed_at = CASE WHEN :status IN ('completed', 'failed')
                                      THEN CURRENT_TIMESTAMP ELSE completed_at END
                   WHERE id = :id"""),
                {"status": status, "id": db_id},
            )


def get_active_batch() -> dict | None:
    """Get the most recent non-terminal scoring batch, if any."""
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT * FROM scoring_batches
               WHERE status IN ('submitted', 'processing')
               ORDER BY submitted_at DESC LIMIT 1""")
        )
        row = result.fetchone()
        return row_to_dict(row) if row else None


def get_batch_by_id(db_id: int) -> dict | None:
    """Get a scoring batch by its DB ID."""
    with get_db() as conn:
        result = conn.execute(
            text("SELECT * FROM scoring_batches WHERE id = :id"),
            {"id": db_id},
        )
        row = result.fetchone()
        return row_to_dict(row) if row else None


def get_most_recent_batch() -> dict | None:
    """Get the most recent scoring batch regardless of status."""
    with get_db() as conn:
        result = conn.execute(
            text("SELECT * FROM scoring_batches ORDER BY submitted_at DESC LIMIT 1")
        )
        row = result.fetchone()
        return row_to_dict(row) if row else None


def add_batch_items(db_batch_id: int, image_ids: list[int]):
    """Add image-to-batch mapping entries."""
    with get_db() as conn:
        for idx, image_id in enumerate(image_ids):
            conn.execute(
                text("""INSERT INTO scoring_batch_items (batch_id, image_id, request_idx)
                   VALUES (:batch_id, :image_id, :request_idx)"""),
                {"batch_id": db_batch_id, "image_id": image_id, "request_idx": idx},
            )


def get_batch_items(db_batch_id: int) -> list[dict]:
    """Get all items in a scoring batch."""
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT * FROM scoring_batch_items
               WHERE batch_id = :batch_id ORDER BY request_idx"""),
            {"batch_id": db_batch_id},
        )
        return [row_to_dict(row) for row in result.fetchall()]


# ---------------------------------------------------------------------------
# Keep flags
# ---------------------------------------------------------------------------

def flag_keep(image_ids: list[int]):
    """Flag images as explicitly kept."""
    if not image_ids:
        return
    with get_db() as conn:
        for image_id in image_ids:
            conn.execute(
                text("INSERT OR IGNORE INTO image_keep_flags (image_id) VALUES (:id)"),
                {"id": image_id},
            )


def unflag_keep(image_ids: list[int]):
    """Remove keep flags from images."""
    if not image_ids:
        return
    placeholders = ",".join([f":p{i}" for i in range(len(image_ids))])
    params = {f"p{i}": v for i, v in enumerate(image_ids)}
    with get_db() as conn:
        conn.execute(
            text(f"DELETE FROM image_keep_flags WHERE image_id IN ({placeholders})"),
            params,
        )


def get_keep_flags() -> set[int]:
    """Get all image IDs that are flagged as keep."""
    with get_db() as conn:
        result = conn.execute(text("SELECT image_id FROM image_keep_flags"))
        return {row._mapping["image_id"] for row in result.fetchall()}
