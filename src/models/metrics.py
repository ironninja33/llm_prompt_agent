"""Generation metrics — session tracking, deletion logging, graveyard ops."""

import logging
import uuid
from datetime import datetime, timezone, timedelta

from sqlalchemy import text

from src.models.database import get_db

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TIMEOUT_MINUTES = 30


def resolve_session(session_id_from_request: str | None) -> str:
    """Validate or create a generation session. Returns valid session_id."""
    from src.models.settings import get_setting

    timeout_min = int(get_setting("session_timeout_minutes") or DEFAULT_SESSION_TIMEOUT_MINUTES)
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_min)

    if session_id_from_request:
        with get_db() as conn:
            row = conn.execute(
                text("SELECT id, last_activity_at FROM generation_sessions WHERE id = :id"),
                {"id": session_id_from_request},
            ).fetchone()

        if row:
            last_activity = row._mapping["last_activity_at"]
            if _parse_timestamp(last_activity) > cutoff:
                with get_db() as conn:
                    conn.execute(
                        text("""UPDATE generation_sessions
                                SET last_activity_at = CURRENT_TIMESTAMP,
                                    generation_count = generation_count + 1
                                WHERE id = :id"""),
                        {"id": session_id_from_request},
                    )
                return session_id_from_request

    # Create new session
    new_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            text("""INSERT INTO generation_sessions (id, generation_count)
                    VALUES (:id, 1)"""),
            {"id": new_id},
        )
    return new_id


def _parse_timestamp(ts) -> datetime:
    """Parse SQLite timestamp string to datetime."""
    if isinstance(ts, str):
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def record_deletion(job_id: str, image_id: int, positive_prompt: str | None,
                    output_folder: str | None, session_id: str | None,
                    lineage_depth: int, reason: str):
    """Record a deletion event in the deletion_log."""
    with get_db() as conn:
        conn.execute(
            text("""INSERT INTO deletion_log
                    (job_id, image_id, positive_prompt, output_folder,
                     session_id, lineage_depth, reason)
                    VALUES (:job_id, :image_id, :prompt, :folder,
                            :session_id, :depth, :reason)"""),
            {
                "job_id": job_id, "image_id": image_id,
                "prompt": positive_prompt, "folder": output_folder,
                "session_id": session_id, "depth": lineage_depth,
                "reason": reason,
            },
        )


def move_to_graveyard(doc_id: str, reason: str) -> bool:
    """Copy a generated prompt's embedding to the deleted_prompts graveyard.

    Only called for quality/wrong_direction deletions.
    The doc_id is either 'gen_{job_id}' or a file_path.
    Returns True if the doc was found and copied.
    """
    from src.models import vector_store

    collection = vector_store._generated_collection
    try:
        result = collection.get(
            ids=[doc_id],
            include=["embeddings", "metadatas", "documents"],
        )
    except Exception:
        return False  # Doc may already be gone

    if not result["ids"]:
        return False  # Not found

    graveyard = vector_store._deleted_collection
    metadata = result["metadatas"][0] if result["metadatas"] else {}
    metadata["deletion_reason"] = reason

    try:
        graveyard.add(
            ids=[doc_id],
            embeddings=result["embeddings"],
            documents=result["documents"],
            metadatas=[metadata],
        )
        return True
    except Exception:
        logger.warning("Failed to add %s to graveyard", doc_id, exc_info=True)
        return False


def get_overall_stats() -> dict:
    """Aggregate stats for the stats overlay."""
    with get_db() as conn:
        total_gens = conn.execute(
            text("SELECT COUNT(*) as cnt FROM generation_jobs WHERE status = 'completed'")
        ).fetchone()._mapping["cnt"]

        del_rows = conn.execute(
            text("SELECT reason, COUNT(*) as cnt FROM deletion_log GROUP BY reason")
        ).fetchall()
        deletions = {r._mapping["reason"]: r._mapping["cnt"] for r in del_rows}

        session_count = conn.execute(
            text("SELECT COUNT(*) as cnt FROM generation_sessions")
        ).fetchone()._mapping["cnt"]

        avg_session = conn.execute(
            text("SELECT AVG(generation_count) as avg FROM generation_sessions WHERE generation_count > 0")
        ).fetchone()._mapping["avg"] or 0

        top_lineage = conn.execute(
            text("""SELECT gs.positive_prompt, gj.lineage_depth
                    FROM generation_jobs gj
                    JOIN generation_settings gs ON gs.job_id = gj.id
                    WHERE gj.lineage_depth > 0
                    ORDER BY gj.lineage_depth DESC LIMIT 5""")
        ).fetchall()

    return {
        "total_generations": total_gens,
        "deletions_by_reason": deletions,
        "session_count": session_count,
        "avg_session_generations": round(avg_session, 1),
        "top_lineage": [
            {"prompt": r._mapping["positive_prompt"][:100], "depth": r._mapping["lineage_depth"]}
            for r in top_lineage
        ],
    }
