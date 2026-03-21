"""Generation job, image, and settings models for ComfyUI integration."""

import json
import os
import uuid
import logging
from sqlalchemy import text
from src.models.database import get_db, row_to_dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: deserialize settings row
# ---------------------------------------------------------------------------

def _deserialize_settings(row) -> dict:
    """Convert a generation_settings Row to a dict with JSON fields parsed."""
    settings = row_to_dict(row)
    # Parse loras from JSON string to list
    if settings.get("loras"):
        try:
            settings["loras"] = json.loads(settings["loras"])
        except (json.JSONDecodeError, TypeError):
            settings["loras"] = []
    else:
        settings["loras"] = []
    # Parse extra_settings from JSON string to dict
    if settings.get("extra_settings"):
        try:
            settings["extra_settings"] = json.loads(settings["extra_settings"])
        except (json.JSONDecodeError, TypeError):
            settings["extra_settings"] = {}
    else:
        settings["extra_settings"] = {}
    return settings


# ---------------------------------------------------------------------------
# Job operations
# ---------------------------------------------------------------------------

def create_job(chat_id: str | None, message_id: int | None, settings: dict,
               source: str = "chat", session_id: str | None = None,
               parent_job_id: str | None = None, lineage_depth: int = 0) -> dict:
    """Create a new generation job with its settings.

    Args:
        chat_id: The chat this job belongs to (None for browser/scan).
        message_id: Optional associated message ID.
        settings: Dict with keys: positive_prompt, negative_prompt, base_model,
                  loras (list), output_folder, seed, num_images, workflow_name,
                  extra_settings (dict), sampler, cfg_scale, scheduler, steps.
        source: One of 'chat', 'scan', 'browser'.
        session_id: Generation session ID for session tracking.
        parent_job_id: Parent job ID for regeneration lineage.
        lineage_depth: How many regenerations deep this job is.

    Returns:
        The created job dict including id, chat_id, message_id, status, and settings.
    """
    job_id = str(uuid.uuid4())
    loras_json = json.dumps(settings.get("loras", [])) if settings.get("loras") is not None else None
    extra_json = json.dumps(settings.get("extra_settings", {})) if settings.get("extra_settings") is not None else None

    with get_db() as conn:
        conn.execute(
            text("""INSERT INTO generation_jobs
                    (id, chat_id, message_id, status, source,
                     session_id, parent_job_id, lineage_depth)
               VALUES (:id, :chat_id, :message_id, 'pending', :source,
                       :session_id, :parent_job_id, :lineage_depth)"""),
            {"id": job_id, "chat_id": chat_id, "message_id": message_id,
             "source": source, "session_id": session_id,
             "parent_job_id": parent_job_id, "lineage_depth": lineage_depth},
        )
        conn.execute(
            text("""INSERT INTO generation_settings
               (job_id, positive_prompt, negative_prompt, base_model, loras,
                output_folder, seed, num_images, workflow_name, extra_settings,
                sampler, cfg_scale, scheduler, steps)
               VALUES (:job_id, :positive_prompt, :negative_prompt, :base_model, :loras,
                :output_folder, :seed, :num_images, :workflow_name, :extra_settings,
                :sampler, :cfg_scale, :scheduler, :steps)"""),
            {
                "job_id": job_id,
                "positive_prompt": settings.get("positive_prompt", ""),
                "negative_prompt": settings.get("negative_prompt"),
                "base_model": settings.get("base_model"),
                "loras": loras_json,
                "output_folder": settings.get("output_folder"),
                "seed": settings.get("seed", -1),
                "num_images": settings.get("num_images", 1),
                "workflow_name": settings.get("workflow_name"),
                "extra_settings": extra_json,
                "sampler": settings.get("sampler"),
                "cfg_scale": settings.get("cfg_scale"),
                "scheduler": settings.get("scheduler"),
                "steps": settings.get("steps"),
            },
        )

    return {
        "id": job_id,
        "chat_id": chat_id,
        "message_id": message_id,
        "status": "pending",
        "source": source,
        "prompt_id": None,
        "session_id": session_id,
        "parent_job_id": parent_job_id,
        "lineage_depth": lineage_depth,
        "created_at": None,
        "completed_at": None,
        "settings": {
            "positive_prompt": settings.get("positive_prompt", ""),
            "negative_prompt": settings.get("negative_prompt"),
            "base_model": settings.get("base_model"),
            "loras": settings.get("loras", []),
            "output_folder": settings.get("output_folder"),
            "seed": settings.get("seed", -1),
            "num_images": settings.get("num_images", 1),
            "workflow_name": settings.get("workflow_name"),
            "extra_settings": settings.get("extra_settings", {}),
            "sampler": settings.get("sampler"),
            "cfg_scale": settings.get("cfg_scale"),
            "scheduler": settings.get("scheduler"),
            "steps": settings.get("steps"),
        },
    }


def get_job(job_id: str) -> dict | None:
    """Get a single generation job by ID, including its settings."""
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT id, chat_id, message_id, prompt_id, status, source,
                      session_id, parent_job_id, lineage_depth,
                      created_at, completed_at
               FROM generation_jobs WHERE id = :id"""),
            {"id": job_id},
        )
        row = result.fetchone()
        if not row:
            return None

        job = row_to_dict(row)

        # Attach settings
        sc = conn.execute(
            text("""SELECT positive_prompt, negative_prompt, base_model, loras,
                      output_folder, seed, num_images, workflow_name, extra_settings,
                      sampler, cfg_scale, scheduler, steps
               FROM generation_settings WHERE job_id = :job_id"""),
            {"job_id": job_id},
        )
        srow = sc.fetchone()
        job["settings"] = _deserialize_settings(srow) if srow else {}

        return job


def update_job_status(job_id: str, status: str, prompt_id: str | None = None) -> bool:
    """Update job status and optionally the ComfyUI prompt_id.

    If status is 'completed' or 'failed', sets completed_at to now.
    Returns True if job was found and updated.
    """
    with get_db() as conn:
        if status in ("completed", "failed"):
            if prompt_id is not None:
                result = conn.execute(
                    text("""UPDATE generation_jobs
                       SET status = :status, prompt_id = :prompt_id, completed_at = CURRENT_TIMESTAMP
                       WHERE id = :id"""),
                    {"status": status, "prompt_id": prompt_id, "id": job_id},
                )
            else:
                result = conn.execute(
                    text("""UPDATE generation_jobs
                       SET status = :status, completed_at = CURRENT_TIMESTAMP
                       WHERE id = :id"""),
                    {"status": status, "id": job_id},
                )
        else:
            if prompt_id is not None:
                result = conn.execute(
                    text("UPDATE generation_jobs SET status = :status, prompt_id = :prompt_id WHERE id = :id"),
                    {"status": status, "prompt_id": prompt_id, "id": job_id},
                )
            else:
                result = conn.execute(
                    text("UPDATE generation_jobs SET status = :status WHERE id = :id"),
                    {"status": status, "id": job_id},
                )
        return result.rowcount > 0


def update_job_message_id(job_id: str, message_id: int) -> bool:
    """Set the message_id on a generation job (used to associate agent-triggered
    jobs with the assistant message after it's saved to the DB).

    Returns True if job was found and updated.
    """
    with get_db() as conn:
        result = conn.execute(
            text("UPDATE generation_jobs SET message_id = :message_id WHERE id = :id"),
            {"message_id": message_id, "id": job_id},
        )
        return result.rowcount > 0


def _backfill_orphan_message_ids(conn, chat_id: str):
    """Associate orphan generation jobs (message_id IS NULL) with the nearest
    preceding assistant message by timestamp.  Updates the DB in-place so the
    fix is permanent.
    """
    orphans = conn.execute(
        text("""SELECT id, created_at FROM generation_jobs
           WHERE chat_id = :chat_id AND message_id IS NULL AND source = 'chat'
           ORDER BY created_at ASC"""),
        {"chat_id": chat_id},
    ).fetchall()
    if not orphans:
        return

    # Load assistant messages for this chat ordered by creation time
    assistants = conn.execute(
        text("""SELECT id, created_at FROM messages
           WHERE chat_id = :chat_id AND role = 'assistant'
           ORDER BY created_at ASC"""),
        {"chat_id": chat_id},
    ).fetchall()
    if not assistants:
        return

    for orphan in orphans:
        o = orphan._mapping
        # Find the latest assistant message created before (or at same time as) the job
        best = None
        for a in assistants:
            am = a._mapping
            if am["created_at"] <= o["created_at"]:
                best = am
            else:
                break
        if best:
            conn.execute(
                text("UPDATE generation_jobs SET message_id = :message_id WHERE id = :id"),
                {"message_id": best["id"], "id": o["id"]},
            )
    logger.info("Backfilled %d orphan generation jobs for chat %s", len(orphans), chat_id)


def get_jobs_for_chat(chat_id: str) -> list[dict]:
    """Get all generation jobs for a chat, including settings and images.

    Ordered by created_at ASC. Each job dict includes:
    - job fields (id, chat_id, message_id, prompt_id, status, source, created_at, completed_at)
    - settings (nested dict)
    - images (list of dicts)
    """
    with get_db() as conn:
        # Auto-fix orphan jobs that lost their message_id
        _backfill_orphan_message_ids(conn, chat_id)

        result = conn.execute(
            text("""SELECT id, chat_id, message_id, prompt_id, status, source, created_at, completed_at
               FROM generation_jobs WHERE chat_id = :chat_id ORDER BY created_at ASC"""),
            {"chat_id": chat_id},
        )
        jobs = []
        for row in result.fetchall():
            job = row_to_dict(row)

            # Attach settings
            sc = conn.execute(
                text("""SELECT positive_prompt, negative_prompt, base_model, loras,
                          output_folder, seed, num_images, workflow_name, extra_settings,
                          sampler, cfg_scale, scheduler, steps
                   FROM generation_settings WHERE job_id = :job_id"""),
                {"job_id": job["id"]},
            )
            srow = sc.fetchone()
            job["settings"] = _deserialize_settings(srow) if srow else {}

            # Attach images
            ic = conn.execute(
                text("""SELECT id, job_id, filename, subfolder, width, height, created_at,
                          file_size, file_path
                   FROM generated_images WHERE job_id = :job_id ORDER BY id ASC"""),
                {"job_id": job["id"]},
            )
            job["images"] = [row_to_dict(irow) for irow in ic.fetchall()]

            jobs.append(job)
        return jobs


def delete_job(job_id: str) -> bool:
    """Delete a generation job (cascades to images and settings).

    Returns True if found and deleted.
    """
    with get_db() as conn:
        result = conn.execute(
            text("DELETE FROM generation_jobs WHERE id = :id"),
            {"id": job_id},
        )
        return result.rowcount > 0


# ---------------------------------------------------------------------------
# Image operations
# ---------------------------------------------------------------------------

def add_generated_image(
    job_id: str,
    filename: str,
    subfolder: str = "",
    width: int | None = None,
    height: int | None = None,
    file_path: str | None = None,
) -> dict:
    """Add a generated image record to a job. Returns the created image dict.

    Typically called with file_path=NULL immediately on generation completion,
    then updated via update_image_file_path() once the path is resolved.
    Uses INSERT OR IGNORE for the (job_id, filename) unique index.
    """
    with get_db() as conn:
        result = conn.execute(
            text("""INSERT OR IGNORE INTO generated_images
                   (job_id, filename, subfolder, width, height, file_path)
               VALUES (:job_id, :filename, :subfolder, :width, :height, :file_path)"""),
            {"job_id": job_id, "filename": filename, "subfolder": subfolder,
             "width": width, "height": height, "file_path": file_path},
        )
        image_id = result.lastrowid

    return {
        "id": image_id,
        "job_id": job_id,
        "filename": filename,
        "subfolder": subfolder,
        "width": width,
        "height": height,
        "file_path": file_path,
        "created_at": None,
    }


def update_image_file_path(job_id: str, filename: str, file_path: str) -> bool:
    """Set file_path on an existing generated_images record.

    Called after resolve_image_path succeeds.  Idempotent — safe to call
    even if fast_register_images already set the path.

    Returns True if a record was updated.
    """
    with get_db() as conn:
        result = conn.execute(
            text("""UPDATE generated_images
                   SET file_path = :file_path
                   WHERE job_id = :job_id AND filename = :filename
                   AND (file_path IS NULL OR file_path = :file_path)"""),
            {"file_path": file_path, "job_id": job_id, "filename": filename},
        )
        return result.rowcount > 0


def get_job_images(job_id: str) -> list[dict]:
    """Get all images for a job."""
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT id, job_id, filename, subfolder, file_path, file_size,
                      width, height, created_at
               FROM generated_images WHERE job_id = :job_id ORDER BY id ASC"""),
            {"job_id": job_id},
        )
        return [row_to_dict(row) for row in result.fetchall()]


def get_image(image_id: int) -> dict | None:
    """Get a single image by ID."""
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT id, job_id, filename, subfolder, width, height,
                      file_size, file_path, created_at
               FROM generated_images WHERE id = :id"""),
            {"id": image_id},
        )
        row = result.fetchone()
        return row_to_dict(row) if row else None


def delete_image(image_id: int) -> bool:
    """Delete a generated image record. Returns True if found and deleted."""
    with get_db() as conn:
        result = conn.execute(
            text("DELETE FROM generated_images WHERE id = :id"),
            {"id": image_id},
        )
        return result.rowcount > 0


def get_images_by_filename(filename: str) -> list[dict]:
    """Get all image records matching a filename."""
    with get_db() as conn:
        result = conn.execute(
            text("SELECT id, job_id, filename, file_path FROM generated_images WHERE filename = :filename"),
            {"filename": filename},
        )
        return [row_to_dict(row) for row in result.fetchall()]


def delete_images_by_filename(filename: str) -> int:
    """Delete all image records matching a filename. Returns count deleted.

    Used for cleanup when images are missing from disk.
    """
    with get_db() as conn:
        result = conn.execute(
            text("DELETE FROM generated_images WHERE filename = :filename"),
            {"filename": filename},
        )
        return result.rowcount


# ---------------------------------------------------------------------------
# Settings operations
# ---------------------------------------------------------------------------

def get_latest_job_settings(output_folder: str | None = None,
                            chat_id: str | None = None) -> dict | None:
    """Get settings from the most recent completed generation job.

    Args:
        output_folder: If provided, only considers jobs that used this folder.
        chat_id: If provided, only considers jobs from this chat.

    Returns None if no completed jobs match.
    """
    with get_db() as conn:
        conditions = ["gj.status = 'completed'"]
        params = {}
        if output_folder:
            conditions.append("gs.output_folder = :output_folder")
            params["output_folder"] = output_folder
        if chat_id:
            conditions.append("gj.chat_id = :chat_id")
            params["chat_id"] = chat_id

        where = " AND ".join(conditions)
        result = conn.execute(
            text(f"""SELECT gs.positive_prompt, gs.negative_prompt, gs.base_model, gs.loras,
                      gs.output_folder, gs.seed, gs.num_images, gs.workflow_name,
                      gs.extra_settings, gs.sampler, gs.cfg_scale, gs.scheduler, gs.steps
               FROM generation_settings gs
               JOIN generation_jobs gj ON gs.job_id = gj.id
               WHERE {where}
               ORDER BY gj.completed_at DESC LIMIT 1"""),
            params,
        )
        row = result.fetchone()
        if not row:
            return None
        return _deserialize_settings(row)


def get_recent_job_prompts(count: int = 1,
                           chat_id: str | None = None) -> list[dict]:
    """Get prompts and summary settings from the most recent completed jobs.

    Args:
        count: Number of recent jobs to retrieve.
        chat_id: If provided, only considers jobs from this chat.

    Returns list of dicts with keys: job_id, positive_prompt, negative_prompt,
    base_model, loras, output_folder, seed.
    """
    with get_db() as conn:
        conditions = ["gj.status = 'completed'"]
        params = {"limit": count}
        if chat_id:
            conditions.append("gj.chat_id = :chat_id")
            params["chat_id"] = chat_id

        where = " AND ".join(conditions)
        result = conn.execute(
            text(f"""SELECT gj.id as job_id, gs.positive_prompt, gs.negative_prompt,
                      gs.base_model, gs.loras, gs.output_folder, gs.seed,
                      gj.lineage_depth
               FROM generation_settings gs
               JOIN generation_jobs gj ON gs.job_id = gj.id
               WHERE {where}
               ORDER BY gj.completed_at DESC LIMIT :limit"""),
            params,
        )
        results = []
        for row in result.fetchall():
            r = row_to_dict(row)
            if r.get("loras"):
                try:
                    r["loras"] = json.loads(r["loras"])
                except (json.JSONDecodeError, TypeError):
                    r["loras"] = []
            else:
                r["loras"] = []
            results.append(r)
        return results


def get_job_settings(job_id: str) -> dict | None:
    """Get generation settings for a job.

    Deserializes loras from JSON to list and extra_settings from JSON to dict.
    """
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT positive_prompt, negative_prompt, base_model, loras,
                      output_folder, seed, num_images, workflow_name, extra_settings,
                      sampler, cfg_scale, scheduler, steps
               FROM generation_settings WHERE job_id = :job_id"""),
            {"job_id": job_id},
        )
        row = result.fetchone()
        if not row:
            return None
        return _deserialize_settings(row)


# ---------------------------------------------------------------------------
# Chat-level queries
# ---------------------------------------------------------------------------

def get_generation_data_for_chat(chat_id: str) -> list[dict]:
    """Get all generation data for rebuilding a chat's generation bubbles.

    Returns list of jobs with their settings and images, grouped by message_id.
    Used when reloading a chat to reconstruct the generation UI.
    """
    with get_db() as conn:
        # Auto-fix orphan jobs that lost their message_id
        _backfill_orphan_message_ids(conn, chat_id)

        result = conn.execute(
            text("""SELECT id, chat_id, message_id, prompt_id, status, source, created_at, completed_at
               FROM generation_jobs WHERE chat_id = :chat_id
               ORDER BY message_id ASC, created_at ASC"""),
            {"chat_id": chat_id},
        )
        jobs = []
        for row in result.fetchall():
            job = row_to_dict(row)

            # Attach settings
            sc = conn.execute(
                text("""SELECT positive_prompt, negative_prompt, base_model, loras,
                          output_folder, seed, num_images, workflow_name, extra_settings,
                          sampler, cfg_scale, scheduler, steps
                   FROM generation_settings WHERE job_id = :job_id"""),
                {"job_id": job["id"]},
            )
            srow = sc.fetchone()
            job["settings"] = _deserialize_settings(srow) if srow else {}

            # Attach images
            ic = conn.execute(
                text("""SELECT id, job_id, filename, subfolder, width, height, created_at,
                          file_size, file_path
                   FROM generated_images WHERE job_id = :job_id ORDER BY id ASC"""),
                {"job_id": job["id"]},
            )
            job["images"] = [row_to_dict(irow) for irow in ic.fetchall()]

            jobs.append(job)
        return jobs


def get_active_jobs() -> list[dict]:
    """Get all jobs with status 'pending', 'queued', or 'running'.

    Used to resume polling on app restart.
    """
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT id, chat_id, message_id, prompt_id, status, source, created_at, completed_at
               FROM generation_jobs WHERE status IN ('pending', 'queued', 'running')
               ORDER BY created_at ASC"""),
        )
        jobs = []
        for row in result.fetchall():
            job = row_to_dict(row)

            # Attach settings
            sc = conn.execute(
                text("""SELECT positive_prompt, negative_prompt, base_model, loras,
                          output_folder, seed, num_images, workflow_name, extra_settings,
                          sampler, cfg_scale, scheduler, steps
                   FROM generation_settings WHERE job_id = :job_id"""),
                {"job_id": job["id"]},
            )
            srow = sc.fetchone()
            job["settings"] = _deserialize_settings(srow) if srow else {}

            jobs.append(job)
        return jobs


# ---------------------------------------------------------------------------
# Browser queries
# ---------------------------------------------------------------------------

def get_image_by_filepath(file_path: str) -> dict | None:
    """Get a generated_images record by its file_path."""
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT gi.id, gi.job_id, gi.filename, gi.subfolder,
                      gi.width, gi.height, gi.created_at, gi.file_size, gi.file_path
               FROM generated_images gi WHERE gi.file_path = :file_path"""),
            {"file_path": file_path},
        )
        row = result.fetchone()
        return row_to_dict(row) if row else None


def get_images_for_directory(path_prefix: str, offset: int = 0, limit: int = 50,
                             sort: str = "date") -> dict:
    """Get images whose file_path is directly inside the given directory.

    Only matches direct children — files in subdirectories are excluded.

    Sort modes:
        "date" (default): newest first (created_at DESC)
        "name": alphabetical by filename (filename ASC)

    Returns dict with keys: images (list of job+settings+image dicts),
    total_count, has_more.
    """
    # Ensure prefix ends with separator for exact directory match
    if not path_prefix.endswith(os.sep):
        path_prefix = path_prefix + os.sep
    # Match files directly in this dir: path_prefix + "%" but NOT path_prefix + "%/%"
    like_pattern = path_prefix + "%"
    not_like_pattern = path_prefix + "%/%"

    with get_db() as conn:
        # Count total (direct children only)
        result = conn.execute(
            text("""SELECT COUNT(*) as cnt FROM generated_images
               WHERE file_path LIKE :like AND file_path NOT LIKE :not_like"""),
            {"like": like_pattern, "not_like": not_like_pattern},
        )
        total_count = result.fetchone()._mapping["cnt"]

        # Fetch paginated images with their job + settings
        if sort == "name":
            order_clause = "ORDER BY gi.filename ASC"
        else:  # "date" (default)
            order_clause = "ORDER BY gj.created_at DESC"

        result = conn.execute(
            text(f"""SELECT gi.id as image_id, gi.job_id, gi.filename, gi.subfolder,
                      gi.width, gi.height, gi.created_at, gi.file_size, gi.file_path,
                      gi.metadata_status,
                      gj.status, gj.source,
                      gs.positive_prompt, gs.negative_prompt, gs.base_model, gs.loras,
                      gs.output_folder, gs.seed, gs.num_images, gs.sampler, gs.cfg_scale,
                      gs.scheduler, gs.steps
               FROM generated_images gi
               JOIN generation_jobs gj ON gi.job_id = gj.id
               LEFT JOIN generation_settings gs ON gi.job_id = gs.job_id
               WHERE gi.file_path LIKE :like AND gi.file_path NOT LIKE :not_like
               {order_clause}
               LIMIT :limit OFFSET :offset"""),
            {"like": like_pattern, "not_like": not_like_pattern,
             "limit": limit, "offset": offset},
        )

        images = _rows_to_image_list(result.fetchall())

        return {
            "images": images,
            "total_count": total_count,
            "has_more": offset + limit < total_count,
        }


def search_by_keywords(keywords: list[str], offset: int = 0, limit: int = 50) -> dict:
    """Search images by keyword matching against positive_prompt.

    Each keyword is matched via SQL LIKE (case-insensitive).
    Returns same format as get_images_for_directory.
    """
    if not keywords:
        return {"images": [], "total_count": 0, "has_more": False}

    conditions = []
    params = {}
    for i, kw in enumerate(keywords):
        conditions.append(f"gs.positive_prompt LIKE :kw{i}")
        params[f"kw{i}"] = f"%{kw}%"

    where_clause = " AND ".join(conditions)

    with get_db() as conn:
        result = conn.execute(
            text(f"""SELECT COUNT(*) as cnt
                FROM generated_images gi
                JOIN generation_settings gs ON gi.job_id = gs.job_id
                WHERE {where_clause}"""),
            params,
        )
        total_count = result.fetchone()._mapping["cnt"]

        query_params = dict(params)
        query_params["limit"] = limit
        query_params["offset"] = offset
        result = conn.execute(
            text(f"""SELECT gi.id as image_id, gi.job_id, gi.filename, gi.subfolder,
                       gi.width, gi.height, gi.created_at, gi.file_size, gi.file_path,
                       gj.status, gj.source,
                       gs.positive_prompt, gs.negative_prompt, gs.base_model, gs.loras,
                       gs.output_folder, gs.seed, gs.num_images, gs.sampler, gs.cfg_scale,
                       gs.scheduler, gs.steps
                FROM generated_images gi
                JOIN generation_jobs gj ON gi.job_id = gj.id
                JOIN generation_settings gs ON gi.job_id = gs.job_id
                WHERE {where_clause}
                ORDER BY gi.created_at DESC
                LIMIT :limit OFFSET :offset"""),
            query_params,
        )

        images = _rows_to_image_list(result.fetchall())

        return {
            "images": images,
            "total_count": total_count,
            "has_more": offset + limit < total_count,
        }


def get_images_by_job_ids(job_ids: list[str]) -> list[dict]:
    """Get images for a list of job IDs, with their settings."""
    if not job_ids:
        return []

    placeholders = ",".join([f":p{i}" for i in range(len(job_ids))])
    params = {f"p{i}": v for i, v in enumerate(job_ids)}
    with get_db() as conn:
        result = conn.execute(
            text(f"""SELECT gi.id as image_id, gi.job_id, gi.filename, gi.subfolder,
                       gi.width, gi.height, gi.created_at, gi.file_size, gi.file_path,
                       gj.status, gj.source,
                       gs.positive_prompt, gs.negative_prompt, gs.base_model, gs.loras,
                       gs.output_folder, gs.seed, gs.num_images, gs.sampler, gs.cfg_scale,
                       gs.scheduler, gs.steps
                FROM generated_images gi
                JOIN generation_jobs gj ON gi.job_id = gj.id
                LEFT JOIN generation_settings gs ON gi.job_id = gs.job_id
                WHERE gi.job_id IN ({placeholders})
                ORDER BY gi.created_at DESC"""),
            params,
        )

        return _rows_to_image_list(result.fetchall())


def _rows_to_image_list(rows) -> list[dict]:
    """Convert joined query rows to a list of image dicts with nested settings."""
    images = []
    for row in rows:
        r = row_to_dict(row)
        loras = []
        if r.get("loras"):
            try:
                loras = json.loads(r["loras"])
            except (json.JSONDecodeError, TypeError):
                pass
        images.append({
            "id": r["image_id"],
            "job_id": r["job_id"],
            "filename": r["filename"],
            "subfolder": r["subfolder"],
            "width": r["width"],
            "height": r["height"],
            "created_at": r["created_at"],
            "file_size": r["file_size"],
            "file_path": r["file_path"],
            "metadata_status": r.get("metadata_status", "complete"),
            "status": r["status"],
            "source": r["source"],
            "settings": {
                "positive_prompt": r.get("positive_prompt", ""),
                "negative_prompt": r.get("negative_prompt"),
                "base_model": r.get("base_model"),
                "loras": loras,
                "output_folder": r.get("output_folder"),
                "seed": r.get("seed"),
                "num_images": r.get("num_images"),
                "sampler": r.get("sampler"),
                "cfg_scale": r.get("cfg_scale"),
                "scheduler": r.get("scheduler"),
                "steps": r.get("steps"),
            },
        })
    return images
