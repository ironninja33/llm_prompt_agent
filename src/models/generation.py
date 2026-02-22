"""Generation job, image, and settings models for ComfyUI integration."""

import json
import uuid
import logging
from src.models.database import get_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: deserialize settings row
# ---------------------------------------------------------------------------

def _deserialize_settings(row) -> dict:
    """Convert a generation_settings Row to a dict with JSON fields parsed."""
    settings = dict(row)
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

def create_job(chat_id: str, message_id: int | None, settings: dict) -> dict:
    """Create a new generation job with its settings.

    Args:
        chat_id: The chat this job belongs to.
        message_id: Optional associated message ID.
        settings: Dict with keys: positive_prompt, negative_prompt, base_model,
                  loras (list), output_folder, seed, num_images, workflow_name,
                  extra_settings (dict).

    Returns:
        The created job dict including id, chat_id, message_id, status, and settings.
    """
    job_id = str(uuid.uuid4())
    loras_json = json.dumps(settings.get("loras", [])) if settings.get("loras") is not None else None
    extra_json = json.dumps(settings.get("extra_settings", {})) if settings.get("extra_settings") is not None else None

    with get_db() as conn:
        conn.execute(
            """INSERT INTO generation_jobs (id, chat_id, message_id, status)
               VALUES (?, ?, ?, 'pending')""",
            (job_id, chat_id, message_id),
        )
        conn.execute(
            """INSERT INTO generation_settings
               (job_id, positive_prompt, negative_prompt, base_model, loras,
                output_folder, seed, num_images, workflow_name, extra_settings)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                settings.get("positive_prompt", ""),
                settings.get("negative_prompt"),
                settings.get("base_model"),
                loras_json,
                settings.get("output_folder"),
                settings.get("seed", -1),
                settings.get("num_images", 1),
                settings.get("workflow_name"),
                extra_json,
            ),
        )

    return {
        "id": job_id,
        "chat_id": chat_id,
        "message_id": message_id,
        "status": "pending",
        "prompt_id": None,
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
        },
    }


def get_job(job_id: str) -> dict | None:
    """Get a single generation job by ID, including its settings."""
    with get_db() as conn:
        cursor = conn.execute(
            """SELECT id, chat_id, message_id, prompt_id, status, created_at, completed_at
               FROM generation_jobs WHERE id = ?""",
            (job_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        job = dict(row)

        # Attach settings
        sc = conn.execute(
            """SELECT positive_prompt, negative_prompt, base_model, loras,
                      output_folder, seed, num_images, workflow_name, extra_settings
               FROM generation_settings WHERE job_id = ?""",
            (job_id,),
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
                cursor = conn.execute(
                    """UPDATE generation_jobs
                       SET status = ?, prompt_id = ?, completed_at = CURRENT_TIMESTAMP
                       WHERE id = ?""",
                    (status, prompt_id, job_id),
                )
            else:
                cursor = conn.execute(
                    """UPDATE generation_jobs
                       SET status = ?, completed_at = CURRENT_TIMESTAMP
                       WHERE id = ?""",
                    (status, job_id),
                )
        else:
            if prompt_id is not None:
                cursor = conn.execute(
                    """UPDATE generation_jobs SET status = ?, prompt_id = ? WHERE id = ?""",
                    (status, prompt_id, job_id),
                )
            else:
                cursor = conn.execute(
                    """UPDATE generation_jobs SET status = ? WHERE id = ?""",
                    (status, job_id),
                )
        return cursor.rowcount > 0


def get_jobs_for_chat(chat_id: str) -> list[dict]:
    """Get all generation jobs for a chat, including settings and images.

    Ordered by created_at ASC. Each job dict includes:
    - job fields (id, chat_id, message_id, prompt_id, status, created_at, completed_at)
    - settings (nested dict)
    - images (list of dicts)
    """
    with get_db() as conn:
        cursor = conn.execute(
            """SELECT id, chat_id, message_id, prompt_id, status, created_at, completed_at
               FROM generation_jobs WHERE chat_id = ? ORDER BY created_at ASC""",
            (chat_id,),
        )
        jobs = []
        for row in cursor.fetchall():
            job = dict(row)

            # Attach settings
            sc = conn.execute(
                """SELECT positive_prompt, negative_prompt, base_model, loras,
                          output_folder, seed, num_images, workflow_name, extra_settings
                   FROM generation_settings WHERE job_id = ?""",
                (job["id"],),
            )
            srow = sc.fetchone()
            job["settings"] = _deserialize_settings(srow) if srow else {}

            # Attach images
            ic = conn.execute(
                """SELECT id, job_id, filename, subfolder, width, height, created_at
                   FROM generated_images WHERE job_id = ? ORDER BY id ASC""",
                (job["id"],),
            )
            job["images"] = [dict(irow) for irow in ic.fetchall()]

            jobs.append(job)
        return jobs


def delete_job(job_id: str) -> bool:
    """Delete a generation job (cascades to images and settings).

    Returns True if found and deleted.
    """
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM generation_jobs WHERE id = ?", (job_id,))
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Image operations
# ---------------------------------------------------------------------------

def add_generated_image(
    job_id: str,
    filename: str,
    subfolder: str = "",
    width: int | None = None,
    height: int | None = None,
) -> dict:
    """Add a generated image record to a job. Returns the created image dict."""
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO generated_images (job_id, filename, subfolder, width, height)
               VALUES (?, ?, ?, ?, ?)""",
            (job_id, filename, subfolder, width, height),
        )
        return {
            "id": cursor.lastrowid,
            "job_id": job_id,
            "filename": filename,
            "subfolder": subfolder,
            "width": width,
            "height": height,
            "created_at": None,
        }


def get_job_images(job_id: str) -> list[dict]:
    """Get all images for a job."""
    with get_db() as conn:
        cursor = conn.execute(
            """SELECT id, job_id, filename, subfolder, width, height, created_at
               FROM generated_images WHERE job_id = ? ORDER BY id ASC""",
            (job_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_image(image_id: int) -> dict | None:
    """Get a single image by ID."""
    with get_db() as conn:
        cursor = conn.execute(
            """SELECT id, job_id, filename, subfolder, width, height, created_at
               FROM generated_images WHERE id = ?""",
            (image_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_image(image_id: int) -> bool:
    """Delete a generated image record. Returns True if found and deleted."""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM generated_images WHERE id = ?", (image_id,))
        return cursor.rowcount > 0


def delete_images_by_filename(filename: str) -> int:
    """Delete all image records matching a filename. Returns count deleted.

    Used for cleanup when images are missing from disk.
    """
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM generated_images WHERE filename = ?", (filename,))
        return cursor.rowcount


# ---------------------------------------------------------------------------
# Settings operations
# ---------------------------------------------------------------------------

def get_job_settings(job_id: str) -> dict | None:
    """Get generation settings for a job.

    Deserializes loras from JSON to list and extra_settings from JSON to dict.
    """
    with get_db() as conn:
        cursor = conn.execute(
            """SELECT positive_prompt, negative_prompt, base_model, loras,
                      output_folder, seed, num_images, workflow_name, extra_settings
               FROM generation_settings WHERE job_id = ?""",
            (job_id,),
        )
        row = cursor.fetchone()
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
        cursor = conn.execute(
            """SELECT id, chat_id, message_id, prompt_id, status, created_at, completed_at
               FROM generation_jobs WHERE chat_id = ?
               ORDER BY message_id ASC, created_at ASC""",
            (chat_id,),
        )
        jobs = []
        for row in cursor.fetchall():
            job = dict(row)

            # Attach settings
            sc = conn.execute(
                """SELECT positive_prompt, negative_prompt, base_model, loras,
                          output_folder, seed, num_images, workflow_name, extra_settings
                   FROM generation_settings WHERE job_id = ?""",
                (job["id"],),
            )
            srow = sc.fetchone()
            job["settings"] = _deserialize_settings(srow) if srow else {}

            # Attach images
            ic = conn.execute(
                """SELECT id, job_id, filename, subfolder, width, height, created_at
                   FROM generated_images WHERE job_id = ? ORDER BY id ASC""",
                (job["id"],),
            )
            job["images"] = [dict(irow) for irow in ic.fetchall()]

            jobs.append(job)
        return jobs


def get_active_jobs() -> list[dict]:
    """Get all jobs with status 'pending', 'queued', or 'running'.

    Used to resume polling on app restart.
    """
    with get_db() as conn:
        cursor = conn.execute(
            """SELECT id, chat_id, message_id, prompt_id, status, created_at, completed_at
               FROM generation_jobs WHERE status IN ('pending', 'queued', 'running')
               ORDER BY created_at ASC""",
        )
        jobs = []
        for row in cursor.fetchall():
            job = dict(row)

            # Attach settings
            sc = conn.execute(
                """SELECT positive_prompt, negative_prompt, base_model, loras,
                          output_folder, seed, num_images, workflow_name, extra_settings
                   FROM generation_settings WHERE job_id = ?""",
                (job["id"],),
            )
            srow = sc.fetchone()
            job["settings"] = _deserialize_settings(srow) if srow else {}

            jobs.append(job)
        return jobs
