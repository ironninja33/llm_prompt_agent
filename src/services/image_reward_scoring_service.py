"""ImageReward scoring service -- local GPU scoring with progress tracking."""

import logging
import threading

from PIL import Image
from sqlalchemy import text

from src.models import scoring as scoring_model
from src.models.database import get_db

logger = logging.getLogger(__name__)

_scoring_lock = threading.Lock()
_scoring_progress = {
    "running": False,
    "phase": "idle",          # idle | loading_model | scoring | unloading | completed | failed
    "scored": 0,
    "total": 0,
    "current_folder": "",
    "error": None,
}


def get_scoring_progress() -> dict:
    """Return current scoring progress snapshot."""
    return dict(_scoring_progress)


def get_unscored_images() -> list[dict]:
    """Query output images with no entry in image_quality_scores.

    Returns [{image_id, file_path, positive_prompt, output_folder}]
    ordered by output_folder, id.
    """
    with get_db() as conn:
        result = conn.execute(text("""
            SELECT gi.id as image_id, gi.file_path,
                   gs.positive_prompt, gs.output_folder
            FROM generated_images gi
            JOIN generation_jobs gj ON gi.job_id = gj.id
            LEFT JOIN generation_settings gs ON gs.job_id = gi.job_id
            LEFT JOIN image_quality_scores iqs ON iqs.image_id = gi.id
            WHERE gi.file_path IS NOT NULL
              AND iqs.image_id IS NULL
            ORDER BY gs.output_folder, gi.id
        """))
        return [dict(row._mapping) for row in result.fetchall()]


def start_scoring_all_unscored():
    """Acquire lock and spawn background thread to score all unscored images.

    No-op if scoring is already in progress.
    """
    if not _scoring_lock.acquire(blocking=False):
        logger.info("Scoring already in progress")
        return

    _scoring_progress["running"] = True
    _scoring_progress["phase"] = "idle"
    _scoring_progress["scored"] = 0
    _scoring_progress["total"] = 0
    _scoring_progress["current_folder"] = ""
    _scoring_progress["error"] = None

    thread = threading.Thread(target=_scoring_thread, daemon=True,
                              name="imagereward-scoring")
    thread.start()


def _scoring_thread():
    """Background thread: load model, score all unscored images, unload model."""
    assessor = None
    try:
        _scoring_progress["phase"] = "loading_model"

        unscored = get_unscored_images()
        _scoring_progress["total"] = len(unscored)
        _scoring_progress["scored"] = 0

        if not unscored:
            _scoring_progress["phase"] = "completed"
            return

        # Import and load ImageReward model
        from src.experiments.image_quality.assessors.image_reward import (
            ImageRewardAssessor,
        )

        assessor = ImageRewardAssessor()
        assessor.load_model(device="cuda")

        _scoring_progress["phase"] = "scoring"

        for item in unscored:
            folder = (item["output_folder"] or "").split("/")[0] or "(root)"
            _scoring_progress["current_folder"] = folder

            try:
                img = Image.open(item["file_path"])
                img.load()
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                result = assessor.score(img, prompt=item["positive_prompt"])
                img.close()

                scoring_model.upsert_quality_score(
                    image_id=item["image_id"],
                    overall=result.normalized_score,
                    raw_score=result.raw_score,
                    model_used="imagereward-v1.0",
                )
            except Exception as e:
                logger.warning("Failed to score image %d: %s",
                               item["image_id"], e)

            _scoring_progress["scored"] += 1

        # Unload model
        _scoring_progress["phase"] = "unloading"
        assessor.unload_model()
        assessor = None

        _scoring_progress["phase"] = "completed"
        logger.info("ImageReward scoring completed: %d images",
                     _scoring_progress["scored"])

    except Exception as e:
        logger.exception("Scoring thread failed: %s", e)
        _scoring_progress["phase"] = "failed"
        _scoring_progress["error"] = str(e)
        # Ensure model is unloaded on failure
        if assessor is not None:
            try:
                assessor.unload_model()
            except Exception:
                pass
    finally:
        _scoring_progress["running"] = False
        _scoring_lock.release()
