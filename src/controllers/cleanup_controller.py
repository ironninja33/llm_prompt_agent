"""Cleanup controller — orchestrate triage, deletion, keep-flagging, and batch scoring."""

import logging
import threading

from src.services import cleanup_service
from src.models import scoring as scoring_model

logger = logging.getLogger(__name__)


def get_folder_summary() -> list[dict]:
    """Get all output folders with image count, disk size, and cleanup score."""
    return cleanup_service.get_folder_summary()


def get_triage_page(folder: str | None, wave: int, offset: int,
                    limit: int, dupe_groups: list[dict] | None = None) -> dict:
    """Return paginated images for a wave/folder with keep scores.

    Images are returned in keep_score ascending order (most deletable first).
    Each image dict includes job_id, image_id, filename, file_path, file_size,
    output_folder, keep_score, wave, seed, positive_prompt (truncated), and
    llm_overall (if scored).
    """
    all_scores = cleanup_service.compute_keep_scores(
        folder_filter=folder, dupe_groups=dupe_groups)

    # Filter to requested wave
    wave_images = [s for s in all_scores if s["wave"] == wave]

    total = len(wave_images)
    page = wave_images[offset:offset + limit]

    return {
        "images": page,
        "total_count": total,
        "has_more": (offset + limit) < total,
        "wave": wave,
        "folder": folder,
    }


def get_wave_counts(folder: str | None = None) -> dict:
    """Get image counts per wave for the wave tabs."""
    all_scores = cleanup_service.compute_keep_scores(folder_filter=folder)
    counts = {1: 0, 2: 0, 3: 0}
    for s in all_scores:
        w = s.get("wave")
        if w in counts:
            counts[w] += 1
    return counts


def delete_images(image_ids: list[int], reason: str = "space") -> dict:
    """Delete images from disk, DB, and ChromaDB."""
    return cleanup_service.delete_images(image_ids, reason=reason)


def delete_visible(folder: str | None, wave: int) -> dict:
    """Delete all images visible in the current folder+wave filter."""
    all_scores = cleanup_service.compute_keep_scores(folder_filter=folder)
    target_ids = [s["image_id"] for s in all_scores if s["wave"] == wave]
    if not target_ids:
        return {"deleted_count": 0, "freed_bytes": 0, "errors": []}
    return cleanup_service.delete_images(target_ids)


def keep_images(image_ids: list[int]):
    """Flag images as explicitly kept (excluded from triage)."""
    scoring_model.flag_keep(image_ids)


def unkeep_images(image_ids: list[int]):
    """Remove keep flags from images."""
    scoring_model.unflag_keep(image_ids)


def start_parse():
    """Launch background metadata parse thread."""
    cleanup_service.start_bulk_parse()


def get_parse_status() -> dict:
    """Return {running, parsed, total}."""
    return cleanup_service.get_parse_progress()


def get_near_duplicates(folder: str | None = None) -> list[dict]:
    """Get near-duplicate image groups."""
    return cleanup_service.detect_near_duplicates(folder_filter=folder)


def get_triage_data(folder: str | None, wave: int) -> dict:
    """Get triage images and near-duplicate groups in one call.

    Computes dupe groups once, then passes them to keep-score computation
    so the expensive dupe detection doesn't run twice.
    """
    dupe_groups = cleanup_service.detect_near_duplicates(folder_filter=folder)

    all_scores = cleanup_service.compute_keep_scores(
        folder_filter=folder, dupe_groups=dupe_groups)
    wave_images = [s for s in all_scores if s["wave"] == wave]

    # Wave counts from all_scores (all waves)
    wave_counts = {1: 0, 2: 0, 3: 0}
    for s in all_scores:
        w = s.get("wave")
        if w in wave_counts:
            wave_counts[w] += 1

    return {
        "images": wave_images,
        "dupe_groups": dupe_groups,
        "wave_counts": wave_counts,
    }


def get_batch_status() -> dict | None:
    """Get the most recent scoring batch status."""
    batch = scoring_model.get_most_recent_batch()
    if not batch:
        return None
    return {
        "id": batch["id"],
        "batch_id": batch["batch_id"],
        "status": batch["status"],
        "total_images": batch["total_images"],
        "scored_count": batch["scored_count"],
        "submitted_at": batch.get("submitted_at"),
        "completed_at": batch.get("completed_at"),
    }


def get_active_batch() -> dict | None:
    """Get the active (non-terminal) scoring batch, if any."""
    batch = scoring_model.get_active_batch()
    if not batch:
        return None
    return {
        "id": batch["id"],
        "batch_id": batch["batch_id"],
        "status": batch["status"],
        "total_images": batch["total_images"],
        "scored_count": batch["scored_count"],
    }


def submit_scoring(mode: str, wave: int,
                   folder: str | None = None) -> dict:
    """Submit a scoring batch.

    mode: 'near-dupes' (score only near-duplicate groups) or 'all' (score all in wave).
    wave: which wave to score.
    """
    from src.services import scoring_service

    # Check for active batch (wave gating)
    active = scoring_model.get_active_batch()
    if active:
        return {"error": "A scoring batch is already active. Wait for it to complete.",
                "batch_id": active["id"]}

    # Determine which image IDs to score
    all_scores = cleanup_service.compute_keep_scores(folder_filter=folder)
    wave_images = [s for s in all_scores if s["wave"] == wave]

    if mode == "near-dupes":
        near_dupes = cleanup_service.detect_near_duplicates(folder_filter=folder)
        dupe_ids = set()
        for group in near_dupes:
            dupe_ids.update(group["image_ids"])
        image_ids = [s["image_id"] for s in wave_images if s["image_id"] in dupe_ids]
    else:
        image_ids = [s["image_id"] for s in wave_images]

    if not image_ids:
        return {"error": "No images to score in this wave/mode"}

    # Submit in background thread
    def _run():
        scoring_service.submit_scoring_batch(image_ids)

    thread = threading.Thread(target=_run, daemon=True, name="scoring-batch")
    thread.start()

    return {"ok": True, "image_count": len(image_ids)}


def poll_scoring_batch() -> dict:
    """Poll the most recent scoring batch for status updates."""
    from src.services import scoring_service

    batch = scoring_model.get_most_recent_batch()
    if not batch:
        return {"status": "none"}

    if batch["status"] in ("completed", "failed"):
        return {
            "status": batch["status"],
            "scored_count": batch["scored_count"],
            "total_images": batch["total_images"],
        }

    return scoring_service.poll_batch_status(batch["id"])


def get_scoring_progress() -> dict:
    """Get upload/submission progress for the current scoring batch."""
    from src.services import scoring_service
    return scoring_service.get_scoring_progress()
