"""Cleanup controller — orchestrate triage, deletion, keep-flagging, and scoring."""

import logging

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
    quality_overall (if scored).
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


def get_triage_data(folder: str | None, wave: int,
                    sort_order: str = "desc") -> dict:
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

    # Sort wave images
    reverse = (sort_order == "desc")
    wave_images.sort(key=lambda s: s["keep_score"], reverse=reverse)

    # Sort dupe group members by quality score (None goes last)
    for group in dupe_groups:
        group["members"].sort(
            key=lambda m: m.get("quality_overall") if m.get("quality_overall") is not None else (-1 if reverse else 2),
            reverse=reverse,
        )

    return {
        "images": wave_images,
        "dupe_groups": dupe_groups,
        "wave_counts": wave_counts,
    }


def start_scoring() -> dict:
    """Start scoring all unscored images with ImageReward.

    Returns {ok: True} or {error: str}.
    """
    from src.services import image_reward_scoring_service

    progress = image_reward_scoring_service.get_scoring_progress()
    if progress["running"]:
        return {"error": "Scoring is already in progress"}

    image_reward_scoring_service.start_scoring_all_unscored()
    return {"ok": True}


def get_scoring_progress() -> dict:
    """Get ImageReward scoring progress."""
    from src.services import image_reward_scoring_service
    return image_reward_scoring_service.get_scoring_progress()
