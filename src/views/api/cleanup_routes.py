"""Cleanup assistant API endpoints."""

import logging

from flask import request, jsonify

from src.views.api import api_bp
from src.controllers import cleanup_controller

logger = logging.getLogger(__name__)


@api_bp.route("/cleanup/folders", methods=["GET"])
def cleanup_folders():
    """Get all output folders with cleanup metrics."""
    folders = cleanup_controller.get_folder_summary()
    return jsonify({"folders": folders})


@api_bp.route("/cleanup/triage", methods=["GET"])
def cleanup_triage():
    """Get paginated triage images for a wave/folder.

    Params: folder (optional), wave (1-3), offset, limit.
    """
    folder = request.args.get("folder") or None
    wave = int(request.args.get("wave", 1))
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 50))

    if wave not in (1, 2, 3):
        return jsonify({"error": "wave must be 1, 2, or 3"}), 400

    result = cleanup_controller.get_triage_page(folder, wave, offset, limit)
    return jsonify(result)


@api_bp.route("/cleanup/wave-counts", methods=["GET"])
def cleanup_wave_counts():
    """Get image counts per wave for tab badges."""
    folder = request.args.get("folder") or None
    counts = cleanup_controller.get_wave_counts(folder)
    return jsonify({"counts": counts})


@api_bp.route("/cleanup/delete", methods=["POST"])
def cleanup_delete():
    """Delete specific images.

    Body: { "image_ids": [int, ...] }
    """
    data = request.get_json()
    if not data or not data.get("image_ids"):
        return jsonify({"error": "image_ids required"}), 400

    image_ids = data["image_ids"]
    result = cleanup_controller.delete_images(image_ids)
    return jsonify(result)


@api_bp.route("/cleanup/delete-visible", methods=["POST"])
def cleanup_delete_visible():
    """Delete all images visible in the current folder+wave filter.

    Body: { "folder": str|null, "wave": int }
    """
    data = request.get_json()
    if not data or "wave" not in data:
        return jsonify({"error": "wave required"}), 400

    folder = data.get("folder") or None
    wave = int(data["wave"])
    result = cleanup_controller.delete_visible(folder, wave)
    return jsonify(result)


@api_bp.route("/cleanup/keep", methods=["POST"])
def cleanup_keep():
    """Flag images as explicitly kept.

    Body: { "image_ids": [int, ...] }
    """
    data = request.get_json()
    if not data or not data.get("image_ids"):
        return jsonify({"error": "image_ids required"}), 400

    cleanup_controller.keep_images(data["image_ids"])
    return jsonify({"ok": True})


@api_bp.route("/cleanup/unkeep", methods=["POST"])
def cleanup_unkeep():
    """Remove keep flags from images.

    Body: { "image_ids": [int, ...] }
    """
    data = request.get_json()
    if not data or not data.get("image_ids"):
        return jsonify({"error": "image_ids required"}), 400

    cleanup_controller.unkeep_images(data["image_ids"])
    return jsonify({"ok": True})


@api_bp.route("/cleanup/parse", methods=["POST"])
def cleanup_parse():
    """Trigger background bulk metadata parse."""
    cleanup_controller.start_parse()
    return jsonify({"ok": True})


@api_bp.route("/cleanup/parse-status", methods=["GET"])
def cleanup_parse_status():
    """Get bulk parse progress."""
    status = cleanup_controller.get_parse_status()
    return jsonify(status)


@api_bp.route("/cleanup/near-dupes", methods=["GET"])
def cleanup_near_dupes():
    """Get near-duplicate image groups.

    Params: folder (optional).
    """
    folder = request.args.get("folder") or None
    groups = cleanup_controller.get_near_duplicates(folder)
    return jsonify({"groups": groups})


@api_bp.route("/cleanup/triage-data", methods=["GET"])
def cleanup_triage_data():
    """Get triage images + near-dupe groups + wave counts in one call.

    Computes dupe detection once and reuses it for scoring, avoiding
    the expensive double-computation that happens when triage and
    near-dupes endpoints are called in parallel.

    Params: folder (optional), wave (1-3).
    """
    folder = request.args.get("folder") or None
    wave = int(request.args.get("wave", 1))

    if wave not in (1, 2, 3):
        return jsonify({"error": "wave must be 1, 2, or 3"}), 400

    result = cleanup_controller.get_triage_data(folder, wave)
    return jsonify(result)


@api_bp.route("/cleanup/batch-status", methods=["GET"])
def cleanup_batch_status():
    """Get the most recent scoring batch status."""
    batch = cleanup_controller.get_batch_status()
    return jsonify({"batch": batch})


@api_bp.route("/cleanup/active-batch", methods=["GET"])
def cleanup_active_batch():
    """Get the active (non-terminal) scoring batch."""
    batch = cleanup_controller.get_active_batch()
    return jsonify({"batch": batch})


@api_bp.route("/cleanup/score", methods=["POST"])
def cleanup_score():
    """Submit a scoring batch to Gemini.

    Body: { "mode": "near-dupes"|"all", "wave": int, "folder": str|null }
    """
    data = request.get_json()
    if not data or "mode" not in data or "wave" not in data:
        return jsonify({"error": "mode and wave required"}), 400

    result = cleanup_controller.submit_scoring(
        mode=data["mode"],
        wave=int(data["wave"]),
        folder=data.get("folder"),
    )

    if "error" in result:
        return jsonify(result), 409 if "batch_id" in result else 400

    return jsonify(result)


@api_bp.route("/cleanup/scoring-progress", methods=["GET"])
def cleanup_scoring_progress():
    """Get upload/submission progress for the current scoring batch."""
    progress = cleanup_controller.get_scoring_progress()
    return jsonify(progress)


@api_bp.route("/cleanup/poll-batch", methods=["POST"])
def cleanup_poll_batch():
    """Poll the scoring batch for status updates (and process results if done)."""
    result = cleanup_controller.poll_scoring_batch()
    return jsonify(result)
