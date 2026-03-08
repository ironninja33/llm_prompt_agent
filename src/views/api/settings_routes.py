"""Settings, data directories, ingestion, clustering, stats, and dataset-map endpoints."""

import logging
import os
import re
import queue

from flask import request, jsonify, Response

from src.views.api import api_bp
from src.views.api.helpers import _sse_event
from src.controllers import settings_controller
from src.services import ingestion_service, clustering_service
from src.models import vector_store, settings as settings_model

logger = logging.getLogger(__name__)


# ── Settings endpoints ───────────────────────────────────────────────────

@api_bp.route("/settings", methods=["GET"])
def get_settings():
    """Get all settings (API key masked)."""
    return jsonify(settings_controller.get_all_settings())


@api_bp.route("/settings", methods=["PUT"])
def update_settings():
    """Update settings."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    settings_controller.update_settings(data)
    return jsonify({"ok": True})


@api_bp.route("/settings/reset-system-prompt", methods=["POST"])
def reset_system_prompt():
    """Reset system prompt to the default from markdown file."""
    prompt = settings_controller.reset_system_prompt()
    return jsonify({"system_prompt": prompt})


@api_bp.route("/settings/models", methods=["GET"])
def list_models():
    """List available Gemini models."""
    models = settings_controller.get_available_models()
    return jsonify(models)


# ── Per-folder cluster k endpoints ────────────────────────────────────────

@api_bp.route("/settings/custom-cluster-k", methods=["GET"])
def get_custom_cluster_k():
    """Check if any per-folder cluster k overrides exist."""
    from src.models.database import get_db
    from sqlalchemy import text

    with get_db() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) as cnt FROM settings WHERE key LIKE 'cluster_k_intra:%'")
        )
        count = result.fetchone()._mapping["cnt"]

    return jsonify({"has_overrides": count > 0})


@api_bp.route("/settings/reset-custom-cluster-k", methods=["POST"])
def reset_custom_cluster_k():
    """Delete all per-folder cluster k overrides."""
    from src.models.database import get_db
    from sqlalchemy import text

    with get_db() as conn:
        result = conn.execute(
            text("DELETE FROM settings WHERE key LIKE 'cluster_k_intra:%'")
        )
        deleted = result.rowcount

    return jsonify({"ok": True, "deleted": deleted})


# ── Summarizer defaults endpoint ──────────────────────────────────────────

@api_bp.route("/summarizer/defaults", methods=["GET"])
def get_summarizer_defaults():
    """Return bundled default summarizer prompt templates."""
    from src.services.summarizer.prompts import load_default_templates
    return jsonify(load_default_templates())


# ── Data directory endpoints ─────────────────────────────────────────────

@api_bp.route("/data-directories", methods=["GET"])
def list_data_directories():
    """List all data directories."""
    dirs = settings_controller.get_data_directories()
    return jsonify(dirs)


@api_bp.route("/data-directories", methods=["POST"])
def add_data_directory():
    """Add a new data directory."""
    data = request.get_json()
    if not data or not data.get("path") or not data.get("dir_type"):
        return jsonify({"error": "path and dir_type are required"}), 400

    try:
        directory = settings_controller.add_data_directory(data["path"], data["dir_type"])
        return jsonify(directory), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@api_bp.route("/data-directories/<int:dir_id>", methods=["PUT"])
def update_data_directory(dir_id):
    """Update a data directory."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    success = settings_controller.update_data_directory(dir_id, **data)
    if success:
        return jsonify({"ok": True})
    return jsonify({"error": "Directory not found"}), 404


@api_bp.route("/data-directories/<int:dir_id>", methods=["DELETE"])
def delete_data_directory(dir_id):
    """Delete a data directory."""
    success = settings_controller.delete_data_directory(dir_id)
    if success:
        return jsonify({"ok": True})
    return jsonify({"error": "Directory not found"}), 404


# ── Ingestion endpoints ─────────────────────────────────────────────────

@api_bp.route("/ingestion/status", methods=["GET"])
def ingestion_status():
    """SSE endpoint for ingestion progress updates."""
    q = queue.Queue()

    def listener(progress):
        """Push progress updates into the queue."""
        try:
            q.put(progress, block=False)
        except queue.Full:
            pass

    ingestion_service.add_status_listener(listener)

    def generate():
        try:
            while True:
                try:
                    progress = q.get(timeout=30)
                    data = {
                        "phase": progress.phase,
                        "message": progress.message,
                        "directories_scanned": progress.directories_scanned,
                        "total_files": progress.total_files,
                        "new_files": progress.new_files,
                        "already_indexed": progress.already_indexed,
                        "current": progress.current,
                        "current_dir": progress.current_dir,
                        "errors": progress.errors,
                        "complete": progress.complete,
                    }

                    if progress.complete or progress.phase in ("complete", "error"):
                        yield _sse_event("ingestion_complete", data)
                        break
                    else:
                        yield _sse_event("ingestion_status", data)

                except queue.Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"

                    # If ingestion isn't running, end the stream
                    if not ingestion_service.is_running():
                        yield _sse_event("ingestion_complete", {
                            "phase": "idle",
                            "message": "No ingestion running",
                            "complete": True,
                        })
                        break

        finally:
            ingestion_service.remove_status_listener(listener)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@api_bp.route("/ingestion/trigger", methods=["POST"])
def trigger_ingestion():
    """Trigger a full re-ingestion."""
    if ingestion_service.is_running():
        return jsonify({"error": "Ingestion already running"}), 409

    ingestion_service.start_ingestion(output_only=False)
    return jsonify({"ok": True, "message": "Ingestion started"})


@api_bp.route("/ingestion/refresh-output", methods=["POST"])
def refresh_output():
    """Re-scan output directories only."""
    if ingestion_service.is_running():
        return jsonify({"error": "Ingestion already running"}), 409

    ingestion_service.start_ingestion(output_only=True)
    return jsonify({"ok": True, "message": "Output refresh started"})


# ── Clustering endpoints ─────────────────────────────────────────────────

@api_bp.route("/clustering/trigger", methods=["POST"])
def trigger_clustering():
    """Trigger a clustering-only run (no re-indexing)."""
    if clustering_service.is_running():
        return jsonify({"error": "Clustering already running"}), 409
    if ingestion_service.is_running():
        return jsonify({"error": "Ingestion is running"}), 409

    clustering_service.start_clustering(cross_folder=True, intra_folder=True, force=True)
    return jsonify({"ok": True, "message": "Clustering started"})


@api_bp.route("/clustering/status", methods=["GET"])
def clustering_status():
    """SSE endpoint for clustering progress updates."""
    q = queue.Queue()

    def listener(progress):
        """Push progress updates into the queue."""
        try:
            q.put(progress, block=False)
        except queue.Full:
            pass

    clustering_service.add_status_listener(listener)

    def generate():
        try:
            while True:
                try:
                    progress = q.get(timeout=30)
                    data = {
                        "phase": progress.phase,
                        "message": progress.message,
                        "current": progress.current,
                        "total": progress.total,
                        "complete": progress.complete,
                    }

                    if progress.complete or progress.phase in ("complete", "error"):
                        yield _sse_event("clustering_complete", data)
                        break
                    else:
                        yield _sse_event("clustering_status", data)

                except queue.Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"

                    # If clustering isn't running, end the stream
                    if not clustering_service.is_running():
                        yield _sse_event("clustering_complete", {
                            "phase": "idle",
                            "message": "No clustering running",
                            "complete": True,
                        })
                        break

        finally:
            clustering_service.remove_status_listener(listener)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@api_bp.route("/dataset-map", methods=["GET"])
def get_dataset_map():
    """Get the dataset map showing themes and folder structure."""
    dataset_map = clustering_service.get_dataset_map()
    return jsonify(dataset_map)


@api_bp.route("/folder/rename", methods=["POST"])
def rename_folder():
    """Rename a concept folder on disk and update all DB/ChromaDB references."""
    data = request.get_json(silent=True) or {}
    old_name = (data.get("old_name") or "").strip()
    new_name = (data.get("new_name") or "").strip()

    if not old_name or not new_name:
        return jsonify({"ok": False, "error": "old_name and new_name are required"}), 400
    if old_name == new_name:
        return jsonify({"ok": False, "error": "New name is the same as the old name"}), 400
    if "/" in new_name or "\\" in new_name or "\0" in new_name:
        return jsonify({"ok": False, "error": "Folder name cannot contain slashes or null bytes"}), 400
    if re.search(r"^\s|\s$", new_name):
        return jsonify({"ok": False, "error": "Folder name cannot have leading/trailing whitespace"}), 400

    # Find all data directories that contain a folder with old_name
    data_dirs = settings_model.get_data_directories(active_only=True)
    parent_dirs = []
    for dd in data_dirs:
        candidate = os.path.join(dd["path"], old_name)
        if os.path.isdir(candidate):
            # Check that destination doesn't already exist
            dest = os.path.join(dd["path"], new_name)
            if os.path.exists(dest):
                return jsonify({"ok": False, "error": f"Destination already exists: {dest}"}), 409
            parent_dirs.append(dd["path"])

    if not parent_dirs:
        return jsonify({"ok": False, "error": f"No data directory contains folder '{old_name}'"}), 404

    result = clustering_service.rename_concept(old_name, new_name, parent_dirs)
    if result["ok"]:
        return jsonify(result)
    return jsonify(result), 500


# ── Stats endpoint ───────────────────────────────────────────────────────

@api_bp.route("/stats", methods=["GET"])
def get_stats():
    """Get database statistics."""
    counts = vector_store.get_collection_counts()
    cluster_stats = clustering_service.get_clustering_stats()
    return jsonify({
        "training_prompts": counts.get("training", 0),
        "generated_prompts": counts.get("generated", 0),
        **cluster_stats,
    })
