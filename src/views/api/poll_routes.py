"""Unified polling endpoint — replaces individual SSE/poll endpoints."""

import logging

from flask import request, jsonify

from src.views.api import api_bp

logger = logging.getLogger(__name__)


@api_bp.route("/poll", methods=["GET"])
def unified_poll():
    """Single endpoint for all periodic status checks.

    Query params:
        m: comma-separated module names to include
        gen_jobs: comma-separated job IDs for generation progress
        browser_path: current browser directory path
        browser_since: timestamp for browser new-file detection
    """
    modules = request.args.get("m", "").split(",")
    modules = [m for m in modules if m]
    result = {}

    if "comfyui" in modules:
        result["comfyui"] = _get_comfyui_status()

    if "generation" in modules:
        result["generation"] = _get_generation_status()

    if "browser" in modules:
        result["browser"] = _get_browser_status()

    if "ingestion" in modules:
        from src.services import ingestion_service
        result["ingestion"] = ingestion_service.get_current_status()

    if "clustering" in modules:
        from src.services import clustering_service
        result["clustering"] = clustering_service.get_current_status()

    if "cleanup_parse" in modules:
        result["cleanup_parse"] = _get_cleanup_parse_status()

    if "cleanup_batch" in modules:
        result["cleanup_batch"] = _get_cleanup_batch_status()

    # Always include active generation job IDs for fresh-tab discovery
    from src.services import comfyui_service
    active_gen_jobs = comfyui_service.get_active_job_ids()
    if active_gen_jobs:
        result["_active_generation_jobs"] = active_gen_jobs

    return jsonify(result)


def _get_comfyui_status() -> dict:
    """Reuse existing comfyui_status logic."""
    from src.services import comfyui_service
    health = comfyui_service.check_health()
    queue_size = 0
    if health.get("ok"):
        queue = comfyui_service.get_queue_status()
        running = len(queue.get("queue_running", []))
        pending = len(queue.get("queue_pending", []))
        queue_size = running + pending
    return {"ok": health.get("ok", False), "queue_size": queue_size}


def _get_generation_status() -> dict:
    """Return cached progress for requested job IDs."""
    from src.services import comfyui_service
    from src.models import generation as gen_model

    job_ids = request.args.get("gen_jobs", "").split(",")
    gen_data = {}
    for job_id in (j for j in job_ids if j):
        cached = comfyui_service.get_cached_job_progress(job_id)
        if cached:
            gen_data[job_id] = cached
        else:
            # Fallback: check DB for terminal state
            try:
                job = gen_model.get_job(job_id)
                if job and job["status"] in ("completed", "failed"):
                    gen_data[job_id] = {"phase": job["status"], "complete": True}
            except Exception:
                pass
    return gen_data


def _get_browser_status() -> dict:
    """Check for new files since timestamp and generation completions."""
    from src.controllers import browser_controller
    from src.services import comfyui_service

    path = request.args.get("browser_path", "")
    since = float(request.args.get("browser_since", "0"))
    result = browser_controller.poll_new_files(path, since)

    # Add generation completion seq scoped to the browsed directory
    abs_path = browser_controller.resolve_virtual_path(path) if path else None
    if abs_path:
        result["completion_seq"] = comfyui_service.get_completion_seq_for_path(abs_path)
    else:
        # Root level: check all output directories
        from src.models.browser import get_root_directories
        best = 0
        for root in get_root_directories():
            seq = comfyui_service.get_completion_seq_for_path(root["path"])
            if seq > best:
                best = seq
        result["completion_seq"] = best

    return result


def _get_cleanup_parse_status() -> dict:
    """Get cleanup parse progress."""
    from src.controllers import cleanup_controller
    return cleanup_controller.get_parse_status()


def _get_cleanup_batch_status() -> dict:
    """Get cleanup batch scoring status."""
    from src.controllers import cleanup_controller
    return {"batch": cleanup_controller.get_batch_status()}
