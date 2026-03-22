"""ComfyUI HTTP client -- communicates with a running ComfyUI server.

Provides health checks, model listing (with TTL cache), job submission,
progress polling, and image retrieval.  All network errors are caught and
returned as structured error values -- callers never see raw exceptions
from ``requests``.

Output images are served directly from the filesystem via the ``output``
data directories stored in :mod:`src.models.settings`, rather than
proxying through ComfyUI's ``/view`` endpoint.  This means images remain
accessible even when ComfyUI is not running, and the same filesystem
access can later be extended into a general image browser.
"""

from __future__ import annotations

import io
import json as _json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeouts (seconds)
# ---------------------------------------------------------------------------
_QUERY_TIMEOUT = 5

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_base_url() -> str:
    """Get ComfyUI base URL from settings, fallback to config default.

    Checks the database setting ``comfyui_base_url`` first; if absent or
    empty, falls back to :data:`src.config.DEFAULT_COMFYUI_BASE_URL`.
    """
    try:
        from src.models.settings import get_setting
        url = get_setting("comfyui_base_url")
    except Exception:
        url = None

    if not url:
        from src.config import DEFAULT_COMFYUI_BASE_URL
        url = DEFAULT_COMFYUI_BASE_URL

    return url.rstrip("/")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_health() -> dict:
    """Check ComfyUI server connectivity via ``GET /system_stats``.

    Returns:
        ``{"ok": True, "message": "Connected to ..."}`` on success, or
        ``{"ok": False, "message": "...error description..."}`` on failure.
    """
    base = _get_base_url()
    try:
        resp = requests.get(f"{base}/system_stats", timeout=_QUERY_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # Extract a useful summary from system_stats if available
        devices = data.get("devices", [])
        device_info = devices[0].get("name", "unknown") if devices else "unknown"
        return {"ok": True, "message": f"Connected to ComfyUI at {base} (device: {device_info})"}
    except requests.ConnectionError:
        return {"ok": False, "message": f"Cannot connect to ComfyUI at {base}"}
    except requests.Timeout:
        return {"ok": False, "message": f"Connection to ComfyUI timed out ({base})"}
    except Exception as exc:
        return {"ok": False, "message": f"ComfyUI health check failed: {exc}"}


# ---------------------------------------------------------------------------
# Model listing with TTL cache (60 seconds)
# ---------------------------------------------------------------------------

_model_cache: dict[str, tuple[float, list[str]]] = {}
_model_cache_lock = threading.Lock()
_MODEL_CACHE_TTL = 60.0  # seconds


def get_available_models(model_type: str) -> list[str]:
    """Fetch available models from ComfyUI.

    Args:
        model_type: One of ``"loras"`` or ``"diffusion_models"``.

    Returns:
        Sorted list of model filenames.  Returns ``[]`` on error.

    Results are cached for 60 seconds (thread-safe).
    """
    now = time.monotonic()

    with _model_cache_lock:
        if model_type in _model_cache:
            cached_time, cached_list = _model_cache[model_type]
            if now - cached_time < _MODEL_CACHE_TTL:
                return cached_list

    base = _get_base_url()
    try:
        resp = requests.get(f"{base}/models/{model_type}", timeout=_QUERY_TIMEOUT)
        resp.raise_for_status()
        models = resp.json()
        if not isinstance(models, list):
            logger.warning("Unexpected response from /models/%s: %s", model_type, type(models))
            return []
        models = sorted(models)
        logger.info("Fetched %d %s models from ComfyUI", len(models), model_type)
    except requests.ConnectionError:
        logger.warning("Cannot connect to ComfyUI for model listing (%s)", model_type)
        return []
    except requests.Timeout:
        logger.warning("Timeout fetching %s models from ComfyUI", model_type)
        return []
    except Exception as exc:
        logger.warning("Error fetching %s models: %s", model_type, exc)
        return []

    with _model_cache_lock:
        _model_cache[model_type] = (now, models)
    return models


def clear_model_cache() -> None:
    """Invalidate the model cache so the next call re-fetches."""
    with _model_cache_lock:
        _model_cache.clear()


# ---------------------------------------------------------------------------
# Output folder listing (from data_directories in the database)
# ---------------------------------------------------------------------------

def get_output_subfolders() -> list[str]:
    """Get all subdirectories (recursively) from ``output``-type data directories.

    Reads data directories with ``dir_type = 'output'`` from the
    :mod:`src.models.settings` module and walks their directory trees
    to find all nested subdirectories.

    Returns:
        Sorted, deduplicated list of relative subfolder paths
        (e.g. ``["base", "base/subdir"]``).  Returns ``[]`` if no
        output directories are configured or none exist on disk.
    """
    try:
        from src.models.settings import get_data_directories
        dirs = get_data_directories(active_only=True)
    except Exception as exc:
        logger.warning("Error reading data directories: %s", exc)
        return []

    output_dirs = [d for d in dirs if d.get("dir_type") == "output"]
    if not output_dirs:
        logger.debug("No active output-type data directories configured")
        return []

    subfolders: set[str] = set()
    for d in output_dirs:
        dir_path = d.get("path", "")
        if not dir_path or not os.path.isdir(dir_path):
            continue
        try:
            for dirpath, dirnames, _ in os.walk(dir_path):
                for dirname in dirnames:
                    full = os.path.join(dirpath, dirname)
                    rel = os.path.relpath(full, dir_path)
                    subfolders.add(rel)
        except OSError as exc:
            logger.warning("Error listing output directory %s: %s", dir_path, exc)

    return sorted(subfolders)


# ---------------------------------------------------------------------------
# Object info (node class definitions)
# ---------------------------------------------------------------------------

def get_object_info() -> dict:
    """Fetch node class definitions from ComfyUI's ``GET /object_info`` endpoint.

    Returns a dict mapping class_type strings to their input/output definitions.
    Each entry contains ``input`` (with ``required`` and ``optional`` dicts),
    ``output``, ``output_name``, etc.

    Returns:
        Dict of node definitions, or empty dict on error.
    """
    base = _get_base_url()
    try:
        resp = requests.get(f"{base}/object_info", timeout=30)
        if resp.status_code == 200:
            return resp.json()
        logger.warning("ComfyUI /object_info returned HTTP %d", resp.status_code)
        return {}
    except requests.ConnectionError:
        logger.warning("Cannot connect to ComfyUI at %s for /object_info", base)
        return {}
    except requests.Timeout:
        logger.warning("Timeout fetching /object_info from ComfyUI")
        return {}
    except Exception as exc:
        logger.warning("Error fetching /object_info: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

@dataclass
class SubmitResult:
    """Result of a workflow submission to ComfyUI."""
    success: bool
    prompt_id: str | None = None
    error: str | None = None


def submit_prompt(
    workflow_json: dict,
    client_id: str | None = None,
    ui_workflow: dict | None = None,
) -> SubmitResult:
    """Submit an API-format workflow to ComfyUI's ``POST /prompt`` endpoint.

    Sends ``prompt``, ``client_id``, and ``extra_data`` containing the
    UI-format workflow in ``extra_pnginfo`` (needed by introspection nodes
    like KJNodes' ``GetWidgetValue`` / ``WidgetToString``).

    Args:
        workflow_json: Workflow dict in **API format** (flat dict keyed by
            node-ID strings).  This is what the execution engine uses.
        client_id: Optional client identifier for WebSocket routing.
        ui_workflow: Workflow dict in **UI format** (has a top-level
            ``"nodes"`` array with ``title``, ``widgets_values``, etc.)
            with the same user settings already applied.  Passed as
            ``extra_pnginfo.workflow`` for introspection nodes.
            If ``None``, the API-format workflow is used as a fallback
            (which may cause errors in nodes that expect UI format).

    Returns:
        :class:`SubmitResult` with ``prompt_id`` on success or ``error``
        on failure.
    """
    base = _get_base_url()

    if client_id is None:
        client_id = str(uuid.uuid4())

    # The ``extra_data.extra_pnginfo.workflow`` field is read by
    # introspection nodes such as KJNodes' ``GetWidgetValue`` /
    # ``WidgetToString`` at execution time.  These nodes expect the
    # **UI format** (with a ``"nodes"`` array containing ``title``,
    # ``widgets_values``, ``inputs``/``outputs`` link arrays, etc.).
    # The caller should supply a UI-format workflow with settings
    # already applied so widget values match the API workflow.
    # If no UI-format workflow is available, we fall back to the API
    # format -- but this will fail for any node that iterates
    # ``workflow["nodes"]``.
    pnginfo_workflow = ui_workflow if ui_workflow is not None else workflow_json

    body: dict[str, Any] = {
        "prompt": workflow_json,
        "client_id": client_id,
        "extra_data": {
            "extra_pnginfo": {
                "workflow": pnginfo_workflow,
            },
        },
    }

    try:
        resp = requests.post(
            f"{base}/prompt",
            json=body,
            timeout=_QUERY_TIMEOUT,
        )

        if resp.status_code == 200:
            data = resp.json()
            prompt_id = data.get("prompt_id")
            if prompt_id:
                logger.info("Submitted prompt to ComfyUI: prompt_id=%s", prompt_id)
                return SubmitResult(success=True, prompt_id=prompt_id)
            else:
                return SubmitResult(
                    success=False,
                    error=f"No prompt_id in response: {data}",
                )
        else:
            # ComfyUI returns error details in the response body
            try:
                error_data = resp.json()
                error_msg = error_data.get("error", {}).get("message", resp.text)
                node_errors = error_data.get("node_errors", {})
                if node_errors:
                    error_msg += f" | Node errors: {node_errors}"
            except Exception:
                error_msg = resp.text

            logger.warning(
                "ComfyUI rejected prompt (HTTP %d): %s",
                resp.status_code,
                error_msg,
            )
            return SubmitResult(success=False, error=error_msg)

    except requests.ConnectionError:
        msg = f"Cannot connect to ComfyUI at {base}"
        logger.warning(msg)
        return SubmitResult(success=False, error=msg)
    except requests.Timeout:
        msg = f"Timeout submitting prompt to ComfyUI ({base})"
        logger.warning(msg)
        return SubmitResult(success=False, error=msg)
    except Exception as exc:
        msg = f"Error submitting prompt: {exc}"
        logger.warning(msg)
        return SubmitResult(success=False, error=msg)


# ---------------------------------------------------------------------------
# Job status & queue polling
# ---------------------------------------------------------------------------

@dataclass
class JobProgress:
    """Snapshot of a single job's progress."""
    status: str  # "queued", "running", "completed", "failed", "unknown"
    progress: float = 0.0  # 0.0 to 1.0
    current_node: str | None = None
    message: str = ""


def get_queue_status() -> dict:
    """Fetch the ComfyUI queue state via ``GET /queue``.

    Returns:
        Raw queue dict from ComfyUI, or ``{}`` on error.
    """
    base = _get_base_url()
    try:
        resp = requests.get(f"{base}/queue", timeout=_QUERY_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("Error fetching queue: %s", exc)
        return {}


def get_history(prompt_id: str) -> dict | None:
    """Fetch execution history for a single prompt via ``GET /history/{prompt_id}``.

    Returns:
        The history entry dict for this prompt, or ``None`` if not found
        or on error.
    """
    base = _get_base_url()
    try:
        resp = requests.get(f"{base}/history/{prompt_id}", timeout=_QUERY_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # The response is keyed by prompt_id
        return data.get(prompt_id)
    except Exception as exc:
        logger.debug("Error fetching history for %s: %s", prompt_id, exc)
        return None


def get_job_progress(prompt_id: str) -> JobProgress:
    """Determine job progress by examining queue and history.

    Checks the queue first (for queued/running status), then falls back
    to history (for completed/failed).

    Returns:
        :class:`JobProgress` describing the current state.
    """
    # 1. Check queue for queued or running status
    queue = get_queue_status()

    # queue_running is a list of entries currently executing
    running = queue.get("queue_running", [])
    for entry in running:
        if isinstance(entry, (list, tuple)) and len(entry) > 1:
            if entry[1] == prompt_id:
                return JobProgress(
                    status="running",
                    progress=0.5,
                    message="Job is currently executing",
                )

    # queue_pending is a list of queued entries
    pending = queue.get("queue_pending", [])
    for idx, entry in enumerate(pending):
        if isinstance(entry, (list, tuple)) and len(entry) > 1:
            if entry[1] == prompt_id:
                return JobProgress(
                    status="queued",
                    progress=0.0,
                    message=f"Job is queued (position {idx + 1}/{len(pending)})",
                )

    # 2. Check history for completed/failed
    history = get_history(prompt_id)
    if history is not None:
        status_info = history.get("status", {})
        completed = status_info.get("completed", False)
        status_str = status_info.get("status_str", "")

        if completed or status_str == "success":
            return JobProgress(
                status="completed",
                progress=1.0,
                message="Job completed successfully",
            )
        elif status_str == "error":
            # Extract error messages from status
            messages = status_info.get("messages", [])
            error_msg = "; ".join(str(m) for m in messages) if messages else "Job failed"
            return JobProgress(
                status="failed",
                progress=0.0,
                message=error_msg,
            )
        else:
            # History exists but status is ambiguous -- likely completed
            outputs = history.get("outputs", {})
            if outputs:
                return JobProgress(
                    status="completed",
                    progress=1.0,
                    message="Job completed",
                )

    return JobProgress(status="unknown", message="Job not found in queue or history")


# ---------------------------------------------------------------------------
# Image retrieval (direct filesystem access)
# ---------------------------------------------------------------------------

def construct_output_path(filename: str, subfolder: str = "") -> str | None:
    """Construct the expected output path from known data.

    Deterministic — no filesystem check.  Uses the first active output
    directory + subfolder + filename.  Returns a normalized absolute path,
    or None if no output directories are configured.
    """
    try:
        from src.models.settings import get_data_directories
        dirs = get_data_directories(active_only=True)
    except Exception:
        return None

    output_dirs = [d for d in dirs if d.get("dir_type") == "output"]
    if not output_dirs:
        return None
    dir_path = output_dirs[0].get("path", "")
    if not dir_path:
        return None
    if subfolder:
        return os.path.normpath(os.path.join(dir_path, subfolder, filename))
    return os.path.normpath(os.path.join(dir_path, filename))


def resolve_image_path(filename: str, subfolder: str = "") -> str | None:
    """Resolve an image filename to an absolute path on disk.

    Searches all active ``output``-type data directories for the file.

    Args:
        filename: The image filename (e.g. ``"image_00001.png"``).
        subfolder: Optional subfolder within the output directory.

    Returns:
        Absolute path to the file, or ``None`` if not found.
    """
    try:
        from src.models.settings import get_data_directories
        dirs = get_data_directories(active_only=True)
    except Exception:
        return None

    output_dirs = [d for d in dirs if d.get("dir_type") == "output"]
    for d in output_dirs:
        dir_path = d.get("path", "")
        if not dir_path:
            continue
        candidate = os.path.join(dir_path, subfolder, filename) if subfolder else os.path.join(dir_path, filename)
        if os.path.isfile(candidate):
            return candidate

    return None


# Keep underscore alias for internal backward compatibility
_resolve_image_path = resolve_image_path


def get_image(
    filename: str,
    subfolder: str = "",
) -> bytes | None:
    """Read an image from the output data directories on disk.

    Searches active ``output``-type data directories for the file.
    Does not require ComfyUI to be running.

    Args:
        filename: The image filename.
        subfolder: Subfolder within the output directory.

    Returns:
        Raw image bytes, or ``None`` if not found.
    """
    filepath = resolve_image_path(filename, subfolder)
    if filepath is None:
        logger.warning("Image not found: %s/%s", subfolder, filename)
        return None

    try:
        with open(filepath, "rb") as fh:
            return fh.read()
    except OSError as exc:
        logger.warning("Error reading image %s: %s", filepath, exc)
        return None


def get_image_thumbnail(
    filename: str,
    subfolder: str = "",
    max_size: int = 256,
) -> bytes | None:
    """Read an image and return a resized JPEG thumbnail.

    Uses PIL/Pillow to resize the image to fit within *max_size* pixels
    on its longest side, then encodes as JPEG.  Does not require ComfyUI
    to be running.

    Returns:
        JPEG bytes, or ``None`` on error.
    """
    image_bytes = get_image(filename, subfolder)
    if image_bytes is None:
        return None

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Convert to RGB if necessary (e.g. RGBA PNGs)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("Error creating thumbnail for %s: %s", filename, exc)
        return None


def delete_image_file(
    filename: str,
    subfolder: str = "",
) -> bool:
    """Delete an image file from the output data directories on disk.

    Searches active ``output``-type data directories for the file and
    deletes it if found.

    Args:
        filename: The image filename.
        subfolder: Subfolder within the output directory.

    Returns:
        True if the file was found and deleted, False otherwise.
    """
    filepath = resolve_image_path(filename, subfolder)
    if filepath is None:
        logger.warning("Image file not found for deletion: %s/%s", subfolder, filename)
        return False

    try:
        os.remove(filepath)
        logger.info("Deleted image file: %s", filepath)
        return True
    except OSError as exc:
        logger.warning("Error deleting image file %s: %s", filepath, exc)
        return False


# ---------------------------------------------------------------------------
# Extract output images from history
# ---------------------------------------------------------------------------

def _extract_output_images(history_data: dict) -> list[dict]:
    """Extract output filenames from a ComfyUI history response.

    Walks the ``outputs`` dict looking for image outputs.  Only images
    with ``type == "output"`` are included -- ``"temp"`` images from
    PreviewImage nodes live in ComfyUI's temp directory and are not
    accessible via the configured output data directories.

    Args:
        history_data: The history entry for a single prompt (from
            :func:`get_history`).

    Returns:
        List of dicts like ``{"filename": ..., "subfolder": ..., "type": ...}``.
    """
    images: list[dict] = []
    outputs = history_data.get("outputs", {})

    for node_id, node_output in outputs.items():
        # Each node output may have an "images" key
        node_images = node_output.get("images", [])
        for img in node_images:
            if isinstance(img, dict) and "filename" in img:
                img_type = img.get("type", "output")
                # Skip temp/preview images -- they live in ComfyUI's
                # temporary directory and are not in our output dirs.
                if img_type != "output":
                    logger.debug(
                        "Skipping non-output image: node=%s filename=%s type=%s",
                        node_id, img["filename"], img_type,
                    )
                    continue
                images.append({
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img_type,
                })

    return images
