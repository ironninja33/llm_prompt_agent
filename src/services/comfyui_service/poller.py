"""Threaded job poller with status listeners for ComfyUI generation jobs.

The listener/callback pattern mirrors :mod:`src.services.ingestion_service`
and :mod:`src.services.clustering_service`.
"""

from __future__ import annotations

import json as _json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from src.services.comfyui_service.client import (
    construct_output_path,
    get_job_progress,
    get_history,
    _extract_output_images,
    _get_base_url,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Progress dataclass & listener infrastructure
# ---------------------------------------------------------------------------

@dataclass
class GenerationProgress:
    """Progress update for a generation job, broadcast to listeners."""
    job_id: str
    prompt_id: str
    phase: str  # "queued", "running", "completed", "failed"
    progress: float  # 0.0 to 1.0
    current_image: int = 0
    total_images: int = 1
    message: str = ""
    complete: bool = False
    output_images: list[dict] | None = None
    queue_position: int = 0  # 0 = not queued/unknown, 1+ = position in queue


_status_listeners: list[Callable[[GenerationProgress], Any]] = []
_listeners_lock = threading.Lock()

# Completion callbacks run *before* status listeners in _emit_completed,
# guaranteeing DB writes (e.g. image record storage) finish before
# SSE events reach the browser.
_completion_callbacks: list[Callable[[GenerationProgress], Any]] = []
_completion_callbacks_lock = threading.Lock()

# In-memory cache of generation job progress for polling
_job_progress_cache: dict[str, dict] = {}
_job_progress_cache_lock = threading.Lock()

# Per-directory completion sequence numbers for browser refresh signalling.
# Maps normalized absolute dir path -> latest seq.  Grows by one entry per
# unique output directory (bounded, typically dozens).
_dir_completion_seqs: dict[str, int] = {}
_seq_counter: int = 0


def add_status_listener(callback: Callable[[GenerationProgress], Any]) -> None:
    """Register a callback for generation progress updates.

    The callback receives a :class:`GenerationProgress` instance.
    """
    with _listeners_lock:
        _status_listeners.append(callback)


def remove_status_listener(callback: Callable[[GenerationProgress], Any]) -> None:
    """Remove a previously registered status listener."""
    with _listeners_lock:
        if callback in _status_listeners:
            _status_listeners.remove(callback)


def add_completion_callback(callback: Callable[[GenerationProgress], Any]) -> None:
    """Register a callback that runs when a job completes, *before* status listeners.

    Use this for DB writes that must be visible before SSE events reach the browser.
    """
    with _completion_callbacks_lock:
        _completion_callbacks.append(callback)


def _notify_listeners(progress: GenerationProgress) -> None:
    """Broadcast a progress update to all registered listeners."""
    # Update in-memory cache for polling
    with _job_progress_cache_lock:
        _job_progress_cache[progress.job_id] = {
            "job_id": progress.job_id,
            "prompt_id": progress.prompt_id,
            "phase": progress.phase,
            "progress": progress.progress,
            "current_image": progress.current_image,
            "total_images": progress.total_images,
            "message": progress.message,
            "complete": progress.complete,
            "output_images": progress.output_images,
            "queue_position": progress.queue_position,
            "_updated_at": time.monotonic(),
        }

    with _listeners_lock:
        listeners = _status_listeners[:]

    for listener in listeners:
        try:
            listener(progress)
        except Exception as exc:
            logger.error("Error in ComfyUI status listener: %s", exc)


def get_cached_job_progress(job_id: str) -> dict | None:
    """Return cached progress for a job, or None if not tracked."""
    with _job_progress_cache_lock:
        return _job_progress_cache.get(job_id)


def get_active_job_ids() -> list[str]:
    """Return job IDs that are currently in progress (not complete)."""
    now = time.monotonic()
    active = []
    stale = []
    with _job_progress_cache_lock:
        for job_id, entry in _job_progress_cache.items():
            if entry.get("complete"):
                # Remove completed entries older than 60s
                if now - entry.get("_updated_at", 0) > 60:
                    stale.append(job_id)
            else:
                active.append(job_id)
        for job_id in stale:
            del _job_progress_cache[job_id]
    return active


def _record_completion_dirs(output_images: list[dict]) -> None:
    """Record which directories received output from a completed generation."""
    global _seq_counter
    dirs: set[str] = set()
    for img in output_images:
        file_path = construct_output_path(
            img.get("filename", ""), img.get("subfolder", ""),
        )
        if file_path:
            dirs.add(os.path.normpath(os.path.dirname(file_path)))
    if dirs:
        _seq_counter += 1
        for d in dirs:
            _dir_completion_seqs[d] = _seq_counter


def get_completion_seq_for_path(abs_dir: str) -> int:
    """Return the highest completion seq for any dir at or under abs_dir.

    Returns 0 if no completions have ever landed in this subtree.
    """
    norm = os.path.normpath(abs_dir)
    norm_prefix = norm + os.sep
    best = 0
    for d, seq in _dir_completion_seqs.items():
        if d == norm or d.startswith(norm_prefix):
            if seq > best:
                best = seq
    return best


# ---------------------------------------------------------------------------
# Notification helpers (Step 2A -- deduplicate _notify_listeners calls)
# ---------------------------------------------------------------------------

def _emit(job_id: str, prompt_id: str, total_images: int, phase: str,
          progress: float = 0.0, **kwargs: Any) -> None:
    """Shorthand for _notify_listeners(GenerationProgress(...))."""
    _notify_listeners(GenerationProgress(
        job_id=job_id, prompt_id=prompt_id, phase=phase,
        progress=progress, total_images=total_images, **kwargs,
    ))


def _emit_completed(job_id: str, prompt_id: str, total_images: int) -> list[dict]:
    """Shared completion logic: fetch history, extract images, emit.

    Completion callbacks run first (DB writes) so the data is committed
    before status listeners (SSE) notify the browser.
    """
    history = get_history(prompt_id)
    output_images = _extract_output_images(history) if history else []

    # Build the progress object once for both callback phases
    progress = GenerationProgress(
        job_id=job_id, prompt_id=prompt_id, phase="completed",
        progress=1.0, current_image=total_images, total_images=total_images,
        message="Generation complete", complete=True, output_images=output_images,
    )

    # Phase 1: completion callbacks (DB writes before SSE)
    with _completion_callbacks_lock:
        callbacks = _completion_callbacks[:]
    for cb in callbacks:
        try:
            cb(progress)
        except Exception as exc:
            logger.error("Error in completion callback: %s", exc)

    # Record which directories received output (for browser poll)
    _record_completion_dirs(output_images)

    # Phase 2: status listeners (SSE, cache)
    _notify_listeners(progress)

    return output_images


# ---------------------------------------------------------------------------
# Threaded job poller
# ---------------------------------------------------------------------------

def poll_job(
    job_id: str,
    prompt_id: str,
    total_images: int = 1,
) -> None:
    """Start a background thread that polls job progress until completion.

    Polls every 1 second, notifying registered listeners with
    :class:`GenerationProgress` updates.  Stops when the job reaches
    ``"completed"`` or ``"failed"`` status, or after a maximum of
    600 polls (~10 minutes).

    Args:
        job_id: Application-level job identifier.
        prompt_id: ComfyUI prompt ID (from :class:`SubmitResult`).
        total_images: Expected number of output images.
    """
    thread = threading.Thread(
        target=_poll_loop,
        args=(job_id, prompt_id, total_images),
        daemon=True,
    )
    thread.start()
    logger.info("Started polling thread for job %s (prompt_id=%s)", job_id, prompt_id)


def _poll_loop(
    job_id: str,
    prompt_id: str,
    total_images: int,
) -> None:
    """Internal polling loop -- tries WebSocket first, falls back to HTTP."""
    try:
        _ws_poll_loop(job_id, prompt_id, total_images)
    except Exception as exc:
        logger.warning("WS poll failed, falling back to HTTP: %s", exc)
        _http_poll_loop(job_id, prompt_id, total_images)


def _ws_poll_loop(
    job_id: str,
    prompt_id: str,
    total_images: int,
) -> None:
    """WebSocket-based progress loop.  Falls back to HTTP polling on failure."""
    try:
        import websocket  # websocket-client library
    except ImportError:
        logger.warning("websocket-client not installed, falling back to HTTP polling")
        _http_poll_loop(job_id, prompt_id, total_images)
        return

    base_url = _get_base_url()
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    client_id = str(uuid.uuid4())
    ws_url = f"{ws_url}/ws?clientId={client_id}"

    # Get initial status via HTTP before entering WS loop
    initial_progress = get_job_progress(prompt_id)

    # Fast path: job already completed before we started polling
    if initial_progress.status == "completed":
        output_images = _emit_completed(job_id, prompt_id, total_images)
        logger.info("Job %s already completed at poll start (%d images)",
                     job_id, len(output_images))
        return
    if initial_progress.status == "failed":
        _emit(job_id, prompt_id, total_images, phase="failed",
              message=initial_progress.message or "Generation failed",
              complete=True)
        logger.warning("Job %s already failed at poll start", job_id)
        return

    last_queue_position = 0
    if initial_progress.status == "queued":
        # Extract position from message like "Job is queued (position 2/5)"
        msg = initial_progress.message
        if "(position " in msg:
            try:
                pos_str = msg.split("(position ")[1].split("/")[0]
                last_queue_position = int(pos_str)
            except (IndexError, ValueError):
                pass

    # Emit initial queued status
    _emit(job_id, prompt_id, total_images, phase="queued",
          queue_position=last_queue_position,
          message=f"Queue position: {last_queue_position}" if last_queue_position > 0
              else "Job submitted, waiting in queue\u2026")

    try:
        ws = websocket.create_connection(ws_url, timeout=10)
    except Exception as exc:
        logger.warning("WebSocket connection failed (%s), falling back to HTTP: %s", ws_url, exc)
        _http_poll_loop(job_id, prompt_id, total_images)
        return

    ws_failed = False
    try:
        ws.settimeout(1.0)  # 1s receive timeout for poll loop responsiveness
        max_queue_time = 1800  # 30 min max waiting in queue
        max_exec_time = 600   # 10 min max once execution starts
        start = time.monotonic()
        exec_start_time = None  # set when execution begins
        execution_started = False
        last_queue_remaining = -1  # track for delta-based position updates
        last_http_check = time.monotonic()
        _HTTP_CHECK_INTERVAL = 3.0  # seconds between HTTP fallback checks

        while True:
            now = time.monotonic()
            if exec_start_time is not None:
                if now - exec_start_time > max_exec_time:
                    break  # execution timeout
            elif now - start > max_queue_time:
                break  # queue timeout

            # HTTP safety-net check — runs every _HTTP_CHECK_INTERVAL
            # regardless of whether WebSocket messages are flowing.
            # Catches completions that WS events miss (e.g. prompt_id
            # filtering, events arriving before WS connects, etc.).
            if now - last_http_check >= _HTTP_CHECK_INTERVAL:
                last_http_check = now
                try:
                    job_prog = get_job_progress(prompt_id)
                    if job_prog.status == "completed":
                        output_images = _emit_completed(job_id, prompt_id, total_images)
                        logger.info("Job %s completed (detected via HTTP fallback)", job_id)
                        return
                    elif job_prog.status == "failed":
                        _emit(job_id, prompt_id, total_images, phase="failed",
                              message=job_prog.message or "Generation failed",
                              complete=True)
                        logger.warning("Job %s failed (detected via HTTP fallback)", job_id)
                        return
                    elif job_prog.status == "running" and not execution_started:
                        # Job started but we missed the WS event
                        execution_started = True
                        if exec_start_time is None:
                            exec_start_time = time.monotonic()
                        _emit(job_id, prompt_id, total_images, phase="running",
                              progress=0.5, message="Generating\u2026")
                except Exception:
                    pass  # HTTP check failed, keep listening on WS

            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                logger.warning("WebSocket recv error for job %s, falling back to HTTP", job_id)
                ws_failed = True
                break

            msg = _json.loads(raw)
            msg_type = msg.get("type")
            data = msg.get("data", {})

            # Filter to our prompt_id where applicable
            msg_prompt_id = data.get("prompt_id")
            if msg_prompt_id and msg_prompt_id != prompt_id:
                continue

            if msg_type == "status":
                queue_remaining = data.get("status", {}).get("exec_info", {}).get("queue_remaining", 0)
                if not execution_started and queue_remaining > 0:
                    # Update position estimate using delta from last known queue_remaining
                    if last_queue_remaining >= 0 and queue_remaining < last_queue_remaining:
                        delta = last_queue_remaining - queue_remaining
                        last_queue_position = max(1, last_queue_position - delta)
                    last_queue_remaining = queue_remaining
                    _emit(job_id, prompt_id, total_images, phase="queued",
                          queue_position=last_queue_position,
                          message=f"Queue position: {last_queue_position}" if last_queue_position > 0
                              else "Waiting in queue\u2026")

            elif msg_type == "execution_start":
                execution_started = True
                if exec_start_time is None:
                    exec_start_time = time.monotonic()
                _emit(job_id, prompt_id, total_images, phase="running",
                      message="Starting generation\u2026")

            elif msg_type == "progress":
                execution_started = True
                if exec_start_time is None:
                    exec_start_time = time.monotonic()
                value = data.get("value", 0)
                max_val = data.get("max", 1)
                pct = value / max_val if max_val > 0 else 0.0
                _emit(job_id, prompt_id, total_images, phase="running",
                      progress=pct, message=f"Step {value}/{max_val}")

            elif msg_type == "executing":
                if data.get("node") is None:
                    # Execution complete -- fetch history for output images
                    time.sleep(0.5)
                    output_images = _emit_completed(job_id, prompt_id, total_images)
                    logger.info("Job %s completed via WebSocket (%d output images)",
                                job_id, len(output_images))
                    return
                else:
                    execution_started = True
                    if exec_start_time is None:
                        exec_start_time = time.monotonic()

    finally:
        ws.close()

    # If we broke out due to WS error, fall back to HTTP polling
    if ws_failed:
        _http_poll_loop(job_id, prompt_id, total_images)
        return

    # Timed out -- do a final completion check before marking failed
    try:
        final_progress = get_job_progress(prompt_id)
        if final_progress.status == "completed":
            output_images = _emit_completed(job_id, prompt_id, total_images)
            logger.info("Job %s completed (detected at timeout via final check)", job_id)
            return
    except Exception:
        pass  # final check failed, proceed to mark as failed

    timeout_type = "execution" if exec_start_time is not None else "queue"
    timeout_duration = "10 minutes" if exec_start_time is not None else "30 minutes"
    _emit(job_id, prompt_id, total_images, phase="failed",
          message=f"Timed out after {timeout_duration} ({timeout_type} timeout)",
          complete=True)
    logger.warning("WebSocket polling %s timeout for job %s", timeout_type, job_id)


def _http_poll_loop(
    job_id: str,
    prompt_id: str,
    total_images: int,
) -> None:
    """HTTP-based polling loop -- fallback when WebSocket is unavailable."""
    max_queue_polls = 1800  # ~30 minutes at 1s interval
    max_exec_polls = 600    # ~10 minutes at 1s interval
    poll_count = 0
    exec_poll_count = None  # None = still queued, 0+ = executing

    # Emit initial queued status
    _emit(job_id, prompt_id, total_images, phase="queued",
          message="Job submitted, waiting in queue\u2026")

    while True:
        # Check appropriate timeout
        if exec_poll_count is not None:
            if exec_poll_count > max_exec_polls:
                break
        elif poll_count > max_queue_polls:
            break
        time.sleep(1.0)
        poll_count += 1

        try:
            job_prog = get_job_progress(prompt_id)
        except Exception as exc:
            logger.debug("Poll error for %s: %s", prompt_id, exc)
            continue

        if job_prog.status == "completed":
            output_images = _emit_completed(job_id, prompt_id, total_images)
            logger.info("Job %s completed (%d output images)", job_id, len(output_images))
            return

        elif job_prog.status == "failed":
            _emit(job_id, prompt_id, total_images, phase="failed",
                  message=job_prog.message or "Generation failed",
                  complete=True)
            logger.warning("Job %s failed: %s", job_id, job_prog.message)
            return

        elif job_prog.status == "running":
            if exec_poll_count is None:
                exec_poll_count = 0  # start execution counter
            exec_poll_count += 1
            _emit(job_id, prompt_id, total_images, phase="running",
                  progress=job_prog.progress,
                  message=job_prog.message or "Generating\u2026")

        elif job_prog.status == "queued":
            _emit(job_id, prompt_id, total_images, phase="queued",
                  message=job_prog.message or "Waiting in queue\u2026")

        # "unknown" status -- keep polling silently
        logger.debug(
            "Poll #%d for job %s: status=%s progress=%.1f%%",
            poll_count, job_id, job_prog.status, job_prog.progress * 100,
        )

    # Timed out -- do a final completion check before marking failed
    try:
        final_progress = get_job_progress(prompt_id)
        if final_progress.status == "completed":
            output_images = _emit_completed(job_id, prompt_id, total_images)
            logger.info("Job %s completed (detected at timeout via final check)", job_id)
            return
    except Exception:
        pass  # final check failed, proceed to mark as failed

    timeout_type = "execution" if exec_poll_count is not None else "queue"
    timeout_duration = "10 minutes" if exec_poll_count is not None else "30 minutes"
    _emit(job_id, prompt_id, total_images, phase="failed",
          message=f"Polling timed out after {timeout_duration} ({timeout_type} timeout)",
          complete=True)
    logger.warning("Polling %s timeout for job %s after %d polls", timeout_type, job_id, poll_count)
