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
# Consolidated job poller — single thread manages all jobs
# ---------------------------------------------------------------------------

# Fixed client ID shared between WS connection and prompt submission
# so ComfyUI routes execution events to our poller's WebSocket.
COMFYUI_CLIENT_ID = str(uuid.uuid4())

_MAX_QUEUE_TIME = 1800   # 30 min waiting in queue
_MAX_EXEC_TIME = 600     # 10 min once execution starts
_SWEEP_INTERVAL = 30.0   # seconds between full HTTP reconciliation sweeps
_UNKNOWN_FAIL_COUNT = 3  # consecutive "unknown" results before marking failed


@dataclass
class JobEntry:
    """Per-job tracking state held in the registry."""
    job_id: str
    prompt_id: str
    total_images: int
    phase: str = "queued"
    registered_at: float = field(default_factory=time.monotonic)
    exec_start_time: float | None = None
    queue_position: int = 0
    last_queue_remaining: int = -1
    unknown_count: int = 0


# Job registry — maps job_id → JobEntry
_job_registry: dict[str, JobEntry] = {}
# Reverse lookup — maps prompt_id → job_id
_prompt_to_job: dict[str, str] = {}
_registry_lock = threading.Lock()

# Poller thread management
_poller_thread: threading.Thread | None = None
_shutdown_event = threading.Event()


def poll_job(
    job_id: str,
    prompt_id: str,
    total_images: int = 1,
) -> None:
    """Register a job for tracking by the consolidated poller.

    This replaces the old per-job thread model.  The job is added to an
    internal registry and picked up by the single poller thread on its
    next iteration.

    Args:
        job_id: Application-level job identifier.
        prompt_id: ComfyUI prompt ID (from :class:`SubmitResult`).
        total_images: Expected number of output images.
    """
    entry = JobEntry(job_id=job_id, prompt_id=prompt_id, total_images=total_images)

    with _registry_lock:
        _job_registry[job_id] = entry
        _prompt_to_job[prompt_id] = job_id

    # Quick HTTP check — job may already be done before we start watching
    try:
        progress = get_job_progress(prompt_id)
        if progress.status == "completed":
            _complete_job(entry)
            logger.info("Job %s already completed at registration (%s)", job_id, prompt_id)
            return
        if progress.status == "failed":
            _fail_job(entry, progress.message or "Generation failed")
            logger.warning("Job %s already failed at registration", job_id)
            return
        # Extract queue position if available
        if progress.status == "queued" and "(position " in progress.message:
            try:
                pos_str = progress.message.split("(position ")[1].split("/")[0]
                entry.queue_position = int(pos_str)
            except (IndexError, ValueError):
                pass
    except Exception:
        pass  # HTTP check failed — poller thread will pick it up

    # Emit initial queued status
    _emit(job_id, prompt_id, total_images, phase="queued",
          queue_position=entry.queue_position,
          message=f"Queue position: {entry.queue_position}" if entry.queue_position > 0
              else "Job submitted, waiting in queue\u2026")

    logger.info("Registered job %s for polling (prompt_id=%s)", job_id, prompt_id)


def start_poller() -> None:
    """Start the consolidated poller thread (idempotent)."""
    global _poller_thread
    if _poller_thread is not None and _poller_thread.is_alive():
        return
    _shutdown_event.clear()
    _poller_thread = threading.Thread(target=_poller_main, daemon=False, name="comfyui-poller")
    _poller_thread.start()
    logger.info("Consolidated ComfyUI poller thread started")


def stop_poller(timeout: float = 10) -> None:
    """Signal the poller thread to stop and wait for it to finish."""
    global _poller_thread
    _shutdown_event.set()
    if _poller_thread is not None and _poller_thread.is_alive():
        print(f"Waiting for ComfyUI poller thread to stop (timeout={timeout}s)...")
        logger.info("Waiting for ComfyUI poller thread to stop...")
        _poller_thread.join(timeout=timeout)
        if _poller_thread.is_alive():
            logger.warning("ComfyUI poller thread did not finish in time")
        else:
            print("ComfyUI poller thread stopped.")
    _poller_thread = None


# ---------------------------------------------------------------------------
# Main poller loop
# ---------------------------------------------------------------------------

def _poller_main() -> None:
    """Single thread that manages all job polling via one WebSocket."""
    ws = None
    last_sweep = 0.0

    try:
        import websocket as _ws_lib
    except ImportError:
        _ws_lib = None
        logger.warning("websocket-client not installed — using HTTP-only polling")

    ws_url = _build_ws_url() if _ws_lib else None

    while not _shutdown_event.is_set():
        # --- Idle when no jobs ---
        with _registry_lock:
            job_count = len(_job_registry)
        if job_count == 0:
            # Close WS when idle to avoid stale connections
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
                ws = None
            _shutdown_event.wait(timeout=1.0)
            continue

        # --- Ensure WebSocket connected ---
        if _ws_lib and (ws is None or not ws.connected):
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
                ws = None
            try:
                ws_url = _build_ws_url()  # fresh client_id on reconnect
                ws = _ws_lib.create_connection(ws_url, timeout=10)
                ws.settimeout(1.0)
                logger.info("WebSocket connected to ComfyUI")
                # Immediate sweep on reconnect to catch anything missed
                _http_reconciliation_sweep()
                last_sweep = time.monotonic()
            except Exception as exc:
                logger.debug("WebSocket connection failed: %s", exc)
                ws = None
                # HTTP-only fallback sweep
                _http_reconciliation_sweep()
                last_sweep = time.monotonic()
                _shutdown_event.wait(timeout=3.0)
                continue

        # --- Receive and handle WS message ---
        if ws is not None:
            try:
                raw = ws.recv()
                _handle_ws_message(raw)
            except Exception as exc:
                exc_name = type(exc).__name__
                if exc_name == "WebSocketTimeoutException":
                    pass  # normal — 1s timeout
                else:
                    logger.warning("WebSocket recv error: %s", exc)
                    try:
                        ws.close()
                    except Exception:
                        pass
                    ws = None  # will reconnect next iteration
        else:
            # No WS available — just sleep briefly
            _shutdown_event.wait(timeout=1.0)

        # --- Periodic HTTP reconciliation sweep ---
        now = time.monotonic()
        if now - last_sweep >= _SWEEP_INTERVAL:
            last_sweep = now
            _http_reconciliation_sweep()
            _check_timeouts()

    # --- Shutdown ---
    if ws is not None:
        try:
            ws.close()
        except Exception:
            pass
    logger.info("ComfyUI poller thread stopped")


def _build_ws_url() -> str:
    """Build a WebSocket URL using the shared client ID."""
    base_url = _get_base_url()
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    return f"{ws_url}/ws?clientId={COMFYUI_CLIENT_ID}"


# ---------------------------------------------------------------------------
# WebSocket message handling
# ---------------------------------------------------------------------------

def _handle_ws_message(raw: str) -> None:
    """Route a single WebSocket message to the correct job(s)."""
    try:
        msg = _json.loads(raw)
    except Exception:
        return

    msg_type = msg.get("type")
    data = msg.get("data", {})
    msg_prompt_id = data.get("prompt_id")

    # "status" messages are broadcast (no prompt_id) — update all queued jobs
    if msg_type == "status":
        queue_remaining = data.get("status", {}).get("exec_info", {}).get("queue_remaining", 0)
        with _registry_lock:
            entries = [e for e in _job_registry.values() if e.phase == "queued"]
        for entry in entries:
            if queue_remaining > 0:
                if entry.last_queue_remaining >= 0 and queue_remaining < entry.last_queue_remaining:
                    delta = entry.last_queue_remaining - queue_remaining
                    entry.queue_position = max(1, entry.queue_position - delta)
                entry.last_queue_remaining = queue_remaining
                _emit(entry.job_id, entry.prompt_id, entry.total_images, phase="queued",
                      queue_position=entry.queue_position,
                      message=f"Queue position: {entry.queue_position}" if entry.queue_position > 0
                          else "Waiting in queue\u2026")
        return

    # All other messages require a prompt_id match
    if not msg_prompt_id:
        return

    with _registry_lock:
        job_id = _prompt_to_job.get(msg_prompt_id)
        if not job_id:
            return  # not our job (external ComfyUI submission)
        entry = _job_registry.get(job_id)
        if not entry:
            return

    if msg_type == "execution_start":
        entry.phase = "running"
        if entry.exec_start_time is None:
            entry.exec_start_time = time.monotonic()
        _emit(entry.job_id, entry.prompt_id, entry.total_images, phase="running",
              message="Starting generation\u2026")

    elif msg_type == "progress":
        entry.phase = "running"
        if entry.exec_start_time is None:
            entry.exec_start_time = time.monotonic()
        value = data.get("value", 0)
        max_val = data.get("max", 1)
        pct = value / max_val if max_val > 0 else 0.0
        _emit(entry.job_id, entry.prompt_id, entry.total_images, phase="running",
              progress=pct, message=f"Step {value}/{max_val}")

    elif msg_type == "executing":
        if data.get("node") is None:
            # Execution complete
            logger.info("WS completion detected for job %s (prompt_id=%s)", entry.job_id, entry.prompt_id)
            _complete_job(entry)
        else:
            entry.phase = "running"
            if entry.exec_start_time is None:
                entry.exec_start_time = time.monotonic()


# ---------------------------------------------------------------------------
# HTTP reconciliation & timeout checks
# ---------------------------------------------------------------------------

def _http_reconciliation_sweep() -> None:
    """Check all active jobs via HTTP. Catches anything WebSocket missed."""
    with _registry_lock:
        entries = list(_job_registry.values())
    if not entries:
        return

    for entry in entries:
        # Skip jobs that were already unregistered by a WS completion
        # between the snapshot and this iteration
        with _registry_lock:
            if entry.job_id not in _job_registry:
                continue

        try:
            progress = get_job_progress(entry.prompt_id)
        except Exception:
            continue

        if progress.status == "completed":
            logger.info("HTTP sweep completion detected for job %s (prompt_id=%s)", entry.job_id, entry.prompt_id)
            _complete_job(entry)
        elif progress.status == "failed":
            _fail_job(entry, progress.message or "Generation failed")
        elif progress.status == "running" and entry.phase == "queued":
            entry.phase = "running"
            entry.exec_start_time = time.monotonic()
            _emit(entry.job_id, entry.prompt_id, entry.total_images, phase="running",
                  progress=progress.progress,
                  message=progress.message or "Generating\u2026")
            entry.unknown_count = 0
        elif progress.status == "unknown":
            entry.unknown_count += 1
            if entry.unknown_count >= _UNKNOWN_FAIL_COUNT:
                _fail_job(entry, "Job not found in ComfyUI")
        else:
            entry.unknown_count = 0


def _check_timeouts() -> None:
    """Check all jobs for queue/execution timeouts."""
    now = time.monotonic()
    with _registry_lock:
        entries = list(_job_registry.values())

    for entry in entries:
        if entry.exec_start_time is not None:
            if now - entry.exec_start_time > _MAX_EXEC_TIME:
                # Final check before failing
                try:
                    progress = get_job_progress(entry.prompt_id)
                    if progress.status == "completed":
                        _complete_job(entry)
                        continue
                except Exception:
                    pass
                _fail_job(entry, f"Timed out after 10 minutes (execution timeout)")
        elif now - entry.registered_at > _MAX_QUEUE_TIME:
            try:
                progress = get_job_progress(entry.prompt_id)
                if progress.status == "completed":
                    _complete_job(entry)
                    continue
            except Exception:
                pass
            _fail_job(entry, f"Timed out after 30 minutes (queue timeout)")


# ---------------------------------------------------------------------------
# Job completion & failure
# ---------------------------------------------------------------------------

def _complete_job(entry: JobEntry) -> None:
    """Handle job completion: emit, unregister."""
    time.sleep(0.5)  # let ComfyUI write history
    output_images = _emit_completed(entry.job_id, entry.prompt_id, entry.total_images)
    logger.info("Job %s completed (%d output images)", entry.job_id, len(output_images))
    _unregister_job(entry.job_id, entry.prompt_id)


def _fail_job(entry: JobEntry, message: str) -> None:
    """Handle job failure: emit, unregister."""
    _emit(entry.job_id, entry.prompt_id, entry.total_images, phase="failed",
          message=message, complete=True)
    logger.warning("Job %s failed: %s", entry.job_id, message)
    _unregister_job(entry.job_id, entry.prompt_id)


def _unregister_job(job_id: str, prompt_id: str) -> None:
    """Remove a job from the registry."""
    with _registry_lock:
        _job_registry.pop(job_id, None)
        _prompt_to_job.pop(prompt_id, None)
