"""Threading, progress, and status management for clustering runs."""

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global clustering state
# ---------------------------------------------------------------------------
_clustering_lock = threading.Lock()
_clustering_running = False
_status_listeners: list = []
_current_status: dict = {"phase": "idle", "complete": True}


@dataclass
class ClusteringProgress:
    """Current state of a clustering run."""
    phase: str = "idle"
    message: str = ""
    current: int = 0
    total: int = 0
    complete: bool = False


def add_status_listener(callback):
    """Register a callback for clustering status updates."""
    _status_listeners.append(callback)


def remove_status_listener(callback):
    """Remove a status listener."""
    if callback in _status_listeners:
        _status_listeners.remove(callback)


def _emit_status(progress: ClusteringProgress):
    """Notify all listeners of a status update."""
    global _current_status
    _current_status = {
        "phase": progress.phase,
        "message": progress.message,
        "current": progress.current,
        "total": progress.total,
        "complete": progress.complete,
    }
    for listener in _status_listeners[:]:
        try:
            listener(progress)
        except Exception as e:
            logger.error(f"Error in clustering status listener: {e}")


def get_current_status() -> dict:
    """Return the latest clustering status snapshot for polling."""
    return _current_status


def is_running() -> bool:
    """Check if clustering is currently running."""
    return _clustering_running


def start_clustering_single(folder_path: str, k: int, source_type: str | None = None):
    """Start single-folder recluster in a background thread."""
    global _clustering_running
    with _clustering_lock:
        if _clustering_running:
            logger.warning("Clustering already running, skipping single-folder recluster")
            return
        _clustering_running = True

    thread = threading.Thread(
        target=_run_clustering_single,
        args=(folder_path, k, source_type),
        daemon=True,
    )
    thread.start()


def _run_clustering_single(folder_path: str, k: int, source_type: str | None = None):
    """Single-folder recluster — runs in a background thread."""
    global _clustering_running
    progress = ClusteringProgress()

    try:
        from .core import generate_intra_folder_clusters

        progress.phase = "intra_folder"
        progress.message = f"Reclustering '{folder_path}' with k={k}..."
        _emit_status(progress)

        generate_intra_folder_clusters(folder_path=folder_path, k=k, force=True, source_type=source_type)

        # Run summarization for this folder
        progress.phase = "summarizing"
        progress.message = f"Generating summaries for '{folder_path}'..."
        _emit_status(progress)
        try:
            from src.services.summarizer.service import run_folder_summarization
            run_folder_summarization(folder_path, source_type)
        except Exception as e:
            logger.warning(f"Folder summarization failed for '{folder_path}': {e}", exc_info=True)

        progress.phase = "complete"
        progress.complete = True
        progress.message = "Reclustering complete."
        _emit_status(progress)

        from src.services.cache_service import cache_manager
        cache_manager.invalidate()

    except Exception as e:
        logger.error(f"Single-folder recluster error: {e}", exc_info=True)
        progress.phase = "error"
        progress.complete = True
        progress.message = f"Reclustering failed: {str(e)}"
        _emit_status(progress)

    finally:
        _clustering_running = False


def start_clustering(cross_folder: bool = True, intra_folder: bool = True, force: bool = False):
    """Start clustering in a background thread."""
    global _clustering_running
    with _clustering_lock:
        if _clustering_running:
            logger.warning("Clustering already running, skipping")
            return
        _clustering_running = True

    thread = threading.Thread(
        target=_run_clustering,
        args=(cross_folder, intra_folder, force),
        daemon=True,
    )
    thread.start()


def _run_clustering(cross_folder: bool, intra_folder: bool, force: bool):
    """Main clustering logic — runs in a background thread."""
    global _clustering_running
    progress = ClusteringProgress()

    try:
        from .core import generate_cross_folder_clusters, generate_intra_folder_clusters

        if cross_folder:
            progress.phase = "cross_folder"
            progress.message = "Running cross-folder clustering..."
            _emit_status(progress)
            generate_cross_folder_clusters()

        if intra_folder:
            progress.phase = "intra_folder"
            progress.message = "Running intra-folder clustering..."
            _emit_status(progress)
            generate_intra_folder_clusters(force=force)

        # Run LLM summarization instead of TF-IDF labeling
        progress.phase = "summarizing"
        progress.message = "Generating LLM summaries..."
        _emit_status(progress)
        try:
            from src.services.summarizer.service import run_summarization
            run_summarization(
                progress_callback=lambda msg: _emit_summarizer_progress(progress, msg),
            )
        except Exception as e:
            logger.warning("LLM summarization failed: %s", e, exc_info=True)

        progress.phase = "complete"
        progress.complete = True
        progress.message = "Clustering complete."
        _emit_status(progress)

        from src.services.cache_service import cache_manager
        cache_manager.invalidate()

    except Exception as e:
        logger.error(f"Clustering error: {e}", exc_info=True)
        progress.phase = "error"
        progress.message = f"Clustering failed: {str(e)}"
        _emit_status(progress)

    finally:
        _clustering_running = False


def _emit_summarizer_progress(progress: ClusteringProgress, message: str):
    """Forward summarizer progress messages to clustering listeners."""
    progress.message = message
    _emit_status(progress)
