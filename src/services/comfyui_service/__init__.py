"""ComfyUI service package -- re-exports all public names for backward compatibility.

Callers can continue to use::

    from src.services import comfyui_service
    comfyui_service.check_health()
"""

# --- client.py exports ---
from src.services.comfyui_service.client import (
    check_health,
    get_available_models,
    clear_model_cache,
    get_output_subfolders,
    get_object_info,
    submit_prompt,
    get_queue_status,
    get_history,
    get_job_progress,
    get_image,
    get_image_thumbnail,
    delete_image_file,
    resolve_image_path,
    _resolve_image_path,
    _extract_output_images,
    _get_base_url,
    SubmitResult,
    JobProgress,
)

# --- poller.py exports ---
from src.services.comfyui_service.poller import (
    GenerationProgress,
    add_status_listener,
    remove_status_listener,
    get_cached_job_progress,
    get_active_job_ids,
    poll_job,
)

__all__ = [
    # client
    "check_health",
    "get_available_models",
    "clear_model_cache",
    "get_output_subfolders",
    "get_object_info",
    "submit_prompt",
    "get_queue_status",
    "get_history",
    "get_job_progress",
    "get_image",
    "get_image_thumbnail",
    "delete_image_file",
    "resolve_image_path",
    "_resolve_image_path",
    "_extract_output_images",
    "_get_base_url",
    "SubmitResult",
    "JobProgress",
    # poller
    "GenerationProgress",
    "add_status_listener",
    "remove_status_listener",
    "get_cached_job_progress",
    "get_active_job_ids",
    "poll_job",
]
