"""Workflow controller — business logic for ComfyUI workflow management.

Handles upload, validation, storage, deletion, and preparation of workflows
for generation submission.

Supports **dual upload**: users upload both API-format and UI-format workflow
JSON files separately.  The API format is submitted to ComfyUI's ``/prompt``
endpoint; the UI format is passed as ``extra_pnginfo`` for introspection
nodes.
"""

import copy
import hashlib
import json
import logging

from src.models import settings as settings_model
from src.services import workflow_manager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_api_workflow(raw_json: str, filename: str) -> dict:
    """Upload an API-format workflow JSON file.

    API format is a flat dict keyed by node-ID strings, each with
    ``class_type`` and ``inputs``.  This is what ComfyUI's ``/prompt``
    endpoint expects.

    Args:
        raw_json: Raw JSON string of the workflow file.
        filename: Original filename.

    Returns:
        Dict with ``status``, ``filename``, ``workflow_name``, ``message``.
    """
    # -- Parse JSON -----------------------------------------------------------
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as e:
        return {"status": "error", "filename": filename, "workflow_name": None,
                "message": f"Invalid JSON: {e}"}

    # -- Validate it's API format ---------------------------------------------
    if not workflow_manager.is_api_format(parsed):
        return {
            "status": "error",
            "filename": filename,
            "workflow_name": None,
            "message": (
                "This looks like a UI-format workflow (has a 'nodes' array). "
                "Please upload it using the UI Workflow input instead, and "
                "export the API format via 'Save (API Format)' in ComfyUI."
            ),
        }

    # -- Content hash — skip if identical to stored ---------------------------
    content_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    existing_hash = settings_model.get_setting("comfyui_workflow_api_hash") or ""
    if existing_hash == content_hash:
        existing_filename = (
            settings_model.get_setting("comfyui_workflow_api_filename") or filename
        )
        return {
            "status": "unchanged",
            "filename": existing_filename,
            "workflow_name": _workflow_name_for(existing_filename),
            "message": "API workflow content is identical — no update needed",
        }

    # -- Validate against WorkflowRegistry ------------------------------------
    defn = workflow_manager.get_definition_for_workflow(filename)
    if not defn:
        return {
            "status": "error",
            "filename": filename,
            "workflow_name": None,
            "message": "No matching workflow definition found for this file",
        }

    # -- Store ----------------------------------------------------------------
    settings_model.update_settings({
        "comfyui_workflow_api_filename": filename,
        "comfyui_workflow_api_json": raw_json,
        "comfyui_workflow_api_hash": content_hash,
    })

    logger.info(
        "API workflow '%s' uploaded (definition=%s, %d nodes)",
        filename, defn.name, len(parsed),
    )

    return {
        "status": "uploaded",
        "filename": filename,
        "workflow_name": defn.name,
        "message": f"API workflow '{filename}' uploaded successfully",
    }


def upload_ui_workflow(raw_json: str, filename: str) -> dict:
    """Upload a UI-format workflow JSON file.

    UI format has a top-level ``"nodes"`` array with ``widgets_values``,
    ``title``, link arrays, etc.  This is the JSON exported via regular
    "Save" in ComfyUI.  It is used for ``extra_pnginfo`` (introspection
    nodes like KJNodes' ``GetWidgetValue``).

    Args:
        raw_json: Raw JSON string of the workflow file.
        filename: Original filename.

    Returns:
        Dict with ``status``, ``filename``, ``workflow_name``, ``message``.
    """
    # -- Parse JSON -----------------------------------------------------------
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as e:
        return {"status": "error", "filename": filename, "workflow_name": None,
                "message": f"Invalid JSON: {e}"}

    # -- Validate it's UI format ----------------------------------------------
    if workflow_manager.is_api_format(parsed):
        return {
            "status": "error",
            "filename": filename,
            "workflow_name": None,
            "message": (
                "This looks like an API-format workflow. "
                "Please upload it using the API Workflow input instead, and "
                "export the UI format via regular 'Save' in ComfyUI."
            ),
        }

    # -- Store ----------------------------------------------------------------
    settings_model.update_settings({
        "comfyui_workflow_ui_filename": filename,
        "comfyui_workflow_ui_json": raw_json,
    })

    # Try to find a matching definition for informational purposes
    defn = workflow_manager.get_definition_for_workflow(filename)
    workflow_name = defn.name if defn else None

    logger.info(
        "UI workflow '%s' uploaded (definition=%s, %d nodes)",
        filename, workflow_name or "none", len(parsed.get("nodes", [])),
    )

    return {
        "status": "uploaded",
        "filename": filename,
        "workflow_name": workflow_name,
        "message": f"UI workflow '{filename}' uploaded successfully",
    }


def get_workflow_info() -> dict:
    """Return information about the currently stored workflows.

    Returns:
        Dict with ``api_filename``, ``ui_filename``, ``has_api_workflow``,
        ``has_ui_workflow``, ``workflow_name``, ``valid``.
    """
    api_filename = settings_model.get_setting("comfyui_workflow_api_filename") or ""
    ui_filename = settings_model.get_setting("comfyui_workflow_ui_filename") or ""
    has_api = bool(api_filename and settings_model.get_setting("comfyui_workflow_api_json"))
    has_ui = bool(ui_filename and settings_model.get_setting("comfyui_workflow_ui_json"))

    # Also check for new-style keys (from the converter era) as fallbacks
    if not has_api:
        new_api_cache = settings_model.get_setting("comfyui_workflow_api_cache") or ""
        new_filename = settings_model.get_setting("comfyui_workflow_filename") or ""
        if new_api_cache and new_filename:
            api_filename = new_filename
            has_api = True
    if not has_ui:
        new_ui_json = settings_model.get_setting("comfyui_workflow_json") or ""
        new_filename = settings_model.get_setting("comfyui_workflow_filename") or ""
        if new_ui_json and new_filename:
            ui_filename = new_filename
            has_ui = True

    # Determine workflow name from API filename (primary) or UI filename
    ref_filename = api_filename or ui_filename
    if ref_filename:
        defn = workflow_manager.get_definition_for_workflow(ref_filename)
        workflow_name = defn.name if defn else None
        valid = defn is not None
    else:
        workflow_name = None
        valid = False

    return {
        "api_filename": api_filename,
        "ui_filename": ui_filename,
        "has_api_workflow": has_api,
        "has_ui_workflow": has_ui,
        "workflow_name": workflow_name,
        "valid": valid,
        # Legacy compat — some frontend code may check these
        "filename": api_filename or ui_filename,
        "has_workflow": has_api,
    }


def delete_workflow() -> None:
    """Clear all workflow-related settings (both old and new keys)."""
    settings_model.update_settings({
        # Dual-upload keys
        "comfyui_workflow_api_filename": "",
        "comfyui_workflow_api_json": "",
        "comfyui_workflow_api_hash": "",
        "comfyui_workflow_ui_filename": "",
        "comfyui_workflow_ui_json": "",
        # Converter-era keys (clean up any leftovers)
        "comfyui_workflow_filename": "",
        "comfyui_workflow_json": "",
        "comfyui_workflow_hash": "",
        "comfyui_workflow_api_cache": "",
        "comfyui_object_info_cache": "",
    })
    logger.info("All workflow data cleared")


def prepare_for_generation(user_settings: dict) -> tuple[dict, dict | None]:
    """Load stored workflows and apply user settings for generation submission.

    Loads the API workflow JSON and UI workflow JSON from settings, finds the
    matching workflow definition, and applies user settings to both.

    Args:
        user_settings: Dict of user-provided generation settings (prompt,
            model, loras, seed, etc.).

    Returns:
        Tuple of ``(prepared_api_workflow, prepared_ui_workflow)``.
        The UI workflow may be ``None`` if not uploaded.

    Raises:
        ValueError: If no API workflow is configured or the workflow
            definition is not found.
    """
    # -- Load stored workflows ------------------------------------------------
    api_json_str = settings_model.get_setting("comfyui_workflow_api_json") or ""
    ui_json_str = settings_model.get_setting("comfyui_workflow_ui_json") or ""
    api_filename = settings_model.get_setting("comfyui_workflow_api_filename") or ""

    # Fallback to converter-era keys if dual-upload keys are empty
    if not api_json_str:
        api_json_str = settings_model.get_setting("comfyui_workflow_api_cache") or ""
    if not api_filename:
        api_filename = settings_model.get_setting("comfyui_workflow_filename") or ""
    if not ui_json_str:
        ui_json_str = settings_model.get_setting("comfyui_workflow_json") or ""

    if not api_json_str:
        raise ValueError(
            "No API-format workflow configured. Upload one in Settings > ComfyUI."
        )

    # -- Find workflow definition ---------------------------------------------
    defn = workflow_manager.get_definition_for_workflow(api_filename) if api_filename else None

    # -- Parse and apply settings to API workflow -----------------------------
    try:
        api_workflow = json.loads(api_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Stored API workflow is invalid JSON: {e}")

    if defn:
        prepared_api = defn.apply_settings(api_workflow, user_settings)
        user_settings["workflow_name"] = defn.name
    else:
        # No definition found — pass through without settings application
        logger.warning(
            "No workflow definition for '%s' — submitting API workflow as-is",
            api_filename,
        )
        prepared_api = api_workflow
        user_settings["workflow_name"] = "unknown"

    # -- Parse and apply settings to UI workflow (for extra_pnginfo) ----------
    prepared_ui = None
    if ui_json_str and defn:
        try:
            ui_workflow = json.loads(ui_json_str)
            prepared_ui = defn.apply_settings(ui_workflow, user_settings)
        except json.JSONDecodeError:
            logger.warning("Stored UI workflow JSON is invalid — omitting from extra_pnginfo")
        except Exception as e:
            logger.warning("Failed to apply settings to UI workflow: %s", e)

    return prepared_api, prepared_ui


def get_sampler_options() -> dict:
    """Get available samplers/schedulers for the active workflow's sampler node.

    Queries ComfyUI's ``/object_info`` endpoint, finds the sampler node type
    used in the stored workflow, and extracts the enum values for
    ``sampler_name`` and ``scheduler``.

    Returns:
        Dict with ``samplers``, ``schedulers`` (lists of strings), and
        ``sampler_node_type`` (the class_type of the sampler node, or None).
    """
    from src.services import comfyui_service

    empty = {"samplers": [], "schedulers": [], "sampler_node_type": None}

    api_json_str = settings_model.get_setting("comfyui_workflow_api_json") or ""
    if not api_json_str:
        return empty

    try:
        api_workflow = json.loads(api_json_str)
    except json.JSONDecodeError:
        return empty

    api_filename = settings_model.get_setting("comfyui_workflow_api_filename") or ""
    defn = workflow_manager.get_definition_for_workflow(api_filename) if api_filename else None
    if not defn or not hasattr(defn, "_find_sampler_node_api"):
        return empty

    _, sampler_node = defn._find_sampler_node_api(api_workflow)
    if not sampler_node:
        return empty

    class_type = sampler_node.get("class_type", "")
    object_info = comfyui_service.get_object_info()
    node_info = object_info.get(class_type, {})
    required_inputs = node_info.get("input", {}).get("required", {})

    samplers: list[str] = []
    schedulers: list[str] = []
    for key, val in required_inputs.items():
        if key in ("sampler_name", "sampler") and isinstance(val, list) and val and isinstance(val[0], list):
            samplers = val[0]
        if key == "scheduler" and isinstance(val, list) and val and isinstance(val[0], list):
            schedulers = val[0]

    return {
        "samplers": samplers,
        "schedulers": schedulers,
        "sampler_node_type": class_type,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _workflow_name_for(filename: str) -> str | None:
    """Look up the workflow definition name for *filename*."""
    defn = workflow_manager.get_definition_for_workflow(filename)
    return defn.name if defn else None
