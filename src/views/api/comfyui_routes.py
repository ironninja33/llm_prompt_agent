"""ComfyUI health, models, and workflow endpoints."""

import logging
from dataclasses import asdict

from flask import request, jsonify

from src.views.api import api_bp
from src.controllers import settings_controller
from src.services import comfyui_service, workflow_manager

logger = logging.getLogger(__name__)


@api_bp.route("/comfyui/health", methods=["GET"])
def comfyui_health():
    """Check ComfyUI server connectivity."""
    result = comfyui_service.check_health()
    return jsonify(result)


@api_bp.route("/comfyui/status", methods=["GET"])
def comfyui_status():
    """Combined health + queue size for header status indicator."""
    health = comfyui_service.check_health()
    queue_size = 0
    if health.get("ok"):
        queue = comfyui_service.get_queue_status()
        running = len(queue.get("queue_running", []))
        pending = len(queue.get("queue_pending", []))
        queue_size = running + pending
    return jsonify({
        "ok": health.get("ok", False),
        "queue_size": queue_size,
    })


@api_bp.route("/comfyui/models/<model_type>", methods=["GET"])
def comfyui_models(model_type):
    """List available ComfyUI models. model_type: 'loras' or 'diffusion_models'."""
    if model_type not in ("loras", "diffusion_models"):
        return jsonify({"error": "Invalid model type"}), 400
    models = comfyui_service.get_available_models(model_type)
    return jsonify(models)


@api_bp.route("/comfyui/output-folders", methods=["GET"])
def comfyui_output_folders():
    """List output subdirectories for folder autocomplete."""
    folders = comfyui_service.get_output_subfolders()
    return jsonify(folders)


@api_bp.route("/comfyui/sampler-options", methods=["GET"])
def comfyui_sampler_options():
    """Get available samplers/schedulers for the active workflow's sampler node."""
    from src.controllers import workflow_controller
    return jsonify(workflow_controller.get_sampler_options())


@api_bp.route("/comfyui/workflow-definitions", methods=["GET"])
def get_workflow_definitions():
    """List available workflow definitions and their field schemas."""
    registry = workflow_manager.get_registry()
    definitions = []
    for d in registry.list_definitions():
        d_copy = dict(d)
        d_copy["fields"] = [asdict(f) for f in d["fields"]]
        definitions.append(d_copy)
    return jsonify(definitions)


@api_bp.route("/comfyui/validate-workflow", methods=["POST"])
def validate_workflow():
    """Validate a workflow file path."""
    data = request.get_json()
    path = data.get("path", "") if data else ""
    result = settings_controller.validate_workflow(path)
    return jsonify(result)


@api_bp.route("/comfyui/workflow/api", methods=["POST"])
def upload_api_workflow():
    """Upload an API-format workflow JSON file."""
    from src.controllers import workflow_controller

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No filename"}), 400
    if not f.filename.lower().endswith(".json"):
        return jsonify({"error": "File must be a .json file"}), 400

    try:
        raw = f.read().decode("utf-8")
    except UnicodeDecodeError as e:
        return jsonify({"error": f"Invalid file encoding: {e}"}), 400

    result = workflow_controller.upload_api_workflow(raw, f.filename)
    status_code = 200 if result.get("status") != "error" else 400
    return jsonify(result), status_code


@api_bp.route("/comfyui/workflow/ui", methods=["POST"])
def upload_ui_workflow():
    """Upload a UI-format workflow JSON file (for extra_pnginfo metadata)."""
    from src.controllers import workflow_controller

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No filename"}), 400
    if not f.filename.lower().endswith(".json"):
        return jsonify({"error": "File must be a .json file"}), 400

    try:
        raw = f.read().decode("utf-8")
    except UnicodeDecodeError as e:
        return jsonify({"error": f"Invalid file encoding: {e}"}), 400

    result = workflow_controller.upload_ui_workflow(raw, f.filename)
    status_code = 200 if result.get("status") != "error" else 400
    return jsonify(result), status_code


@api_bp.route("/comfyui/workflow", methods=["GET"])
def get_workflow_info():
    """Get info about the currently stored workflow."""
    from src.controllers import workflow_controller
    return jsonify(workflow_controller.get_workflow_info())


@api_bp.route("/comfyui/workflow", methods=["DELETE"])
def delete_workflow():
    """Remove the stored workflow."""
    from src.controllers import workflow_controller
    workflow_controller.delete_workflow()
    return jsonify({"status": "deleted"})
