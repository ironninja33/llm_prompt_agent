"""REST API endpoints and SSE streaming routes."""

import json
import logging
import queue
import threading
from dataclasses import asdict

from flask import Blueprint, request, jsonify, Response

from src.controllers import chat_controller, generation_controller, settings_controller
from src.services import comfyui_service, ingestion_service, clustering_service, workflow_manager
from src.models import generation as gen_model, settings, vector_store

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


# ── Placeholder SVG for missing/deleted images ───────────────────────────

MISSING_IMAGE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">'
    '<rect width="256" height="256" fill="#3a3a4a"/>'
    '<line x1="80" y1="80" x2="176" y2="176" stroke="#888" stroke-width="6" stroke-linecap="round"/>'
    '<line x1="176" y1="80" x2="80" y2="176" stroke="#888" stroke-width="6" stroke-linecap="round"/>'
    '<text x="128" y="210" text-anchor="middle" fill="#888" font-family="sans-serif" font-size="14">'
    'Image not found</text>'
    '</svg>'
)


# ── Helper: SSE formatting ───────────────────────────────────────────────

def _sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ── Chat endpoints ───────────────────────────────────────────────────────

@api_bp.route("/chats", methods=["GET"])
def list_chats():
    """List all chats sorted by most recent."""
    chats = chat_controller.get_all_chats()
    return jsonify(chats)


@api_bp.route("/chats", methods=["POST"])
def create_chat():
    """Create a new chat session."""
    chat = chat_controller.create_chat()
    return jsonify(chat), 201


@api_bp.route("/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    """Delete a chat and all its messages."""
    success = chat_controller.delete_chat(chat_id)
    if success:
        return jsonify({"ok": True})
    return jsonify({"error": "Chat not found"}), 404


@api_bp.route("/chats/<chat_id>/messages", methods=["GET"])
def get_messages(chat_id):
    """Get all messages for a chat."""
    messages = chat_controller.get_messages(chat_id)
    return jsonify(messages)


@api_bp.route("/chats/<chat_id>/messages", methods=["POST"])
def send_message(chat_id):
    """Send a user message and stream the agent response via SSE.

    Supports both JSON (text only) and multipart/form-data (with attachments).
    """
    attachment_data = []

    if request.content_type and "multipart/form-data" in request.content_type:
        content = request.form.get("content", "").strip()
        files = request.files.getlist("attachments")

        if not content:
            return jsonify({"error": "content is required"}), 400

        for f in files:
            raw = f.read()
            attachment_data.append({
                "filename": f.filename,
                "content_type": f.content_type or "image/png",
                "data": raw,
            })
    else:
        data = request.get_json()
        if not data or not data.get("content"):
            return jsonify({"error": "content is required"}), 400
        content = data["content"]

    def generate():
        try:
            for event in chat_controller.send_message(
                chat_id, content, attachments=attachment_data
            ):
                event_type = event.get("type", "unknown")
                yield _sse_event(event_type, event)
        except Exception as e:
            logger.error(f"SSE stream error: {e}", exc_info=True)
            yield _sse_event("error", {"message": str(e)})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@api_bp.route("/chats/<chat_id>/messages/<int:message_id>", methods=["PUT"])
def edit_message(chat_id, message_id):
    """Edit and resubmit a user message. Streams the new response via SSE."""
    data = request.get_json()
    if not data or not data.get("content"):
        return jsonify({"error": "content is required"}), 400

    content = data["content"]

    def generate():
        try:
            for event in chat_controller.edit_and_resubmit(chat_id, message_id, content):
                event_type = event.get("type", "unknown")
                yield _sse_event(event_type, event)
        except Exception as e:
            logger.error(f"SSE stream error: {e}", exc_info=True)
            yield _sse_event("error", {"message": str(e)})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


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


# ── ComfyUI endpoints ────────────────────────────────────────────────────

@api_bp.route("/comfyui/health", methods=["GET"])
def comfyui_health():
    """Check ComfyUI server connectivity."""
    result = comfyui_service.check_health()
    return jsonify(result)


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


@api_bp.route("/comfyui/workflow-definitions", methods=["GET"])
def get_workflow_definitions():
    """List available workflow definitions and their field schemas."""
    from dataclasses import asdict as _asdict

    registry = workflow_manager.get_registry()
    definitions = []
    for d in registry.list_definitions():
        d_copy = dict(d)
        d_copy["fields"] = [_asdict(f) for f in d["fields"]]
        definitions.append(d_copy)
    return jsonify(definitions)


@api_bp.route("/comfyui/validate-workflow", methods=["POST"])
def validate_workflow():
    """Validate a workflow file path."""
    data = request.get_json()
    path = data.get("path", "") if data else ""
    result = settings_controller.validate_workflow(path)
    return jsonify(result)


@api_bp.route("/comfyui/workflow", methods=["POST"])
def upload_workflow():
    """Upload an API-format workflow JSON file and store it in the database.

    Accepts multipart/form-data with a 'file' field containing the JSON file.
    The uploaded file must be in **API format** (exported via "Save (API Format)"
    in ComfyUI).  Stores the JSON content and filename in settings.
    If the uploaded content is identical (by hash) to what's already stored,
    it's treated as a no-op.
    """
    import hashlib
    import json as json_mod

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No filename"}), 400
    if not f.filename.lower().endswith(".json"):
        return jsonify({"error": "File must be a .json file"}), 400

    try:
        raw = f.read().decode("utf-8")
        # Validate it's actually JSON
        parsed = json_mod.loads(raw)
    except (UnicodeDecodeError, json_mod.JSONDecodeError) as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    # Verify it's API format (flat dict keyed by node-ID strings)
    if not workflow_manager.is_api_format(parsed):
        return jsonify({
            "error": "This file appears to be in UI format. Please upload the API-format workflow here (exported via 'Save (API Format)' in ComfyUI). Use the separate UI Workflow upload for the UI-format file.",
        }), 400

    content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # Check for duplicate
    existing_hash = settings.get_setting("comfyui_workflow_api_hash")
    if existing_hash == content_hash:
        return jsonify({
            "status": "unchanged",
            "filename": settings.get_setting("comfyui_workflow_api_filename") or f.filename,
            "message": "API workflow content is identical to current — no update needed",
        })

    # Validate against workflow definitions
    defn = workflow_manager.get_definition_for_workflow(f.filename)
    if not defn:
        return jsonify({
            "error": "No matching workflow definition found for this file",
        }), 400

    # Store in settings
    settings.update_settings({
        "comfyui_workflow_api_filename": f.filename,
        "comfyui_workflow_api_json": raw,
        "comfyui_workflow_api_hash": content_hash,
    })

    return jsonify({
        "status": "uploaded",
        "filename": f.filename,
        "workflow_name": defn.name,
        "message": f"API workflow '{f.filename}' uploaded successfully",
    })


@api_bp.route("/comfyui/workflow", methods=["GET"])
def get_workflow_info():
    """Get info about the currently stored workflows (API and UI format)."""
    api_filename = settings.get_setting("comfyui_workflow_api_filename") or ""
    has_api = bool(settings.get_setting("comfyui_workflow_api_json")) and bool(api_filename)

    ui_filename = settings.get_setting("comfyui_workflow_ui_filename") or ""
    has_ui = bool(settings.get_setting("comfyui_workflow_ui_json")) and bool(ui_filename)

    result = {
        "filename": api_filename,
        "has_workflow": has_api,
        "ui_filename": ui_filename,
        "has_ui_workflow": has_ui,
    }

    if api_filename:
        defn = workflow_manager.get_definition_for_workflow(api_filename)
        result["workflow_name"] = defn.name if defn else None
        result["valid"] = defn is not None
    else:
        result["workflow_name"] = None
        result["valid"] = False

    return jsonify(result)


@api_bp.route("/comfyui/workflow", methods=["DELETE"])
def delete_workflow():
    """Remove all stored workflows (API and UI format)."""
    settings.update_settings({
        "comfyui_workflow_api_filename": "",
        "comfyui_workflow_api_json": "",
        "comfyui_workflow_api_hash": "",
        "comfyui_workflow_ui_filename": "",
        "comfyui_workflow_ui_json": "",
    })
    return jsonify({"status": "deleted"})


@api_bp.route("/comfyui/workflow-ui", methods=["POST"])
def upload_ui_workflow():
    """Upload a UI-format workflow JSON file and store it in the database.

    Accepts multipart/form-data with a 'file' field containing the JSON file.
    The uploaded file must be in **UI format** (exported via regular "Save" /
    "Export" in ComfyUI's graph editor).  This format has a top-level ``"nodes"``
    array and is required by introspection nodes like KJNodes' GetWidgetValue.
    """
    import json as json_mod

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No filename"}), 400
    if not f.filename.lower().endswith(".json"):
        return jsonify({"error": "File must be a .json file"}), 400

    try:
        raw = f.read().decode("utf-8")
        parsed = json_mod.loads(raw)
    except (UnicodeDecodeError, json_mod.JSONDecodeError) as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    # Verify it's UI format (has "nodes" array)
    if workflow_manager.is_api_format(parsed):
        return jsonify({
            "error": "This file appears to be in API format. Please upload the UI-format workflow here (exported via regular 'Save' in ComfyUI). Use the API Workflow upload for the API-format file.",
        }), 400

    # Store in settings
    settings.update_settings({
        "comfyui_workflow_ui_filename": f.filename,
        "comfyui_workflow_ui_json": raw,
    })

    return jsonify({
        "status": "uploaded",
        "filename": f.filename,
        "message": f"UI workflow '{f.filename}' uploaded successfully",
    })


@api_bp.route("/comfyui/workflow-ui", methods=["DELETE"])
def delete_ui_workflow():
    """Remove the stored UI-format workflow."""
    settings.update_settings({
        "comfyui_workflow_ui_filename": "",
        "comfyui_workflow_ui_json": "",
    })
    return jsonify({"status": "deleted"})


# ── Generation endpoints ─────────────────────────────────────────────────

@api_bp.route("/generate", methods=["POST"])
def submit_generation():
    """Submit a generation job."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    chat_id = data.get("chat_id")
    message_id = data.get("message_id")
    settings = data.get("settings", {})

    if not chat_id:
        return jsonify({"error": "chat_id is required"}), 400
    if not settings.get("positive_prompt"):
        return jsonify({"error": "positive_prompt is required"}), 400

    try:
        job = generation_controller.submit_generation(chat_id, message_id, settings)
        status_code = 201 if job.get("status") != "failed" else 500
        return jsonify(job), status_code
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/generate/<job_id>/status", methods=["GET"])
def generation_status_sse(job_id):
    """SSE endpoint for generation job progress updates."""
    q = queue.Queue()

    def listener(progress):
        if progress.job_id == job_id:
            try:
                q.put(progress, block=False)
            except queue.Full:
                pass

    comfyui_service.add_status_listener(listener)

    def generate():
        try:
            while True:
                try:
                    progress = q.get(timeout=30)
                    data = {
                        "job_id": progress.job_id,
                        "prompt_id": progress.prompt_id,
                        "phase": progress.phase,
                        "progress": progress.progress,
                        "current_image": progress.current_image,
                        "total_images": progress.total_images,
                        "message": progress.message,
                        "complete": progress.complete,
                    }

                    if progress.complete:
                        # Include output images in the final event
                        if progress.output_images:
                            data["output_images"] = progress.output_images
                        yield _sse_event("generation_complete", data)
                        break
                    else:
                        yield _sse_event("generation_status", data)

                except queue.Empty:
                    yield ": keepalive\n\n"

                    # Check if job still exists and is active
                    job = gen_model.get_job(job_id)
                    if not job or job["status"] in ("completed", "failed"):
                        data = {
                            "job_id": job_id,
                            "phase": job["status"] if job else "failed",
                            "complete": True,
                            "message": "Job finished" if job else "Job not found",
                        }
                        yield _sse_event("generation_complete", data)
                        break
        finally:
            comfyui_service.remove_status_listener(listener)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@api_bp.route("/generate/chat/<chat_id>", methods=["GET"])
def get_chat_generations(chat_id):
    """Get all generations for a chat (for rebuilding UI on reload)."""
    data = generation_controller.get_chat_generations(chat_id)
    return jsonify(data)


@api_bp.route("/generate/image/<job_id>/<int:image_id>", methods=["GET"])
def get_generated_image(job_id, image_id):
    """Proxy a generated image from ComfyUI or filesystem."""
    image = gen_model.get_image(image_id)
    if not image or image.get("job_id") != job_id:
        return jsonify({"error": "Image not found"}), 404

    image_bytes = comfyui_service.get_image(
        image["filename"],
        image.get("subfolder", ""),
    )

    if not image_bytes:
        _cleanup_missing_image(job_id, image_id)
        return Response(
            MISSING_IMAGE_SVG,
            mimetype="image/svg+xml",
            headers={"Cache-Control": "no-cache"},
        )

    # Determine content type from filename
    ext = (
        image["filename"].rsplit(".", 1)[-1].lower()
        if "." in image["filename"]
        else "png"
    )
    content_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(ext, "image/png")

    return Response(
        image_bytes,
        mimetype=content_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@api_bp.route("/generate/image/<job_id>/<int:image_id>", methods=["DELETE"])
def delete_generated_image(job_id, image_id):
    """Delete a generated image file from disk and clean up database records."""
    image = gen_model.get_image(image_id)
    if not image or image.get("job_id") != job_id:
        return jsonify({"error": "Image not found"}), 404

    # Delete the actual file from disk
    comfyui_service.delete_image_file(
        image["filename"],
        image.get("subfolder", ""),
    )

    # Clean up DB records (image row, vector store if no images remain)
    _cleanup_missing_image(job_id, image_id)

    # If the job has no remaining images, remove the job too
    remaining = gen_model.get_job_images(job_id)
    if not remaining:
        gen_model.delete_job(job_id)

    return jsonify({"ok": True})


@api_bp.route("/generate/thumbnail/<job_id>/<int:image_id>", methods=["GET"])
def get_generated_thumbnail(job_id, image_id):
    """Proxy a thumbnail of a generated image."""
    image = gen_model.get_image(image_id)
    if not image or image.get("job_id") != job_id:
        return jsonify({"error": "Image not found"}), 404

    thumb_bytes = comfyui_service.get_image_thumbnail(
        image["filename"],
        image.get("subfolder", ""),
    )

    if not thumb_bytes:
        _cleanup_missing_image(job_id, image_id)
        return Response(
            MISSING_IMAGE_SVG,
            mimetype="image/svg+xml",
            headers={"Cache-Control": "no-cache"},
        )

    return Response(
        thumb_bytes,
        mimetype="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _cleanup_missing_image(job_id: str, image_id: int):
    """Clean up database records for a missing image file.

    Deletes the image record from generated_images and, if the job has
    no remaining images, removes the embedded prompt from the vector store.
    """
    # Delete the image record
    deleted = gen_model.delete_image(image_id)
    if deleted:
        logger.warning(
            "Deleted missing image record id=%d for job %s", image_id, job_id
        )

    # Check if the job still has remaining images
    remaining = gen_model.get_job_images(job_id)
    if not remaining:
        # Remove embedded prompt from vector store (doc_id format: gen_{job_id})
        doc_id = f"gen_{job_id}"
        removed = vector_store.delete_document(doc_id, "output")
        if removed:
            logger.warning(
                "Deleted vector store document %s — no images remain for job %s",
                doc_id,
                job_id,
            )
