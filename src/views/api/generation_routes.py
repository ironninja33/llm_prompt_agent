"""Generation submit, status SSE, image serving, and deletion endpoints."""

import io
import logging
import os
import queue

from flask import request, jsonify, Response

from src.views.api import api_bp
from src.views.api.helpers import _sse_event, MISSING_IMAGE_SVG, _cleanup_missing_image
from src.controllers import generation_controller
from src.services import comfyui_service
from src.models import generation as gen_model, vector_store

logger = logging.getLogger(__name__)


def _read_image_bytes(image: dict) -> bytes | None:
    """Read image bytes, preferring file_path for scanned images."""
    file_path = image.get("file_path")
    if file_path and os.path.isfile(file_path):
        try:
            with open(file_path, "rb") as fh:
                return fh.read()
        except OSError:
            pass
    # Fallback to ComfyUI output directory search
    return comfyui_service.get_image(
        image["filename"],
        image.get("subfolder", ""),
    )


def _get_thumbnail_bytes(image: dict) -> bytes | None:
    """Get thumbnail bytes, preferring file_path for scanned images."""
    file_path = image.get("file_path")
    if file_path and os.path.isfile(file_path):
        try:
            from PIL import Image as PILImage

            with open(file_path, "rb") as fh:
                raw = fh.read()
            img = PILImage.open(io.BytesIO(raw))
            img.thumbnail((256, 256), PILImage.Resampling.LANCZOS)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except Exception:
            pass
    # Fallback to ComfyUI output directory search
    return comfyui_service.get_image_thumbnail(
        image["filename"],
        image.get("subfolder", ""),
    )


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
                        "queue_position": progress.queue_position,
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

    image_bytes = _read_image_bytes(image)

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


@api_bp.route("/generate/job/<job_id>", methods=["DELETE"])
def delete_generation_job(job_id):
    """Delete a generation job (e.g. a failed job with no images)."""
    job = gen_model.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    # If the job has images, clean them up first
    images = gen_model.get_job_images(job_id)
    for image in images:
        comfyui_service.delete_image_file(
            image["filename"],
            image.get("subfolder", ""),
        )

    # Remove vector store entry if present
    doc_id = f"gen_{job_id}"
    vector_store.delete_document(doc_id, "output")

    # Delete the job (cascades to images and settings)
    gen_model.delete_job(job_id)

    return jsonify({"ok": True})


@api_bp.route("/generate/image/<job_id>/<int:image_id>", methods=["DELETE"])
def delete_generated_image(job_id, image_id):
    """Delete a generated image file from disk and clean up database records."""
    image = gen_model.get_image(image_id)
    if not image or image.get("job_id") != job_id:
        return jsonify({"error": "Image not found"}), 404

    # Find all records with the same filename (scan + generation duplicates).
    # One of them should have the correct file_path for disk deletion.
    duplicates = gen_model.get_images_by_filename(image["filename"])
    duplicate_job_ids = {dup["job_id"] for dup in duplicates if dup["id"] != image_id}

    # Resolve the file on disk: try file_path from this record, then from
    # any duplicate, then fall back to the subfolder-based resolver.
    file_path = image.get("file_path")
    if not file_path or not os.path.isfile(file_path):
        for dup in duplicates:
            if dup.get("file_path") and os.path.isfile(dup["file_path"]):
                file_path = dup["file_path"]
                break

    if file_path and os.path.isfile(file_path):
        try:
            os.remove(file_path)
            logger.info("Deleted image file: %s", file_path)
        except OSError as exc:
            logger.warning("Error deleting image file %s: %s", file_path, exc)
            return jsonify({"error": "Failed to delete image file from disk"}), 500
    else:
        deleted_from_disk = comfyui_service.delete_image_file(
            image["filename"],
            image.get("subfolder", ""),
        )
        if not deleted_from_disk:
            return jsonify({"error": "Failed to delete image file from disk"}), 500

    # Clean up DB records (image row, vector store if no images remain)
    _cleanup_missing_image(job_id, image_id)

    # Remove duplicate records — the file is gone so these are orphaned
    if duplicate_job_ids:
        gen_model.delete_images_by_filename(image["filename"])

    # If the original job has no remaining images, remove it too
    remaining = gen_model.get_job_images(job_id)
    if not remaining:
        gen_model.delete_job(job_id)

    # Clean up orphan jobs from duplicates
    for dup_job_id in duplicate_job_ids:
        remaining = gen_model.get_job_images(dup_job_id)
        if not remaining:
            gen_model.delete_job(dup_job_id)

    return jsonify({"ok": True})


@api_bp.route("/generate/thumbnail/<job_id>/<int:image_id>", methods=["GET"])
def get_generated_thumbnail(job_id, image_id):
    """Proxy a thumbnail of a generated image."""
    image = gen_model.get_image(image_id)
    if not image or image.get("job_id") != job_id:
        return jsonify({"error": "Image not found"}), 404

    thumb_bytes = _get_thumbnail_bytes(image)

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
