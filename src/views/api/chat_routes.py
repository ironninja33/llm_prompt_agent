"""Chat CRUD and message SSE endpoints."""

import logging
import os
import queue
import re
import uuid

from flask import request, jsonify, Response, send_from_directory

from src.views.api import api_bp
from src.views.api.helpers import _sse_event
from src.controllers import chat_controller
from src.agent import runner
from src.config import BASE_DIR

logger = logging.getLogger(__name__)

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")


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
    attachment_urls = []  # Persistent URLs for message metadata

    if request.content_type and "multipart/form-data" in request.content_type:
        content = request.form.get("content", "").strip()
        files = request.files.getlist("attachments")

        if not content:
            return jsonify({"error": "content is required"}), 400

        for f in files:
            raw = f.read()
            filename = f.filename or "image.png"

            # Detect generated image attachments by filename pattern
            # (generated_{jobId}_{imageId}.png) and use existing API URL
            gen_match = re.match(r"generated_([^_]+)_(\d+)\.png$", filename)
            if gen_match:
                job_id, image_id = gen_match.group(1), gen_match.group(2)
                full_url = f"/api/generate/image/{job_id}/{image_id}"
            else:
                full_url = _save_upload(chat_id, filename, raw)

            attachment_data.append({
                "filename": filename,
                "content_type": f.content_type or "image/png",
                "data": raw,
            })
            attachment_urls.append(full_url)
    else:
        data = request.get_json()
        if not data or not data.get("content"):
            return jsonify({"error": "content is required"}), 400
        content = data["content"]

    agent_run = chat_controller.send_message(
        chat_id, content,
        attachments=attachment_data,
        attachment_urls=attachment_urls if attachment_urls else None,
    )

    if agent_run is None:
        return jsonify({"error": "Chat not found"}), 404

    return Response(
        _stream_from_queue(agent_run),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@api_bp.route("/uploads/<chat_id>/<filename>", methods=["GET"])
def serve_upload(chat_id, filename):
    """Serve an uploaded attachment file."""
    upload_dir = os.path.join(UPLOADS_DIR, chat_id)
    if not os.path.isdir(upload_dir):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(upload_dir, filename)


def _save_upload(chat_id: str, original_filename: str, data: bytes) -> str:
    """Save an uploaded file to disk and return its serving URL."""
    upload_dir = os.path.join(UPLOADS_DIR, chat_id)
    os.makedirs(upload_dir, exist_ok=True)

    # Generate a unique filename to avoid collisions
    ext = os.path.splitext(original_filename)[1] or ".png"
    unique_name = f"{uuid.uuid4().hex[:12]}{ext}"
    filepath = os.path.join(upload_dir, unique_name)

    with open(filepath, "wb") as f:
        f.write(data)

    return f"/api/uploads/{chat_id}/{unique_name}"


@api_bp.route("/chats/<chat_id>/cancel", methods=["POST"])
def cancel_message(chat_id):
    """Cancel a streaming response."""
    # Signal the background agent to stop
    runner.cancel_run(chat_id)

    data = request.get_json()
    message_id = data.get("message_id") if data else None
    if message_id:
        from src.models import chat as chat_model
        chat_model.delete_messages_after(chat_id, message_id)

    return jsonify({"ok": True})


@api_bp.route("/chats/<chat_id>/messages/<int:message_id>", methods=["PUT"])
def edit_message(chat_id, message_id):
    """Edit and resubmit a user message. Streams the new response via SSE."""
    data = request.get_json()
    if not data or not data.get("content"):
        return jsonify({"error": "content is required"}), 400

    content = data["content"]

    agent_run = chat_controller.edit_and_resubmit(chat_id, message_id, content)

    if agent_run is None:
        return jsonify({"error": "Chat not found"}), 404

    return Response(
        _stream_from_queue(agent_run),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _stream_from_queue(agent_run):
    """Read events from the AgentRun queue and yield SSE strings.

    If the client disconnects, the background thread continues to completion.
    Yields keepalive comments on queue timeout to detect dead connections.
    """
    try:
        while True:
            try:
                event = agent_run.events.get(timeout=0.5)
            except queue.Empty:
                # Keepalive — lets Flask detect disconnected clients
                yield ": keepalive\n\n"
                continue

            if event is None:
                # Sentinel — agent run finished
                break

            event_type = event.get("type", "unknown")
            yield _sse_event(event_type, event)
    except GeneratorExit:
        # Client disconnected — agent thread continues in background
        logger.debug("SSE client disconnected for chat %s; agent continues", agent_run.chat_id)
    except Exception as e:
        logger.error("SSE stream error: %s", e, exc_info=True)
        yield _sse_event("error", {"message": str(e)})
