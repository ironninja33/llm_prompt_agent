"""Chat CRUD and message SSE endpoints."""

import logging

from flask import request, jsonify, Response

from src.views.api import api_bp
from src.views.api.helpers import _sse_event
from src.controllers import chat_controller

logger = logging.getLogger(__name__)


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
