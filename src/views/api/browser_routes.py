"""Browser API endpoints — directory listing, search, and browser-initiated generation."""

import logging

from flask import request, jsonify

from src.views.api import api_bp
from src.controllers import browser_controller

logger = logging.getLogger(__name__)


@api_bp.route("/browser/listing", methods=["GET"])
def browser_listing():
    """Directory listing. Params: path (optional), offset, limit."""
    path = request.args.get("path", "")
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 50))
    sort = request.args.get("sort", "date")

    if not path:
        # Root listing
        roots = browser_controller.get_root_listing(sort=sort)
        return jsonify({
            "directories": roots,
            "images": [],
            "total_image_count": 0,
            "has_more": False,
            "breadcrumb": [{"name": "Root", "path": ""}],
        })

    result = browser_controller.get_directory_contents(path, offset, limit, sort=sort)
    result["breadcrumb"] = browser_controller.get_breadcrumb(path)
    return jsonify(result)


@api_bp.route("/browser/poll", methods=["GET"])
def browser_poll():
    """Check for new files since timestamp. Params: path, since."""
    path = request.args.get("path", "")
    since = float(request.args.get("since", 0))

    result = browser_controller.poll_new_files(path, since)
    return jsonify(result)


@api_bp.route("/browser/search", methods=["GET"])
def browser_search():
    """Search images. Params: q, mode (keyword|embedding), offset, limit."""
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "keyword")
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 50))

    if not query:
        return jsonify({"images": [], "total_count": 0, "has_more": False})

    if mode == "embedding":
        result = browser_controller.search_embedding(query, offset, limit)
    else:
        result = browser_controller.search_keyword(query, offset, limit)

    return jsonify(result)


@api_bp.route("/browser/generate", methods=["POST"])
def browser_generate():
    """Submit generation without chat context. Body: { settings }."""
    from src.controllers import generation_controller

    data = request.get_json()
    if not data or not data.get("settings"):
        return jsonify({"error": "settings required"}), 400

    settings = data["settings"]
    job = generation_controller.submit_generation(
        chat_id=None, message_id=None, settings=settings, source="browser"
    )

    return jsonify(job)
