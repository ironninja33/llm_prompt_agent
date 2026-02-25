"""REST API endpoints and SSE streaming routes."""

from flask import Blueprint

api_bp = Blueprint("api", __name__)

# Import sub-modules so their @api_bp.route() decorators register
from src.views.api import chat_routes      # noqa: F401, E402
from src.views.api import settings_routes  # noqa: F401, E402
from src.views.api import comfyui_routes   # noqa: F401, E402
from src.views.api import generation_routes  # noqa: F401, E402
