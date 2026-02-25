"""Main routes — serves page templates."""

from flask import Blueprint, render_template

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Serve the chat page."""
    return render_template("chat.html", active_page="chat")


@main_bp.route("/browser")
def browser():
    """Serve the image browser page."""
    return render_template("browser.html", active_page="browser")
