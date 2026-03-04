"""Flask application factory and entry point."""

import logging
import os

from flask import Flask

from src.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, BASE_DIR


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        static_folder=os.path.join(BASE_DIR, "src", "static"),
        template_folder=os.path.join(BASE_DIR, "src", "views", "templates"),
    )

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    # Initialize database
    from src.models.database import initialize_database
    logger.info("Initializing database...")
    initialize_database()

    # Initialize ChromaDB
    from src.models.vector_store import initialize as init_vector_store
    logger.info("Initializing vector store...")
    init_vector_store()

    # One-time fixup: repair subfolder concepts in ChromaDB (needs vector store)
    from src.models.settings import get_setting, update_setting
    if get_setting("subfolder_concepts_fixed") != "true":
        from src.models.database import fix_subfolder_concepts
        fix_subfolder_concepts()
        update_setting("subfolder_concepts_fixed", "true")

    # Initialize LLM service (if API key is set)
    from src.services.llm_service import initialize as init_llm
    api_key = get_setting("gemini_api_key")
    if api_key:
        logger.info("Initializing Gemini client...")
        init_llm(api_key)
    else:
        logger.warning("No Gemini API key set. Configure it in Settings.")

    # Register routes
    from src.views.api import api_bp
    from src.views.routes import main_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    # Skip background work in the reloader parent process — only the child
    # (WERKZEUG_RUN_MAIN=true) or non-debug mode should poll and ingest.
    _is_reloader_parent = app.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true"

    # Initialize generation listener
    from src.controllers import generation_controller
    if not _is_reloader_parent:
        logger.info("Initializing generation listener...")
        generation_controller.initialize()

    # Start background ingestion
    from src.services.ingestion_service import start_ingestion
    if not _is_reloader_parent:
        logger.info("Starting background data ingestion...")
        start_ingestion()

    logger.info("Application initialized successfully")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, threaded=True)
