"""Application configuration."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database paths
SQLITE_DB_PATH = os.path.join(BASE_DIR, "app.db")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# Default model settings
DEFAULT_MODEL_AGENT = "gemini-3-pro-preview"
DEFAULT_MODEL_EMBEDDING = "gemini-embedding-001"
DEFAULT_MODEL_SUMMARY = "gemini-2.5-flash-lite"

# Gemini rate limit (requests per minute)
DEFAULT_GEMINI_RATE_LIMIT = 3000

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Embedding batch size
EMBEDDING_BATCH_SIZE = 10
