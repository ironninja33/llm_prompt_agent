"""SQLite database connection management and schema migrations."""

import sqlite3
import os
import logging
from contextlib import contextmanager

from src.config import SQLITE_DB_PATH, DEFAULT_COMFYUI_BASE_URL

logger = logging.getLogger(__name__)

# Schema migrations in order. Each is (version, description, sql_statements)
MIGRATIONS = [
    (1, "Initial schema", [
        """CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS data_directories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            dir_type TEXT NOT NULL CHECK (dir_type IN ('training', 'output')),
            active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT 'New Chat',
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS agent_state (
            chat_id TEXT PRIMARY KEY REFERENCES chats(id) ON DELETE CASCADE,
            state_json TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)""",
        """CREATE INDEX IF NOT EXISTS idx_messages_chat_created ON messages(chat_id, created_at)""",
    ]),
    (2, "Add clustering tables", [
        """CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_type TEXT NOT NULL CHECK (cluster_type IN ('cross_folder', 'intra_folder')),
            folder_path TEXT,
            cluster_index INTEGER NOT NULL,
            label TEXT NOT NULL,
            centroid TEXT,
            prompt_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE INDEX IF NOT EXISTS idx_clusters_type ON clusters(cluster_type)""",
        """CREATE INDEX IF NOT EXISTS idx_clusters_folder ON clusters(folder_path)""",
        """CREATE TABLE IF NOT EXISTS cluster_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            cluster_id INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
            distance REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE INDEX IF NOT EXISTS idx_cluster_assignments_doc ON cluster_assignments(doc_id)""",
        """CREATE INDEX IF NOT EXISTS idx_cluster_assignments_cluster ON cluster_assignments(cluster_id)""",
        """CREATE TABLE IF NOT EXISTS clustering_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_type TEXT NOT NULL CHECK (run_type IN ('cross_folder', 'intra_folder', 'full')),
            folder_path TEXT,
            total_prompts INTEGER DEFAULT 0,
            num_clusters INTEGER DEFAULT 0,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )""",
    ]),
    (3, "Image generation tables", [
        """CREATE TABLE IF NOT EXISTS generation_jobs (
            id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
            prompt_id TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'queued', 'running', 'completed', 'failed')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS generated_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES generation_jobs(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            subfolder TEXT DEFAULT '',
            width INTEGER,
            height INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS generation_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL UNIQUE REFERENCES generation_jobs(id) ON DELETE CASCADE,
            positive_prompt TEXT NOT NULL,
            negative_prompt TEXT,
            base_model TEXT,
            loras TEXT,
            output_folder TEXT,
            seed INTEGER DEFAULT -1,
            num_images INTEGER DEFAULT 1,
            workflow_name TEXT,
            extra_settings TEXT
        )""",
        """CREATE INDEX IF NOT EXISTS idx_gen_jobs_chat ON generation_jobs(chat_id)""",
        """CREATE INDEX IF NOT EXISTS idx_gen_jobs_message ON generation_jobs(message_id)""",
        """CREATE INDEX IF NOT EXISTS idx_gen_images_job ON generated_images(job_id)""",
        """CREATE INDEX IF NOT EXISTS idx_gen_settings_job ON generation_settings(job_id)""",
    ]),
]


def get_connection() -> sqlite3.Connection:
    """Get a new SQLite connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """Context manager for database connections with auto-commit/rollback."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_current_version(conn: sqlite3.Connection) -> int:
    """Get the current schema version."""
    try:
        cursor = conn.execute("SELECT MAX(version) FROM schema_version")
        row = cursor.fetchone()
        return row[0] if row[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


def initialize_database():
    """Initialize the database and run any pending migrations."""
    os.makedirs(os.path.dirname(SQLITE_DB_PATH) if os.path.dirname(SQLITE_DB_PATH) else ".", exist_ok=True)

    with get_db() as conn:
        current_version = get_current_version(conn)
        logger.info(f"Current database schema version: {current_version}")

        for version, description, statements in MIGRATIONS:
            if version > current_version:
                logger.info(f"Applying migration {version}: {description}")
                for sql in statements:
                    conn.execute(sql)
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (version,)
                )
                logger.info(f"Migration {version} applied successfully")

        # Insert default settings if they don't exist
        _insert_default_settings(conn)


def _insert_default_settings(conn: sqlite3.Connection):
    """Insert default settings if they don't already exist."""
    from src.config import DEFAULT_MODEL_AGENT, DEFAULT_MODEL_EMBEDDING, DEFAULT_MODEL_SUMMARY, DEFAULT_GEMINI_RATE_LIMIT
    from src.agent.system_prompt import DEFAULT_SYSTEM_PROMPT

    defaults = {
        "gemini_api_key": "",
        "model_agent": DEFAULT_MODEL_AGENT,
        "model_embedding": DEFAULT_MODEL_EMBEDDING,
        "model_summary": DEFAULT_MODEL_SUMMARY,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "gemini_rate_limit": str(DEFAULT_GEMINI_RATE_LIMIT),
        "query_k_similar": "10",
        "query_k_theme_intra": "5",
        "query_k_theme_cross": "5",
        "query_k_random": "3",
        "cluster_k_intra": "5",
        "cluster_k_cross": "15",
        "cluster_min_folder_size": "20",
        "cluster_label_terms": "3",  # Number of TF-IDF terms used in cluster labels
        "comfyui_base_url": DEFAULT_COMFYUI_BASE_URL,
        "comfyui_default_model": "",
        "comfyui_default_negative": "",
        "comfyui_workflow_path": "",
        "comfyui_workflow_api_filename": "",
        "comfyui_workflow_api_json": "",
        "comfyui_workflow_api_hash": "",
        "comfyui_workflow_ui_filename": "",
        "comfyui_workflow_ui_json": "",
    }

    for key, value in defaults.items():
        conn.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
            (key, value)
        )
