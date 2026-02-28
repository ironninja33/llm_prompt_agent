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
    (4, "Browser support: nullable chat_id, source column, sampler settings, tool calls, thumbnail cache", [
        # -- Recreate generation_jobs with nullable chat_id + source column --
        """CREATE TABLE generation_jobs_new (
            id TEXT PRIMARY KEY,
            chat_id TEXT REFERENCES chats(id) ON DELETE SET NULL,
            message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
            prompt_id TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'queued', 'running', 'completed', 'failed')),
            source TEXT NOT NULL DEFAULT 'chat'
                CHECK (source IN ('chat', 'scan', 'browser')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )""",
        """INSERT INTO generation_jobs_new (id, chat_id, message_id, prompt_id, status, source, created_at, completed_at)
            SELECT id, chat_id, message_id, prompt_id, status, 'chat', created_at, completed_at
            FROM generation_jobs""",
        """DROP TABLE generation_jobs""",
        """ALTER TABLE generation_jobs_new RENAME TO generation_jobs""",
        """CREATE INDEX idx_gen_jobs_chat ON generation_jobs(chat_id)""",
        """CREATE INDEX idx_gen_jobs_message ON generation_jobs(message_id)""",
        """CREATE INDEX idx_gen_jobs_source ON generation_jobs(source)""",
        """CREATE INDEX idx_gen_jobs_status ON generation_jobs(status)""",
        # -- Add sampler/CFG/scheduler/steps columns to generation_settings --
        """ALTER TABLE generation_settings ADD COLUMN sampler TEXT""",
        """ALTER TABLE generation_settings ADD COLUMN cfg_scale REAL""",
        """ALTER TABLE generation_settings ADD COLUMN scheduler TEXT""",
        """ALTER TABLE generation_settings ADD COLUMN steps INTEGER""",
        # -- Add file metadata columns to generated_images --
        """ALTER TABLE generated_images ADD COLUMN file_size INTEGER""",
        """ALTER TABLE generated_images ADD COLUMN file_path TEXT""",
        # -- Add tool_calls table --
        """CREATE TABLE tool_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            tool_name TEXT NOT NULL,
            parameters TEXT,
            response_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE INDEX idx_tool_calls_message ON tool_calls(message_id)""",
        # -- Add thumbnail_cache table --
        """CREATE TABLE thumbnail_cache (
            file_path TEXT PRIMARY KEY,
            thumbnail BLOB NOT NULL,
            width INTEGER,
            height INTEGER,
            source_mtime REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
    ]),
    (5, "Lazy loading: metadata_status column and unique file_path index", [
        """ALTER TABLE generated_images ADD COLUMN metadata_status TEXT NOT NULL DEFAULT 'complete'""",
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_gen_images_file_path ON generated_images(file_path)""",
    ]),
    (6, "Re-parse scanned images to extract seeds from metadata", [
        """DELETE FROM generation_settings WHERE job_id IN (
            SELECT id FROM generation_jobs WHERE source = 'scan'
        )""",
        """UPDATE generated_images SET metadata_status = 'pending' WHERE job_id IN (
            SELECT id FROM generation_jobs WHERE source = 'scan'
        )""",
    ]),
    (7, "Fix duplicate image records: merge scan duplicates into generation originals", [
        # For images generated through the app, add_generated_image left file_path NULL.
        # fast_register_images then created scan duplicates with file_path set.
        # Fix: copy file_path/file_size from scan duplicate to the generation original,
        # then delete the scan duplicate and its orphan job.
        # Step 1: Delete scan duplicate image records (frees unique file_path)
        """DELETE FROM generated_images WHERE id IN (
            SELECT b.id
            FROM generated_images a
            JOIN generated_images b ON a.filename = b.filename
            JOIN generation_jobs gj ON b.job_id = gj.id
            WHERE a.file_path IS NULL AND b.file_path IS NOT NULL AND gj.source = 'scan'
        )""",
        # Step 2: Update generation originals with file_path from a fresh filesystem scan
        # (handled by fast_register_images on next browse — it detects NULL file_path records)
        # Step 3: Clean up orphan scan jobs (no images left)
        """DELETE FROM generation_jobs WHERE source = 'scan'
           AND id NOT IN (SELECT job_id FROM generated_images)""",
    ]),
    (8, "Normalize timestamps: convert float epoch values to ISO-8601 text", [
        # created_at: convert real (float epoch) values to ISO-8601 text
        """UPDATE generation_jobs
           SET created_at = datetime(created_at, 'unixepoch')
           WHERE typeof(created_at) = 'real'""",
        # completed_at: same treatment
        """UPDATE generation_jobs
           SET completed_at = datetime(completed_at, 'unixepoch')
           WHERE typeof(completed_at) = 'real'""",
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
        "comfyui_workflow_filename": "",
        "comfyui_workflow_json": "",
        "comfyui_workflow_hash": "",
        "comfyui_workflow_api_cache": "",
        "comfyui_object_info_cache": "",
        "thumbnail_size_chat": "large",
        "thumbnail_size_browser": "medium",
        "search_mode": "keyword",
        "viewer_sidebar_visible": "true",
        "comfyui_default_sampler": "",
        "comfyui_default_cfg": "",
        "comfyui_default_scheduler": "",
        "comfyui_default_steps": "",
    }

    for key, value in defaults.items():
        conn.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
            (key, value)
        )
