"""SQLAlchemy Core database connection management and schema migrations."""

import os
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text

from src.config import SQLITE_DB_PATH, DEFAULT_COMFYUI_BASE_URL

logger = logging.getLogger(__name__)

# Singleton engine
_engine = None


def get_engine():
    """Return the singleton SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is not None:
        return _engine

    db_url = os.environ.get("DATABASE_URL", f"sqlite:///{SQLITE_DB_PATH}")

    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    # SQLite pragmas applied on every new raw DBAPI connection
    if db_url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return _engine


@contextmanager
def get_db():
    """Context manager yielding a SQLAlchemy Connection with auto-commit/rollback."""
    engine = get_engine()
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            yield conn
            trans.commit()
        except Exception:
            trans.rollback()
            raise


def row_to_dict(row) -> dict:
    """Convert a SQLAlchemy Row to a plain dict."""
    return dict(row._mapping)


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
        """DELETE FROM generated_images WHERE id IN (
            SELECT b.id
            FROM generated_images a
            JOIN generated_images b ON a.filename = b.filename
            JOIN generation_jobs gj ON b.job_id = gj.id
            WHERE a.file_path IS NULL AND b.file_path IS NOT NULL AND gj.source = 'scan'
        )""",
        """DELETE FROM generation_jobs WHERE source = 'scan'
           AND id NOT IN (SELECT job_id FROM generated_images)""",
    ]),
    (8, "Normalize timestamps: convert float epoch values to ISO-8601 text", [
        """UPDATE generation_jobs
           SET created_at = datetime(created_at, 'unixepoch')
           WHERE typeof(created_at) = 'real'""",
        """UPDATE generation_jobs
           SET completed_at = datetime(completed_at, 'unixepoch')
           WHERE typeof(completed_at) = 'real'""",
    ]),
    (9, "Deduplicate generated_images and add unique constraint on (job_id, filename)", [
        # Remove duplicate rows (keep the lowest id per job_id+filename pair)
        """DELETE FROM generated_images WHERE id NOT IN (
            SELECT MIN(id) FROM generated_images GROUP BY job_id, filename
        )""",
        # Prevent future duplicates at the DB level
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_gen_images_job_filename
           ON generated_images(job_id, filename)""",
    ]),
    (10, "Re-run dedup cleanup (migration 9 index creation may have been skipped)", [
        """DELETE FROM generated_images WHERE id NOT IN (
            SELECT MIN(id) FROM generated_images GROUP BY job_id, filename
        )""",
        # Drop and recreate to ensure it exists even if migration 9's attempt failed
        """DROP INDEX IF EXISTS idx_gen_images_job_filename""",
        """CREATE UNIQUE INDEX idx_gen_images_job_filename
           ON generated_images(job_id, filename)""",
    ]),
    (11, "Fix truncated output_folder for scanned images in nested directories", [
        # Actual fixup is done in Python by _fix_truncated_output_folders()
        # because it needs os.path.relpath which isn't available in SQL.
    ]),
    (12, "Cleanup assistant: quality scores, batch tracking, keep flags", [
        """CREATE TABLE IF NOT EXISTS image_quality_scores (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id    INTEGER NOT NULL UNIQUE REFERENCES generated_images(id) ON DELETE CASCADE,
            overall     REAL NOT NULL,
            character   REAL NOT NULL,
            composition REAL NOT NULL,
            artifacts   REAL NOT NULL,
            theme       REAL NOT NULL,
            detail      REAL NOT NULL,
            expression  REAL NOT NULL,
            notes       TEXT,
            model_used  TEXT NOT NULL,
            scored_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS scoring_batches (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id    TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'submitted'
                        CHECK (status IN ('submitted', 'processing', 'completed', 'failed')),
            total_images INTEGER NOT NULL,
            scored_count INTEGER DEFAULT 0,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS scoring_batch_items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id    INTEGER NOT NULL REFERENCES scoring_batches(id) ON DELETE CASCADE,
            image_id    INTEGER NOT NULL REFERENCES generated_images(id) ON DELETE CASCADE,
            request_idx INTEGER NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_scoring_batch_items_batch
           ON scoring_batch_items(batch_id)""",
        """CREATE TABLE IF NOT EXISTS image_keep_flags (
            image_id    INTEGER PRIMARY KEY REFERENCES generated_images(id) ON DELETE CASCADE,
            flagged_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
    ]),
]


def get_current_version(conn) -> int:
    """Get the current schema version."""
    try:
        result = conn.execute(text("SELECT MAX(version) FROM schema_version"))
        row = result.fetchone()
        return row[0] if row[0] is not None else 0
    except Exception:
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
                    conn.execute(text(sql))
                conn.execute(
                    text("INSERT INTO schema_version (version) VALUES (:v)"),
                    {"v": version}
                )
                logger.info(f"Migration {version} applied successfully")

        # One-time Python fixup for migration 11
        if current_version < 11:
            _fix_truncated_output_folders(conn)

        # Insert default settings if they don't exist
        _insert_default_settings(conn)


def _fix_truncated_output_folders(conn):
    """Fix output_folder values that were truncated to just the leaf directory.

    Scanned images in nested directories like ``character/subdir`` had
    output_folder set to ``"subdir"`` instead of ``"character/subdir"``.
    Recompute from file_path relative to the configured output root.
    """
    # Get output root directories
    result = conn.execute(
        text("SELECT path FROM data_directories WHERE dir_type = 'output' AND active = 1")
    )
    output_roots = [os.path.normpath(row[0]) for row in result.fetchall()]
    if not output_roots:
        return

    # Find all images with file_path set and a generation_settings record
    result = conn.execute(
        text("""SELECT gs.job_id, gs.output_folder, gi.file_path
           FROM generation_settings gs
           JOIN generated_images gi ON gs.job_id = gi.job_id
           WHERE gi.file_path IS NOT NULL""")
    )
    rows = result.fetchall()

    fixed = 0
    for row in rows:
        job_id = row[0]
        stored_folder = row[1] or ""
        filepath = row[2]

        dirpath = os.path.normpath(os.path.dirname(filepath))
        correct_folder = None
        for root in output_roots:
            if dirpath.startswith(root):
                rel = os.path.relpath(dirpath, root)
                correct_folder = rel if rel != "." else ""
                break

        if correct_folder is not None and correct_folder != stored_folder:
            conn.execute(
                text("UPDATE generation_settings SET output_folder = :folder WHERE job_id = :job_id"),
                {"folder": correct_folder, "job_id": job_id},
            )
            fixed += 1

    if fixed:
        logger.info(f"Fixed {fixed} truncated output_folder values")


def _insert_default_settings(conn):
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
        "cluster_label_terms": "3",
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
        "auto_organize_output": "false",
    }

    for key, value in defaults.items():
        conn.execute(
            text("INSERT OR IGNORE INTO settings (key, value) VALUES (:key, :value)"),
            {"key": key, "value": value}
        )
