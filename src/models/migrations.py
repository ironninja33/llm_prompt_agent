"""Schema migrations list for the application database."""

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
    (13, "Folder summaries for two-tier dataset map", [
        """CREATE TABLE IF NOT EXISTS folder_summaries (
            folder_path TEXT PRIMARY KEY,
            summary     TEXT NOT NULL,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
    ]),
    (14, "Fix subfolder concepts in ChromaDB vector store", [
        # Remove cluster assignments for subfolder clusters
        """DELETE FROM cluster_assignments WHERE cluster_id IN (
            SELECT id FROM clusters WHERE folder_path LIKE '%/%'
        )""",
        # Remove the subfolder cluster entries themselves
        """DELETE FROM clusters WHERE folder_path LIKE '%/%'""",
        # ChromaDB metadata fixup is done in Python by _fix_subfolder_concepts()
    ]),
    (15, "Add source_type column to clusters for per-source-type intra-folder clustering", [
        """ALTER TABLE clusters ADD COLUMN source_type TEXT DEFAULT 'training'""",
    ]),
    (16, "Add sequence and iteration columns to tool_calls for ordering and metrics", [
        """ALTER TABLE tool_calls ADD COLUMN sequence INTEGER NOT NULL DEFAULT 0""",
        """ALTER TABLE tool_calls ADD COLUMN iteration INTEGER NOT NULL DEFAULT 1""",
    ]),
    (17, "Generation metrics: sessions, lineage, deletion log", [
        """CREATE TABLE generation_sessions (
            id TEXT PRIMARY KEY,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            generation_count INTEGER DEFAULT 0
        )""",
        """ALTER TABLE generation_jobs ADD COLUMN session_id TEXT""",
        """ALTER TABLE generation_jobs ADD COLUMN parent_job_id TEXT""",
        """ALTER TABLE generation_jobs ADD COLUMN lineage_depth INTEGER DEFAULT 0""",
        """CREATE INDEX idx_gen_jobs_session ON generation_jobs(session_id)""",
        """CREATE INDEX idx_gen_jobs_parent ON generation_jobs(parent_job_id)""",
        """CREATE TABLE deletion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            image_id INTEGER NOT NULL,
            positive_prompt TEXT,
            output_folder TEXT,
            session_id TEXT,
            lineage_depth INTEGER DEFAULT 0,
            reason TEXT NOT NULL CHECK (reason IN ('quality', 'wrong_direction', 'duplicate', 'space')),
            deleted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE INDEX idx_deletion_log_reason ON deletion_log(reason)""",
        """CREATE INDEX idx_deletion_log_folder ON deletion_log(output_folder)""",
    ]),
]
