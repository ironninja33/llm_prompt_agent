-- Reference only. Canonical schema defined by migrations.py.
-- This file represents the final state after all 18 migrations are applied.
-- It is not executed by the application.

CREATE TABLE schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE data_directories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL UNIQUE,
    dir_type    TEXT NOT NULL CHECK (dir_type IN ('training', 'output')),
    active      INTEGER DEFAULT 1,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chats (
    id          TEXT PRIMARY KEY,
    title       TEXT DEFAULT 'New Chat',
    summary     TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content     TEXT NOT NULL,
    metadata    TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_messages_chat_id      ON messages(chat_id);
CREATE INDEX idx_messages_chat_created ON messages(chat_id, created_at);

CREATE TABLE agent_state (
    chat_id     TEXT PRIMARY KEY REFERENCES chats(id) ON DELETE CASCADE,
    state_json  TEXT NOT NULL,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE clusters (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_type    TEXT NOT NULL CHECK (cluster_type IN ('cross_folder', 'intra_folder')),
    folder_path     TEXT,
    cluster_index   INTEGER NOT NULL,
    label           TEXT NOT NULL,
    centroid        TEXT,
    prompt_count    INTEGER DEFAULT 0,
    source_type     TEXT DEFAULT 'training',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_clusters_type   ON clusters(cluster_type);
CREATE INDEX idx_clusters_folder ON clusters(folder_path);

CREATE TABLE cluster_assignments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL,
    source_type TEXT NOT NULL,
    cluster_id  INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    distance    REAL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cluster_assignments_doc     ON cluster_assignments(doc_id);
CREATE INDEX idx_cluster_assignments_cluster ON cluster_assignments(cluster_id);

CREATE TABLE clustering_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type        TEXT NOT NULL CHECK (run_type IN ('cross_folder', 'intra_folder', 'full')),
    folder_path     TEXT,
    total_prompts   INTEGER DEFAULT 0,
    num_clusters    INTEGER DEFAULT 0,
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at    TIMESTAMP
);

CREATE TABLE generation_jobs (
    id          TEXT PRIMARY KEY,
    chat_id     TEXT REFERENCES chats(id) ON DELETE SET NULL,
    message_id  INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    prompt_id   TEXT,
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'queued', 'running', 'completed', 'failed')),
    source      TEXT NOT NULL DEFAULT 'chat'
                CHECK (source IN ('chat', 'scan', 'browser')),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_gen_jobs_chat    ON generation_jobs(chat_id);
CREATE INDEX idx_gen_jobs_message ON generation_jobs(message_id);
CREATE INDEX idx_gen_jobs_source  ON generation_jobs(source);
CREATE INDEX idx_gen_jobs_status  ON generation_jobs(status);

CREATE TABLE generated_images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL REFERENCES generation_jobs(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    subfolder       TEXT DEFAULT '',
    width           INTEGER,
    height          INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_size       INTEGER,
    file_path       TEXT,
    metadata_status TEXT NOT NULL DEFAULT 'complete'
);

CREATE INDEX  idx_gen_images_job          ON generated_images(job_id);
CREATE UNIQUE INDEX idx_gen_images_file_path     ON generated_images(file_path);
CREATE UNIQUE INDEX idx_gen_images_job_filename  ON generated_images(job_id, filename);

CREATE TABLE generation_settings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL UNIQUE REFERENCES generation_jobs(id) ON DELETE CASCADE,
    positive_prompt TEXT NOT NULL,
    negative_prompt TEXT,
    base_model      TEXT,
    loras           TEXT,
    output_folder   TEXT,
    seed            INTEGER DEFAULT -1,
    num_images      INTEGER DEFAULT 1,
    workflow_name   TEXT,
    extra_settings  TEXT,
    sampler         TEXT,
    cfg_scale       REAL,
    scheduler       TEXT,
    steps           INTEGER
);

CREATE INDEX idx_gen_settings_job ON generation_settings(job_id);

CREATE TABLE tool_calls (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id       INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    tool_name        TEXT NOT NULL,
    parameters       TEXT,
    response_summary TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sequence         INTEGER NOT NULL DEFAULT 0,
    iteration        INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX idx_tool_calls_message ON tool_calls(message_id);

CREATE TABLE thumbnail_cache (
    file_path    TEXT PRIMARY KEY,
    thumbnail    BLOB NOT NULL,
    width        INTEGER,
    height       INTEGER,
    source_mtime REAL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE image_quality_scores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id    INTEGER NOT NULL UNIQUE REFERENCES generated_images(id) ON DELETE CASCADE,
    overall     REAL NOT NULL,
    raw_score   REAL,
    model_used  TEXT NOT NULL,
    dimensions  TEXT,
    notes       TEXT,
    scored_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE scoring_batches (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id     TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'submitted'
                 CHECK (status IN ('submitted', 'processing', 'completed', 'failed')),
    total_images INTEGER NOT NULL,
    scored_count INTEGER DEFAULT 0,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE scoring_batch_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id    INTEGER NOT NULL REFERENCES scoring_batches(id) ON DELETE CASCADE,
    image_id    INTEGER NOT NULL REFERENCES generated_images(id) ON DELETE CASCADE,
    request_idx INTEGER NOT NULL
);

CREATE INDEX idx_scoring_batch_items_batch ON scoring_batch_items(batch_id);

CREATE TABLE image_keep_flags (
    image_id   INTEGER PRIMARY KEY REFERENCES generated_images(id) ON DELETE CASCADE,
    flagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE folder_summaries (
    folder_path TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
