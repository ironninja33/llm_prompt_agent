"""Repair script: remove corrupted training_prompts data from ChromaDB.

Deletes the corrupted HNSW index and all SQLite entries for the
training_prompts collection, while leaving generated_prompts and
deleted_prompts untouched.  The ingestion service will re-ingest
training data from source files on next app startup.

Usage:
    conda run -n llm_prompt_agent python repair_training_prompts.py
"""

import os
import shutil
import sqlite3
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
BACKUP_PATH = os.path.join(BASE_DIR, "chroma_db.bak")
SQLITE_PATH = os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")

# training_prompts identifiers
COLLECTION_ID = "5ac66879-56ea-400c-9c72-feb622caf3f8"
METADATA_SEGMENT = "8b27ce24-6caf-43e4-815c-c4711485be3c"
VECTOR_SEGMENT = "d026bee0-c5f8-4b02-a66a-337605a79d99"
HNSW_DIR = os.path.join(CHROMA_DB_PATH, VECTOR_SEGMENT)
QUEUE_TOPIC = f"persistent://default/default/{COLLECTION_ID}"


def main():
    # --- Pre-flight checks ---
    if not os.path.isdir(CHROMA_DB_PATH):
        print(f"ERROR: chroma_db not found at {CHROMA_DB_PATH}")
        sys.exit(1)

    if not os.path.isfile(SQLITE_PATH):
        print(f"ERROR: chroma.sqlite3 not found at {SQLITE_PATH}")
        sys.exit(1)

    # --- Step 1: Back up ---
    if os.path.exists(BACKUP_PATH):
        print(f"Backup already exists at {BACKUP_PATH} — skipping backup.")
    else:
        print(f"Step 1: Backing up chroma_db -> chroma_db.bak ...")
        shutil.copytree(CHROMA_DB_PATH, BACKUP_PATH)
        print(f"  Backup created ({sum(f.stat().st_size for f in os.scandir(BACKUP_PATH) if f.is_file()) // 1024} KB top-level files)")

    # --- Step 2: Delete corrupted HNSW index directory ---
    print(f"\nStep 2: Deleting corrupted HNSW index directory ...")
    if os.path.isdir(HNSW_DIR):
        files = os.listdir(HNSW_DIR)
        shutil.rmtree(HNSW_DIR)
        print(f"  Removed {HNSW_DIR}/ ({len(files)} files)")
    else:
        print(f"  HNSW directory not found (already removed?)")

    # --- Step 3: Clean SQLite tables ---
    print(f"\nStep 3: Cleaning SQLite tables ...")
    db = sqlite3.connect(SQLITE_PATH)
    cursor = db.cursor()

    # 3a: embedding_metadata
    cursor.execute("""
        DELETE FROM embedding_metadata
        WHERE id IN (SELECT id FROM embeddings WHERE segment_id = ?)
    """, (METADATA_SEGMENT,))
    print(f"  3a. embedding_metadata:           {cursor.rowcount} rows deleted")

    # 3b: embedding_fulltext_search_content (FTS content table)
    cursor.execute("""
        DELETE FROM embedding_fulltext_search_content
        WHERE id IN (SELECT rowid FROM embeddings WHERE segment_id = ?)
    """, (METADATA_SEGMENT,))
    print(f"  3b. embedding_fulltext_search_content: {cursor.rowcount} rows deleted")

    # 3c: Rebuild FTS index
    cursor.execute("""
        INSERT INTO embedding_fulltext_search(embedding_fulltext_search) VALUES('rebuild')
    """)
    print(f"  3c. embedding_fulltext_search:     rebuilt")

    # 3d: embeddings
    cursor.execute("""
        DELETE FROM embeddings WHERE segment_id = ?
    """, (METADATA_SEGMENT,))
    print(f"  3d. embeddings:                    {cursor.rowcount} rows deleted")

    # 3e: embeddings_queue
    cursor.execute("""
        DELETE FROM embeddings_queue WHERE topic = ?
    """, (QUEUE_TOPIC,))
    print(f"  3e. embeddings_queue:              {cursor.rowcount} rows deleted")

    # 3f: max_seq_id for both segments
    cursor.execute("DELETE FROM max_seq_id WHERE segment_id = ?", (METADATA_SEGMENT,))
    meta_del = cursor.rowcount
    cursor.execute("DELETE FROM max_seq_id WHERE segment_id = ?", (VECTOR_SEGMENT,))
    vec_del = cursor.rowcount
    print(f"  3f. max_seq_id:                    {meta_del + vec_del} rows deleted")

    db.commit()

    # --- Step 4: Vacuum ---
    print(f"\nStep 4: Vacuuming database ...")
    size_before = os.path.getsize(SQLITE_PATH)
    cursor.execute("VACUUM")
    db.close()
    size_after = os.path.getsize(SQLITE_PATH)
    print(f"  {size_before // (1024*1024)} MB -> {size_after // (1024*1024)} MB")

    # --- Verify ---
    print(f"\nVerification:")
    db = sqlite3.connect(SQLITE_PATH)
    r = db.execute("SELECT COUNT(*) FROM embeddings WHERE segment_id = ?", (METADATA_SEGMENT,)).fetchone()
    print(f"  training_prompts embeddings remaining: {r[0]}")
    r = db.execute("SELECT COUNT(*) FROM embeddings_queue WHERE topic = ?", (QUEUE_TOPIC,)).fetchone()
    print(f"  training_prompts queue entries remaining: {r[0]}")
    print(f"  HNSW directory exists: {os.path.isdir(HNSW_DIR)}")

    # Verify other collections are intact
    for name, seg in [("generated_prompts", "64d56819-1a77-4222-8c25-317b63ab38b1"),
                      ("deleted_prompts", "289c5c7c-ef42-4ee0-ae86-dc311372c0d2")]:
        r = db.execute("SELECT COUNT(*) FROM embeddings WHERE segment_id = ?", (seg,)).fetchone()
        print(f"  {name} embeddings: {r[0]} (unchanged)")
    db.close()

    print(f"\nDone. Run ./run.sh to restart — ingestion will re-embed training data.")


if __name__ == "__main__":
    main()
