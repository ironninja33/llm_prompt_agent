#!/usr/bin/env python3
"""Repair corrupted ChromaDB: extract all data, nuke, rebuild.

No Gemini API calls. Vectors recovered from HNSW binary + WAL queue.
Document text for training_prompts re-read from source files on disk.

Usage:
    conda run -n llm_prompt_agent python scripts/repair_chroma_full.py
"""

import json
import os
import pickle
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
BACKUP_PATH = f"{CHROMA_DB_PATH}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# HNSW binary layout for 3072-dim embeddings
EMBEDDING_DIM = 3072
ELEMENT_SIZE = 12428
VEC_OFFSET = 132

# Segment UUIDs (from: SELECT s.id, s.collection, s.scope, c.name FROM segments s JOIN collections c ON s.collection = c.id)
TRAINING_VECTOR_SEGMENT = "c670fd46-44cf-4f53-b40d-4dceebcc8ddf"
TRAINING_META_SEGMENT = "65cca24b-4b11-491c-8017-5e3c70ec420a"
TRAINING_COLLECTION_UUID = "3392c8c5-97aa-4623-9bfb-dfc648f61ec2"
TRAINING_WAL_TOPIC = f"persistent://default/default/{TRAINING_COLLECTION_UUID}"


def extract_working_collections():
    """Extract generated_prompts and deleted_prompts via ChromaDB API.

    Runs in a SUBPROCESS to ensure the PersistentClient and all its locks
    are fully released before we proceed to nuke and rebuild.
    """
    # Write a helper script to a temp file, run it, read the results
    helper = f'''
import chromadb, json, sys
import numpy as np

client = chromadb.PersistentClient(path="{CHROMA_DB_PATH}")
results = {{}}

for name in ["generated_prompts", "deleted_prompts"]:
    col = client.get_collection(name)
    count = col.count()
    print(f"  {{name}}: {{count}} docs", file=sys.stderr)

    if count == 0:
        results[name] = {{"ids":[],"documents":[],"embeddings":[],"metadatas":[]}}
        continue

    all_ids, all_docs, all_embs, all_metas = [], [], [], []
    offset = 0
    while offset < count:
        batch = col.get(limit=500, offset=offset,
                        include=["documents","metadatas","embeddings"])
        if not batch["ids"]:
            break
        all_ids.extend(batch["ids"])
        all_docs.extend(batch["documents"])
        # Convert numpy arrays to plain lists for JSON serialization
        for emb in batch["embeddings"]:
            if isinstance(emb, np.ndarray):
                all_embs.append(emb.tolist())
            else:
                all_embs.append(emb)
        all_metas.extend(batch["metadatas"])
        offset += len(batch["ids"])

    results[name] = {{
        "ids": all_ids,
        "documents": all_docs,
        "embeddings": all_embs,
        "metadatas": all_metas,
    }}
    print(f"    Extracted {{len(all_ids)}} docs with embeddings", file=sys.stderr)

# Save data BEFORE stopping client (stop could segfault on training_prompts)
with open(sys.argv[1], "w") as f:
    json.dump(results, f)

try:
    client._system.stop()
except Exception:
    pass
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        out_path = tf.name

    try:
        result = subprocess.run(
            [sys.executable, "-c", helper, out_path],
            capture_output=False, timeout=120,
        )
        if result.returncode not in (0, -11):  # 0=ok, -11=SIGSEGV during stop() is acceptable
            raise RuntimeError(f"Extraction subprocess failed with code {result.returncode}")

        with open(out_path) as f:
            return json.load(f)
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def extract_hnsw_vectors():
    """Read training_prompts vectors directly from HNSW binary file."""
    seg_dir = os.path.join(CHROMA_DB_PATH, TRAINING_VECTOR_SEGMENT)
    data_file = os.path.join(seg_dir, "data_level0.bin")
    pickle_file = os.path.join(seg_dir, "index_metadata.pickle")

    if not os.path.exists(pickle_file):
        print("  WARNING: index_metadata.pickle not found, skipping HNSW")
        return {}

    with open(pickle_file, "rb") as f:
        meta = pickle.load(f)

    label_to_id = meta.get("label_to_id", {})
    print(f"  Pickle: {len(label_to_id)} label-to-id mappings")

    file_size = os.path.getsize(data_file)
    max_elements = file_size // ELEMENT_SIZE
    print(f"  Binary: {file_size:,} bytes = {max_elements} elements")

    with open(data_file, "rb") as f:
        data = f.read()

    vectors = {}
    for label, doc_id in label_to_id.items():
        if label >= max_elements:
            continue
        start = label * ELEMENT_SIZE + VEC_OFFSET
        end = start + EMBEDDING_DIM * 4
        vec = np.frombuffer(data[start:end], dtype=np.float32).copy()
        if np.any(vec != 0):
            vectors[doc_id] = vec

    print(f"  Recovered {len(vectors)} vectors from HNSW")
    return vectors


def extract_wal_entries():
    """Read training_prompts entries from the WAL queue in SQLite."""
    db_path = os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")
    conn = sqlite3.connect(db_path)

    rows = conn.execute(
        "SELECT id, vector, metadata FROM embeddings_queue "
        "WHERE topic = ? AND operation = 0",
        (TRAINING_WAL_TOPIC,),
    ).fetchall()
    conn.close()

    wal = {}
    for doc_id, vec_blob, meta_json in rows:
        meta = json.loads(meta_json) if meta_json else {}
        document = meta.pop("chroma:document", "")
        vec = np.frombuffer(vec_blob, dtype=np.float32).copy() if vec_blob else None
        wal[doc_id] = {"vector": vec, "document": document, "metadata": meta}

    print(f"  Recovered {len(wal)} entries from WAL queue")
    return wal


def get_all_training_ids():
    """Get the canonical list of training_prompts doc_ids from embeddings table."""
    db_path = os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT embedding_id FROM embeddings WHERE segment_id = ?",
        (TRAINING_META_SEGMENT,),
    ).fetchall()
    conn.close()
    ids = [r[0] for r in rows]
    print(f"  {len(ids)} doc_ids in embeddings table")
    return ids


def merge_training_data(all_ids, hnsw_vecs, wal_entries):
    """Merge HNSW vectors, WAL entries, and source files into complete dataset."""
    merged = {}
    stats = {"wal": 0, "hnsw": 0, "disk_text": 0, "no_vec": 0, "no_text": 0}

    for doc_id in all_ids:
        entry = {"vector": None, "document": None, "metadata": None}

        # Vector: WAL wins over HNSW
        if doc_id in wal_entries and wal_entries[doc_id]["vector"] is not None:
            entry["vector"] = wal_entries[doc_id]["vector"]
            entry["document"] = wal_entries[doc_id]["document"]
            entry["metadata"] = wal_entries[doc_id]["metadata"]
            stats["wal"] += 1
        elif doc_id in hnsw_vecs:
            entry["vector"] = hnsw_vecs[doc_id]
            stats["hnsw"] += 1
        else:
            stats["no_vec"] += 1

        # Document text: read from disk if not from WAL
        if not entry["document"]:
            try:
                with open(doc_id, "r", encoding="utf-8") as f:
                    entry["document"] = f.read().strip()
                stats["disk_text"] += 1
            except Exception:
                stats["no_text"] += 1

        # Metadata: derive from path if not from WAL
        if entry["metadata"] is None:
            parts = doc_id.replace("\\", "/").split("/")
            entry["metadata"] = {
                "concept": parts[-2] if len(parts) >= 2 else "unknown",
                "base_dir": "",
                "source_file": parts[-1],
                "dir_type": "training",
            }

        merged[doc_id] = entry

    print(f"  Vectors from WAL:  {stats['wal']}")
    print(f"  Vectors from HNSW: {stats['hnsw']}")
    print(f"  Text from disk:    {stats['disk_text']}")
    print(f"  Missing vectors:   {stats['no_vec']}")
    print(f"  Missing text:      {stats['no_text']}")
    return merged, stats


def rebuild(collections_data, training_data):
    """Delete chroma_db, create fresh, insert everything."""
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # --- generated_prompts & deleted_prompts ---
    descs = {
        "generated_prompts": "Prompts extracted from generated output images",
        "deleted_prompts": "Graveyard: embeddings from quality/wrong_direction deletions",
        "training_prompts": "Prompts from training data text files",
    }
    for name in ["generated_prompts", "deleted_prompts"]:
        data = collections_data[name]
        col = client.get_or_create_collection(name=name, metadata={"description": descs[name]})
        if not data["ids"]:
            print(f"  {name}: 0 docs (empty)")
            continue
        for i in range(0, len(data["ids"]), 500):
            end = min(i + 500, len(data["ids"]))
            col.add(
                ids=data["ids"][i:end],
                documents=data["documents"][i:end],
                embeddings=data["embeddings"][i:end],
                metadatas=data["metadatas"][i:end],
            )
        print(f"  {name}: {len(data['ids'])} docs inserted")

    # --- training_prompts ---
    col = client.get_or_create_collection(
        name="training_prompts", metadata={"description": descs["training_prompts"]}
    )
    inserted = 0
    skipped = 0
    batch_ids, batch_docs, batch_embs, batch_metas = [], [], [], []

    for doc_id, entry in training_data.items():
        if entry["vector"] is None or not entry["document"]:
            skipped += 1
            continue
        batch_ids.append(doc_id)
        batch_docs.append(entry["document"])
        batch_embs.append(entry["vector"].tolist())
        batch_metas.append(entry["metadata"])

        if len(batch_ids) >= 500:
            col.add(ids=batch_ids, documents=batch_docs,
                    embeddings=batch_embs, metadatas=batch_metas)
            inserted += len(batch_ids)
            batch_ids, batch_docs, batch_embs, batch_metas = [], [], [], []

    if batch_ids:
        col.add(ids=batch_ids, documents=batch_docs,
                embeddings=batch_embs, metadatas=batch_metas)
        inserted += len(batch_ids)

    print(f"  training_prompts: {inserted} docs inserted, {skipped} skipped (missing vec/text)")

    # Flush cleanly
    try:
        client._system.stop()
    except Exception:
        pass

    return inserted, skipped


def verify():
    """Load the rebuilt ChromaDB in a subprocess and verify each collection."""
    helper = f'''
import chromadb, sys

client = chromadb.PersistentClient(path="{CHROMA_DB_PATH}")
ok = True

for name in ["training_prompts", "generated_prompts", "deleted_prompts"]:
    col = client.get_collection(name)
    count = col.count()
    if count > 0:
        sample = col.get(limit=1, include=["documents", "embeddings", "metadatas"])
        has_emb = sample["embeddings"] is not None and len(sample["embeddings"]) > 0
        emb_dim = len(sample["embeddings"][0]) if has_emb else 0
        print(f"  {{name}}: {{count}} docs | emb_dim={{emb_dim}} | embeddings_ok={{has_emb}}")
        if not has_emb:
            ok = False
    else:
        print(f"  {{name}}: 0 docs")

try:
    client._system.stop()
except Exception:
    pass

sys.exit(0 if ok else 1)
'''
    result = subprocess.run(
        [sys.executable, "-c", helper],
        capture_output=False, timeout=60,
    )
    return result.returncode == 0


def main():
    print("=" * 60)
    print("ChromaDB Full Repair (no Gemini API calls)")
    print("=" * 60)

    # --- Step 1: Extract working collections ---
    print("\n[1/6] Extracting working collections (generated + deleted)...")
    collections_data = extract_working_collections()

    # --- Step 2: Extract training_prompts from HNSW ---
    print("\n[2/6] Extracting training_prompts vectors from HNSW binary...")
    hnsw_vecs = extract_hnsw_vectors()

    # --- Step 3: Extract training_prompts from WAL ---
    print("\n[3/6] Extracting training_prompts from WAL queue...")
    wal_entries = extract_wal_entries()

    # --- Step 4: Merge ---
    print("\n[4/6] Merging training data...")
    all_ids = get_all_training_ids()
    training_data, stats = merge_training_data(all_ids, hnsw_vecs, wal_entries)

    total_recoverable = stats["wal"] + stats["hnsw"]
    print(f"\n  Total recoverable: {total_recoverable}/{len(all_ids)}")

    if total_recoverable == 0:
        print("ERROR: No vectors recovered. Aborting.")
        sys.exit(1)

    # --- Step 5: Backup & nuke ---
    print(f"\n[5/6] Backup -> {os.path.basename(BACKUP_PATH)}")
    shutil.copytree(CHROMA_DB_PATH, BACKUP_PATH)
    backup_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(BACKUP_PATH)
        for f in fns
    )
    print(f"  Backup: {backup_size // 1024 // 1024} MB")
    print(f"  Deleting {CHROMA_DB_PATH}...")
    shutil.rmtree(CHROMA_DB_PATH)

    # --- Step 6: Rebuild ---
    print("\n[6/6] Rebuilding ChromaDB...")
    inserted, skipped = rebuild(collections_data, training_data)

    # --- Verify ---
    print("\n--- Verification ---")
    ok = verify()

    print("\n" + "=" * 60)
    if ok:
        print("REPAIR SUCCESSFUL")
    else:
        print("REPAIR COMPLETED WITH WARNINGS")
    print(f"Backup: {BACKUP_PATH}")
    if stats["no_vec"] > 0:
        print(f"Note: {stats['no_vec']} docs missing vectors will be re-embedded on next app startup")
    print("=" * 60)


if __name__ == "__main__":
    main()
