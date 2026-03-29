"""Recover ChromaDB from corrupted HNSW index.

Reads vectors from data_level0.bin + embeddings queue, reads docs/metadata
from the SQLite metadata segments, then rebuilds both collections from scratch.
"""
import os
import sys
import sqlite3
import pickle
import shutil
from collections import defaultdict

import numpy as np

CHROMA_PATH = "chroma_db"
DIM = 3072

SEGMENTS = {
    "training": {
        "collection_name": "training_prompts",
        "collection_id": "eaf47480-43cf-4e94-bfb1-a96758009d46",
        "hnsw_dir": "6d5b85ba-9da5-42b3-b304-01ab0d26e4cc",
        "meta_seg": "aaa92d37-6154-46e7-900e-9c63cfb8d2c6",
    },
    "generated": {
        "collection_name": "generated_prompts",
        "collection_id": "02be2d1d-c040-465e-a4ad-a67337f26e4a",
        "hnsw_dir": "50923684-7a2b-4ce3-b047-dda12045a1b9",
        "meta_seg": "ec4f24a0-8336-4a8b-bcf0-456d9b6ed01d",
    },
}


def read_hnsw_vectors(hnsw_dir):
    """Read vectors from an HNSW data_level0.bin using the pickle label mapping."""
    meta_path = os.path.join(hnsw_dir, "index_metadata.pickle")
    data_path = os.path.join(hnsw_dir, "data_level0.bin")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    label_to_id = meta["label_to_id"]
    num_elements = meta["total_elements_added"]
    file_size = os.path.getsize(data_path)
    elem_size = file_size // num_elements
    maxM0 = (elem_size - DIM * 4 - 4) // 4
    vec_offset = 4 + maxM0 * 4

    raw = np.fromfile(data_path, dtype=np.uint8)
    vectors = {}
    for label, doc_id in label_to_id.items():
        start = label * elem_size + vec_offset
        vec = np.frombuffer(raw[start : start + DIM * 4], dtype=np.float32).copy()
        vectors[doc_id] = vec

    return vectors


def main():
    # Step 1: Read HNSW vectors
    all_vecs = {}
    for stype, info in SEGMENTS.items():
        hnsw_path = os.path.join(CHROMA_PATH, info["hnsw_dir"])
        vecs = read_hnsw_vectors(hnsw_path)
        all_vecs[stype] = vecs
        print(f"  {stype} HNSW: {len(vecs)} vectors")

    # Step 2: Read queue vectors (override HNSW — queue is newer)
    conn = sqlite3.connect(os.path.join(CHROMA_PATH, "chroma.sqlite3"))
    queue_count = {"training": 0, "generated": 0}
    for doc_id, vec_bytes, topic in conn.execute(
        "SELECT id, vector, topic FROM embeddings_queue WHERE vector IS NOT NULL"
    ):
        vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
        for stype, info in SEGMENTS.items():
            if info["collection_id"] in topic:
                all_vecs[stype][doc_id] = vec
                queue_count[stype] += 1
                break
    print(f"  Queue: {queue_count}")

    # Step 3: Read docs and metadata from SQLite metadata segments
    seg_to_type = {info["meta_seg"]: stype for stype, info in SEGMENTS.items()}

    emb_rows = conn.execute("SELECT id, embedding_id, segment_id FROM embeddings").fetchall()
    id_to_doc = {r[0]: r[1] for r in emb_rows}
    id_to_seg = {r[0]: r[2] for r in emb_rows}

    meta_rows = conn.execute(
        "SELECT id, key, string_value, int_value, float_value FROM embedding_metadata"
    ).fetchall()
    conn.close()

    docs = {}
    metas = defaultdict(dict)
    doc_types = {}

    for internal_id, key, sval, ival, fval in meta_rows:
        doc_id = id_to_doc.get(internal_id)
        if not doc_id:
            continue
        seg = id_to_seg.get(internal_id)
        doc_types[doc_id] = seg_to_type.get(seg, "unknown")

        if key == "chroma:document":
            docs[doc_id] = sval
        elif not key.startswith("chroma:"):
            if sval is not None:
                metas[doc_id][key] = sval
            elif ival is not None:
                metas[doc_id][key] = ival
            elif fval is not None:
                metas[doc_id][key] = fval

    training_ids = [d for d, t in doc_types.items() if t == "training"]
    generated_ids = [d for d, t in doc_types.items() if t == "generated"]
    print(f"  Metadata: {len(training_ids)} training, {len(generated_ids)} generated, {len(docs)} docs")

    # Step 4: Check coverage
    for stype, ids in [("training", training_ids), ("generated", generated_ids)]:
        vecs = all_vecs[stype]
        has_all = sum(1 for d in ids if d in vecs and d in docs)
        missing_vec = sum(1 for d in ids if d not in vecs)
        missing_doc = sum(1 for d in ids if d not in docs)
        print(f"  {stype}: {has_all} complete, {missing_vec} missing vectors, {missing_doc} missing docs")

    total_missing = sum(1 for d in training_ids + generated_ids if d not in all_vecs.get(doc_types.get(d, ""), {}))
    if total_missing > 0:
        print(f"\n  WARNING: {total_missing} docs have no vectors — they will need re-embedding after rebuild")

    # Step 5: Confirm before destructive action
    if os.environ.get("CHROMA_RECOVER_CONFIRM") != "yes":
        resp = input("\nProceed with rebuild? This will delete and recreate chroma_db/. [y/N] ")
        if resp.lower() != "y":
            print("Aborted.")
            return

    # Step 6: Delete chroma_db and rebuild
    backup_path = CHROMA_PATH + ".backup"
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    shutil.copytree(CHROMA_PATH, backup_path)
    print(f"  Backed up to {backup_path}/")

    shutil.rmtree(CHROMA_PATH)

    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    for stype, info in SEGMENTS.items():
        col = client.get_or_create_collection(
            name=info["collection_name"],
            metadata={"description": f"Rebuilt {stype} collection"},
        )

        ids = training_ids if stype == "training" else generated_ids
        vecs = all_vecs[stype]

        # Filter to only docs with both vector and document text
        batch_ids = []
        batch_docs = []
        batch_vecs = []
        batch_metas = []
        skipped = 0

        for doc_id in ids:
            if doc_id not in vecs or doc_id not in docs:
                skipped += 1
                continue
            batch_ids.append(doc_id)
            batch_docs.append(docs[doc_id])
            batch_vecs.append(vecs[doc_id].tolist())
            batch_metas.append(dict(metas.get(doc_id, {})))

            # Insert in chunks of 500
            if len(batch_ids) >= 500:
                col.add(ids=batch_ids, documents=batch_docs, embeddings=batch_vecs, metadatas=batch_metas)
                print(f"    {stype}: inserted {col.count()}...", flush=True)
                batch_ids, batch_docs, batch_vecs, batch_metas = [], [], [], []

        if batch_ids:
            col.add(ids=batch_ids, documents=batch_docs, embeddings=batch_vecs, metadatas=batch_metas)

        print(f"  {stype}: {col.count()} docs inserted ({skipped} skipped)")

    # Recreate deleted_prompts collection (empty)
    client.get_or_create_collection(
        name="deleted_prompts",
        metadata={"description": "Graveyard: embeddings from quality/wrong_direction deletions"},
    )

    # Verify
    print("\nVerification:")
    for stype, info in SEGMENTS.items():
        col = client.get_collection(info["collection_name"])
        try:
            result = col.get(limit=5, include=["embeddings", "documents"])
            print(f"  {stype}: {col.count()} docs, embedding fetch OK")
        except Exception as e:
            print(f"  {stype}: {col.count()} docs, embedding fetch FAILED: {e}")

    print(f"\nDone! Backup at {backup_path}/")


if __name__ == "__main__":
    main()
