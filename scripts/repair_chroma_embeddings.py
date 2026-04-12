#!/usr/bin/env python3
"""Repair corrupted ChromaDB embeddings by recovering vectors from HNSW data file and WAL queue.

Standalone script — does not require Flask or the full application stack.

Usage:
    conda run -n llm_prompt_agent python scripts/repair_chroma_embeddings.py
"""

import os
import pickle
import sqlite3
import struct
import sys
import time

import chromadb
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "training_prompts"

# HNSW data_level0.bin layout (determined by inspecting the header)
HNSW_SIZE_PER_ELEMENT = 12428
HNSW_VEC_OFFSET = 132  # bytes of link-list overhead before the vector data
EMBEDDING_DIM = 3072

# Batch sizes
INSERT_BATCH_SIZE = 500


def log(msg: str):
    print(f"[repair] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: Scan — identify corrupted docs AND extract working embeddings
# ---------------------------------------------------------------------------
def scan_collection(collection) -> tuple[dict[str, list[float]], list[str], dict[str, dict]]:
    """Scan every document. Return (working_vectors, corrupted_ids, all_docs).

    working_vectors: {doc_id: embedding} for docs whose embeddings are readable.
    corrupted_ids: list of doc IDs with broken embeddings.
    all_docs: {doc_id: {"text": ..., "metadata": ...}} for every doc.
    """
    count = collection.count()
    log(f"Collection has {count} documents. Scanning...")

    # Get all IDs + text + metadata in pages (works without embeddings)
    all_docs: dict[str, dict] = {}
    all_ids_ordered: list[str] = []
    page_size = 5000
    offset = 0
    while offset < count:
        result = collection.get(
            limit=page_size, offset=offset, include=["documents", "metadatas"]
        )
        if not result["ids"]:
            break
        for doc_id, text, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        ):
            all_docs[doc_id] = {"text": text, "metadata": meta}
            all_ids_ordered.append(doc_id)
        offset += len(result["ids"])
    log(f"  Fetched text+metadata for {len(all_docs)} documents")

    # Test each doc individually to separate working vs corrupted,
    # and extract working embeddings at the same time.
    working_vectors: dict[str, list[float]] = {}
    corrupted_ids: list[str] = []

    for i, doc_id in enumerate(all_ids_ordered):
        try:
            result = collection.get(ids=[doc_id], include=["embeddings"])
            emb = result["embeddings"]
            # chromadb 1.x returns numpy arrays
            if isinstance(emb, np.ndarray):
                working_vectors[doc_id] = emb[0].tolist()
            else:
                working_vectors[doc_id] = list(emb[0])
        except Exception:
            corrupted_ids.append(doc_id)

        if (i + 1) % 500 == 0:
            log(f"  Scanned {i + 1}/{len(all_ids_ordered)}: "
                f"{len(working_vectors)} ok, {len(corrupted_ids)} corrupted")

    log(f"  Scan complete: {len(working_vectors)} working, "
        f"{len(corrupted_ids)} corrupted")
    return working_vectors, corrupted_ids, all_docs


# ---------------------------------------------------------------------------
# Phase 2: Recover corrupted vectors from HNSW data file
# ---------------------------------------------------------------------------
def get_hnsw_segment_dir(collection_name: str) -> str:
    """Find the HNSW segment directory for the given collection."""
    db_path = os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT s.id FROM segments s "
        "JOIN collections c ON s.collection = c.id "
        "WHERE c.name = ? AND s.scope = 'VECTOR'",
        (collection_name,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise RuntimeError(f"No HNSW segment found for '{collection_name}'")
    segment_dir = os.path.join(CHROMA_DB_PATH, row[0])
    if not os.path.isdir(segment_dir):
        raise RuntimeError(f"Segment directory not found: {segment_dir}")
    return segment_dir


def read_hnsw_vectors(segment_dir: str) -> np.ndarray:
    """Read all raw vectors from data_level0.bin as a (N, dim) array."""
    meta_path = os.path.join(segment_dir, "index_metadata.pickle")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    num_elements = meta["total_elements_added"]
    log(f"  HNSW reports {num_elements} elements")

    data_path = os.path.join(segment_dir, "data_level0.bin")
    vec_bytes = EMBEDDING_DIM * 4
    vectors = np.empty((num_elements, EMBEDDING_DIM), dtype=np.float32)

    with open(data_path, "rb") as f:
        for idx in range(num_elements):
            f.seek(idx * HNSW_SIZE_PER_ELEMENT + HNSW_VEC_OFFSET)
            raw = f.read(vec_bytes)
            if len(raw) < vec_bytes:
                log(f"  WARNING: Short read at HNSW index {idx}")
                vectors = vectors[:idx]
                break
            vectors[idx] = np.frombuffer(raw, dtype=np.float32)

    log(f"  Read {len(vectors)} vectors from HNSW data file")
    return vectors


def match_hnsw_to_corrupted(
    hnsw_vectors: np.ndarray,
    working_vectors: dict[str, list[float]],
    corrupted_ids: list[str],
    collection,
) -> dict[str, list[float]]:
    """Find which HNSW vectors belong to corrupted docs.

    Strategy:
    1. Build fingerprint array from working vectors (first 8 floats)
    2. Mark which HNSW vectors match working docs (by fingerprint)
    3. The unmatched HNSW vectors must belong to corrupted or stale docs
    4. Query ChromaDB with each unmatched vector to find the owning doc
    """
    corrupted_set = set(corrupted_ids)
    num_hnsw = len(hnsw_vectors)
    num_working = len(working_vectors)

    # Build fingerprint arrays for fast matching (first 8 floats)
    fp_len = 8
    log(f"  Building fingerprint index for {num_working} working vectors...")
    working_fps = np.array(
        [v[:fp_len] for v in working_vectors.values()], dtype=np.float32
    )  # (num_working, fp_len)
    hnsw_fps = hnsw_vectors[:, :fp_len]  # (num_hnsw, fp_len)

    # For each HNSW vector, check if it matches any working vector
    log(f"  Matching {num_hnsw} HNSW vectors against {num_working} working fingerprints...")
    matched_hnsw = np.zeros(num_hnsw, dtype=bool)

    # Process in chunks to limit memory usage
    chunk_size = 500
    for start in range(0, num_hnsw, chunk_size):
        end = min(start + chunk_size, num_hnsw)
        # (chunk, 1, fp_len) - (1, num_working, fp_len) → (chunk, num_working, fp_len)
        diffs = np.abs(
            hnsw_fps[start:end, np.newaxis, :] - working_fps[np.newaxis, :, :]
        )
        max_diffs = diffs.max(axis=2)  # (chunk, num_working)
        min_per_hnsw = max_diffs.min(axis=1)  # (chunk,)
        matched_hnsw[start:end] = min_per_hnsw < 1e-6

        if end % 2000 == 0 or end == num_hnsw:
            log(f"    Fingerprinted {end}/{num_hnsw}")

    unmatched_indices = np.where(~matched_hnsw)[0]
    log(f"  {len(unmatched_indices)} HNSW vectors unmatched "
        f"(belong to corrupted/stale docs)")

    # Query ChromaDB with each unmatched vector to identify the owner
    recovered: dict[str, list[float]] = {}
    stale_count = 0
    batch_size = 50

    for batch_start in range(0, len(unmatched_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(unmatched_indices))
        batch_indices = unmatched_indices[batch_start:batch_end]
        batch_vecs = [hnsw_vectors[idx].tolist() for idx in batch_indices]

        result = collection.query(
            query_embeddings=batch_vecs,
            n_results=1,
            include=["distances"],
        )

        for i, (ids, dists) in enumerate(
            zip(result["ids"], result["distances"])
        ):
            doc_id = ids[0]
            dist = dists[0]

            if dist > 0.01:
                # Stale HNSW entry — vector for a deleted doc
                stale_count += 1
                continue

            if doc_id in corrupted_set and doc_id not in recovered:
                recovered[doc_id] = batch_vecs[i]

        if (batch_end) % 500 == 0 or batch_end == len(unmatched_indices):
            log(f"    Queried {batch_end}/{len(unmatched_indices)}, "
                f"recovered {len(recovered)}, stale {stale_count}")

    log(f"  Recovered {len(recovered)} corrupted vectors from HNSW "
        f"({stale_count} stale entries skipped)")
    return recovered


# ---------------------------------------------------------------------------
# Phase 3: Extract queue vectors from SQLite
# ---------------------------------------------------------------------------
def get_collection_id(collection_name: str) -> str:
    """Look up the ChromaDB internal collection UUID from SQLite."""
    db_path = os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM collections WHERE name = ?", (collection_name,)
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise RuntimeError(f"Collection '{collection_name}' not found in SQLite")
    return row[0]


def extract_queue_vectors(
    collection_id: str, already_recovered: set[str]
) -> dict[str, list[float]]:
    """Extract vectors from embeddings_queue for docs not yet recovered."""
    db_path = os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    topic = f"persistent://default/default/{collection_id}"
    cursor.execute(
        "SELECT id, vector FROM embeddings_queue "
        "WHERE topic = ? AND vector IS NOT NULL",
        (topic,),
    )

    queue_vectors: dict[str, list[float]] = {}
    for doc_id, vec_blob in cursor.fetchall():
        if doc_id in already_recovered:
            continue
        num_floats = len(vec_blob) // 4
        vec = list(struct.unpack(f"{num_floats}f", vec_blob))
        queue_vectors[doc_id] = vec

    conn.close()
    log(f"  Extracted {len(queue_vectors)} additional vectors from WAL queue")
    return queue_vectors


# ---------------------------------------------------------------------------
# Phase 4: Rebuild the collection
# ---------------------------------------------------------------------------
def rebuild_collection(
    client: chromadb.PersistentClient,
    docs: dict[str, dict],
    vectors: dict[str, list[float]],
):
    """Delete and recreate the collection with recovered data."""
    log("Deleting old collection...")
    client.delete_collection(COLLECTION_NAME)

    log("Creating fresh collection...")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Training prompts (repaired)"},
    )

    doc_ids = list(docs.keys())
    total = len(doc_ids)
    log(f"Inserting {total} documents in batches of {INSERT_BATCH_SIZE}...")

    for batch_start in range(0, total, INSERT_BATCH_SIZE):
        batch_end = min(batch_start + INSERT_BATCH_SIZE, total)
        batch_ids = doc_ids[batch_start:batch_end]

        batch_texts = [docs[did]["text"] for did in batch_ids]
        batch_metas = [docs[did]["metadata"] for did in batch_ids]
        batch_embs = [vectors[did] for did in batch_ids]

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embs,
            metadatas=batch_metas,
        )

        if batch_end % 1000 == 0 or batch_end == total:
            log(f"  Inserted {batch_end}/{total}")

    return collection


# ---------------------------------------------------------------------------
# Phase 5: Verify the rebuilt collection
# ---------------------------------------------------------------------------
def verify_collection(collection, expected_count: int) -> bool:
    """Verify the rebuilt collection is healthy."""
    actual_count = collection.count()
    if actual_count != expected_count:
        log(f"FAIL: Expected {expected_count} docs, got {actual_count}")
        return False
    log(f"Count verified: {actual_count}")

    # Test paginated embedding retrieval (the operation that was failing)
    page_size = 500
    offset = 0
    total_fetched = 0
    while offset < actual_count:
        try:
            result = collection.get(
                limit=page_size,
                offset=offset,
                include=["embeddings", "documents", "metadatas"],
            )
            batch_size = len(result["ids"])
            if batch_size == 0:
                break
            total_fetched += batch_size
            offset += batch_size
        except Exception as e:
            log(f"FAIL: get(include=['embeddings']) failed at offset {offset}: {e}")
            return False

        if total_fetched % 2000 == 0 or total_fetched >= actual_count:
            log(f"  Verified {total_fetched}/{actual_count} embeddings readable")

    if total_fetched != actual_count:
        log(f"FAIL: Only fetched {total_fetched} out of {actual_count}")
        return False

    log("All embeddings verified readable")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    log(f"ChromaDB path: {CHROMA_DB_PATH}")

    if not os.path.isdir(CHROMA_DB_PATH):
        log(f"ERROR: ChromaDB directory not found: {CHROMA_DB_PATH}")
        sys.exit(1)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    total_docs = collection.count()
    log(f"Collection '{COLLECTION_NAME}' has {total_docs} documents")

    # Phase 1: Scan for corruption and extract working embeddings
    log(f"\n{'='*60}")
    log("Phase 1: Scanning collection...")
    log(f"{'='*60}")
    working_vectors, corrupted_ids, all_docs = scan_collection(collection)

    if not corrupted_ids:
        log("No corrupted embeddings found — nothing to repair.")
        sys.exit(0)

    log(f"\nFound {len(corrupted_ids)} corrupted embeddings. Recovering...")

    # Phase 2: Recover from HNSW
    log(f"\n{'='*60}")
    log("Phase 2: Recovering vectors from HNSW data file...")
    log(f"{'='*60}")
    segment_dir = get_hnsw_segment_dir(COLLECTION_NAME)
    hnsw_vectors = read_hnsw_vectors(segment_dir)
    hnsw_recovered = match_hnsw_to_corrupted(
        hnsw_vectors, working_vectors, corrupted_ids, collection
    )

    # Phase 3: Recover remaining from queue
    log(f"\n{'='*60}")
    log("Phase 3: Recovering remaining vectors from WAL queue...")
    log(f"{'='*60}")
    collection_id = get_collection_id(COLLECTION_NAME)
    already_have = set(working_vectors.keys()) | set(hnsw_recovered.keys())
    queue_recovered = extract_queue_vectors(collection_id, already_have)

    # Merge all recovered vectors
    all_vectors = dict(working_vectors)
    all_vectors.update(hnsw_recovered)
    all_vectors.update(queue_recovered)

    # Check coverage
    all_doc_ids = set(all_docs.keys())
    recovered_ids = set(all_vectors.keys())
    missing = all_doc_ids - recovered_ids

    log(f"\nRecovery summary:")
    log(f"  Total documents:      {len(all_doc_ids)}")
    log(f"  Working (from get):   {len(working_vectors)}")
    log(f"  Recovered from HNSW:  {len(hnsw_recovered)}")
    log(f"  Recovered from queue: {len(queue_recovered)}")
    log(f"  Still missing:        {len(missing)}")

    if missing:
        log(f"\nWARNING: {len(missing)} documents have unrecoverable vectors.")
        log("These will be dropped from the rebuilt collection.")
        log("First 10:")
        for doc_id in sorted(missing)[:10]:
            log(f"  {doc_id}")
        if len(missing) > 10:
            log(f"  ... and {len(missing) - 10} more")

        # Remove missing docs from all_docs so rebuild doesn't fail
        for doc_id in missing:
            del all_docs[doc_id]
        total_docs -= len(missing)

    # Phase 4: Rebuild
    log(f"\n{'='*60}")
    log(f"Phase 4: Rebuilding collection ({total_docs} documents)...")
    log(f"{'='*60}")
    rebuild_collection(client, all_docs, all_vectors)
    collection = client.get_collection(COLLECTION_NAME)

    # Phase 5: Verify
    log(f"\n{'='*60}")
    log("Phase 5: Verifying rebuilt collection...")
    log(f"{'='*60}")
    if not verify_collection(collection, total_docs):
        log("\nERROR: Verification failed!")
        sys.exit(1)

    elapsed = time.time() - start_time
    log(f"\nRepair complete in {elapsed:.1f}s.")
    if missing:
        log(f"  {total_docs} documents recovered. "
            f"{len(missing)} unrecoverable documents dropped.")
    else:
        log(f"  All {total_docs} documents recovered and verified.")


if __name__ == "__main__":
    main()
