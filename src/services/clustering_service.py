"""Clustering service — KMeans clustering of prompt embeddings with TF-IDF labeling.

Operates on ChromaDB (vector_store) for embeddings/documents, SQLite (database)
for cluster metadata/assignments/runs, and settings for configuration. Fully
decoupled from the agent loop and LLM interface.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models import vector_store, settings
from src.models.database import get_db

logger = logging.getLogger(__name__)

# Regex to split prompts on punctuation boundaries (commas, periods,
# semicolons, colons, parentheses, brackets, pipes, etc.).  Sequences of
# punctuation and surrounding whitespace are consumed together so that
# tokens adjacent to punctuation don't form cross-boundary ngrams.
_PUNCT_SPLIT_RE = re.compile(r'[\s]*[,\.;:!?\(\)\[\]\{\}|/\\]+[\s]*')

# Token pattern: 2+ word-characters (letters, digits, underscore).
_TOKEN_RE = re.compile(r'\b\w\w+\b', re.UNICODE)


def _make_punctuation_aware_analyzer(
    ngram_range: tuple[int, int],
    stop_words: set[str] | None = None,
) -> callable:
    """Return an analyzer function that respects punctuation as ngram boundaries.

    Instead of generating ngrams across the entire document token stream,
    this splits the document on punctuation first, then generates ngrams
    *within* each punctuation-delimited segment independently.  Punctuation
    itself never appears as a term.
    """
    # Build a stop-word set (sklearn's built-in "english" list)
    if stop_words is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop_words = ENGLISH_STOP_WORDS

    min_n, max_n = ngram_range

    def _analyzer(doc: str) -> list[str]:
        # Split on punctuation into sub-phrases
        segments = _PUNCT_SPLIT_RE.split(doc.lower())
        ngrams: list[str] = []
        for segment in segments:
            tokens = _TOKEN_RE.findall(segment)
            # Remove stop words
            tokens = [t for t in tokens if t not in stop_words]
            if not tokens:
                continue
            # Generate ngrams within this segment only
            for n in range(min_n, max_n + 1):
                for i in range(len(tokens) - n + 1):
                    ngrams.append(" ".join(tokens[i:i + n]))
        return ngrams

    return _analyzer


# ---------------------------------------------------------------------------
# Global clustering state
# ---------------------------------------------------------------------------
_clustering_lock = threading.Lock()
_clustering_running = False
_status_listeners: list = []


@dataclass
class ClusteringProgress:
    """Current state of a clustering run."""
    phase: str = "idle"
    message: str = ""
    current: int = 0
    total: int = 0
    complete: bool = False


def add_status_listener(callback):
    """Register a callback for clustering status updates."""
    _status_listeners.append(callback)


def remove_status_listener(callback):
    """Remove a status listener."""
    if callback in _status_listeners:
        _status_listeners.remove(callback)


def _emit_status(progress: ClusteringProgress):
    """Notify all listeners of a status update."""
    for listener in _status_listeners[:]:
        try:
            listener(progress)
        except Exception as e:
            logger.error(f"Error in clustering status listener: {e}")


def is_running() -> bool:
    """Check if clustering is currently running."""
    return _clustering_running


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_all_embeddings() -> tuple[list[str], list[list[float]], list[str], list[dict]]:
    """Fetch all embeddings, documents, and metadata from both ChromaDB collections.

    Returns:
        Tuple of (ids, embeddings, documents, metadatas).
    """
    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    for collection in [vector_store._training_collection, vector_store._generated_collection]:
        if collection is None:
            continue
        count = collection.count()
        if count == 0:
            continue

        page_size = 5000
        offset = 0
        while offset < count:
            result = collection.get(
                limit=page_size,
                offset=offset,
                include=["embeddings", "documents", "metadatas"],
            )
            batch_ids = result["ids"]
            if not batch_ids:
                break

            all_ids.extend(batch_ids)
            all_embeddings.extend(result["embeddings"])
            all_documents.extend(result["documents"])
            all_metadatas.extend(result["metadatas"])
            offset += len(batch_ids)

    return all_ids, all_embeddings, all_documents, all_metadatas


def _fetch_embeddings_by_concept(concept_name: str) -> tuple[list[str], list[list[float]], list[str], list[dict]]:
    """Fetch embeddings and documents for a specific concept from both collections.

    Args:
        concept_name: The concept/folder name to filter by.

    Returns:
        Tuple of (ids, embeddings, documents, metadatas).
    """
    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    for collection in [vector_store._training_collection, vector_store._generated_collection]:
        if collection is None:
            continue
        count = collection.count()
        if count == 0:
            continue

        try:
            result = collection.get(
                where={"concept": concept_name},
                include=["embeddings", "documents", "metadatas"],
            )
            if result["ids"]:
                all_ids.extend(result["ids"])
                all_embeddings.extend(result["embeddings"])
                all_documents.extend(result["documents"])
                all_metadatas.extend(result["metadatas"])
        except Exception as e:
            logger.error(f"Error fetching embeddings for concept '{concept_name}': {e}")

    return all_ids, all_embeddings, all_documents, all_metadatas


def _get_all_concepts() -> list[dict]:
    """Get all unique concept names with counts and source types from ChromaDB.

    Returns:
        List of dicts with keys: concept, source_type, count.
    """
    return vector_store.list_concepts()


def _generate_unique_cluster_labels(
    cluster_texts: dict[int, list[str]],
    top_n: int | None = None,
) -> dict[int, str]:
    """Generate unique phrase labels for a set of clusters using TF-IDF bigrams/trigrams.

    Each cluster gets a label composed of distinctive phrases.  Phrases are
    guaranteed unique across clusters within the corpus — once a phrase is
    assigned to one cluster it will not appear in any other cluster's label.

    The function fits a single TF-IDF model across *all* cluster documents,
    then scores each phrase per cluster by *distinctiveness* (the cluster's
    mean TF-IDF for the phrase divided by the global mean).  Phrases are
    assigned greedily — the most distinctive phrase for a cluster is claimed
    first, then excluded from further consideration.

    Args:
        cluster_texts: Mapping of cluster_index → list of document texts.
        top_n: Number of phrases per label.  Reads ``cluster_label_terms``
               from settings when *None*.

    Returns:
        Mapping of cluster_index → comma-separated label string.
    """
    if top_n is None:
        label_terms_str = settings.get_setting("cluster_label_terms")
        top_n = int(label_terms_str) if label_terms_str else 3

    labels: dict[int, str] = {}
    if not cluster_texts:
        return labels

    # ── Pre-filter: handle degenerate clusters (empty / single-doc) ──
    all_docs: list[str] = []           # flat list of filtered docs
    doc_cluster_map: list[int] = []    # parallel: which cluster each doc belongs to
    viable_clusters: list[int] = []    # clusters with ≥ 2 filtered docs

    for cluster_idx, texts in cluster_texts.items():
        filtered = [t for t in texts if t and t.strip()]
        if not filtered:
            labels[cluster_idx] = "empty"
            continue
        if len(filtered) == 1:
            words = filtered[0].split()[:top_n]
            labels[cluster_idx] = ", ".join(words) if words else "single"
            continue
        viable_clusters.append(cluster_idx)
        for doc in filtered:
            all_docs.append(doc)
            doc_cluster_map.append(cluster_idx)

    if not viable_clusters:
        return labels

    # ── Fit TF-IDF (bigrams + trigrams first, fallback to unigrams+bigrams) ──
    vectorizer = None
    tfidf_matrix = None
    feature_names = None

    for ngram_range, min_df, max_feats in [
        ((2, 3), 2, 3000),
        ((2, 3), 1, 3000),
        ((1, 3), 1, 2000),
    ]:
        try:
            analyzer = _make_punctuation_aware_analyzer(ngram_range)
            v = TfidfVectorizer(
                analyzer=analyzer,
                max_features=max_feats,
                min_df=min_df,
                max_df=0.95,
            )
            m = v.fit_transform(all_docs)
            if len(v.get_feature_names_out()) >= len(viable_clusters) * top_n:
                vectorizer, tfidf_matrix, feature_names = v, m, v.get_feature_names_out()
                break
            # Not enough features — try next fallback
            vectorizer, tfidf_matrix, feature_names = v, m, v.get_feature_names_out()
        except ValueError:
            continue

    if vectorizer is None or tfidf_matrix is None:
        for ci in viable_clusters:
            labels[ci] = "unknown"
        return labels

    # ── Compute per-cluster mean TF-IDF ──
    cluster_mean: dict[int, np.ndarray] = {}
    doc_arr = np.array(doc_cluster_map)

    for ci in viable_clusters:
        mask = doc_arr == ci
        rows = tfidf_matrix[mask]
        cluster_mean[ci] = np.asarray(rows.mean(axis=0)).flatten()

    # Global mean across all clusters (unweighted per cluster, not per doc)
    global_mean = np.mean(list(cluster_mean.values()), axis=0)

    # ── Greedy unique-phrase assignment ──
    used_features: set[int] = set()

    # Process clusters smallest-first so smaller clusters get first pick at
    # distinguishing phrases (larger clusters have more alternatives).
    sorted_clusters = sorted(viable_clusters, key=lambda ci: len(cluster_texts[ci]))

    for ci in sorted_clusters:
        mean_tfidf = cluster_mean[ci]

        # Distinctiveness = cluster mean / (global mean + ε)
        distinctiveness = mean_tfidf / (global_mean + 1e-10)

        # Zero out already-claimed features
        for fi in used_features:
            distinctiveness[fi] = 0.0

        ranked = distinctiveness.argsort()[::-1]
        selected: list[int] = []
        for fi in ranked:
            if len(selected) >= top_n:
                break
            if fi not in used_features and mean_tfidf[fi] > 0:
                selected.append(fi)
                used_features.add(fi)

        if selected:
            labels[ci] = ", ".join(feature_names[i] for i in selected)
        else:
            labels[ci] = "unknown"

    return labels


# ---------------------------------------------------------------------------
# Cross-folder clustering
# ---------------------------------------------------------------------------

def generate_cross_folder_clusters(k: int | None = None):
    """Run KMeans clustering across all prompts in both ChromaDB collections.

    Args:
        k: Number of clusters. Falls back to ``cluster_k_cross`` setting.
    """
    progress = ClusteringProgress(phase="cross_folder", message="Starting cross-folder clustering...")
    _emit_status(progress)

    # 1. Read k from settings if not provided
    if k is None:
        k_str = settings.get_setting("cluster_k_cross")
        k = int(k_str) if k_str else 15

    # 2. Fetch all embeddings
    progress.message = "Fetching all embeddings from ChromaDB..."
    _emit_status(progress)

    doc_ids, embeddings_list, documents, metadatas = _fetch_all_embeddings()
    n_samples = len(doc_ids)

    if n_samples == 0:
        progress.message = "No documents found — skipping cross-folder clustering."
        _emit_status(progress)
        return

    if n_samples <= 1:
        progress.message = "Only 1 document found — skipping cross-folder clustering."
        _emit_status(progress)
        return

    # Reduce k if not enough samples
    effective_k = min(k, n_samples)

    progress.message = f"Clustering {n_samples} documents into {effective_k} clusters..."
    progress.total = n_samples
    _emit_status(progress)

    # 3. KMeans
    embeddings_np = np.array(embeddings_list)
    kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings_np)
    centroids = kmeans.cluster_centers_

    # 4-5. Generate unique labels for all clusters in batch
    cluster_texts_map: dict[int, list[str]] = {}
    for cluster_idx in range(effective_k):
        member_mask = labels == cluster_idx
        cluster_texts_map[cluster_idx] = [documents[i] for i in range(n_samples) if member_mask[i]]

    unique_labels = _generate_unique_cluster_labels(cluster_texts_map)

    cluster_data: list[dict] = []
    for cluster_idx in range(effective_k):
        member_mask = labels == cluster_idx
        cluster_data.append({
            "cluster_index": cluster_idx,
            "label": unique_labels.get(cluster_idx, "unknown"),
            "centroid": json.dumps(centroids[cluster_idx].tolist()),
            "prompt_count": int(member_mask.sum()),
        })

    # Build assignment list
    assignment_data: list[dict] = []
    for i in range(n_samples):
        cluster_idx = int(labels[i])
        centroid = centroids[cluster_idx]
        distance = float(np.linalg.norm(embeddings_np[i] - centroid))
        source_type = metadatas[i].get("dir_type", "training") if metadatas[i] else "training"
        assignment_data.append({
            "doc_id": doc_ids[i],
            "source_type": source_type,
            "cluster_index": cluster_idx,
            "distance": distance,
        })

    # 6-8. Store in SQLite
    progress.message = "Saving cross-folder clusters to database..."
    _emit_status(progress)

    started_at = datetime.utcnow().isoformat()

    with get_db() as conn:
        # Clear old cross_folder data
        conn.execute(
            "DELETE FROM cluster_assignments WHERE cluster_id IN "
            "(SELECT id FROM clusters WHERE cluster_type = 'cross_folder')"
        )
        conn.execute("DELETE FROM clusters WHERE cluster_type = 'cross_folder'")

        # Insert new clusters
        cluster_id_map: dict[int, int] = {}  # cluster_index -> db id
        for cd in cluster_data:
            cursor = conn.execute(
                "INSERT INTO clusters (cluster_type, folder_path, cluster_index, label, centroid, prompt_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("cross_folder", None, cd["cluster_index"], cd["label"], cd["centroid"], cd["prompt_count"]),
            )
            cluster_id_map[cd["cluster_index"]] = cursor.lastrowid

        # Insert assignments
        for ad in assignment_data:
            db_cluster_id = cluster_id_map[ad["cluster_index"]]
            conn.execute(
                "INSERT INTO cluster_assignments (doc_id, source_type, cluster_id, distance) "
                "VALUES (?, ?, ?, ?)",
                (ad["doc_id"], ad["source_type"], db_cluster_id, ad["distance"]),
            )

        # 9. Record the run
        conn.execute(
            "INSERT INTO clustering_runs (run_type, folder_path, total_prompts, num_clusters, started_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            ("cross_folder", None, n_samples, effective_k, started_at),
        )

    # 10. Emit completion
    progress.message = f"Cross-folder clustering complete: {effective_k} clusters from {n_samples} documents."
    progress.current = n_samples
    _emit_status(progress)


# ---------------------------------------------------------------------------
# Intra-folder clustering
# ---------------------------------------------------------------------------

def generate_intra_folder_clusters(folder_path: str | None = None, k: int | None = None, force: bool = False):
    """Run KMeans clustering within individual folders/concepts.

    Args:
        folder_path: If provided, only cluster this specific concept. Otherwise
                     cluster all concepts meeting minimum size.
        k: Number of clusters per folder. Falls back to ``cluster_k_intra`` setting.
        force: If True, skip size and freshness checks.
    """
    progress = ClusteringProgress(phase="intra_folder", message="Starting intra-folder clustering...")
    _emit_status(progress)

    # 1. Read settings
    if k is None:
        k_str = settings.get_setting("cluster_k_intra")
        k = int(k_str) if k_str else 5

    min_size_str = settings.get_setting("cluster_min_folder_size")
    min_folder_size = int(min_size_str) if min_size_str else 20

    # 2. Determine which concepts to cluster
    if folder_path:
        concepts_to_cluster = [{"concept": folder_path}]
    else:
        all_concepts = _get_all_concepts()
        # Aggregate counts per concept (across source types)
        concept_counts: dict[str, int] = {}
        for c in all_concepts:
            concept_counts[c["concept"]] = concept_counts.get(c["concept"], 0) + c["count"]
        concepts_to_cluster = [{"concept": name} for name in concept_counts]

    progress.total = len(concepts_to_cluster)
    progress.message = f"Processing {len(concepts_to_cluster)} concept(s)..."
    _emit_status(progress)

    # 3. Cluster each concept
    for idx, concept_info in enumerate(concepts_to_cluster):
        concept_name = concept_info["concept"]
        progress.current = idx + 1
        progress.message = f"Clustering concept '{concept_name}' ({idx + 1}/{len(concepts_to_cluster)})..."
        _emit_status(progress)

        # Fetch embeddings for this concept
        doc_ids, embeddings_list, documents, metadatas = _fetch_embeddings_by_concept(concept_name)
        n_samples = len(doc_ids)

        # Skip if too few prompts (unless forced)
        if n_samples < min_folder_size and not force:
            logger.info(
                f"Skipping concept '{concept_name}': {n_samples} docs < min_folder_size {min_folder_size}"
            )
            continue

        if n_samples <= 1:
            logger.info(f"Skipping concept '{concept_name}': only {n_samples} document(s)")
            continue

        # Check freshness unless forced
        if not force:
            with get_db() as conn:
                # Check if intra_folder clustering exists for this folder
                cursor = conn.execute(
                    "SELECT id FROM clusters WHERE cluster_type = 'intra_folder' AND folder_path = ? LIMIT 1",
                    (concept_name,),
                )
                existing_cluster = cursor.fetchone()

                if existing_cluster:
                    # Check if there are any doc_ids not yet assigned to intra clusters for this folder
                    assigned_cursor = conn.execute(
                        "SELECT DISTINCT ca.doc_id FROM cluster_assignments ca "
                        "JOIN clusters c ON ca.cluster_id = c.id "
                        "WHERE c.cluster_type = 'intra_folder' AND c.folder_path = ?",
                        (concept_name,),
                    )
                    assigned_ids = {row["doc_id"] for row in assigned_cursor.fetchall()}
                    unassigned = [did for did in doc_ids if did not in assigned_ids]

                    if not unassigned:
                        logger.info(f"Skipping concept '{concept_name}': all docs already assigned, no new docs")
                        continue

        # Run KMeans
        effective_k = min(k, n_samples)
        embeddings_np = np.array(embeddings_list)
        kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(embeddings_np)
        centroids = kmeans.cluster_centers_

        # Generate unique labels for all clusters in this folder in batch
        cluster_texts_map: dict[int, list[str]] = {}
        for cluster_idx in range(effective_k):
            member_mask = labels == cluster_idx
            cluster_texts_map[cluster_idx] = [documents[i] for i in range(n_samples) if member_mask[i]]

        unique_labels = _generate_unique_cluster_labels(cluster_texts_map)

        cluster_data: list[dict] = []
        for cluster_idx in range(effective_k):
            member_mask = labels == cluster_idx
            cluster_data.append({
                "cluster_index": cluster_idx,
                "label": unique_labels.get(cluster_idx, "unknown"),
                "centroid": json.dumps(centroids[cluster_idx].tolist()),
                "prompt_count": int(member_mask.sum()),
            })

        # Build assignments
        assignment_data: list[dict] = []
        for i in range(n_samples):
            cluster_idx = int(labels[i])
            centroid = centroids[cluster_idx]
            distance = float(np.linalg.norm(embeddings_np[i] - centroid))
            source_type = metadatas[i].get("dir_type", "training") if metadatas[i] else "training"
            assignment_data.append({
                "doc_id": doc_ids[i],
                "source_type": source_type,
                "cluster_index": cluster_idx,
                "distance": distance,
            })

        # Store in SQLite
        started_at = datetime.utcnow().isoformat()

        with get_db() as conn:
            # Clear old intra_folder clusters for this concept
            conn.execute(
                "DELETE FROM cluster_assignments WHERE cluster_id IN "
                "(SELECT id FROM clusters WHERE cluster_type = 'intra_folder' AND folder_path = ?)",
                (concept_name,),
            )
            conn.execute(
                "DELETE FROM clusters WHERE cluster_type = 'intra_folder' AND folder_path = ?",
                (concept_name,),
            )

            # Insert clusters
            cluster_id_map: dict[int, int] = {}
            for cd in cluster_data:
                cursor = conn.execute(
                    "INSERT INTO clusters (cluster_type, folder_path, cluster_index, label, centroid, prompt_count) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("intra_folder", concept_name, cd["cluster_index"], cd["label"], cd["centroid"], cd["prompt_count"]),
                )
                cluster_id_map[cd["cluster_index"]] = cursor.lastrowid

            # Insert assignments
            for ad in assignment_data:
                db_cluster_id = cluster_id_map[ad["cluster_index"]]
                conn.execute(
                    "INSERT INTO cluster_assignments (doc_id, source_type, cluster_id, distance) "
                    "VALUES (?, ?, ?, ?)",
                    (ad["doc_id"], ad["source_type"], db_cluster_id, ad["distance"]),
                )

            # Record run
            conn.execute(
                "INSERT INTO clustering_runs (run_type, folder_path, total_prompts, num_clusters, started_at, completed_at) "
                "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                ("intra_folder", concept_name, n_samples, effective_k, started_at),
            )

    progress.message = "Intra-folder clustering complete."
    progress.current = progress.total
    _emit_status(progress)


# ---------------------------------------------------------------------------
# Incremental assignment
# ---------------------------------------------------------------------------

def assign_new_docs_to_clusters(
    doc_ids: list[str],
    embeddings: list[list[float]],
    source_types: list[str],
    concepts: list[str],
):
    """Assign newly ingested documents to existing clusters by nearest centroid.

    For each document, finds the nearest cross-folder cluster centroid and
    creates an assignment. If the document's concept has intra-folder clusters,
    also finds the nearest intra-folder centroid and creates an assignment.

    Args:
        doc_ids: List of document IDs.
        embeddings: Corresponding embedding vectors.
        source_types: Source type per document (``"training"`` or ``"output"``).
        concepts: Concept/folder name per document.
    """
    if not doc_ids:
        return

    # Load cross-folder centroids
    cross_clusters: list[dict] = []
    intra_clusters_by_folder: dict[str, list[dict]] = {}

    with get_db() as conn:
        # Cross-folder clusters
        cursor = conn.execute(
            "SELECT id, centroid FROM clusters WHERE cluster_type = 'cross_folder' AND centroid IS NOT NULL"
        )
        for row in cursor.fetchall():
            cross_clusters.append({
                "id": row["id"],
                "centroid": np.array(json.loads(row["centroid"])),
            })

        # Intra-folder clusters — group by folder_path
        cursor = conn.execute(
            "SELECT id, folder_path, centroid FROM clusters "
            "WHERE cluster_type = 'intra_folder' AND centroid IS NOT NULL"
        )
        for row in cursor.fetchall():
            folder = row["folder_path"]
            if folder not in intra_clusters_by_folder:
                intra_clusters_by_folder[folder] = []
            intra_clusters_by_folder[folder].append({
                "id": row["id"],
                "centroid": np.array(json.loads(row["centroid"])),
            })

    if not cross_clusters and not intra_clusters_by_folder:
        logger.debug("No existing clusters to assign new docs to.")
        return

    with get_db() as conn:
        for doc_id, embedding, source_type, concept in zip(doc_ids, embeddings, source_types, concepts):
            emb = np.array(embedding)

            # Assign to nearest cross-folder cluster
            if cross_clusters:
                best_cross_id = None
                best_cross_dist = float("inf")
                for cc in cross_clusters:
                    dist = float(np.linalg.norm(emb - cc["centroid"]))
                    if dist < best_cross_dist:
                        best_cross_dist = dist
                        best_cross_id = cc["id"]

                if best_cross_id is not None:
                    conn.execute(
                        "INSERT INTO cluster_assignments (doc_id, source_type, cluster_id, distance) "
                        "VALUES (?, ?, ?, ?)",
                        (doc_id, source_type, best_cross_id, best_cross_dist),
                    )
                    # Update prompt_count
                    conn.execute(
                        "UPDATE clusters SET prompt_count = prompt_count + 1 WHERE id = ?",
                        (best_cross_id,),
                    )

            # Assign to nearest intra-folder cluster (if concept has clusters)
            if concept in intra_clusters_by_folder:
                intra_clusters = intra_clusters_by_folder[concept]
                best_intra_id = None
                best_intra_dist = float("inf")
                for ic in intra_clusters:
                    dist = float(np.linalg.norm(emb - ic["centroid"]))
                    if dist < best_intra_dist:
                        best_intra_dist = dist
                        best_intra_id = ic["id"]

                if best_intra_id is not None:
                    conn.execute(
                        "INSERT INTO cluster_assignments (doc_id, source_type, cluster_id, distance) "
                        "VALUES (?, ?, ?, ?)",
                        (doc_id, source_type, best_intra_id, best_intra_dist),
                    )
                    conn.execute(
                        "UPDATE clusters SET prompt_count = prompt_count + 1 WHERE id = ?",
                        (best_intra_id,),
                    )


# ---------------------------------------------------------------------------
# Dataset map
# ---------------------------------------------------------------------------

def get_dataset_map() -> dict:
    """Build a structured dataset map combining ChromaDB and cluster data.

    Returns:
        Dictionary with ``cross_folder_themes``, ``folders``, and ``stats`` keys.
    """
    # Get total docs from ChromaDB
    counts = vector_store.get_collection_counts()
    total_chromadb_docs = counts.get("training", 0) + counts.get("generated", 0)

    # Get concept info from ChromaDB
    concept_list = vector_store.list_concepts()
    # Aggregate by concept
    folder_info: dict[str, dict] = {}
    for c in concept_list:
        name = c["concept"]
        if name not in folder_info:
            folder_info[name] = {"name": name, "source_type": c["source_type"], "total_prompts": 0}
        folder_info[name]["total_prompts"] += c["count"]
        # Keep most specific source_type (if mixed, use "training")
        if c["source_type"] == "training":
            folder_info[name]["source_type"] = "training"

    with get_db() as conn:
        # Cross-folder themes
        cursor = conn.execute(
            "SELECT id, label, prompt_count FROM clusters WHERE cluster_type = 'cross_folder' ORDER BY id"
        )
        cross_folder_themes = [
            {"id": row["id"], "label": row["label"], "prompt_count": row["prompt_count"]}
            for row in cursor.fetchall()
        ]

        # Intra-folder themes grouped by folder
        cursor = conn.execute(
            "SELECT id, folder_path, label, prompt_count FROM clusters "
            "WHERE cluster_type = 'intra_folder' ORDER BY folder_path, id"
        )
        intra_by_folder: dict[str, list[dict]] = {}
        for row in cursor.fetchall():
            fp = row["folder_path"]
            if fp not in intra_by_folder:
                intra_by_folder[fp] = []
            intra_by_folder[fp].append({
                "id": row["id"],
                "label": row["label"],
                "prompt_count": row["prompt_count"],
            })

        # Count docs with cross_folder assignment
        cursor = conn.execute(
            "SELECT COUNT(DISTINCT ca.doc_id) as cnt FROM cluster_assignments ca "
            "JOIN clusters c ON ca.cluster_id = c.id WHERE c.cluster_type = 'cross_folder'"
        )
        cross_assigned_count = cursor.fetchone()["cnt"]

        # Count docs incrementally assigned to intra clusters
        # (assigned to intra clusters but not from a full intra run — approximation:
        #  docs in intra assignments that were not part of the latest intra run)
        cursor = conn.execute(
            "SELECT COUNT(DISTINCT ca.doc_id) as cnt FROM cluster_assignments ca "
            "JOIN clusters c ON ca.cluster_id = c.id WHERE c.cluster_type = 'intra_folder'"
        )
        intra_assigned_count = cursor.fetchone()["cnt"]

        # Total clusters
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM clusters")
        total_clusters = cursor.fetchone()["cnt"]

        # Total intra themes
        cursor = conn.execute(
            "SELECT COUNT(*) as cnt FROM clusters WHERE cluster_type = 'intra_folder'"
        )
        total_intra = cursor.fetchone()["cnt"]

    new_since_last_cross = total_chromadb_docs - cross_assigned_count

    # Build folders list
    folders = []
    for name, info in sorted(folder_info.items()):
        folder_entry = {
            "name": info["name"],
            "source_type": info["source_type"],
            "total_prompts": info["total_prompts"],
            "intra_themes": intra_by_folder.get(name, []),
        }
        folders.append(folder_entry)

    return {
        "cross_folder_themes": cross_folder_themes,
        "folders": folders,
        "stats": {
            "total_prompts": total_chromadb_docs,
            "total_cross_themes": len(cross_folder_themes),
            "total_intra_themes": total_intra,
            "new_since_last_cross_cluster": max(0, new_since_last_cross),
            "assigned_to_existing_intra": intra_assigned_count,
        },
    }


# ---------------------------------------------------------------------------
# Themed prompt retrieval
# ---------------------------------------------------------------------------

def get_themed_prompts(
    query_embedding: list[float],
    k_similar: int | None = None,
    k_intra: int | None = None,
    k_cross: int | None = None,
    k_random: int = 0,
    k_opposite: int = 0,
    source_type: str | None = None,
) -> dict:
    """Unified query returning prompts from multiple retrieval strategies.

    Args:
        query_embedding: The query vector to search against.
        k_similar: Number of direct similar results. Defaults to ``query_k_similar`` setting.
        k_intra: Number of prompts per intra-folder theme cluster. Defaults to ``query_k_theme_intra``.
        k_cross: Number of prompts per cross-folder theme cluster. Defaults to ``query_k_theme_cross``.
        k_random: Number of random prompts to include.
        k_opposite: Number of diverse/opposite prompts to include.
        source_type: Filter by ``'training'`` or ``'output'``, or ``None`` for both.

    Returns:
        Dictionary with ``direct_similar``, ``intra_theme_matches``,
        ``cross_theme_matches``, ``random``, ``opposite``, and ``total_count`` keys.
    """
    # Read defaults from settings
    if k_similar is None:
        v = settings.get_setting("query_k_similar")
        k_similar = int(v) if v else 10
    if k_intra is None:
        v = settings.get_setting("query_k_theme_intra")
        k_intra = int(v) if v else 5
    if k_cross is None:
        v = settings.get_setting("query_k_theme_cross")
        k_cross = int(v) if v else 5

    result: dict = {
        "direct_similar": [],
        "intra_theme_matches": [],
        "cross_theme_matches": [],
        "random": [],
        "opposite": [],
        "total_count": 0,
    }

    query_emb = np.array(query_embedding)

    # 1. Direct similar
    similar_results = vector_store.search_similar(query_embedding, k=k_similar, source_type=source_type)
    result["direct_similar"] = [
        {
            "text": r["document"],
            "concept": r["metadata"].get("concept", ""),
            "source": r["metadata"].get("dir_type", ""),
            "distance": r["distance"],
        }
        for r in similar_results
    ]

    # 2. Intra-folder themed
    result["intra_theme_matches"] = _get_theme_matches(
        query_emb, cluster_type="intra_folder", k_per_cluster=k_intra, top_clusters=3,
        source_type=source_type,
    )

    # 3. Cross-folder themed
    result["cross_theme_matches"] = _get_theme_matches(
        query_emb, cluster_type="cross_folder", k_per_cluster=k_cross, top_clusters=3,
        source_type=source_type,
    )

    # 4. Random
    if k_random > 0:
        random_results = vector_store.get_random(k=k_random, source_type=source_type)
        result["random"] = [
            {
                "text": r["document"],
                "concept": r["metadata"].get("concept", ""),
                "source": r["metadata"].get("dir_type", ""),
                "distance": r["distance"],
            }
            for r in random_results
        ]

    # 5. Opposite
    if k_opposite > 0:
        opposite_results = vector_store.search_diverse(query_embedding, k=k_opposite, offset=100, source_type=source_type)
        result["opposite"] = [
            {
                "text": r["document"],
                "concept": r["metadata"].get("concept", ""),
                "source": r["metadata"].get("dir_type", ""),
                "distance": r["distance"],
            }
            for r in opposite_results
        ]

    # Total count
    total = len(result["direct_similar"]) + len(result["random"]) + len(result["opposite"])
    for tm in result["intra_theme_matches"]:
        total += len(tm.get("prompts", []))
    for tm in result["cross_theme_matches"]:
        total += len(tm.get("prompts", []))
    result["total_count"] = total

    return result


def _get_theme_matches(
    query_emb: np.ndarray,
    cluster_type: str,
    k_per_cluster: int,
    top_clusters: int = 3,
    source_type: str | None = None,
) -> list[dict]:
    """Find the closest cluster centroids and return their assigned prompts.

    Args:
        query_emb: Query embedding as numpy array.
        cluster_type: ``"cross_folder"`` or ``"intra_folder"``.
        k_per_cluster: Maximum prompts to return per cluster.
        top_clusters: Number of nearest clusters to use.
        source_type: Filter by ``'training'`` or ``'output'``, or ``None`` for both.

    Returns:
        List of theme match dicts with ``theme_label``, ``theme_id``, and ``prompts``.
    """
    # Load centroids from SQLite
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT id, label, centroid FROM clusters WHERE cluster_type = ? AND centroid IS NOT NULL",
            (cluster_type,),
        )
        clusters = []
        for row in cursor.fetchall():
            try:
                centroid = np.array(json.loads(row["centroid"]))
                clusters.append({
                    "id": row["id"],
                    "label": row["label"],
                    "centroid": centroid,
                })
            except (json.JSONDecodeError, ValueError):
                continue

    if not clusters:
        return []

    # Find top N closest clusters to query
    distances = [(c, float(np.linalg.norm(query_emb - c["centroid"]))) for c in clusters]
    distances.sort(key=lambda x: x[1])
    nearest = distances[:top_clusters]

    theme_matches: list[dict] = []
    for cluster_info, _ in nearest:
        cluster_id = cluster_info["id"]
        cluster_label = cluster_info["label"]

        # Get assigned doc IDs
        with get_db() as conn:
            if source_type:
                cursor = conn.execute(
                    "SELECT doc_id, source_type FROM cluster_assignments WHERE cluster_id = ? AND source_type = ? ORDER BY distance ASC LIMIT ?",
                    (cluster_id, source_type, k_per_cluster),
                )
            else:
                cursor = conn.execute(
                    "SELECT doc_id, source_type FROM cluster_assignments WHERE cluster_id = ? ORDER BY distance ASC LIMIT ?",
                    (cluster_id, k_per_cluster),
                )
            assignments = cursor.fetchall()

        if not assignments:
            continue

        # Fetch documents from ChromaDB by ID
        doc_id_list = [a["doc_id"] for a in assignments]
        source_type_map = {a["doc_id"]: a["source_type"] for a in assignments}

        prompts = _fetch_docs_by_ids(doc_id_list, source_type_map)
        if prompts:
            theme_matches.append({
                "theme_label": cluster_label,
                "theme_id": cluster_id,
                "prompts": prompts,
            })

    return theme_matches


def _fetch_docs_by_ids(doc_ids: list[str], source_type_map: dict[str, str]) -> list[dict]:
    """Fetch documents from ChromaDB by their IDs.

    Args:
        doc_ids: List of document IDs to fetch.
        source_type_map: Mapping of doc_id to source_type for collection selection.

    Returns:
        List of prompt dicts with ``text``, ``concept``, ``source``, ``distance`` keys.
    """
    prompts: list[dict] = []

    # Group IDs by collection
    training_ids = [did for did in doc_ids if source_type_map.get(did, "training") == "training"]
    output_ids = [did for did in doc_ids if source_type_map.get(did, "training") == "output"]

    for ids_batch, collection in [
        (training_ids, vector_store._training_collection),
        (output_ids, vector_store._generated_collection),
    ]:
        if not ids_batch or collection is None:
            continue
        try:
            result = collection.get(
                ids=ids_batch,
                include=["documents", "metadatas"],
            )
            for i in range(len(result["ids"])):
                meta = result["metadatas"][i] if result["metadatas"] else {}
                prompts.append({
                    "text": result["documents"][i],
                    "concept": meta.get("concept", ""),
                    "source": meta.get("dir_type", ""),
                    "distance": 0,
                })
        except Exception as e:
            logger.error(f"Error fetching docs by IDs from collection: {e}")

    return prompts


# ---------------------------------------------------------------------------
# Background thread entry point
# ---------------------------------------------------------------------------

def start_clustering(cross_folder: bool = True, intra_folder: bool = True, force: bool = False):
    """Start clustering in a background thread.

    Args:
        cross_folder: Whether to run cross-folder clustering.
        intra_folder: Whether to run intra-folder clustering.
        force: If True, re-cluster even if data hasn't changed.
    """
    global _clustering_running
    with _clustering_lock:
        if _clustering_running:
            logger.warning("Clustering already running, skipping")
            return
        _clustering_running = True

    thread = threading.Thread(
        target=_run_clustering,
        args=(cross_folder, intra_folder, force),
        daemon=True,
    )
    thread.start()


def _run_clustering(cross_folder: bool, intra_folder: bool, force: bool):
    """Main clustering logic — runs in a background thread."""
    global _clustering_running
    progress = ClusteringProgress()

    try:
        if cross_folder:
            progress.phase = "cross_folder"
            progress.message = "Running cross-folder clustering..."
            _emit_status(progress)
            generate_cross_folder_clusters()

        if intra_folder:
            progress.phase = "intra_folder"
            progress.message = "Running intra-folder clustering..."
            _emit_status(progress)
            generate_intra_folder_clusters(force=force)

        progress.phase = "complete"
        progress.complete = True
        progress.message = "Clustering complete."
        _emit_status(progress)

    except Exception as e:
        logger.error(f"Clustering error: {e}", exc_info=True)
        progress.phase = "error"
        progress.message = f"Clustering failed: {str(e)}"
        _emit_status(progress)

    finally:
        _clustering_running = False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_clustering_stats() -> dict:
    """Return clustering statistics for header display.

    Returns:
        Dictionary with ``new_since_last_cross_cluster``,
        ``assigned_to_existing_intra``, ``last_cross_cluster_run``,
        and ``total_clusters`` keys.
    """
    counts = vector_store.get_collection_counts()
    total_chromadb_docs = counts.get("training", 0) + counts.get("generated", 0)

    with get_db() as conn:
        # Docs assigned to cross-folder clusters
        cursor = conn.execute(
            "SELECT COUNT(DISTINCT ca.doc_id) as cnt FROM cluster_assignments ca "
            "JOIN clusters c ON ca.cluster_id = c.id WHERE c.cluster_type = 'cross_folder'"
        )
        cross_assigned = cursor.fetchone()["cnt"]

        # Docs assigned to intra-folder clusters
        cursor = conn.execute(
            "SELECT COUNT(DISTINCT ca.doc_id) as cnt FROM cluster_assignments ca "
            "JOIN clusters c ON ca.cluster_id = c.id WHERE c.cluster_type = 'intra_folder'"
        )
        intra_assigned = cursor.fetchone()["cnt"]

        # Last cross-folder run
        cursor = conn.execute(
            "SELECT completed_at FROM clustering_runs "
            "WHERE run_type = 'cross_folder' ORDER BY completed_at DESC LIMIT 1"
        )
        last_run_row = cursor.fetchone()
        last_cross_run = last_run_row["completed_at"] if last_run_row else None

        # Total clusters
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM clusters")
        total_clusters = cursor.fetchone()["cnt"]

        # Separate cluster counts by type
        cursor = conn.execute(
            "SELECT COUNT(*) as cnt FROM clusters WHERE cluster_type = 'cross_folder'"
        )
        cross_folder_clusters = cursor.fetchone()["cnt"]

        cursor = conn.execute(
            "SELECT COUNT(*) as cnt FROM clusters WHERE cluster_type = 'intra_folder'"
        )
        intra_folder_clusters = cursor.fetchone()["cnt"]

    return {
        "new_since_last_cross_cluster": max(0, total_chromadb_docs - cross_assigned),
        "assigned_to_existing_intra": intra_assigned,
        "last_cross_cluster_run": last_cross_run,
        "total_clusters": total_clusters,
        "cross_folder_clusters": cross_folder_clusters,
        "intra_folder_clusters": intra_folder_clusters,
    }
