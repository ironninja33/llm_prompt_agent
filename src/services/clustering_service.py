"""Clustering service — KMeans clustering of prompt embeddings with TF-IDF labeling.

Operates on ChromaDB (vector_store) for embeddings/documents, SQLite (database)
for cluster metadata/assignments/runs, and settings for configuration. Fully
decoupled from the agent loop and LLM interface.
"""

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import text

from src.models import vector_store, settings
from src.models.database import get_db, row_to_dict

logger = logging.getLogger(__name__)

# Regex to split prompts on punctuation boundaries (commas, periods,
# semicolons, colons, parentheses, brackets, pipes, etc.).  Sequences of
# punctuation and surrounding whitespace are consumed together so that
# tokens adjacent to punctuation don't form cross-boundary ngrams.
_PUNCT_SPLIT_RE = re.compile(r'[\s]*[,\.;:!?\(\)\[\]\{\}|/\\]+[\s]*')

# Token pattern: 2+ word-characters (letters, digits, underscore).
_TOKEN_RE = re.compile(r'\b\w\w+\b', re.UNICODE)

# Domain-specific stop words for TF-IDF cluster labeling.
# These terms are too generic or redundant to be useful as cluster labels.
_DOMAIN_STOP_WORDS: set[str] = {
    # ── Technical image generation terms ──
    "photorealistic", "photorealism", "photograph", "photography",
    "photo", "image", "picture", "render", "rendered", "rendering",
    "realistic", "realism", "hyperrealistic",
    "resolution", "quality", "masterpiece", "masterwork",
    "detailed", "detail", "details", "intricate",
    "high", "ultra", "extremely", "very", "highly", "incredibly",
    "8k", "4k", "hd", "uhd",
    "depicts", "depicting", "depicted", "shows", "showing", "features",
    "wearing", "wears", "worn", "dressed",
    "beautiful", "gorgeous", "stunning", "attractive",
    "professional", "cinematic",
    # ── Common prompt boilerplate ──
    "style", "aesthetic", "look", "looking", "looks",
    "scene", "shot", "view", "camera", "angle",
    "background", "foreground",
    # ── Overly generic descriptors ──
    "woman", "women", "man", "men", "person", "people",
    "young", "adult", "female", "male",
    "face", "body", "hair", "eyes", "skin",
    "standing", "sitting", "posing", "posed",
}


_DEFAULT_TIERS_TRAINING = [
    {"max_prompts": 40, "k": 2},
    {"max_prompts": 80, "k": 3},
    {"max_prompts": 150, "k": 4},
    {"max_prompts": None, "k": 5},
]

_DEFAULT_TIERS_OUTPUT = [
    {"max_prompts": 30, "k": 3},
    {"max_prompts": 100, "k": 7},
    {"max_prompts": 300, "k": 10},
    {"max_prompts": None, "k": 15},
]


def _compute_adaptive_k(n_prompts: int, source_type: str) -> int:
    """Compute default cluster count based on folder size and type.

    Reads configurable tier lists from settings. Training folders are narrow
    (one subject), need fewer clusters. Output folders are diverse (many
    subjects/styles), need more.
    """
    key = "adaptive_k_output" if source_type == "output" else "adaptive_k_training"
    tiers_json = settings.get_setting(key)
    if tiers_json:
        tiers = json.loads(tiers_json)
    else:
        tiers = _DEFAULT_TIERS_OUTPUT if source_type == "output" else _DEFAULT_TIERS_TRAINING

    for tier in tiers:
        if tier["max_prompts"] is None or n_prompts < tier["max_prompts"]:
            return tier["k"]
    return tiers[-1]["k"]


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


def _fetch_embeddings_by_concept(
    concept_name: str,
    source_type: str | None = None,
) -> tuple[list[str], list[list[float]], list[str], list[dict]]:
    """Fetch embeddings and documents for a specific concept.

    Args:
        concept_name: The concept/folder name to filter by.
        source_type: ``"training"`` to fetch only from the training collection,
            ``"output"`` for only the generated collection, or ``None`` for both.

    Returns:
        Tuple of (ids, embeddings, documents, metadatas).
    """
    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    if source_type == "training":
        collections = [vector_store._training_collection]
    elif source_type == "output":
        collections = [vector_store._generated_collection]
    else:
        collections = [vector_store._training_collection, vector_store._generated_collection]

    for collection in collections:
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

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    combined_stop_words = ENGLISH_STOP_WORDS | _DOMAIN_STOP_WORDS

    for ngram_range, min_df, max_feats in [
        ((2, 3), 2, 3000),
        ((2, 3), 1, 3000),
        ((1, 3), 1, 2000),
    ]:
        try:
            analyzer = _make_punctuation_aware_analyzer(ngram_range, stop_words=combined_stop_words)
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


def _generate_folder_summaries(
    folder_texts: dict[str, list[str]],
    top_n: int = 3,
) -> dict[str, str]:
    """Generate summary terms for each folder using TF-IDF.

    Each folder's prompts are concatenated into one "document", then TF-IDF
    is computed across all folder-documents.  The top terms per folder are
    its summary — what distinguishes it from other folders.

    Args:
        folder_texts: Mapping of folder_name -> list of prompt texts.
        top_n: Number of summary terms per folder.

    Returns:
        Mapping of folder_name -> comma-separated summary string.
    """
    if not folder_texts:
        return {}

    folder_names = sorted(folder_texts.keys())
    corpus = [" ".join(folder_texts[name]) for name in folder_names]

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    combined_stops = ENGLISH_STOP_WORDS | _DOMAIN_STOP_WORDS
    analyzer = _make_punctuation_aware_analyzer((2, 3), stop_words=combined_stops)

    try:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            max_features=2000,
            min_df=1,
            max_df=0.85,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        return {name: "" for name in folder_names}

    summaries = {}
    for i, name in enumerate(folder_names):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n]
        terms = [feature_names[j] for j in top_indices if row[j] > 0]
        summaries[name] = ", ".join(terms) if terms else ""

    return summaries


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
            text("DELETE FROM cluster_assignments WHERE cluster_id IN "
                 "(SELECT id FROM clusters WHERE cluster_type = 'cross_folder')")
        )
        conn.execute(text("DELETE FROM clusters WHERE cluster_type = 'cross_folder'"))

        # Insert new clusters
        cluster_id_map: dict[int, int] = {}  # cluster_index -> db id
        for cd in cluster_data:
            result = conn.execute(
                text("INSERT INTO clusters (cluster_type, folder_path, cluster_index, label, centroid, prompt_count) "
                     "VALUES (:cluster_type, :folder_path, :cluster_index, :label, :centroid, :prompt_count)"),
                {"cluster_type": "cross_folder", "folder_path": None,
                 "cluster_index": cd["cluster_index"], "label": cd["label"],
                 "centroid": cd["centroid"], "prompt_count": cd["prompt_count"]},
            )
            cluster_id_map[cd["cluster_index"]] = result.lastrowid

        # Insert assignments
        for ad in assignment_data:
            db_cluster_id = cluster_id_map[ad["cluster_index"]]
            conn.execute(
                text("INSERT INTO cluster_assignments (doc_id, source_type, cluster_id, distance) "
                     "VALUES (:doc_id, :source_type, :cluster_id, :distance)"),
                {"doc_id": ad["doc_id"], "source_type": ad["source_type"],
                 "cluster_id": db_cluster_id, "distance": ad["distance"]},
            )

        # 9. Record the run
        conn.execute(
            text("INSERT INTO clustering_runs (run_type, folder_path, total_prompts, num_clusters, started_at, completed_at) "
                 "VALUES (:run_type, :folder_path, :total_prompts, :num_clusters, :started_at, CURRENT_TIMESTAMP)"),
            {"run_type": "cross_folder", "folder_path": None,
             "total_prompts": n_samples, "num_clusters": effective_k, "started_at": started_at},
        )

    # 10. Emit completion
    progress.message = f"Cross-folder clustering complete: {effective_k} clusters from {n_samples} documents."
    progress.current = n_samples
    _emit_status(progress)


# ---------------------------------------------------------------------------
# Intra-folder clustering
# ---------------------------------------------------------------------------

def generate_intra_folder_clusters(
    folder_path: str | None = None,
    k: int | None = None,
    force: bool = False,
    source_type: str | None = None,
):
    """Run KMeans clustering within individual folders/concepts, split by source_type.

    Args:
        folder_path: If provided, only cluster this specific concept. Otherwise
                     cluster all concepts meeting minimum size.
        k: Number of clusters per folder. Falls back to adaptive tiers from settings.
        force: If True, skip size and freshness checks.
        source_type: If provided, only cluster this source_type for the given
            folder_path. When ``None`` (full run), clusters each (concept, source_type)
            pair independently.
    """
    progress = ClusteringProgress(phase="intra_folder", message="Starting intra-folder clustering...")
    _emit_status(progress)

    # 1. Read settings
    # Track whether k was explicitly passed (single-folder recluster) vs adaptive default
    k_explicit = k is not None

    min_size_str = settings.get_setting("cluster_min_folder_size")
    min_folder_size = int(min_size_str) if min_size_str else 20

    # 2. Determine which (concept, source_type) pairs to cluster
    if folder_path and source_type:
        entries_to_cluster = [{"concept": folder_path, "source_type": source_type}]
    elif folder_path:
        # Cluster both source types for this concept
        entries_to_cluster = [
            {"concept": folder_path, "source_type": "training"},
            {"concept": folder_path, "source_type": "output"},
        ]
    else:
        # Full run: iterate all (concept, source_type) pairs from ChromaDB
        all_concepts = _get_all_concepts()
        entries_to_cluster = [
            {"concept": c["concept"], "source_type": c["source_type"]}
            for c in all_concepts
        ]

    progress.total = len(entries_to_cluster)
    progress.message = f"Processing {len(entries_to_cluster)} concept/source pair(s)..."
    _emit_status(progress)

    # 3. Cluster each (concept, source_type) pair
    for idx, entry in enumerate(entries_to_cluster):
        concept_name = entry["concept"]
        entry_source = entry["source_type"]
        progress.current = idx + 1
        progress.message = f"Clustering '{concept_name}' ({entry_source}) ({idx + 1}/{len(entries_to_cluster)})..."
        _emit_status(progress)

        # Fetch embeddings for this concept + source_type
        doc_ids, embeddings_list, documents, metadatas = _fetch_embeddings_by_concept(
            concept_name, source_type=entry_source,
        )
        n_samples = len(doc_ids)

        # Skip if too few prompts (unless forced)
        if n_samples < min_folder_size and not force:
            logger.info(
                f"Skipping '{concept_name}' ({entry_source}): {n_samples} docs < min_folder_size {min_folder_size}"
            )
            continue

        if n_samples <= 1:
            logger.info(f"Skipping '{concept_name}' ({entry_source}): only {n_samples} document(s)")
            continue

        # Check freshness unless forced
        if not force:
            with get_db() as conn:
                result = conn.execute(
                    text("SELECT id FROM clusters WHERE cluster_type = 'intra_folder' "
                         "AND folder_path = :folder_path AND source_type = :source_type LIMIT 1"),
                    {"folder_path": concept_name, "source_type": entry_source},
                )
                existing_cluster = result.fetchone()

                if existing_cluster:
                    assigned_result = conn.execute(
                        text("SELECT DISTINCT ca.doc_id FROM cluster_assignments ca "
                             "JOIN clusters c ON ca.cluster_id = c.id "
                             "WHERE c.cluster_type = 'intra_folder' AND c.folder_path = :folder_path "
                             "AND c.source_type = :source_type"),
                        {"folder_path": concept_name, "source_type": entry_source},
                    )
                    assigned_ids = {row._mapping["doc_id"] for row in assigned_result.fetchall()}
                    unassigned = [did for did in doc_ids if did not in assigned_ids]

                    if not unassigned:
                        logger.info(f"Skipping '{concept_name}' ({entry_source}): all docs already assigned")
                        continue

        # Determine effective k: explicit k param > per-folder override > adaptive tiers
        if not k_explicit:
            per_folder_k_str = settings.get_setting(f"cluster_k_intra:{concept_name}:{entry_source}")
            if per_folder_k_str:
                effective_base_k = int(per_folder_k_str)
            else:
                effective_base_k = _compute_adaptive_k(n_samples, entry_source)
        else:
            effective_base_k = k

        # Run KMeans
        effective_k = min(effective_base_k, n_samples)
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
            doc_source = metadatas[i].get("dir_type", "training") if metadatas[i] else "training"
            assignment_data.append({
                "doc_id": doc_ids[i],
                "source_type": doc_source,
                "cluster_index": cluster_idx,
                "distance": distance,
            })

        # Store in SQLite
        started_at = datetime.utcnow().isoformat()

        with get_db() as conn:
            # Clear old intra_folder clusters for this (concept, source_type)
            conn.execute(
                text("DELETE FROM cluster_assignments WHERE cluster_id IN "
                     "(SELECT id FROM clusters WHERE cluster_type = 'intra_folder' "
                     "AND folder_path = :folder_path AND source_type = :source_type)"),
                {"folder_path": concept_name, "source_type": entry_source},
            )
            conn.execute(
                text("DELETE FROM clusters WHERE cluster_type = 'intra_folder' "
                     "AND folder_path = :folder_path AND source_type = :source_type"),
                {"folder_path": concept_name, "source_type": entry_source},
            )

            # Insert clusters
            cluster_id_map: dict[int, int] = {}
            for cd in cluster_data:
                result = conn.execute(
                    text("INSERT INTO clusters (cluster_type, folder_path, cluster_index, label, centroid, prompt_count, source_type) "
                         "VALUES (:cluster_type, :folder_path, :cluster_index, :label, :centroid, :prompt_count, :source_type)"),
                    {"cluster_type": "intra_folder", "folder_path": concept_name,
                     "cluster_index": cd["cluster_index"], "label": cd["label"],
                     "centroid": cd["centroid"], "prompt_count": cd["prompt_count"],
                     "source_type": entry_source},
                )
                cluster_id_map[cd["cluster_index"]] = result.lastrowid

            # Insert assignments
            for ad in assignment_data:
                db_cluster_id = cluster_id_map[ad["cluster_index"]]
                conn.execute(
                    text("INSERT INTO cluster_assignments (doc_id, source_type, cluster_id, distance) "
                         "VALUES (:doc_id, :source_type, :cluster_id, :distance)"),
                    {"doc_id": ad["doc_id"], "source_type": ad["source_type"],
                     "cluster_id": db_cluster_id, "distance": ad["distance"]},
                )

            # Record run
            conn.execute(
                text("INSERT INTO clustering_runs (run_type, folder_path, total_prompts, num_clusters, started_at, completed_at) "
                     "VALUES (:run_type, :folder_path, :total_prompts, :num_clusters, :started_at, CURRENT_TIMESTAMP)"),
                {"run_type": "intra_folder", "folder_path": concept_name,
                 "total_prompts": n_samples, "num_clusters": effective_k, "started_at": started_at},
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
        result = conn.execute(
            text("SELECT id, centroid FROM clusters WHERE cluster_type = 'cross_folder' AND centroid IS NOT NULL")
        )
        for row in result.fetchall():
            r = row._mapping
            cross_clusters.append({
                "id": r["id"],
                "centroid": np.array(json.loads(r["centroid"])),
            })

        # Intra-folder clusters — group by folder_path
        result = conn.execute(
            text("SELECT id, folder_path, centroid FROM clusters "
                 "WHERE cluster_type = 'intra_folder' AND centroid IS NOT NULL")
        )
        for row in result.fetchall():
            r = row._mapping
            folder = r["folder_path"]
            if folder not in intra_clusters_by_folder:
                intra_clusters_by_folder[folder] = []
            intra_clusters_by_folder[folder].append({
                "id": r["id"],
                "centroid": np.array(json.loads(r["centroid"])),
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
                        text("INSERT INTO cluster_assignments (doc_id, source_type, cluster_id, distance) "
                             "VALUES (:doc_id, :source_type, :cluster_id, :distance)"),
                        {"doc_id": doc_id, "source_type": source_type,
                         "cluster_id": best_cross_id, "distance": best_cross_dist},
                    )
                    # Update prompt_count
                    conn.execute(
                        text("UPDATE clusters SET prompt_count = prompt_count + 1 WHERE id = :id"),
                        {"id": best_cross_id},
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
                        text("INSERT INTO cluster_assignments (doc_id, source_type, cluster_id, distance) "
                             "VALUES (:doc_id, :source_type, :cluster_id, :distance)"),
                        {"doc_id": doc_id, "source_type": source_type,
                         "cluster_id": best_intra_id, "distance": best_intra_dist},
                    )
                    conn.execute(
                        text("UPDATE clusters SET prompt_count = prompt_count + 1 WHERE id = :id"),
                        {"id": best_intra_id},
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
    # Aggregate by (concept, source_type) so training and output show separately
    folder_info: dict[tuple[str, str], dict] = {}
    for c in concept_list:
        name = c["concept"]
        key = (name, c["source_type"])
        if key not in folder_info:
            folder_info[key] = {"name": name, "source_type": c["source_type"], "total_prompts": 0}
        folder_info[key]["total_prompts"] += c["count"]

    with get_db() as conn:
        # Cross-folder themes
        result = conn.execute(
            text("SELECT id, label, prompt_count FROM clusters WHERE cluster_type = 'cross_folder' ORDER BY id")
        )
        cross_folder_themes = [
            {"label": row._mapping["label"], "prompt_count": row._mapping["prompt_count"]}
            for row in result.fetchall()
        ]

        # Intra-folder themes grouped by (folder, source_type)
        result = conn.execute(
            text("SELECT id, folder_path, source_type, label, prompt_count FROM clusters "
                 "WHERE cluster_type = 'intra_folder' ORDER BY folder_path, source_type, id")
        )
        intra_by_folder_source: dict[tuple[str, str], list[dict]] = {}
        for row in result.fetchall():
            r = row._mapping
            key = (r["folder_path"], r["source_type"] or "training")
            if key not in intra_by_folder_source:
                intra_by_folder_source[key] = []
            intra_by_folder_source[key].append({
                "label": r["label"],
                "prompt_count": r["prompt_count"],
            })

        # Count docs with cross_folder assignment
        result = conn.execute(
            text("SELECT COUNT(DISTINCT ca.doc_id) as cnt FROM cluster_assignments ca "
                 "JOIN clusters c ON ca.cluster_id = c.id WHERE c.cluster_type = 'cross_folder'")
        )
        cross_assigned_count = result.fetchone()._mapping["cnt"]

        # Count docs incrementally assigned to intra clusters
        result = conn.execute(
            text("SELECT COUNT(DISTINCT ca.doc_id) as cnt FROM cluster_assignments ca "
                 "JOIN clusters c ON ca.cluster_id = c.id WHERE c.cluster_type = 'intra_folder'")
        )
        intra_assigned_count = result.fetchone()._mapping["cnt"]

        # Total clusters
        result = conn.execute(text("SELECT COUNT(*) as cnt FROM clusters"))
        total_clusters = result.fetchone()._mapping["cnt"]

        # Total intra themes
        result = conn.execute(
            text("SELECT COUNT(*) as cnt FROM clusters WHERE cluster_type = 'intra_folder'")
        )
        total_intra = result.fetchone()._mapping["cnt"]

        # Folder summaries
        result = conn.execute(text("SELECT folder_path, summary FROM folder_summaries"))
        summaries = {r._mapping["folder_path"]: r._mapping["summary"] for r in result.fetchall()}

    new_since_last_cross = total_chromadb_docs - cross_assigned_count

    # Build folders list
    from src.models.browser import parse_concept_name

    folders = []
    for (name, source_type), info in sorted(folder_info.items()):
        parsed = parse_concept_name(name)
        folder_entry = {
            "name": info["name"],
            "category": parsed["category"],
            "display_name": parsed["display_name"],
            "source_type": info["source_type"],
            "total_prompts": info["total_prompts"],
            "intra_themes": intra_by_folder_source.get((name, source_type), []),
            "summary": summaries.get(name, ""),
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


def get_dataset_overview() -> dict:
    """Lightweight dataset overview — no intra-folder cluster details.

    Returns folder names, source types, prompt counts, per-folder summary
    terms, cross-folder themes, and aggregate stats.  Used for LLM cache
    pre-loading and the ``get_dataset_overview`` agent tool.
    """
    counts = vector_store.get_collection_counts()
    total_docs = counts.get("training", 0) + counts.get("generated", 0)

    concept_list = vector_store.list_concepts()
    folder_info: dict[tuple[str, str], dict] = {}
    for c in concept_list:
        name = c["concept"]
        key = (name, c["source_type"])
        if key not in folder_info:
            folder_info[key] = {"name": name, "source_type": c["source_type"], "total_prompts": 0}
        folder_info[key]["total_prompts"] += c["count"]

    with get_db() as conn:
        # Cross-folder themes (no id)
        result = conn.execute(
            text("SELECT label, prompt_count FROM clusters "
                 "WHERE cluster_type = 'cross_folder' ORDER BY prompt_count DESC")
        )
        cross_themes = [
            {"label": r._mapping["label"], "prompt_count": r._mapping["prompt_count"]}
            for r in result.fetchall()
        ]

        # Folder summaries
        result = conn.execute(text("SELECT folder_path, summary FROM folder_summaries"))
        summaries = {r._mapping["folder_path"]: r._mapping["summary"] for r in result.fetchall()}

        # Stats
        result = conn.execute(
            text("SELECT COUNT(*) as cnt FROM clusters WHERE cluster_type = 'intra_folder'")
        )
        total_intra = result.fetchone()._mapping["cnt"]

    folders = []
    from src.models.browser import parse_concept_name

    for (name, source_type), info in sorted(folder_info.items()):
        parsed = parse_concept_name(name)
        folders.append({
            "name": info["name"],
            "category": parsed["category"],
            "display_name": parsed["display_name"],
            "source_type": info["source_type"],
            "total_prompts": info["total_prompts"],
            "summary": summaries.get(name, ""),
        })

    return {
        "cross_folder_themes": cross_themes,
        "folders": folders,
        "stats": {
            "total_prompts": total_docs,
            "total_cross_themes": len(cross_themes),
            "total_intra_themes": total_intra,
        },
    }


def get_folder_themes(folder_name: str, source_type: str = "training") -> dict:
    """Get intra-folder cluster themes for a specific folder and source type.

    Args:
        folder_name: Concept/folder name.
        source_type: ``"training"`` or ``"output"``.

    Returns:
        Dict with ``folder`` name, ``source_type``, and ``themes`` list of
        ``{label, prompt_count}`` dicts.
    """
    with get_db() as conn:
        result = conn.execute(
            text("SELECT label, prompt_count FROM clusters "
                 "WHERE cluster_type = 'intra_folder' AND folder_path = :fp "
                 "AND source_type = :st "
                 "ORDER BY prompt_count DESC"),
            {"fp": folder_name, "st": source_type},
        )
        themes = [
            {"label": r._mapping["label"], "prompt_count": r._mapping["prompt_count"]}
            for r in result.fetchall()
        ]

    return {"folder": folder_name, "source_type": source_type, "themes": themes}


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
        result = conn.execute(
            text("SELECT id, label, centroid FROM clusters WHERE cluster_type = :cluster_type AND centroid IS NOT NULL"),
            {"cluster_type": cluster_type},
        )
        clusters = []
        for row in result.fetchall():
            r = row._mapping
            try:
                centroid = np.array(json.loads(r["centroid"]))
                clusters.append({
                    "id": r["id"],
                    "label": r["label"],
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
                result = conn.execute(
                    text("SELECT doc_id, source_type FROM cluster_assignments WHERE cluster_id = :cluster_id AND source_type = :source_type ORDER BY distance ASC LIMIT :limit"),
                    {"cluster_id": cluster_id, "source_type": source_type, "limit": k_per_cluster},
                )
            else:
                result = conn.execute(
                    text("SELECT doc_id, source_type FROM cluster_assignments WHERE cluster_id = :cluster_id ORDER BY distance ASC LIMIT :limit"),
                    {"cluster_id": cluster_id, "limit": k_per_cluster},
                )
            assignments = result.fetchall()

        if not assignments:
            continue

        # Fetch documents from ChromaDB by ID
        doc_id_list = [a._mapping["doc_id"] for a in assignments]
        source_type_map = {a._mapping["doc_id"]: a._mapping["source_type"] for a in assignments}

        prompts = _fetch_docs_by_ids(doc_id_list, source_type_map)
        if prompts:
            theme_matches.append({
                "theme_label": cluster_label,
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

def start_clustering_single(folder_path: str, k: int, source_type: str | None = None):
    """Start single-folder recluster in a background thread.

    Args:
        folder_path: Concept/folder name to recluster.
        k: Number of clusters.
        source_type: ``"training"`` or ``"output"`` to recluster only one
            source type. ``None`` reclusters both.
    """
    global _clustering_running
    with _clustering_lock:
        if _clustering_running:
            logger.warning("Clustering already running, skipping single-folder recluster")
            return
        _clustering_running = True

    thread = threading.Thread(
        target=_run_clustering_single,
        args=(folder_path, k, source_type),
        daemon=True,
    )
    thread.start()


def _run_clustering_single(folder_path: str, k: int, source_type: str | None = None):
    """Single-folder recluster — runs in a background thread."""
    global _clustering_running
    progress = ClusteringProgress()

    try:
        progress.phase = "intra_folder"
        progress.message = f"Reclustering '{folder_path}' with k={k}..."
        _emit_status(progress)

        generate_intra_folder_clusters(folder_path=folder_path, k=k, force=True, source_type=source_type)

        progress.phase = "complete"
        progress.complete = True
        progress.message = "Reclustering complete."
        _emit_status(progress)

        # Invalidate LLM cache since dataset map changed
        from src.services.cache_service import cache_manager
        cache_manager.invalidate()

    except Exception as e:
        logger.error(f"Single-folder recluster error: {e}", exc_info=True)
        progress.phase = "error"
        progress.complete = True
        progress.message = f"Reclustering failed: {str(e)}"
        _emit_status(progress)

    finally:
        _clustering_running = False


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

        # Generate folder-level summary terms for the two-tier dataset map
        progress.phase = "folder_summaries"
        progress.message = "Generating folder summary terms..."
        _emit_status(progress)
        try:
            all_concepts = _get_all_concepts()
            folder_texts: dict[str, list[str]] = {}
            for concept_info in all_concepts:
                concept_name = concept_info["concept"]
                _, _, documents, _ = _fetch_embeddings_by_concept(concept_name)
                if documents:
                    folder_texts[concept_name] = documents

            summaries = _generate_folder_summaries(folder_texts)
            with get_db() as conn:
                conn.execute(text("DELETE FROM folder_summaries"))
                for folder, summary in summaries.items():
                    conn.execute(
                        text("INSERT INTO folder_summaries (folder_path, summary) "
                             "VALUES (:fp, :s)"),
                        {"fp": folder, "s": summary},
                    )
            progress.message = f"Generated summaries for {len(summaries)} folders."
            _emit_status(progress)
        except Exception as e:
            logger.warning("Folder summary generation failed: %s", e, exc_info=True)

        progress.phase = "complete"
        progress.complete = True
        progress.message = "Clustering complete."
        _emit_status(progress)

        # Invalidate LLM cache since dataset map changed
        from src.services.cache_service import cache_manager
        cache_manager.invalidate()

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
        result = conn.execute(
            text("SELECT COUNT(DISTINCT ca.doc_id) as cnt FROM cluster_assignments ca "
                 "JOIN clusters c ON ca.cluster_id = c.id WHERE c.cluster_type = 'cross_folder'")
        )
        cross_assigned = result.fetchone()._mapping["cnt"]

        # Docs assigned to intra-folder clusters
        result = conn.execute(
            text("SELECT COUNT(DISTINCT ca.doc_id) as cnt FROM cluster_assignments ca "
                 "JOIN clusters c ON ca.cluster_id = c.id WHERE c.cluster_type = 'intra_folder'")
        )
        intra_assigned = result.fetchone()._mapping["cnt"]

        # Last cross-folder run
        result = conn.execute(
            text("SELECT completed_at FROM clustering_runs "
                 "WHERE run_type = 'cross_folder' ORDER BY completed_at DESC LIMIT 1")
        )
        last_run_row = result.fetchone()
        last_cross_run = last_run_row._mapping["completed_at"] if last_run_row else None

        # Total clusters
        result = conn.execute(text("SELECT COUNT(*) as cnt FROM clusters"))
        total_clusters = result.fetchone()._mapping["cnt"]

        # Separate cluster counts by type
        result = conn.execute(
            text("SELECT COUNT(*) as cnt FROM clusters WHERE cluster_type = 'cross_folder'")
        )
        cross_folder_clusters = result.fetchone()._mapping["cnt"]

        result = conn.execute(
            text("SELECT COUNT(*) as cnt FROM clusters WHERE cluster_type = 'intra_folder'")
        )
        intra_folder_clusters = result.fetchone()._mapping["cnt"]

    return {
        "new_since_last_cross_cluster": max(0, total_chromadb_docs - cross_assigned),
        "assigned_to_existing_intra": intra_assigned,
        "last_cross_cluster_run": last_cross_run,
        "total_clusters": total_clusters,
        "cross_folder_clusters": cross_folder_clusters,
        "intra_folder_clusters": intra_folder_clusters,
    }


# ── Folder rename ────────────────────────────────────────────────────────

def rename_concept(old_concept: str, new_concept: str, parent_dirs: list[str]) -> dict:
    """Rename a concept folder on disk and update all DB/ChromaDB references.

    Args:
        old_concept: Current folder name (e.g. "watercolor_landscapes").
        new_concept: New folder name (e.g. "style__watercolor_landscapes").
        parent_dirs: Absolute paths to data directories containing the folder.

    Returns:
        {"ok": True} on success, {"ok": False, "error": str} on failure.
    """
    # 1. Rename physical directories
    renamed_dirs = []
    for parent in parent_dirs:
        old_dir = os.path.join(parent, old_concept)
        new_dir = os.path.join(parent, new_concept)
        if not os.path.isdir(old_dir):
            continue
        try:
            os.rename(old_dir, new_dir)
            renamed_dirs.append((parent, old_dir, new_dir))
        except OSError as e:
            # Roll back any renames already done
            for _, rd_old, rd_new in renamed_dirs:
                try:
                    os.rename(rd_new, rd_old)
                except OSError:
                    pass
            return {"ok": False, "error": f"Failed to rename {old_dir}: {e}"}

    if not renamed_dirs:
        return {"ok": False, "error": f"No directories found for concept '{old_concept}'"}

    # 2. Update SQLite
    try:
        with get_db() as conn:
            _rename_sqlite(conn, old_concept, new_concept, renamed_dirs)
    except Exception as e:
        logger.error(f"SQLite update failed during rename: {e}")
        # Roll back filesystem
        for _, rd_old, rd_new in renamed_dirs:
            try:
                os.rename(rd_new, rd_old)
            except OSError:
                pass
        return {"ok": False, "error": f"Database update failed: {e}"}

    # 3. Update ChromaDB
    try:
        _rename_chromadb(old_concept, new_concept, renamed_dirs)
    except Exception as e:
        logger.warning(f"ChromaDB update failed during rename (filesystem+SQL already committed): {e}")

    # Invalidate LLM cache since folder names changed in dataset overview
    from src.services.cache_service import cache_manager
    cache_manager.invalidate()

    logger.info(f"Renamed concept '{old_concept}' -> '{new_concept}' across {len(renamed_dirs)} dir(s)")
    return {"ok": True}


def _rename_sqlite(conn, old_concept: str, new_concept: str, renamed_dirs: list[tuple]):
    """Update all SQLite tables for a concept rename."""
    conn.execute(
        text("UPDATE clusters SET folder_path = :new WHERE folder_path = :old"),
        {"new": new_concept, "old": old_concept},
    )
    conn.execute(
        text("UPDATE clustering_runs SET folder_path = :new WHERE folder_path = :old"),
        {"new": new_concept, "old": old_concept},
    )

    # folder_summaries (PK = folder_path)
    row = conn.execute(
        text("SELECT summary FROM folder_summaries WHERE folder_path = :old"),
        {"old": old_concept},
    ).fetchone()
    if row:
        conn.execute(
            text("INSERT OR REPLACE INTO folder_summaries (folder_path, summary, updated_at) "
                 "VALUES (:new, :summary, CURRENT_TIMESTAMP)"),
            {"new": new_concept, "summary": row._mapping["summary"]},
        )
        conn.execute(
            text("DELETE FROM folder_summaries WHERE folder_path = :old"),
            {"old": old_concept},
        )

    for parent, old_dir, new_dir in renamed_dirs:
        old_prefix = old_dir + os.sep
        new_prefix = new_dir + os.sep

        # generated_images.file_path
        conn.execute(
            text("UPDATE generated_images SET file_path = :new_prefix || SUBSTR(file_path, :old_len + 1) "
                 "WHERE file_path LIKE :old_like"),
            {"new_prefix": new_prefix, "old_len": len(old_prefix), "old_like": old_prefix + "%"},
        )

        # generated_images.subfolder
        conn.execute(
            text("UPDATE generated_images SET subfolder = :new_sub || SUBSTR(subfolder, :old_len + 1) "
                 "WHERE subfolder LIKE :old_like"),
            {"new_sub": new_concept, "old_len": len(old_concept), "old_like": old_concept + "%"},
        )

        # generation_settings.output_folder
        conn.execute(
            text("UPDATE generation_settings SET output_folder = :new_sub || SUBSTR(output_folder, :old_len + 1) "
                 "WHERE output_folder LIKE :old_like"),
            {"new_sub": new_concept, "old_len": len(old_concept), "old_like": old_concept + "%"},
        )

        # cluster_assignments.doc_id
        conn.execute(
            text("UPDATE cluster_assignments SET doc_id = :new_prefix || SUBSTR(doc_id, :old_len + 1) "
                 "WHERE doc_id LIKE :old_like"),
            {"new_prefix": new_prefix, "old_len": len(old_prefix), "old_like": old_prefix + "%"},
        )

        # thumbnail_cache.file_path
        conn.execute(
            text("UPDATE thumbnail_cache SET file_path = :new_prefix || SUBSTR(file_path, :old_len + 1) "
                 "WHERE file_path LIKE :old_like"),
            {"new_prefix": new_prefix, "old_len": len(old_prefix), "old_like": old_prefix + "%"},
        )

    # settings: cluster_k_intra:<old> -> cluster_k_intra:<new>
    old_key = f"cluster_k_intra:{old_concept}"
    new_key = f"cluster_k_intra:{new_concept}"
    row = conn.execute(
        text("SELECT value FROM settings WHERE key = :old_key"),
        {"old_key": old_key},
    ).fetchone()
    if row:
        conn.execute(
            text("INSERT OR REPLACE INTO settings (key, value, updated_at) "
                 "VALUES (:new_key, :value, CURRENT_TIMESTAMP)"),
            {"new_key": new_key, "value": row._mapping["value"]},
        )
        conn.execute(
            text("DELETE FROM settings WHERE key = :old_key"),
            {"old_key": old_key},
        )


def _rename_chromadb(old_concept: str, new_concept: str, renamed_dirs: list[tuple]):
    """Update ChromaDB: concept metadata + re-key file-path doc IDs."""
    for source_type in ("training", "output"):
        collection = vector_store._get_collection(source_type)
        count = collection.count()
        if count == 0:
            continue

        all_data = collection.get(
            limit=count,
            include=["metadatas", "embeddings", "documents"],
            where={"concept": old_concept},
        )
        if not all_data["ids"]:
            continue

        rekey_indices = []
        metadata_only_indices = []

        for i, doc_id in enumerate(all_data["ids"]):
            needs_rekey = any(
                doc_id.startswith(old_dir + os.sep) or doc_id.startswith(old_dir + "/")
                for _, old_dir, _ in renamed_dirs
            )
            if needs_rekey:
                rekey_indices.append(i)
            else:
                metadata_only_indices.append(i)

        # Update metadata-only docs
        if metadata_only_indices:
            update_ids = [all_data["ids"][i] for i in metadata_only_indices]
            update_metas = []
            for i in metadata_only_indices:
                meta = dict(all_data["metadatas"][i])
                meta["concept"] = new_concept
                update_metas.append(meta)
            for cs in range(0, len(update_ids), 500):
                ce = cs + 500
                collection.update(ids=update_ids[cs:ce], metadatas=update_metas[cs:ce])

        # Re-key docs with file-path IDs
        if rekey_indices:
            old_ids = []
            new_ids = []
            new_docs = []
            new_embeds = []
            new_metas = []

            for i in rekey_indices:
                old_id = all_data["ids"][i]
                new_id = old_id
                for _, old_dir, new_dir in renamed_dirs:
                    for sep in (os.sep, "/"):
                        prefix = old_dir + sep
                        if old_id.startswith(prefix):
                            new_id = new_dir + sep + old_id[len(prefix):]
                            break
                    if new_id != old_id:
                        break

                meta = dict(all_data["metadatas"][i])
                meta["concept"] = new_concept
                old_ids.append(old_id)
                new_ids.append(new_id)
                new_docs.append(all_data["documents"][i])
                new_embeds.append(all_data["embeddings"][i])
                new_metas.append(meta)

            for cs in range(0, len(old_ids), 500):
                ce = cs + 500
                collection.delete(ids=old_ids[cs:ce])
            for cs in range(0, len(new_ids), 500):
                ce = cs + 500
                collection.add(
                    ids=new_ids[cs:ce],
                    documents=new_docs[cs:ce],
                    embeddings=new_embeds[cs:ce],
                    metadatas=new_metas[cs:ce],
                )
