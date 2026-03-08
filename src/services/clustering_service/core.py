"""KMeans clustering core — cross-folder, intra-folder, and incremental assignment."""

import json
import logging
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy import text

from src.models import settings
from src.models.database import get_db

from .data import _fetch_all_embeddings, _fetch_embeddings_by_concept, _get_all_concepts
from .orchestration import ClusteringProgress, _emit_status

logger = logging.getLogger(__name__)


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
    """Compute default cluster count based on folder size and type."""
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


# ---------------------------------------------------------------------------
# Cross-folder clustering
# ---------------------------------------------------------------------------

def generate_cross_folder_clusters(k: int | None = None):
    """Run KMeans clustering across all prompts in both ChromaDB collections."""
    progress = ClusteringProgress(phase="cross_folder", message="Starting cross-folder clustering...")
    _emit_status(progress)

    if k is None:
        k_str = settings.get_setting("cluster_k_cross")
        k = int(k_str) if k_str else 15

    progress.message = "Fetching all embeddings from ChromaDB..."
    _emit_status(progress)

    doc_ids, embeddings_list, documents, metadatas = _fetch_all_embeddings()
    n_samples = len(doc_ids)

    if n_samples <= 1:
        progress.message = f"Only {n_samples} document(s) found — skipping cross-folder clustering."
        _emit_status(progress)
        return

    effective_k = min(k, n_samples)

    progress.message = f"Clustering {n_samples} documents into {effective_k} clusters..."
    progress.total = n_samples
    _emit_status(progress)

    embeddings_np = np.array(embeddings_list)
    kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings_np)
    centroids = kmeans.cluster_centers_

    cluster_data: list[dict] = []
    for cluster_idx in range(effective_k):
        member_mask = labels == cluster_idx
        cluster_data.append({
            "cluster_index": cluster_idx,
            "label": "pending",
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

    progress.message = "Saving cross-folder clusters to database..."
    _emit_status(progress)

    started_at = datetime.utcnow().isoformat()

    with get_db() as conn:
        conn.execute(
            text("DELETE FROM cluster_assignments WHERE cluster_id IN "
                 "(SELECT id FROM clusters WHERE cluster_type = 'cross_folder')")
        )
        conn.execute(text("DELETE FROM clusters WHERE cluster_type = 'cross_folder'"))

        cluster_id_map: dict[int, int] = {}
        for cd in cluster_data:
            result = conn.execute(
                text("INSERT INTO clusters (cluster_type, folder_path, cluster_index, label, centroid, prompt_count) "
                     "VALUES (:cluster_type, :folder_path, :cluster_index, :label, :centroid, :prompt_count)"),
                {"cluster_type": "cross_folder", "folder_path": None,
                 "cluster_index": cd["cluster_index"], "label": cd["label"],
                 "centroid": cd["centroid"], "prompt_count": cd["prompt_count"]},
            )
            cluster_id_map[cd["cluster_index"]] = result.lastrowid

        for ad in assignment_data:
            db_cluster_id = cluster_id_map[ad["cluster_index"]]
            conn.execute(
                text("INSERT INTO cluster_assignments (doc_id, source_type, cluster_id, distance) "
                     "VALUES (:doc_id, :source_type, :cluster_id, :distance)"),
                {"doc_id": ad["doc_id"], "source_type": ad["source_type"],
                 "cluster_id": db_cluster_id, "distance": ad["distance"]},
            )

        conn.execute(
            text("INSERT INTO clustering_runs (run_type, folder_path, total_prompts, num_clusters, started_at, completed_at) "
                 "VALUES (:run_type, :folder_path, :total_prompts, :num_clusters, :started_at, CURRENT_TIMESTAMP)"),
            {"run_type": "cross_folder", "folder_path": None,
             "total_prompts": n_samples, "num_clusters": effective_k, "started_at": started_at},
        )

    progress.message = f"Cross-folder clustering complete: {effective_k} clusters from {n_samples} documents."
    progress.current = n_samples
    _emit_status(progress)


# ---------------------------------------------------------------------------
# Intra-folder clustering
# ---------------------------------------------------------------------------

def _delete_intra_clusters(folder_path: str, source_type: str):
    """Delete any existing intra_folder clusters and their assignments for a (folder, source_type) pair."""
    with get_db() as conn:
        conn.execute(
            text("DELETE FROM cluster_assignments WHERE cluster_id IN "
                 "(SELECT id FROM clusters WHERE cluster_type = 'intra_folder' "
                 "AND folder_path = :fp AND source_type = :st)"),
            {"fp": folder_path, "st": source_type},
        )
        conn.execute(
            text("DELETE FROM clusters WHERE cluster_type = 'intra_folder' "
                 "AND folder_path = :fp AND source_type = :st"),
            {"fp": folder_path, "st": source_type},
        )


def generate_intra_folder_clusters(
    folder_path: str | None = None,
    k: int | None = None,
    force: bool = False,
    source_type: str | None = None,
):
    """Run KMeans clustering within individual folders/concepts, split by source_type."""
    progress = ClusteringProgress(phase="intra_folder", message="Starting intra-folder clustering...")
    _emit_status(progress)

    k_explicit = k is not None

    min_size_str = settings.get_setting("cluster_min_folder_size")
    min_folder_size = int(min_size_str) if min_size_str else 20

    if folder_path and source_type:
        entries_to_cluster = [{"concept": folder_path, "source_type": source_type}]
    elif folder_path:
        entries_to_cluster = [
            {"concept": folder_path, "source_type": "training"},
            {"concept": folder_path, "source_type": "output"},
        ]
    else:
        all_concepts = _get_all_concepts()
        entries_to_cluster = [
            {"concept": c["concept"], "source_type": c["source_type"]}
            for c in all_concepts
        ]

    progress.total = len(entries_to_cluster)
    progress.message = f"Processing {len(entries_to_cluster)} concept/source pair(s)..."
    _emit_status(progress)

    for idx, entry in enumerate(entries_to_cluster):
        concept_name = entry["concept"]
        entry_source = entry["source_type"]
        progress.current = idx + 1
        progress.message = f"Clustering '{concept_name}' ({entry_source}) ({idx + 1}/{len(entries_to_cluster)})..."
        _emit_status(progress)

        doc_ids, embeddings_list, documents, metadatas = _fetch_embeddings_by_concept(
            concept_name, source_type=entry_source,
        )
        n_samples = len(doc_ids)

        if n_samples < min_folder_size and not force:
            logger.info(
                f"Skipping '{concept_name}' ({entry_source}): {n_samples} docs < min_folder_size {min_folder_size}"
            )
            _delete_intra_clusters(concept_name, entry_source)
            continue

        if n_samples <= 1:
            logger.info(f"Skipping '{concept_name}' ({entry_source}): only {n_samples} document(s)")
            _delete_intra_clusters(concept_name, entry_source)
            continue

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

        if not k_explicit:
            per_folder_k_str = settings.get_setting(f"cluster_k_intra:{concept_name}:{entry_source}")
            if per_folder_k_str:
                effective_base_k = int(per_folder_k_str)
            else:
                effective_base_k = _compute_adaptive_k(n_samples, entry_source)
        else:
            effective_base_k = k

        effective_k = min(effective_base_k, n_samples)
        embeddings_np = np.array(embeddings_list)
        kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(embeddings_np)
        centroids = kmeans.cluster_centers_

        cluster_data: list[dict] = []
        for cluster_idx in range(effective_k):
            member_mask = labels == cluster_idx
            cluster_data.append({
                "cluster_index": cluster_idx,
                "label": "pending",
                "centroid": json.dumps(centroids[cluster_idx].tolist()),
                "prompt_count": int(member_mask.sum()),
            })

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

        started_at = datetime.utcnow().isoformat()

        with get_db() as conn:
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

            for ad in assignment_data:
                db_cluster_id = cluster_id_map[ad["cluster_index"]]
                conn.execute(
                    text("INSERT INTO cluster_assignments (doc_id, source_type, cluster_id, distance) "
                         "VALUES (:doc_id, :source_type, :cluster_id, :distance)"),
                    {"doc_id": ad["doc_id"], "source_type": ad["source_type"],
                     "cluster_id": db_cluster_id, "distance": ad["distance"]},
                )

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
    """Assign newly ingested documents to existing clusters by nearest centroid."""
    if not doc_ids:
        return

    cross_clusters: list[dict] = []
    intra_clusters_by_folder: dict[str, list[dict]] = {}

    with get_db() as conn:
        result = conn.execute(
            text("SELECT id, centroid FROM clusters WHERE cluster_type = 'cross_folder' AND centroid IS NOT NULL")
        )
        for row in result.fetchall():
            r = row._mapping
            cross_clusters.append({
                "id": r["id"],
                "centroid": np.array(json.loads(r["centroid"])),
            })

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
                    conn.execute(
                        text("UPDATE clusters SET prompt_count = prompt_count + 1 WHERE id = :id"),
                        {"id": best_cross_id},
                    )

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
