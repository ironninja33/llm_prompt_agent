"""Load cross-folder cluster and folder data from the database and ChromaDB."""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import text

from src.models.database import get_db
from src.models import vector_store

logger = logging.getLogger(__name__)


@dataclass
class CrossFolderInput:
    cluster_id: int
    tfidf_label: str
    contributing_folders: list[dict] = field(default_factory=list)
    sample_prompts: list[str] = field(default_factory=list)


@dataclass
class IntraClusterInfo:
    cluster_id: int
    label: str
    sample_prompts: list[str] = field(default_factory=list)


@dataclass
class FolderInput:
    folder_path: str
    tfidf_summary: str
    concept: str
    data_root: str = ""
    sample_prompts: list[str] = field(default_factory=list)
    intra_clusters: list[IntraClusterInfo] = field(default_factory=list)


def _resolve_data_root(
    folder_path: str,
    source_type: str,
    training_cluster_id: int | None,
    training_dirs: list[tuple[str, str]],
    output_label: str,
) -> str:
    """Resolve a single (folder, source_type) pair to its data root label."""
    if source_type == "output":
        return output_label

    if training_cluster_id is None:
        return "training"

    # Look up a sample doc_id to match against data directory paths
    with get_db() as conn:
        result = conn.execute(
            text("SELECT doc_id FROM cluster_assignments "
                 "WHERE cluster_id = :cid AND source_type = 'training' LIMIT 1"),
            {"cid": training_cluster_id},
        )
        row = result.fetchone()

    if row:
        doc_id = row._mapping["doc_id"]
        for dir_path, dir_label in training_dirs:
            if doc_id.startswith(dir_path):
                return dir_label

    return "training"


def _load_data_dir_info() -> tuple[list[tuple[str, str]], str]:
    """Load data directories and return (training_dirs, output_label)."""
    with get_db() as conn:
        result = conn.execute(text("SELECT path, dir_type FROM data_directories"))
        data_dirs = [(r._mapping["path"], r._mapping["dir_type"]) for r in result.fetchall()]

    training_dirs = [(p, os.path.basename(p.rstrip("/"))) for p, dt in data_dirs if dt == "training"]
    output_label = os.path.basename(
        next((p.rstrip("/") for p, dt in data_dirs if dt == "output"), "output")
    )
    return training_dirs, output_label


def load_cross_folder_inputs(
    max_clusters: int | None,
    sample_prompts: int,
    distance_threshold: float,
    top_k_folders: int,
) -> list[CrossFolderInput]:
    """Load cross-folder cluster data with contributing folders and sample prompts."""

    # Load cross-folder clusters
    with get_db() as conn:
        result = conn.execute(
            text("SELECT id, label, centroid FROM clusters "
                 "WHERE cluster_type = 'cross_folder' AND centroid IS NOT NULL")
        )
        cross_clusters = []
        for row in result.fetchall():
            r = row._mapping
            try:
                centroid = np.array(json.loads(r["centroid"]))
                cross_clusters.append({
                    "id": r["id"],
                    "label": r["label"],
                    "centroid": centroid,
                })
            except (json.JSONDecodeError, ValueError):
                continue

    if not cross_clusters:
        logger.warning("No cross-folder clusters found.")
        return []

    # Random sample if max_clusters is set
    if max_clusters is not None and max_clusters < len(cross_clusters):
        cross_clusters = random.sample(cross_clusters, max_clusters)

    # Load all intra-folder cluster centroids
    with get_db() as conn:
        result = conn.execute(
            text("SELECT id, folder_path, source_type, label, centroid, prompt_count FROM clusters "
                 "WHERE cluster_type = 'intra_folder' AND centroid IS NOT NULL")
        )
        intra_clusters = []
        for row in result.fetchall():
            r = row._mapping
            try:
                centroid = np.array(json.loads(r["centroid"]))
                intra_clusters.append({
                    "id": r["id"],
                    "folder_path": r["folder_path"],
                    "source_type": r["source_type"] or "training",
                    "label": r["label"],
                    "centroid": centroid,
                    "prompt_count": r["prompt_count"],
                })
            except (json.JSONDecodeError, ValueError):
                continue

    inputs = []
    for cc in cross_clusters:
        # Find top-k intra-folder clusters within distance threshold
        distances = []
        for ic in intra_clusters:
            dist = float(np.linalg.norm(cc["centroid"] - ic["centroid"]))
            if dist <= distance_threshold:
                distances.append((ic, dist))
        distances.sort(key=lambda x: x[1])
        contributing = distances[:top_k_folders]

        contributing_folders = [
            {
                "folder_path": ic["folder_path"],
                "source_type": ic["source_type"],
                "label": ic["label"],
                "distance": dist,
            }
            for ic, dist in contributing
        ]

        # Get sample prompts nearest to centroid
        prompts = _get_sample_prompts_for_cluster(cc["id"], sample_prompts)

        inputs.append(CrossFolderInput(
            cluster_id=cc["id"],
            tfidf_label=cc["label"],
            contributing_folders=contributing_folders,
            sample_prompts=prompts,
        ))

    return inputs


def load_folder_inputs(
    max_folders: int | None,
    sample_prompts: int,
    include_intra_clusters: bool = True,
) -> list[FolderInput]:
    """Load folder data with TF-IDF summaries and sample prompts.

    Folders that exist in multiple data roots (e.g. training and output)
    are split into separate entries.
    """

    # Load folder summaries
    with get_db() as conn:
        result = conn.execute(text("SELECT folder_path, summary FROM folder_summaries"))
        summaries = {
            r._mapping["folder_path"]: r._mapping["summary"]
            for r in result.fetchall()
        }

    if not summaries:
        logger.warning("No folder summaries found.")
        return []

    folder_paths = list(summaries.keys())

    # Random sample if max_folders is set
    if max_folders is not None and max_folders < len(folder_paths):
        folder_paths = random.sample(folder_paths, max_folders)

    # Load intra-folder clusters grouped by (folder_path, source_type)
    with get_db() as conn:
        result = conn.execute(
            text("SELECT id, folder_path, source_type, label, centroid, prompt_count FROM clusters "
                 "WHERE cluster_type = 'intra_folder' AND centroid IS NOT NULL")
        )
        # Key: (folder_path, source_type) -> list of cluster dicts
        intra_by_key: dict[tuple[str, str], list[dict]] = {}
        for row in result.fetchall():
            r = row._mapping
            folder = r["folder_path"]
            stype = r["source_type"] or "training"
            key = (folder, stype)
            if key not in intra_by_key:
                intra_by_key[key] = []
            try:
                centroid = np.array(json.loads(r["centroid"]))
                intra_by_key[key].append({
                    "id": r["id"],
                    "label": r["label"] or "",
                    "centroid": centroid,
                    "prompt_count": r["prompt_count"] or 1,
                    "source_type": stype,
                })
            except (json.JSONDecodeError, ValueError):
                continue

    # Build list of (folder_path, source_type) entries to process
    folder_entries: list[tuple[str, str]] = []
    for folder_path in folder_paths:
        source_types = sorted({st for (fp, st) in intra_by_key if fp == folder_path})
        if not source_types:
            folder_entries.append((folder_path, "training"))
        else:
            for stype in source_types:
                folder_entries.append((folder_path, stype))

    training_dirs, output_label = _load_data_dir_info()

    inputs = []
    for folder_path, source_type in folder_entries:
        tfidf_summary = summaries.get(folder_path, "")
        intra = intra_by_key.get((folder_path, source_type), [])

        # Compute folder centroid as weighted mean of intra-folder cluster centroids
        if intra:
            weights = np.array([c["prompt_count"] for c in intra], dtype=float)
            centroids = np.array([c["centroid"] for c in intra])
            folder_centroid = np.average(centroids, axis=0, weights=weights)
        else:
            folder_centroid = None

        # Resolve data root
        training_cid = next((c["id"] for c in intra if c["source_type"] == "training"), None)
        data_root = _resolve_data_root(
            folder_path, source_type, training_cid, training_dirs, output_label,
        )

        # Sample prompts nearest to folder centroid
        prompts = _get_sample_prompts_for_folder(
            folder_path, intra, folder_centroid, sample_prompts
        )

        # Build intra-cluster info if requested
        intra_cluster_infos = []
        if include_intra_clusters and intra:
            for cluster in intra:
                cluster_prompts = _get_sample_prompts_for_cluster(
                    cluster["id"], sample_prompts
                )
                intra_cluster_infos.append(IntraClusterInfo(
                    cluster_id=cluster["id"],
                    label=cluster["label"],
                    sample_prompts=cluster_prompts,
                ))

        inputs.append(FolderInput(
            folder_path=folder_path,
            tfidf_summary=tfidf_summary,
            concept=folder_path.split("__", 1)[0] if "__" in folder_path else "",
            data_root=data_root,
            sample_prompts=prompts,
            intra_clusters=intra_cluster_infos,
        ))

    return inputs


def _get_sample_prompts_for_cluster(cluster_id: int, k: int) -> list[str]:
    """Get the k nearest prompts to a cluster's centroid via cluster_assignments."""
    with get_db() as conn:
        result = conn.execute(
            text("SELECT doc_id, source_type FROM cluster_assignments "
                 "WHERE cluster_id = :cluster_id ORDER BY distance ASC LIMIT :limit"),
            {"cluster_id": cluster_id, "limit": k},
        )
        assignments = result.fetchall()

    if not assignments:
        return []

    doc_ids = list(dict.fromkeys(a._mapping["doc_id"] for a in assignments))
    source_map = {a._mapping["doc_id"]: a._mapping["source_type"] for a in assignments}
    return _fetch_prompt_texts(doc_ids, source_map)


def _get_sample_prompts_for_folder(
    folder_path: str,
    intra_clusters: list[dict],
    folder_centroid: np.ndarray | None,
    k: int,
) -> list[str]:
    """Get sample prompts for a folder, re-ranked by distance to folder centroid."""
    if not intra_clusters:
        return []

    cluster_ids = [c["id"] for c in intra_clusters]

    # Load ~3x candidates from cluster_assignments
    candidate_limit = k * 3
    with get_db() as conn:
        placeholders = ",".join(f":id{i}" for i in range(len(cluster_ids)))
        params = {f"id{i}": cid for i, cid in enumerate(cluster_ids)}
        params["limit"] = candidate_limit
        result = conn.execute(
            text(f"SELECT doc_id, source_type FROM cluster_assignments "
                 f"WHERE cluster_id IN ({placeholders}) "
                 f"ORDER BY distance ASC LIMIT :limit"),
            params,
        )
        assignments = result.fetchall()

    if not assignments:
        return []

    doc_ids = list(dict.fromkeys(a._mapping["doc_id"] for a in assignments))
    source_map = {a._mapping["doc_id"]: a._mapping["source_type"] for a in assignments}

    if folder_centroid is None:
        # No centroid to re-rank, just take the nearest k
        return _fetch_prompt_texts(doc_ids[:k], source_map)

    # Fetch embeddings to re-rank by folder centroid distance
    texts_with_embeddings = _fetch_prompt_texts_with_embeddings(doc_ids, source_map)
    if not texts_with_embeddings:
        return _fetch_prompt_texts(doc_ids[:k], source_map)

    # Re-rank by distance to folder centroid
    ranked = []
    for text_str, embedding in texts_with_embeddings:
        if embedding is not None and len(embedding) > 0:
            dist = float(np.linalg.norm(folder_centroid - np.array(embedding)))
            ranked.append((text_str, dist))

    ranked.sort(key=lambda x: x[1])
    return [t for t, _ in ranked[:k]]


def get_all_prompts_for_cluster(cluster_id: int) -> list[str]:
    """Get ALL prompt texts assigned to a cross-folder cluster (no LIMIT)."""
    with get_db() as conn:
        result = conn.execute(
            text("SELECT doc_id, source_type FROM cluster_assignments "
                 "WHERE cluster_id = :cluster_id ORDER BY distance ASC"),
            {"cluster_id": cluster_id},
        )
        assignments = result.fetchall()

    if not assignments:
        return []

    doc_ids = list(dict.fromkeys(a._mapping["doc_id"] for a in assignments))
    source_map = {a._mapping["doc_id"]: a._mapping["source_type"] for a in assignments}
    return _fetch_prompt_texts(doc_ids, source_map)


def get_all_prompts_for_folder(folder_path: str, source_type: str | None = None) -> list[str]:
    """Get ALL prompt texts for a folder's intra-folder clusters."""
    # Find all intra_folder cluster IDs for this folder_path
    with get_db() as conn:
        if source_type:
            result = conn.execute(
                text("SELECT id FROM clusters "
                     "WHERE cluster_type = 'intra_folder' "
                     "AND folder_path = :folder_path "
                     "AND source_type = :source_type"),
                {"folder_path": folder_path, "source_type": source_type},
            )
        else:
            result = conn.execute(
                text("SELECT id FROM clusters "
                     "WHERE cluster_type = 'intra_folder' "
                     "AND folder_path = :folder_path"),
                {"folder_path": folder_path},
            )
        cluster_ids = [r._mapping["id"] for r in result.fetchall()]

    if not cluster_ids:
        return []

    # Query all cluster_assignments for those cluster IDs
    with get_db() as conn:
        placeholders = ",".join(f":id{i}" for i in range(len(cluster_ids)))
        params = {f"id{i}": cid for i, cid in enumerate(cluster_ids)}
        result = conn.execute(
            text(f"SELECT doc_id, source_type FROM cluster_assignments "
                 f"WHERE cluster_id IN ({placeholders}) "
                 f"ORDER BY distance ASC"),
            params,
        )
        assignments = result.fetchall()

    if not assignments:
        return []

    doc_ids = list(dict.fromkeys(a._mapping["doc_id"] for a in assignments))
    source_map = {a._mapping["doc_id"]: a._mapping["source_type"] for a in assignments}
    return _fetch_prompt_texts(doc_ids, source_map)


def _fetch_prompt_texts(doc_ids: list[str], source_map: dict[str, str]) -> list[str]:
    """Fetch prompt text strings from ChromaDB by doc IDs."""
    training_ids = [did for did in doc_ids if source_map.get(did, "training") == "training"]
    output_ids = [did for did in doc_ids if source_map.get(did, "training") == "output"]

    texts = []
    for ids_batch, collection in [
        (training_ids, vector_store._training_collection),
        (output_ids, vector_store._generated_collection),
    ]:
        if not ids_batch or collection is None:
            continue
        try:
            result = collection.get(ids=ids_batch, include=["documents"])
            for doc in result["documents"]:
                if doc:
                    texts.append(doc)
        except Exception as e:
            logger.error(f"Error fetching docs: {e}")

    return texts


def _fetch_prompt_texts_with_embeddings(
    doc_ids: list[str], source_map: dict[str, str]
) -> list[tuple[str, list[float] | None]]:
    """Fetch prompt texts and embeddings from ChromaDB."""
    training_ids = [did for did in doc_ids if source_map.get(did, "training") == "training"]
    output_ids = [did for did in doc_ids if source_map.get(did, "training") == "output"]

    results = []
    for ids_batch, collection in [
        (training_ids, vector_store._training_collection),
        (output_ids, vector_store._generated_collection),
    ]:
        if not ids_batch or collection is None:
            continue
        try:
            result = collection.get(ids=ids_batch, include=["documents", "embeddings"])
            for i in range(len(result["ids"])):
                doc = result["documents"][i] if result["documents"] else None
                emb = result["embeddings"][i] if result["embeddings"] else None
                if doc:
                    results.append((doc, emb))
        except Exception as e:
            logger.error(f"Error fetching docs with embeddings: {e}")

    return results
