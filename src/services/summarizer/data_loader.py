"""Load cross-folder cluster and folder data from the database and ChromaDB."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import text

from src.models.database import get_db
from src.models import vector_store

logger = logging.getLogger(__name__)


@dataclass
class CrossFolderInput:
    cluster_id: int
    current_label: str
    contributing_folders: list[dict] = field(default_factory=list)
    sample_prompts: list[str] = field(default_factory=list)


@dataclass
class IntraClusterInfo:
    cluster_id: int
    label: str
    sample_prompts: list[str] = field(default_factory=list)
    difference_prompts: list[str] = field(default_factory=list)


@dataclass
class FolderInput:
    folder_path: str
    current_summary: str
    concept: str
    data_root: str = ""
    sample_prompts: list[str] = field(default_factory=list)
    intra_clusters: list[IntraClusterInfo] = field(default_factory=list)


def load_cross_folder_inputs(
    sample_prompts: int = 10,
    distance_threshold: float = 1.5,
    top_k_folders: int = 5,
) -> list[CrossFolderInput]:
    """Load cross-folder cluster data with contributing folders and sample prompts."""

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

        prompts = _get_sample_prompts_for_cluster(cc["id"], sample_prompts)

        inputs.append(CrossFolderInput(
            cluster_id=cc["id"],
            current_label=cc["label"],
            contributing_folders=contributing_folders,
            sample_prompts=prompts,
        ))

    return inputs


def load_folder_inputs(
    sample_prompts: int = 10,
    intra_sample_prompts: int = 5,
    difference_prompts: int = 3,
    include_intra_clusters: bool = True,
    folder_path: str | None = None,
    source_type: str | None = None,
) -> list[FolderInput]:
    """Load folder data with summaries and sample prompts.

    Args:
        sample_prompts: Number of sample prompts for folder-level summaries.
        intra_sample_prompts: Number of sample prompts per intra-folder cluster.
        include_intra_clusters: Whether to include intra-cluster details.
        folder_path: If set, only load this specific folder.
        source_type: If set, only load this source type.
    """

    with get_db() as conn:
        result = conn.execute(text("SELECT folder_path, summary FROM folder_summaries"))
        summaries = {
            r._mapping["folder_path"]: r._mapping["summary"]
            for r in result.fetchall()
        }

    if folder_path:
        folder_paths = [folder_path]
    else:
        # Always discover folders from clusters table to catch newly added folders
        with get_db() as conn:
            result = conn.execute(
                text("SELECT DISTINCT folder_path FROM clusters WHERE cluster_type = 'intra_folder'")
            )
            cluster_folders = {r._mapping["folder_path"] for r in result.fetchall()}
        # Merge with any folders that have summaries (covers edge cases)
        folder_paths = sorted(cluster_folders | set(summaries.keys()))
        if not folder_paths:
            logger.warning("No folders found for summarization.")
            return []

    # Load intra-folder clusters grouped by (folder_path, source_type)
    with get_db() as conn:
        result = conn.execute(
            text("SELECT id, folder_path, source_type, label, centroid, prompt_count FROM clusters "
                 "WHERE cluster_type = 'intra_folder' AND centroid IS NOT NULL")
        )
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

    # Build list of (folder_path, source_type) entries
    folder_entries: list[tuple[str, str]] = []
    for fp in folder_paths:
        if source_type:
            folder_entries.append((fp, source_type))
        else:
            source_types = sorted({st for (f, st) in intra_by_key if f == fp})
            if not source_types:
                folder_entries.append((fp, "training"))
            else:
                for stype in source_types:
                    folder_entries.append((fp, stype))

    inputs = []
    for fp, st in folder_entries:
        current_summary = summaries.get(fp, "")
        intra = intra_by_key.get((fp, st), [])

        # Compute folder centroid as weighted mean
        if intra:
            weights = np.array([c["prompt_count"] for c in intra], dtype=float)
            centroids = np.array([c["centroid"] for c in intra])
            folder_centroid = np.average(centroids, axis=0, weights=weights)
        else:
            folder_centroid = None

        # Sample prompts nearest to folder centroid
        prompts = _get_sample_prompts_for_folder(fp, intra, folder_centroid, sample_prompts)

        # Build intra-cluster info
        intra_cluster_infos = []
        if include_intra_clusters and intra:
            for cluster in intra:
                cluster_prompts = _get_sample_prompts_for_cluster(cluster["id"], intra_sample_prompts)

                # Gather difference prompts from other clusters
                other_clusters = [c for c in intra if c["id"] != cluster["id"]]
                diff_prompts = _get_difference_prompts(other_clusters, difference_prompts)

                intra_cluster_infos.append(IntraClusterInfo(
                    cluster_id=cluster["id"],
                    label=cluster["label"],
                    sample_prompts=cluster_prompts,
                    difference_prompts=diff_prompts,
                ))

        inputs.append(FolderInput(
            folder_path=fp,
            current_summary=current_summary,
            concept=fp.split("__", 1)[0] if "__" in fp else "",
            sample_prompts=prompts,
            intra_clusters=intra_cluster_infos,
        ))

    return inputs


def _get_difference_prompts(other_clusters: list[dict], k: int) -> list[str]:
    """Draw sample prompts from other intra-folder clusters for differentiation.

    Draws the prompt closest to each cluster's centroid (1 per cluster).
    If k > len(other_clusters), k is capped so each cluster contributes at
    most one prompt and there are no duplicate draws.
    """
    if not other_clusters:
        return []

    k = min(k, len(other_clusters))
    sorted_clusters = sorted(other_clusters, key=lambda c: c["prompt_count"], reverse=True)
    selected = sorted_clusters[:k]
    prompts = []
    for cluster in selected:
        result = _get_sample_prompts_for_cluster(cluster["id"], 1)
        prompts.extend(result)
    return prompts


def _get_sample_prompts_for_cluster(cluster_id: int, k: int) -> list[str]:
    """Get the k nearest prompts to a cluster's centroid."""
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
        return _fetch_prompt_texts(doc_ids[:k], source_map)

    # Fetch embeddings to re-rank
    texts_with_embeddings = _fetch_prompt_texts_with_embeddings(doc_ids, source_map)
    if not texts_with_embeddings:
        return _fetch_prompt_texts(doc_ids[:k], source_map)

    ranked = []
    for text_str, embedding in texts_with_embeddings:
        if embedding is not None and len(embedding) > 0:
            dist = float(np.linalg.norm(folder_centroid - np.array(embedding)))
            ranked.append((text_str, dist))

    ranked.sort(key=lambda x: x[1])
    return [t for t, _ in ranked[:k]]


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
