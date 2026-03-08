"""Themed prompt retrieval — unified query across multiple retrieval strategies."""

import json
import logging

import numpy as np
from sqlalchemy import text

from src.models import vector_store, settings
from src.models.database import get_db

from .data import _fetch_docs_by_ids

logger = logging.getLogger(__name__)


def get_themed_prompts(
    query_embedding: list[float],
    k_similar: int | None = None,
    k_intra: int | None = None,
    k_cross: int | None = None,
    k_random: int = 0,
    k_opposite: int = 0,
    source_type: str | None = None,
) -> dict:
    """Unified query returning prompts from multiple retrieval strategies."""
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
    """Find the closest cluster centroids and return their assigned prompts."""
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

    distances = [(c, float(np.linalg.norm(query_emb - c["centroid"]))) for c in clusters]
    distances.sort(key=lambda x: x[1])
    nearest = distances[:top_clusters]

    theme_matches: list[dict] = []
    for cluster_info, _ in nearest:
        cluster_id = cluster_info["id"]
        cluster_label = cluster_info["label"]

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

        doc_id_list = [a._mapping["doc_id"] for a in assignments]
        source_type_map = {a._mapping["doc_id"]: a._mapping["source_type"] for a in assignments}

        prompts = _fetch_docs_by_ids(doc_id_list, source_type_map)
        if prompts:
            theme_matches.append({
                "theme_label": cluster_label,
                "prompts": prompts,
            })

    return theme_matches
