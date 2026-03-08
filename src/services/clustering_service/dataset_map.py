"""Dataset map and overview endpoints."""

import json
import logging

import numpy as np
from sqlalchemy import text

from src.models import vector_store
from src.models.database import get_db

logger = logging.getLogger(__name__)


def get_dataset_map() -> dict:
    """Build a structured dataset map combining ChromaDB and cluster data.

    Returns:
        Dictionary with ``cross_folder_themes``, ``folders``, and ``stats`` keys.
    """
    counts = vector_store.get_collection_counts()
    total_chromadb_docs = counts.get("training", 0) + counts.get("generated", 0)

    concept_list = vector_store.list_concepts()
    folder_info: dict[tuple[str, str], dict] = {}
    for c in concept_list:
        name = c["concept"]
        key = (name, c["source_type"])
        if key not in folder_info:
            folder_info[key] = {"name": name, "source_type": c["source_type"], "total_prompts": 0}
        folder_info[key]["total_prompts"] += c["count"]

    with get_db() as conn:
        # Cross-folder themes with contributing folders
        result = conn.execute(
            text("SELECT id, label, prompt_count FROM clusters WHERE cluster_type = 'cross_folder' ORDER BY id")
        )
        cross_cluster_rows = result.fetchall()

        cross_folder_themes = []
        for row in cross_cluster_rows:
            r = row._mapping
            cluster_id = r["id"]

            # Find top contributing folders for this cross-folder cluster
            contrib_result = conn.execute(
                text("""
                    SELECT c2.folder_path, c2.source_type, c2.label AS intra_label,
                           COUNT(*) AS shared_count
                    FROM cluster_assignments ca_cross
                    JOIN cluster_assignments ca_intra ON ca_cross.doc_id = ca_intra.doc_id
                    JOIN clusters c2 ON ca_intra.cluster_id = c2.id
                    WHERE ca_cross.cluster_id = :cross_id
                      AND c2.cluster_type = 'intra_folder'
                    GROUP BY c2.folder_path, c2.source_type
                    ORDER BY shared_count DESC
                    LIMIT 5
                """),
                {"cross_id": cluster_id},
            )
            contributing_folders = []
            for cf_row in contrib_result.fetchall():
                cf = cf_row._mapping
                contributing_folders.append({
                    "folder_path": cf["folder_path"],
                    "source_type": cf["source_type"] or "training",
                    "count": cf["shared_count"],
                })

            cross_folder_themes.append({
                "label": r["label"],
                "prompt_count": r["prompt_count"],
                "contributing_folders": contributing_folders,
            })

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

        result = conn.execute(
            text("SELECT COUNT(DISTINCT ca.doc_id) as cnt FROM cluster_assignments ca "
                 "JOIN clusters c ON ca.cluster_id = c.id WHERE c.cluster_type = 'intra_folder'")
        )
        intra_assigned_count = result.fetchone()._mapping["cnt"]

        result = conn.execute(text("SELECT COUNT(*) as cnt FROM clusters"))
        total_clusters = result.fetchone()._mapping["cnt"]

        result = conn.execute(
            text("SELECT COUNT(*) as cnt FROM clusters WHERE cluster_type = 'intra_folder'")
        )
        total_intra = result.fetchone()._mapping["cnt"]

        # Folder summaries
        result = conn.execute(text("SELECT folder_path, summary FROM folder_summaries"))
        summaries = {r._mapping["folder_path"]: r._mapping["summary"] for r in result.fetchall()}

    new_since_last_cross = total_chromadb_docs - cross_assigned_count

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
    """Lightweight dataset overview — no intra-folder cluster details."""
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
        result = conn.execute(
            text("SELECT label, prompt_count FROM clusters "
                 "WHERE cluster_type = 'cross_folder' ORDER BY prompt_count DESC")
        )
        cross_themes = [
            {"label": r._mapping["label"], "prompt_count": r._mapping["prompt_count"]}
            for r in result.fetchall()
        ]

        result = conn.execute(text("SELECT folder_path, summary FROM folder_summaries"))
        summaries = {r._mapping["folder_path"]: r._mapping["summary"] for r in result.fetchall()}

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
    """Get intra-folder cluster themes for a specific folder and source type."""
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
