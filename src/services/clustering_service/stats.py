"""Clustering statistics."""

from sqlalchemy import text

from src.models import vector_store
from src.models.database import get_db


def get_clustering_stats() -> dict:
    """Return clustering statistics for header display."""
    counts = vector_store.get_collection_counts()
    total_chromadb_docs = counts.get("training", 0) + counts.get("generated", 0)

    with get_db() as conn:
        result = conn.execute(
            text("SELECT COUNT(DISTINCT ca.doc_id) as cnt FROM cluster_assignments ca "
                 "JOIN clusters c ON ca.cluster_id = c.id WHERE c.cluster_type = 'cross_folder'")
        )
        cross_assigned = result.fetchone()._mapping["cnt"]

        result = conn.execute(
            text("SELECT COUNT(DISTINCT ca.doc_id) as cnt FROM cluster_assignments ca "
                 "JOIN clusters c ON ca.cluster_id = c.id WHERE c.cluster_type = 'intra_folder'")
        )
        intra_assigned = result.fetchone()._mapping["cnt"]

        result = conn.execute(
            text("SELECT completed_at FROM clustering_runs "
                 "WHERE run_type = 'cross_folder' ORDER BY completed_at DESC LIMIT 1")
        )
        last_run_row = result.fetchone()
        last_cross_run = last_run_row._mapping["completed_at"] if last_run_row else None

        result = conn.execute(text("SELECT COUNT(*) as cnt FROM clusters"))
        total_clusters = result.fetchone()._mapping["cnt"]

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
