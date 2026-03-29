"""Clustering service package — re-exports all public functions.

All existing imports (``from src.services import clustering_service``) continue
working since Python treats ``clustering_service/`` as a package.
"""

from .orchestration import (
    ClusteringProgress,
    add_status_listener,
    remove_status_listener,
    is_running,
    start_clustering,
    get_current_status,
)

from .core import (
    generate_cross_folder_clusters,
    generate_intra_folder_clusters,
    assign_new_docs_to_clusters,
)

from .dataset_map import (
    get_dataset_map,
    get_dataset_overview,
    get_folder_themes,
)

from .query import get_themed_prompts

from .stats import get_clustering_stats

from .rename import rename_concept

__all__ = [
    "ClusteringProgress",
    "add_status_listener",
    "remove_status_listener",
    "is_running",
    "start_clustering",
    "generate_cross_folder_clusters",
    "generate_intra_folder_clusters",
    "assign_new_docs_to_clusters",
    "get_dataset_map",
    "get_dataset_overview",
    "get_folder_themes",
    "get_themed_prompts",
    "get_clustering_stats",
    "rename_concept",
    "get_current_status",
]
