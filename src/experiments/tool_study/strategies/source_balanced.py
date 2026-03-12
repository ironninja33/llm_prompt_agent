"""Source-balanced strategy — force a minimum training/output ratio."""

from __future__ import annotations

from src.models import vector_store
from .baseline import SearchResult


class SourceBalancedStrategy:
    """Search each collection separately and interleave with a configurable ratio.

    This ensures a minimum percentage of training data in results, preventing
    output data from dominating search results.
    """

    name = "source_balanced"

    def __init__(self, training_ratio: float = 0.5, **kwargs):
        self.training_ratio = max(0.0, min(1.0, training_ratio))

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        **kwargs,
    ) -> list[SearchResult]:
        k_training = max(1, round(k * self.training_ratio))
        k_output = k - k_training

        training_results = vector_store.search_similar(
            query_embedding, k=k_training, source_type="training"
        )
        output_results = vector_store.search_similar(
            query_embedding, k=k_output, source_type="output"
        )

        # Interleave by distance while maintaining ratio
        combined = []
        for r in training_results:
            combined.append(("training", r))
        for r in output_results:
            combined.append(("output", r))

        # Sort by distance
        combined.sort(key=lambda x: x[1]["distance"])

        return [
            SearchResult(
                rank=i + 1,
                distance=r["distance"],
                source=source,
                concept=r["metadata"].get("concept", ""),
                text=r["document"],
                doc_id=r["id"],
            )
            for i, (source, r) in enumerate(combined[:k])
        ]
