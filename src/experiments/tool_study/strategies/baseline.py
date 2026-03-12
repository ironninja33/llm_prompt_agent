"""Baseline strategy — current search_similar behavior for comparison."""

from __future__ import annotations

from dataclasses import dataclass

from src.models import vector_store


@dataclass
class SearchResult:
    """A single search result with metadata."""
    rank: int
    distance: float
    source: str
    concept: str
    text: str
    doc_id: str


class BaselineStrategy:
    """Wraps the current vector_store.search_similar as-is."""

    name = "baseline"

    def __init__(self, **kwargs):
        pass

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        **kwargs,
    ) -> list[SearchResult]:
        """Run the current search_similar with no modifications."""
        results = vector_store.search_similar(
            query_embedding, k=k, source_type=None
        )

        return [
            SearchResult(
                rank=i + 1,
                distance=r["distance"],
                source=r["metadata"].get("dir_type", ""),
                concept=r["metadata"].get("concept", ""),
                text=r["document"],
                doc_id=r["id"],
            )
            for i, r in enumerate(results)
        ]
