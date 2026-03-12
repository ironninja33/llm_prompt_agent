"""Dedup-aware strategy — filter results similar to recently generated prompts."""

from __future__ import annotations

import numpy as np

from src.models import vector_store
from .baseline import SearchResult


class DedupAwareStrategy:
    """Over-fetch results, then filter out those too similar to an exclusion list.

    Simulates filtering out the agent's own recently generated prompts during
    refinement queries.
    """

    name = "dedup_aware"

    def __init__(self, similarity_threshold: float = 0.85, overfetch_factor: int = 3, **kwargs):
        self.similarity_threshold = similarity_threshold
        self.overfetch_factor = overfetch_factor

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        exclusion_embeddings: list[list[float]] | None = None,
        **kwargs,
    ) -> list[SearchResult]:
        if not exclusion_embeddings:
            # No exclusions — behave like baseline
            results = vector_store.search_similar(query_embedding, k=k)
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

        # Over-fetch to have room after filtering
        fetch_k = k * self.overfetch_factor
        results = vector_store.search_similar(query_embedding, k=fetch_k)

        # Pre-compute exclusion array
        excl_matrix = np.array(exclusion_embeddings)  # shape: (n_excl, dim)
        excl_norms = np.linalg.norm(excl_matrix, axis=1, keepdims=True)
        excl_normalized = excl_matrix / np.where(excl_norms > 0, excl_norms, 1.0)

        filtered = []
        for r in results:
            # ChromaDB doesn't return embeddings in query results,
            # so we approximate: if the result text is very similar to
            # exclusion texts by embedding, skip it.
            # We need to get the embedding for this result.
            # For efficiency, we'll use the document text distance as a proxy.
            # In production, we'd store/fetch embeddings. For the experiment,
            # we check if the result has an embedding we can compare.
            filtered.append(r)

        # Since ChromaDB doesn't return embeddings in query results,
        # we use a text-based heuristic: get embeddings for filtered results
        # This is the expensive path, only used when exclusions are provided
        if filtered:
            # Get result embeddings from ChromaDB by ID
            result_embeddings = _fetch_embeddings_by_ids(
                [r["id"] for r in filtered],
                [r["metadata"].get("dir_type", "training") for r in filtered],
            )

            truly_filtered = []
            for r, emb in zip(filtered, result_embeddings):
                if emb is None:
                    truly_filtered.append(r)
                    continue

                result_vec = np.array(emb)
                result_norm = np.linalg.norm(result_vec)
                if result_norm == 0:
                    truly_filtered.append(r)
                    continue

                result_normalized = result_vec / result_norm
                # Cosine similarities with all exclusion embeddings
                similarities = excl_normalized @ result_normalized
                max_sim = float(np.max(similarities))

                if max_sim < self.similarity_threshold:
                    truly_filtered.append(r)

            filtered = truly_filtered

        return [
            SearchResult(
                rank=i + 1,
                distance=r["distance"],
                source=r["metadata"].get("dir_type", ""),
                concept=r["metadata"].get("concept", ""),
                text=r["document"],
                doc_id=r["id"],
            )
            for i, r in enumerate(filtered[:k])
        ]


def _fetch_embeddings_by_ids(
    doc_ids: list[str],
    source_types: list[str],
) -> list[list[float] | None]:
    """Fetch embeddings from ChromaDB by document ID."""
    embeddings = []
    for doc_id, source_type in zip(doc_ids, source_types):
        try:
            st = "training" if source_type == "training" else "output"
            collection = vector_store._get_collection(st)
            result = collection.get(ids=[doc_id], include=["embeddings"])
            if result["embeddings"] and len(result["embeddings"]) > 0:
                embeddings.append(result["embeddings"][0])
            else:
                embeddings.append(None)
        except Exception:
            embeddings.append(None)
    return embeddings
