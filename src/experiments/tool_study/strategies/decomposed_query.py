"""Decomposed query strategy — per-concept retrieval for multi-concept queries."""

from __future__ import annotations

import re

from src.models import vector_store
from .baseline import SearchResult

# Suffixes commonly appended to concept display names that don't carry
# semantic weight for matching purposes (e.g. "aletta_ocean_20", "aletta_ocean_recent").
_STRIP_SUFFIXES = re.compile(r"[\s_](?:\d+|recent|old|new|alt|v\d+)$", re.IGNORECASE)

# Minimum character length for a display-name token to be considered a
# meaningful match signal.  Avoids false positives on tiny tokens like
# "dim" matching "dimly" or "cum" matching "document".
_MIN_TOKEN_LEN = 5


class DecomposedQueryStrategy:
    """Parse query to identify known concepts, search per-concept, then merge.

    Most promising for multi-concept queries like "[character] [concept1] [concept2]"
    where a single embedding search would blend the concepts and miss individual matches.
    """

    name = "decomposed"

    def __init__(self, k_per_concept: int = 5, **kwargs):
        self.k_per_concept = k_per_concept
        self._concept_cache: list[tuple[str, str]] | None = None  # (full_name, canonical)

    def _load_concepts(self) -> list[tuple[str, str]]:
        """Load concept names and build canonical match forms.

        Returns list of (full_concept_name, canonical_display) tuples.
        Canonical display has underscores → spaces, trailing numeric/descriptor
        suffixes stripped, and is lowercased.
        """
        if self._concept_cache is None:
            concepts = vector_store.list_concepts()
            seen = set()
            cache = []
            for c in concepts:
                name = c["concept"]
                if name in seen:
                    continue
                seen.add(name)

                parts = name.split("__", 1)
                display = parts[1] if len(parts) > 1 else parts[0]
                canonical = display.replace("_", " ").lower().strip()
                # Strip trailing numeric/descriptor suffixes:
                # "aletta ocean 20" → "aletta ocean"
                canonical = _STRIP_SUFFIXES.sub("", canonical).strip()
                cache.append((name, canonical))
            self._concept_cache = cache
        return self._concept_cache

    def _match_concepts(self, query: str) -> list[str]:
        """Find concept names that appear in the query.

        Uses two matching tiers:
        1. Exact substring — canonical display name found verbatim in query.
        2. Token overlap — for multi-word display names, all significant tokens
           (length >= _MIN_TOKEN_LEN) appear in the query.

        Single-word display names shorter than _MIN_TOKEN_LEN are skipped to
        avoid false positives ("dim" → "dimly lit", "cum" → "document").
        """
        concepts = self._load_concepts()
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        matched = []

        for full_name, canonical in concepts:
            if not canonical:
                continue

            # Tier 1: exact substring match
            if canonical in query_lower:
                matched.append(full_name)
                continue

            # Tier 2: token-overlap match for multi-word names
            canon_tokens = canonical.split()
            if len(canon_tokens) < 2:
                # Single-word: only match if long enough and is a whole word
                if len(canonical) >= _MIN_TOKEN_LEN and canonical in query_tokens:
                    matched.append(full_name)
                continue

            # Multi-word: require all significant tokens present
            significant = [t for t in canon_tokens if len(t) >= _MIN_TOKEN_LEN]
            if significant and all(t in query_tokens for t in significant):
                matched.append(full_name)

        return matched

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        **kwargs,
    ) -> list[SearchResult]:
        matched_concepts = self._match_concepts(kwargs.get("query_text", ""))

        if not matched_concepts:
            # No concepts detected — fall back to standard search
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

        # Per-concept focused search
        all_results: dict[str, dict] = {}  # doc_id → result dict

        for concept_name in matched_concepts:
            concept_results = vector_store.search_similar(
                query_embedding,
                k=self.k_per_concept,
                concept=concept_name,
            )
            for r in concept_results:
                doc_id = r["id"]
                if doc_id not in all_results or r["distance"] < all_results[doc_id]["distance"]:
                    all_results[doc_id] = r

        # Also do a general search for non-concept aspects
        general_results = vector_store.search_similar(query_embedding, k=k)
        for r in general_results:
            doc_id = r["id"]
            if doc_id not in all_results or r["distance"] < all_results[doc_id]["distance"]:
                all_results[doc_id] = r

        # Deduplicate and rank by minimum distance
        sorted_results = sorted(all_results.values(), key=lambda r: r["distance"])

        return [
            SearchResult(
                rank=i + 1,
                distance=r["distance"],
                source=r["metadata"].get("dir_type", ""),
                concept=r["metadata"].get("concept", ""),
                text=r["document"],
                doc_id=r["id"],
            )
            for i, r in enumerate(sorted_results[:k])
        ]
