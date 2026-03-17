"""Cluster-based diverse prompt retrieval tool.

Adapted from the cluster_diverse experiment strategy for production use.
Strategy code is copied (not imported) to avoid production→experiments dependency.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import text

from src.models import vector_store
from src.models.database import get_db
from src.services import embedding_service

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.92


@dataclass
class _SearchResult:
    """A single search result with metadata."""
    rank: int
    distance: float
    source: str
    concept: str
    text: str
    doc_id: str


@dataclass
class _AcceptState:
    """Mutable state shared across the fetch and redistribution loops."""
    seen_ids: set[str] = field(default_factory=set)
    accepted_embeddings: list[np.ndarray] = field(default_factory=list)
    output_count: int = 0
    max_output: int = 0
    results: list[_SearchResult] = field(default_factory=list)

    def _is_similar_to_accepted(self, embedding: np.ndarray) -> bool:
        if not self.accepted_embeddings:
            return False
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return False
        normed = embedding / norm
        return any(float(np.dot(normed, ae)) >= SIMILARITY_THRESHOLD
                   for ae in self.accepted_embeddings)

    def _track_embedding(self, embedding: np.ndarray) -> None:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            self.accepted_embeddings.append(embedding / norm)

    def try_accept(self, r: dict) -> bool:
        """Try to accept a candidate result. Returns True if accepted."""
        if r["id"] in self.seen_ids:
            return False

        emb = r.get("embedding")
        if emb is not None and self._is_similar_to_accepted(np.array(emb)):
            return False

        source = r["metadata"].get("dir_type", "")
        is_output = source != "training"
        if is_output and self.output_count >= self.max_output:
            return False

        self.seen_ids.add(r["id"])
        self.results.append(_SearchResult(
            rank=0,
            distance=r["distance"],
            source=source,
            concept=r["metadata"].get("concept", ""),
            text=r["document"],
            doc_id=r["id"],
        ))
        if emb is not None:
            self._track_embedding(np.array(emb))
        if is_output:
            self.output_count += 1
        return True


class _ClusterDiverseStrategy:
    """Diverse retrieval using cluster centroids for concept identification
    and slot-based allocation for balanced results."""

    def __init__(
        self,
        top_clusters: int = 6,
        min_slots: int = 1,
        discovery_limit: int = 2,
        max_output_ratio: float = 0.5,
    ):
        self.top_clusters = top_clusters
        self.min_slots = min_slots
        self.discovery_limit = discovery_limit
        self.max_output_ratio = max_output_ratio
        self._intra_centroids_cache: list[dict] | None = None
        self._cross_centroids_cache: list[dict] | None = None
        self._cross_concept_composition_cache: dict[int, dict[str, int]] | None = None

    def _load_intra_centroids(self) -> list[dict]:
        """Load intra-folder cluster centroids from DB (cached)."""
        if self._intra_centroids_cache is not None:
            return self._intra_centroids_cache

        centroids = []
        with get_db() as conn:
            result = conn.execute(text(
                "SELECT id, folder_path, source_type, centroid FROM clusters "
                "WHERE cluster_type = 'intra_folder' AND centroid IS NOT NULL"
            ))
            for row in result.fetchall():
                r = row._mapping
                try:
                    centroid = np.array(json.loads(r["centroid"]))
                    centroids.append({
                        "cluster_id": r["id"],
                        "folder_path": r["folder_path"],
                        "source_type": r["source_type"],
                        "centroid": centroid,
                    })
                except (json.JSONDecodeError, ValueError):
                    continue

        self._intra_centroids_cache = centroids
        logger.info(f"Loaded {len(centroids)} intra-folder centroids")
        return centroids

    def _load_cross_centroids(self) -> list[dict]:
        """Load cross-folder cluster centroids from DB (cached)."""
        if self._cross_centroids_cache is not None:
            return self._cross_centroids_cache

        centroids = []
        with get_db() as conn:
            result = conn.execute(text(
                "SELECT id, centroid FROM clusters "
                "WHERE cluster_type = 'cross_folder' AND centroid IS NOT NULL"
            ))
            for row in result.fetchall():
                r = row._mapping
                try:
                    centroid = np.array(json.loads(r["centroid"]))
                    centroids.append({
                        "cluster_id": r["id"],
                        "centroid": centroid,
                    })
                except (json.JSONDecodeError, ValueError):
                    continue

        self._cross_centroids_cache = centroids
        logger.info(f"Loaded {len(centroids)} cross-folder centroids")
        return centroids

    def _load_cross_concept_composition(self) -> dict[int, dict[str, int]]:
        """Precompute concept composition per cross-folder cluster (cached)."""
        if self._cross_concept_composition_cache is not None:
            return self._cross_concept_composition_cache

        composition: dict[int, dict[str, int]] = {}
        with get_db() as conn:
            result = conn.execute(text("""
                SELECT ca_cross.cluster_id, c_intra.folder_path AS concept, COUNT(*) AS doc_count
                FROM cluster_assignments ca_cross
                JOIN cluster_assignments ca_intra ON ca_cross.doc_id = ca_intra.doc_id
                JOIN clusters c_intra ON ca_intra.cluster_id = c_intra.id
                    AND c_intra.cluster_type = 'intra_folder'
                JOIN clusters c_cross ON ca_cross.cluster_id = c_cross.id
                    AND c_cross.cluster_type = 'cross_folder'
                GROUP BY ca_cross.cluster_id, c_intra.folder_path
            """))
            for row in result.fetchall():
                r = row._mapping
                cluster_id = r["cluster_id"]
                if cluster_id not in composition:
                    composition[cluster_id] = {}
                composition[cluster_id][r["concept"]] = r["doc_count"]

        self._cross_concept_composition_cache = composition
        logger.info(f"Loaded concept composition for {len(composition)} cross-folder clusters")
        return composition

    def _rank_concepts(self, query_embedding: np.ndarray) -> list[dict]:
        """Phase 1: Find top concepts via intra-folder centroid distance."""
        centroids = self._load_intra_centroids()
        if not centroids:
            return []

        scored = []
        for c in centroids:
            dist = float(np.linalg.norm(query_embedding - c["centroid"]))
            scored.append({
                "folder_path": c["folder_path"],
                "distance": dist,
                "source_type": c["source_type"],
                "cluster_id": c["cluster_id"],
            })
        scored.sort(key=lambda x: x["distance"])

        seen: set[str] = set()
        deduped = []
        for item in scored:
            if item["folder_path"] not in seen:
                seen.add(item["folder_path"])
                deduped.append(item)

        return deduped[:self.top_clusters]

    def _discover_concepts(
        self,
        query_embedding: np.ndarray,
        existing_concepts: set[str],
    ) -> list[dict]:
        """Phase 2: Find additional concepts via cross-folder clusters."""
        cross_centroids = self._load_cross_centroids()
        if not cross_centroids:
            return []

        composition = self._load_cross_concept_composition()

        scored = []
        for c in cross_centroids:
            dist = float(np.linalg.norm(query_embedding - c["centroid"]))
            scored.append((c["cluster_id"], dist))
        scored.sort(key=lambda x: x[1])

        discovered = []
        for cluster_id, _ in scored[:2]:
            concept_counts = composition.get(cluster_id, {})
            if not concept_counts:
                continue

            total_docs = sum(concept_counts.values())
            for concept, doc_count in concept_counts.items():
                if concept in existing_concepts:
                    continue
                if doc_count < 3 and (doc_count / max(total_docs, 1)) < 0.05:
                    continue
                discovered.append({
                    "folder_path": concept,
                    "doc_count": doc_count,
                })
                existing_concepts.add(concept)

        discovered.sort(key=lambda x: x["doc_count"], reverse=True)
        return discovered[:self.discovery_limit]

    def _allocate_slots(
        self,
        ranked_concepts: list[dict],
        k: int,
    ) -> list[tuple[str, int]]:
        """Phase 3: Distribute k result slots across concepts."""
        if not ranked_concepts:
            return []

        max_concepts = k // max(self.min_slots, 1)
        concepts = ranked_concepts[:max_concepts]

        slots = {c["folder_path"]: self.min_slots for c in concepts}
        remaining = k - self.min_slots * len(concepts)

        if remaining > 0:
            inv_distances = [1.0 / max(c.get("distance", 1.0), 0.01) for c in concepts]
            total_inv = sum(inv_distances)

            fractions = [(c["folder_path"], (inv_d / total_inv) * remaining)
                         for c, inv_d in zip(concepts, inv_distances)]
            for fp, frac in fractions:
                slots[fp] += int(frac)

            leftover = k - sum(slots.values())
            if leftover > 0:
                by_remainder = sorted(fractions, key=lambda x: x[1] - int(x[1]), reverse=True)
                for i in range(min(leftover, len(by_remainder))):
                    slots[by_remainder[i][0]] += 1

        return [(fp, count) for fp, count in slots.items()]

    def _fetch_concept_results(
        self,
        query_embedding: list[float],
        concept_path: str,
        limit: int,
        state: _AcceptState,
    ) -> int:
        """Fetch and accept up to `limit` results for a concept. Returns count added."""
        overfetch = max(limit * 3, 6)
        raw = vector_store.search_similar(
            query_embedding, k=overfetch, concept=concept_path,
            include_embeddings=True,
        )
        deduped = vector_store.deduplicate_by_similarity(raw, threshold=SIMILARITY_THRESHOLD)

        added = 0
        for r in deduped:
            if added >= limit:
                break
            if state.try_accept(r):
                added += 1
        return added

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[_SearchResult]:
        """Full pipeline: rank concepts -> discover -> allocate slots -> fetch -> merge."""
        query_emb = np.array(query_embedding)

        ranked = self._rank_concepts(query_emb)

        if not ranked:
            logger.warning("No intra-folder centroids found, falling back to baseline search")
            results = vector_store.search_similar(query_embedding, k=k)
            return [
                _SearchResult(
                    rank=i + 1,
                    distance=r["distance"],
                    source=r["metadata"].get("dir_type", ""),
                    concept=r["metadata"].get("concept", ""),
                    text=r["document"],
                    doc_id=r["id"],
                )
                for i, r in enumerate(results)
            ]

        existing_concepts = {c["folder_path"] for c in ranked}
        discovered = self._discover_concepts(query_emb, existing_concepts)

        max_ranked_dist = max(c["distance"] for c in ranked)
        all_concepts = list(ranked)
        for d in discovered:
            all_concepts.append({
                "folder_path": d["folder_path"],
                "distance": max_ranked_dist + 0.1,
                "source_type": "discovery",
            })

        slot_allocation = self._allocate_slots(all_concepts, k)
        state = _AcceptState(max_output=math.floor(k * self.max_output_ratio))

        unfilled_slots = 0
        for concept_path, num_slots in slot_allocation:
            added = self._fetch_concept_results(
                query_embedding, concept_path, num_slots, state,
            )
            unfilled_slots += num_slots - added

        for concept_info in all_concepts:
            if unfilled_slots <= 0:
                break
            before = len(state.results)
            self._fetch_concept_results(
                query_embedding, concept_info["folder_path"], unfilled_slots, state,
            )
            unfilled_slots -= len(state.results) - before

        state.results.sort(key=lambda r: r.distance)
        for i, r in enumerate(state.results[:k]):
            r.rank = i + 1
        return state.results[:k]


# Module-level singleton — centroids cached on first search call
_strategy = _ClusterDiverseStrategy()


def _query_diverse_prompts(args: dict) -> dict:
    """Tool function: cluster-based diverse prompt retrieval."""
    query = args.get("query", "")
    k = args.get("k", 10)
    source_type = args.get("source_type")

    query_embedding = embedding_service.embed(query)
    results = _strategy.search(query_embedding, k=k)

    if source_type:
        if source_type == "training":
            results = [r for r in results if r.source == "training"]
        else:
            results = [r for r in results if r.source != "training"]

    return {
        "query": query,
        "count": len(results),
        "prompts": [
            {"text": r.text, "concept": r.concept,
             "source": r.source, "distance": round(r.distance, 4)}
            for r in results
        ],
    }
