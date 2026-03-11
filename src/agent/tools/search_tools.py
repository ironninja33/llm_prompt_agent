"""Search and exploration tool implementations."""

from src.services import embedding_service
from src.models import vector_store


def _search_similar(args: dict) -> dict:
    query = args.get("query", "")
    k = args.get("k", 10)
    source_type = args.get("source_type")
    concept = args.get("concept")

    # Validate concept exists in target collection(s) before embedding
    if concept:
        concept_counts = _check_concept_availability(concept, source_type)
        if not any(c["count"] > 0 for c in concept_counts):
            available_in = [
                c for c in _check_concept_availability(concept, source_type=None)
                if c["count"] > 0
            ]
            msg = f"Concept '{concept}' has 0 documents"
            if source_type:
                msg += f" in {source_type} collection"
            if available_in:
                details = ", ".join(
                    f"{c['source_type']} ({c['count']})" for c in available_in
                )
                msg += f". Found in: {details}"
            else:
                msg += ". This concept does not exist in any collection"
            msg += ". Try removing the source_type or concept filter."
            return {"query": query, "count": 0, "prompts": [], "warning": msg}

    query_embedding = embedding_service.embed(query)
    results = vector_store.search_similar(
        query_embedding, k=k, source_type=source_type, concept=concept
    )

    return {
        "query": query,
        "count": len(results),
        "prompts": [
            {
                "text": r["document"],
                "concept": r["metadata"].get("concept", ""),
                "source": r["metadata"].get("dir_type", ""),
                "distance": round(r["distance"], 4),
            }
            for r in results
        ],
    }


def _check_concept_availability(
    concept: str, source_type: str | None = None
) -> list[dict]:
    """Check how many documents exist for a concept in each collection."""
    collections = []
    if source_type:
        collections.append((vector_store._get_collection(source_type), source_type))
    else:
        collections.append((vector_store._training_collection, "training"))
        collections.append((vector_store._generated_collection, "output"))

    results = []
    for collection, stype in collections:
        try:
            matched = collection.get(
                where={"concept": concept}, include=[], limit=1
            )
            count = len(matched["ids"])
            # If we got 1, do a full count
            if count > 0:
                matched = collection.get(
                    where={"concept": concept}, include=[]
                )
                count = len(matched["ids"])
            results.append({"source_type": stype, "count": count})
        except Exception:
            results.append({"source_type": stype, "count": 0})
    return results


def _search_diverse(args: dict) -> dict:
    query = args.get("query", "")
    k = args.get("k", 10)
    source_type = args.get("source_type")

    query_embedding = embedding_service.embed(query)
    results = vector_store.search_diverse(query_embedding, k=k, source_type=source_type)

    return {
        "query": query,
        "count": len(results),
        "prompts": [
            {
                "text": r["document"],
                "concept": r["metadata"].get("concept", ""),
                "source": r["metadata"].get("dir_type", ""),
            }
            for r in results
        ],
    }


def _get_random(args: dict) -> dict:
    k = args.get("k", 10)
    source_type = args.get("source_type")

    results = vector_store.get_random(k=k, source_type=source_type)

    return {
        "count": len(results),
        "prompts": [
            {
                "text": r["document"],
                "concept": r["metadata"].get("concept", ""),
                "source": r["metadata"].get("dir_type", ""),
            }
            for r in results
        ],
    }


def _get_opposite(args: dict) -> dict:
    query = args.get("query", "")
    k = args.get("k", 10)
    source_type = args.get("source_type")

    query_embedding = embedding_service.embed(query)
    # Use a large offset to get the most distant prompts
    results = vector_store.search_diverse(query_embedding, k=k, offset=100, source_type=source_type)

    return {
        "query": query,
        "count": len(results),
        "prompts": [
            {
                "text": r["document"],
                "concept": r["metadata"].get("concept", ""),
                "source": r["metadata"].get("dir_type", ""),
            }
            for r in results
        ],
    }


def _list_concepts(args: dict) -> dict:
    source_type = args.get("source_type")
    concepts = vector_store.list_concepts(source_type=source_type)

    return {
        "count": len(concepts),
        "concepts": concepts,
    }


def _get_dataset_overview(args: dict) -> dict:
    """Get the lightweight dataset overview (no intra-folder cluster details)."""
    from src.services import clustering_service
    return clustering_service.get_dataset_overview()


def _get_folder_themes(args: dict) -> dict:
    """Get intra-folder cluster themes for a specific folder and source type."""
    from src.services import clustering_service

    folder_name = args.get("folder_name", "")
    source_type = args.get("source_type", "training")
    if not folder_name:
        return {"error": "folder_name is required"}
    if source_type not in ("training", "output"):
        return {"error": "source_type must be 'training' or 'output'"}

    result = clustering_service.get_folder_themes(folder_name, source_type=source_type)

    # If no themes found, try fuzzy matching against known folder names
    if not result.get("themes"):
        known_concepts = vector_store.list_concepts(source_type=source_type)
        known_names = sorted({c["concept"] for c in known_concepts})

        # Try suffix match: user may pass just the display_name part
        matches = [n for n in known_names if n.endswith(f"__{folder_name}") or n == folder_name]
        if len(matches) == 1:
            result = clustering_service.get_folder_themes(matches[0], source_type=source_type)
        elif not matches:
            # No themes and no fuzzy match — suggest available names
            result["suggestions"] = known_names[:20]
            result["hint"] = (
                "No themes found. Use the full 'category__display_name' format from the dataset overview. "
                f"Available {source_type} folders: {', '.join(known_names[:10])}"
            )

    return result


def _query_themed_prompts(args: dict) -> dict:
    """Execute a themed prompt query across multiple sources."""
    from src.services import clustering_service

    query = args.get("query", "")
    include_random = args.get("include_random", False)
    include_opposite = args.get("include_opposite", False)
    source_type = args.get("source_type")

    # Generate embedding for the query
    query_embedding = embedding_service.embed(query)

    # Read k-values from settings
    from src.models.settings import get_setting
    k_random = int(get_setting("query_k_random") or "3") if include_random else 0
    k_opposite = int(get_setting("query_k_random") or "3") if include_opposite else 0

    result = clustering_service.get_themed_prompts(
        query_embedding=query_embedding,
        k_random=k_random,
        k_opposite=k_opposite,
        source_type=source_type,
    )

    return result


def _query_dataset_map(args: dict) -> dict:
    """Search for dataset folders matching a query string."""
    from src.services import clustering_service

    query = args.get("query", "").lower()
    k = args.get("k", 10)
    source_type = args.get("source_type")

    overview = clustering_service.get_dataset_overview()
    query_terms = query.split()

    scored = []
    for folder in overview.get("folders", []):
        if source_type and folder.get("source_type") != source_type:
            continue

        # Build searchable text from folder metadata
        searchable = " ".join([
            folder.get("name", ""),
            folder.get("category", ""),
            folder.get("display_name", ""),
            folder.get("summary", ""),
        ]).lower()

        # Score by term matches (full + partial)
        score = sum(1 for t in query_terms if t in searchable)
        partial = sum(0.5 for t in query_terms
                      for w in searchable.split() if t in w and t != w)
        total = score + partial

        if total > 0:
            scored.append((total, folder))

    scored.sort(key=lambda x: -x[0])

    # Enrich top results with intra-folder themes
    results = []
    for _, folder in scored[:k]:
        themes = clustering_service.get_folder_themes(
            folder["name"], source_type=folder.get("source_type", "training")
        )
        results.append({
            "name": folder["name"],
            "category": folder.get("category", ""),
            "display_name": folder.get("display_name", ""),
            "source_type": folder.get("source_type", ""),
            "total_prompts": folder.get("total_prompts", 0),
            "summary": folder.get("summary", ""),
            "top_themes": [t["label"] for t in themes.get("themes", [])[:3]],
        })

    return {"query": query, "count": len(results), "folders": results}
