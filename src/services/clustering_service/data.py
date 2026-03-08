"""ChromaDB data fetching utilities for clustering."""

import logging

from src.models import vector_store

logger = logging.getLogger(__name__)


def _fetch_all_embeddings() -> tuple[list[str], list[list[float]], list[str], list[dict]]:
    """Fetch all embeddings, documents, and metadata from both ChromaDB collections.

    Returns:
        Tuple of (ids, embeddings, documents, metadatas).
    """
    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    for collection in [vector_store._training_collection, vector_store._generated_collection]:
        if collection is None:
            continue
        count = collection.count()
        if count == 0:
            continue

        page_size = 5000
        offset = 0
        while offset < count:
            result = collection.get(
                limit=page_size,
                offset=offset,
                include=["embeddings", "documents", "metadatas"],
            )
            batch_ids = result["ids"]
            if not batch_ids:
                break

            all_ids.extend(batch_ids)
            all_embeddings.extend(result["embeddings"])
            all_documents.extend(result["documents"])
            all_metadatas.extend(result["metadatas"])
            offset += len(batch_ids)

    return all_ids, all_embeddings, all_documents, all_metadatas


def _fetch_embeddings_by_concept(
    concept_name: str,
    source_type: str | None = None,
) -> tuple[list[str], list[list[float]], list[str], list[dict]]:
    """Fetch embeddings and documents for a specific concept.

    Args:
        concept_name: The concept/folder name to filter by.
        source_type: ``"training"`` to fetch only from the training collection,
            ``"output"`` for only the generated collection, or ``None`` for both.

    Returns:
        Tuple of (ids, embeddings, documents, metadatas).
    """
    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    if source_type == "training":
        collections = [vector_store._training_collection]
    elif source_type == "output":
        collections = [vector_store._generated_collection]
    else:
        collections = [vector_store._training_collection, vector_store._generated_collection]

    for collection in collections:
        if collection is None:
            continue
        count = collection.count()
        if count == 0:
            continue

        try:
            result = collection.get(
                where={"concept": concept_name},
                include=["embeddings", "documents", "metadatas"],
            )
            if result["ids"]:
                all_ids.extend(result["ids"])
                all_embeddings.extend(result["embeddings"])
                all_documents.extend(result["documents"])
                all_metadatas.extend(result["metadatas"])
        except Exception as e:
            logger.error(f"Error fetching embeddings for concept '{concept_name}': {e}")

    return all_ids, all_embeddings, all_documents, all_metadatas


def _get_all_concepts() -> list[dict]:
    """Get all unique concept names with counts and source types from ChromaDB.

    Returns:
        List of dicts with keys: concept, source_type, count.
    """
    return vector_store.list_concepts()


def _fetch_docs_by_ids(doc_ids: list[str], source_type_map: dict[str, str]) -> list[dict]:
    """Fetch documents from ChromaDB by their IDs.

    Args:
        doc_ids: List of document IDs to fetch.
        source_type_map: Mapping of doc_id to source_type for collection selection.

    Returns:
        List of prompt dicts with ``text``, ``concept``, ``source``, ``distance`` keys.
    """
    prompts: list[dict] = []

    # Group IDs by collection
    training_ids = [did for did in doc_ids if source_type_map.get(did, "training") == "training"]
    output_ids = [did for did in doc_ids if source_type_map.get(did, "training") == "output"]

    for ids_batch, collection in [
        (training_ids, vector_store._training_collection),
        (output_ids, vector_store._generated_collection),
    ]:
        if not ids_batch or collection is None:
            continue
        try:
            result = collection.get(
                ids=ids_batch,
                include=["documents", "metadatas"],
            )
            for i in range(len(result["ids"])):
                meta = result["metadatas"][i] if result["metadatas"] else {}
                prompts.append({
                    "text": result["documents"][i],
                    "concept": meta.get("concept", ""),
                    "source": meta.get("dir_type", ""),
                    "distance": 0,
                })
        except Exception as e:
            logger.error(f"Error fetching docs by IDs from collection: {e}")

    return prompts
