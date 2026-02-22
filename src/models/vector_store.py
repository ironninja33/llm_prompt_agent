"""ChromaDB vector store wrapper for prompt storage and retrieval."""

from __future__ import annotations

import json
import os
import random
import logging
from typing import Optional

import chromadb

from src.config import CHROMA_DB_PATH

logger = logging.getLogger(__name__)

# Module-level client (initialized once)
_client: chromadb.PersistentClient | None = None
_training_collection = None
_generated_collection = None


def initialize():
    """Initialize the ChromaDB client and collections."""
    global _client, _training_collection, _generated_collection

    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    _training_collection = _client.get_or_create_collection(
        name="training_prompts",
        metadata={"description": "Prompts from training data text files"}
    )

    _generated_collection = _client.get_or_create_collection(
        name="generated_prompts",
        metadata={"description": "Prompts extracted from generated output images"}
    )

    logger.info(
        f"ChromaDB initialized. Training: {_training_collection.count()} docs, "
        f"Generated: {_generated_collection.count()} docs"
    )


def _get_collection(source_type: str):
    """Get the appropriate collection based on source type."""
    if source_type == "training":
        return _training_collection
    elif source_type == "output":
        return _generated_collection
    else:
        raise ValueError(f"Unknown source_type: {source_type}")


def document_exists(doc_id: str, source_type: str) -> bool:
    """Check if a document with the given ID already exists."""
    collection = _get_collection(source_type)
    try:
        result = collection.get(ids=[doc_id])
        return len(result["ids"]) > 0
    except Exception:
        return False


def add_document(
    doc_id: str,
    text: str,
    embedding: list[float],
    source_type: str,
    metadata: dict,
):
    """Add a document to the appropriate collection."""
    collection = _get_collection(source_type)

    clean_meta = {}
    for k, v in metadata.items():
        if v is None:
            clean_meta[k] = ""
        elif isinstance(v, (list, dict)):
            clean_meta[k] = json.dumps(v)
        else:
            clean_meta[k] = v

    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[clean_meta],
    )


def add_documents_batch(
    doc_ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    source_type: str,
    metadatas: list[dict],
):
    """Add multiple documents at once."""
    collection = _get_collection(source_type)

    clean_metadatas = []
    for metadata in metadatas:
        clean_meta = {}
        for k, v in metadata.items():
            if v is None:
                clean_meta[k] = ""
            elif isinstance(v, (list, dict)):
                clean_meta[k] = json.dumps(v)
            else:
                clean_meta[k] = v
        clean_metadatas.append(clean_meta)

    collection.add(
        ids=doc_ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=clean_metadatas,
    )


def search_similar(
    query_embedding: list[float],
    k: int = 10,
    source_type: str | None = None,
    concept: str | None = None,
) -> list[dict]:
    """Search for similar prompts by embedding."""
    where_filter = None
    if concept:
        where_filter = {"concept": concept}

    collections = []
    if source_type:
        collections.append(_get_collection(source_type))
    else:
        collections = [_training_collection, _generated_collection]

    all_results = []
    for collection in collections:
        if collection.count() == 0:
            continue
        try:
            actual_k = min(k, collection.count())
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k,
                where=where_filter,
            )
            for i in range(len(result["ids"][0])):
                all_results.append({
                    "id": result["ids"][0][i],
                    "document": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i] if result["metadatas"] else {},
                    "distance": result["distances"][0][i] if result["distances"] else 0,
                })
        except Exception as e:
            logger.error(f"Error searching collection: {e}")

    all_results.sort(key=lambda x: x["distance"])
    return all_results[:k]


def search_diverse(
    query_embedding: list[float],
    k: int = 10,
    offset: int = 40,
    source_type: str | None = None,
) -> list[dict]:
    """Search for diverse/distant prompts."""
    collections = []
    if source_type:
        collections.append(_get_collection(source_type))
    else:
        collections = [_training_collection, _generated_collection]

    all_results = []
    fetch_count = offset + k

    for collection in collections:
        if collection.count() == 0:
            continue
        try:
            actual_fetch = min(fetch_count, collection.count())
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_fetch,
            )
            for i in range(len(result["ids"][0])):
                all_results.append({
                    "id": result["ids"][0][i],
                    "document": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i] if result["metadatas"] else {},
                    "distance": result["distances"][0][i] if result["distances"] else 0,
                })
        except Exception as e:
            logger.error(f"Error searching collection: {e}")

    all_results.sort(key=lambda x: x["distance"])
    return all_results[offset:offset + k] if len(all_results) > offset else all_results[-k:]


def get_random(
    k: int = 10,
    source_type: str | None = None,
) -> list[dict]:
    """Get random prompts from the database."""
    collections = []
    if source_type:
        collections.append(_get_collection(source_type))
    else:
        collections = [_training_collection, _generated_collection]

    all_docs = []
    for collection in collections:
        count = collection.count()
        if count == 0:
            continue
        try:
            all_data = collection.get(
                limit=count,
                include=["documents", "metadatas"]
            )
            for i in range(len(all_data["ids"])):
                all_docs.append({
                    "id": all_data["ids"][i],
                    "document": all_data["documents"][i],
                    "metadata": all_data["metadatas"][i] if all_data["metadatas"] else {},
                    "distance": 0,
                })
        except Exception as e:
            logger.error(f"Error getting random docs: {e}")

    if not all_docs:
        return []
    return random.sample(all_docs, min(k, len(all_docs)))


def list_concepts(source_type: str | None = None) -> list[dict]:
    """List all unique concept names with counts."""
    collections = []
    if source_type:
        collections.append((_get_collection(source_type), source_type))
    else:
        collections = [
            (_training_collection, "training"),
            (_generated_collection, "output"),
        ]

    concepts = {}
    for collection, stype in collections:
        count = collection.count()
        if count == 0:
            continue
        try:
            all_data = collection.get(limit=count, include=["metadatas"])
            for meta in all_data["metadatas"]:
                concept = meta.get("concept", "unknown")
                key = (concept, stype)
                concepts[key] = concepts.get(key, 0) + 1
        except Exception as e:
            logger.error(f"Error listing concepts: {e}")

    return [
        {"concept": concept, "source_type": stype, "count": count}
        for (concept, stype), count in sorted(concepts.items())
    ]


def get_collection_counts() -> dict:
    """Get document counts for each collection."""
    return {
        "training": _training_collection.count() if _training_collection else 0,
        "generated": _generated_collection.count() if _generated_collection else 0,
    }


def get_existing_ids(source_type: str) -> set[str]:
    """Get all existing document IDs for a collection.

    Paginates through the collection to ensure all IDs are retrieved,
    even for large collections where a single get() call might not
    return everything.
    """
    collection = _get_collection(source_type)
    count = collection.count()
    if count == 0:
        return set()

    try:
        all_ids: set[str] = set()
        page_size = 5000
        offset = 0

        while offset < count:
            result = collection.get(
                limit=page_size,
                offset=offset,
            )
            batch_ids = result["ids"]
            if not batch_ids:
                break
            all_ids.update(batch_ids)
            offset += len(batch_ids)

        logger.debug(
            f"Retrieved {len(all_ids)} existing IDs from {source_type} "
            f"collection (reported count: {count})"
        )
        return all_ids
    except Exception as e:
        logger.error(f"Error getting existing IDs: {e}")
        return set()


def delete_document(doc_id: str, source_type: str) -> bool:
    """Delete a single document by ID from the given collection.

    Returns True if the document was found and deleted.
    """
    collection = _get_collection(source_type)
    try:
        if not document_exists(doc_id, source_type):
            return False
        collection.delete(ids=[doc_id])
        logger.info(f"Deleted document {doc_id} from {source_type} collection")
        return True
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        return False


def delete_documents_by_directory(dir_path: str, source_type: str) -> int:
    """Delete all documents whose IDs start with the given directory path.

    Returns the number of documents deleted.
    """
    collection = _get_collection(source_type)
    count = collection.count()
    if count == 0:
        return 0

    try:
        # Ensure consistent trailing separator for prefix matching
        prefix = dir_path.rstrip("/") + "/"
        result = collection.get(limit=count)
        ids_to_delete = [
            doc_id for doc_id in result["ids"] if doc_id.startswith(prefix)
        ]

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info(
                f"Deleted {len(ids_to_delete)} documents from {source_type} "
                f"collection for directory: {dir_path}"
            )

        return len(ids_to_delete)
    except Exception as e:
        logger.error(f"Error deleting documents for directory {dir_path}: {e}")
        return 0
