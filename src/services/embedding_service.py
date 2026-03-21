"""Embedding service — thin wrapper around LLM service for embeddings."""

import logging
from src.config import DEFAULT_MODEL_EMBEDDING
from src.services import llm_service
from src.models.settings import get_setting

logger = logging.getLogger(__name__)


def get_embedding_model() -> str:
    """Get the configured embedding model name."""
    return get_setting("model_embedding") or DEFAULT_MODEL_EMBEDDING


def embed(text: str, status_callback=None) -> list[float]:
    """Generate an embedding for a single text using the configured model."""
    model = get_embedding_model()
    return llm_service.embed(text, model=model, status_callback=status_callback)


def embed_batch(texts: list[str], status_callback=None) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    model = get_embedding_model()
    return llm_service.embed_batch(texts, model=model, status_callback=status_callback)
