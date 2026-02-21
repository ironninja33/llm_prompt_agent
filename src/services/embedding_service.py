"""Embedding service — thin wrapper around LLM service for embeddings."""

import logging
from src.services import llm_service
from src.models.settings import get_setting

logger = logging.getLogger(__name__)


def get_embedding_model() -> str:
    """Get the configured embedding model name."""
    return get_setting("model_embedding") or "gemini-embedding-001"


def embed(text: str) -> list[float]:
    """Generate an embedding for a single text using the configured model."""
    model = get_embedding_model()
    return llm_service.embed(text, model=model)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    model = get_embedding_model()
    return llm_service.embed_batch(texts, model=model)
