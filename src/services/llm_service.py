"""Gemini LLM service for generation, embedding, and model listing."""

from __future__ import annotations

import logging
from typing import Generator

from google import genai
from google.genai import types

from src.services import rate_limiter

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def initialize(api_key: str):
    """Initialize the Gemini client with the given API key."""
    global _client
    if not api_key:
        logger.warning("No Gemini API key provided. LLM calls will fail.")
        _client = None
        return
    _client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialized")


def get_client() -> genai.Client | None:
    """Return the current Gemini client."""
    return _client


def generate_stream(
    model: str,
    messages: list[dict],
    system_prompt: str | None = None,
    tools: list | None = None,
    cached_content: str | None = None,
) -> Generator:
    """Stream a response from Gemini.

    Args:
        model: Model name (e.g. 'gemini-3-pro-preview').
        messages: List of {'role': 'user'|'model', 'parts': [...]}.
        system_prompt: Optional system instruction.
        tools: Optional list of tool declarations.
        cached_content: Optional cache resource name. When provided,
            system_prompt and tools should be None (they're in the cache).

    Yields:
        Response chunks from the streaming API.
    """
    if not _client:
        raise RuntimeError("Gemini client not initialized. Set your API key in settings.")

    rate_limiter.acquire()

    config = types.GenerateContentConfig(
        system_instruction=system_prompt if system_prompt else None,
        tools=tools if tools else None,
        cached_content=cached_content,
    )

    contents = _build_contents(messages)

    response = _client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    )
    return response


def generate(
    model: str,
    messages: list[dict],
    system_prompt: str | None = None,
    tools: list | None = None,
    cached_content: str | None = None,
) -> types.GenerateContentResponse:
    """Non-streaming generation (for summarization, etc.)."""
    if not _client:
        raise RuntimeError("Gemini client not initialized. Set your API key in settings.")

    rate_limiter.acquire()

    config = types.GenerateContentConfig(
        system_instruction=system_prompt if system_prompt else None,
        tools=tools if tools else None,
        cached_content=cached_content,
    )

    contents = _build_contents(messages)

    response = _client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return response


def embed(text: str, model: str = "gemini-embedding-001") -> list[float]:
    """Generate an embedding vector for a single text."""
    if not _client:
        raise RuntimeError("Gemini client not initialized.")

    rate_limiter.acquire()

    result = _client.models.embed_content(
        model=model,
        contents=text,
    )
    return result.embeddings[0].values


def embed_batch(texts: list[str], model: str = "gemini-embedding-001") -> list[list[float]]:
    """Generate embedding vectors for a batch of texts."""
    if not _client:
        raise RuntimeError("Gemini client not initialized.")

    rate_limiter.acquire()

    # Gemini supports batch embedding
    result = _client.models.embed_content(
        model=model,
        contents=texts,
    )
    return [e.values for e in result.embeddings]


def list_models() -> list[dict]:
    """List available Gemini models."""
    if not _client:
        return []

    try:
        rate_limiter.acquire()
        models = _client.models.list()
        result = []
        for m in models:
            result.append({
                "name": m.name,
                "display_name": m.display_name,
                "description": m.description or "",
            })
        return result
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return []


def summarize_chat(model: str, messages: list[dict]) -> str:
    """Summarize a chat to a short 4-5 word title."""
    if not _client:
        return "New Chat"

    # Build a summary prompt from the first exchange
    summary_prompt = "Summarize this conversation in exactly 4-5 words for a chat title. Return only the title, nothing else.\n\n"
    for msg in messages[:4]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        summary_prompt += f"{role}: {content[:200]}\n"

    try:
        rate_limiter.acquire()
        response = _client.models.generate_content(
            model=model,
            contents=summary_prompt,
        )
        title = response.text.strip().strip('"').strip("'")
        # Ensure it's not too long
        words = title.split()
        if len(words) > 6:
            title = " ".join(words[:5])
        return title
    except Exception as e:
        logger.error(f"Error summarizing chat: {e}")
        return "New Chat"


def _build_contents(messages: list[dict]) -> list[types.Content]:
    """Convert our message format to Gemini Content objects."""
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        # Map our roles to Gemini roles
        if role in ("assistant", "model"):
            gemini_role = "model"
        elif role == "tool":
            gemini_role = "tool"
        else:
            gemini_role = "user"

        parts = msg.get("parts")
        if parts:
            # Already in parts format
            contents.append(types.Content(role=gemini_role, parts=parts))
        else:
            content_text = msg.get("content", "")
            contents.append(types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=content_text)]
            ))
    return contents
