"""Gemini LLM service for generation, embedding, and model listing."""

from __future__ import annotations

import logging
import re
import time
from typing import Generator

from google import genai
from google.genai import types

from src.config import DEFAULT_MODEL_EMBEDDING
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


def _call_with_retry(fn, max_retries: int = 3, status_callback=None):
    """Call *fn* with retry on transient Gemini errors.

    Handles:
    - 429 (rate limited): parses retry delay from error message, fallback 60s
    - 5xx (server errors): exponential backoff starting at 5s
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except genai.errors.ClientError as exc:
            if exc.code != 429 or attempt == max_retries:
                raise
            # Try to parse "retry after Xs" or "retry in Xs" from the message
            match = re.search(r"retry\s+(?:after|in)\s+(\d+)", str(exc), re.IGNORECASE)
            delay = int(match.group(1)) if match else 60
            msg = f"API rate limited, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
            logger.warning(msg)
            if status_callback:
                try:
                    status_callback(msg)
                except Exception:
                    pass
            time.sleep(delay)
        except genai.errors.ServerError as exc:
            if attempt == max_retries:
                raise
            delay = 5 * (3 ** attempt)  # 5s, 15s, 45s
            msg = f"API server error ({exc.code}), retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
            logger.warning(msg)
            if status_callback:
                try:
                    status_callback(msg)
                except Exception:
                    pass
            time.sleep(delay)


def embed(
    text: str,
    model: str = DEFAULT_MODEL_EMBEDDING,
    status_callback=None,
) -> list[float]:
    """Generate an embedding vector for a single text."""
    if not _client:
        raise RuntimeError("Gemini client not initialized.")

    rate_limiter.acquire(count=1, status_callback=status_callback)

    result = _call_with_retry(
        lambda: _client.models.embed_content(model=model, contents=text),
        status_callback=status_callback,
    )
    return result.embeddings[0].values


def embed_batch(
    texts: list[str],
    model: str = DEFAULT_MODEL_EMBEDDING,
    status_callback=None,
) -> list[list[float]]:
    """Generate embedding vectors for a batch of texts."""
    if not _client:
        raise RuntimeError("Gemini client not initialized.")

    rate_limiter.acquire(count=len(texts), status_callback=status_callback)

    result = _call_with_retry(
        lambda: _client.models.embed_content(model=model, contents=texts),
        status_callback=status_callback,
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
