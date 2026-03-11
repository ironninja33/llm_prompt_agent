"""Gemini explicit caching for static agent content."""

import logging
import threading

from google.genai import types
from sqlalchemy import text

from src.models.database import get_db

logger = logging.getLogger(__name__)

# SQLite settings keys for cache persistence
_DB_KEY_NAME = "gemini_cache_name"
_DB_KEY_MODEL = "gemini_cache_model"


class CacheManager:
    """Lazily creates and reuses a Gemini cached-content object.

    The cache holds: system prompt + tool declarations (no conversation
    content).  The dataset overview is injected as droppable messages
    by the agent loop and excluded once the agent has accumulated state.

    The cache is invalidated explicitly when content changes (settings
    update, ingestion, reclustering, folder rename/delete), or on
    Gemini TTL expiry.
    """

    def __init__(self):
        self._cache_name: str | None = None
        self._cache_model: str | None = None
        self._lock = threading.Lock()

    def get_or_create(
        self,
        client,          # genai.Client
        model: str,
        system_prompt: str,
        tools: list,
        ttl: str = "3600s",
    ) -> str:
        """Return the cache name, creating a new cache if needed.

        The cache contains only the system prompt and tool declarations.
        The dataset overview is handled separately by the agent loop.

        Args:
            client: Initialized genai.Client.
            model: Model name (e.g. "gemini-2.5-pro").
            system_prompt: Base system prompt (no agent state).
            tools: TOOL_DECLARATIONS list.
            ttl: Cache time-to-live.

        Returns:
            Cache resource name string for use in generate_content.
        """
        with self._lock:
            # Fast path: in-memory cache exists and model matches
            if self._cache_name and self._cache_model == model:
                try:
                    client.caches.get(name=self._cache_name)
                    self._refresh_ttl(client, self._cache_name, ttl)
                    logger.debug("Reusing Gemini cache: %s", self._cache_name)
                    return self._cache_name
                except Exception:
                    # Cache expired or deleted on Gemini's side
                    self._cache_name = None
                    self._cache_model = None

            # Recovery path: try to restore from SQLite
            if not self._cache_name:
                db_name, db_model = self._load_from_db()
                if db_name and db_model == model:
                    try:
                        client.caches.get(name=db_name)
                        self._cache_name = db_name
                        self._cache_model = db_model
                        self._refresh_ttl(client, db_name, ttl)
                        logger.info("Recovered Gemini cache from DB: %s", db_name)
                        return db_name
                    except Exception:
                        logger.debug("DB cache expired or invalid, will recreate")
                        self._clear_db()

            # Create new cache with system prompt and tools only
            cache = client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_prompt,
                    tools=tools,
                    ttl=ttl,
                ),
            )

            self._cache_name = cache.name
            self._cache_model = model
            self._save_to_db(cache.name, model)
            logger.info("Created Gemini cache: %s", cache.name)
            return cache.name

    def invalidate(self):
        """Mark cache as stale. Next get_or_create() will build a new one."""
        with self._lock:
            if self._cache_name:
                logger.info("Invalidating cache: %s", self._cache_name)
            self._cache_name = None
            self._cache_model = None
            self._clear_db()

    # -- SQLite persistence helpers --

    def _save_to_db(self, cache_name: str, model: str):
        """Persist cache metadata to the settings table."""
        try:
            with get_db() as conn:
                for key, value in [
                    (_DB_KEY_NAME, cache_name),
                    (_DB_KEY_MODEL, model),
                ]:
                    conn.execute(
                        text("INSERT OR REPLACE INTO settings (key, value) VALUES (:key, :value)"),
                        {"key": key, "value": value},
                    )
            logger.debug("Saved cache metadata to DB")
        except Exception:
            logger.warning("Failed to persist cache metadata to DB", exc_info=True)

    def _load_from_db(self) -> tuple[str | None, str | None]:
        """Load cache metadata from the settings table."""
        try:
            with get_db() as conn:
                result = conn.execute(
                    text("SELECT key, value FROM settings WHERE key IN (:k1, :k2)"),
                    {"k1": _DB_KEY_NAME, "k2": _DB_KEY_MODEL},
                )
                rows = {row[0]: row[1] for row in result.fetchall()}
            return (
                rows.get(_DB_KEY_NAME),
                rows.get(_DB_KEY_MODEL),
            )
        except Exception:
            logger.warning("Failed to load cache metadata from DB", exc_info=True)
            return None, None

    def _clear_db(self):
        """Remove cache metadata from the settings table."""
        try:
            with get_db() as conn:
                conn.execute(
                    text("DELETE FROM settings WHERE key IN (:k1, :k2)"),
                    {"k1": _DB_KEY_NAME, "k2": _DB_KEY_MODEL},
                )
        except Exception:
            logger.warning("Failed to clear cache metadata from DB", exc_info=True)

    def _refresh_ttl(self, client, cache_name: str, ttl: str):
        """Refresh the TTL on an existing cache so it doesn't expire mid-session."""
        try:
            client.caches.update(
                name=cache_name,
                config=types.UpdateCachedContentConfig(ttl=ttl),
            )
        except Exception:
            logger.debug("Could not refresh cache TTL: %s", cache_name, exc_info=True)


# Module-level singleton
cache_manager = CacheManager()
