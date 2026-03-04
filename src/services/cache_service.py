"""Gemini explicit caching for static agent content."""

import hashlib
import json
import logging
import threading

from google.genai import types

logger = logging.getLogger(__name__)


class CacheManager:
    """Lazily creates and reuses a Gemini cached-content object.

    The cache holds: system prompt + tool declarations + dataset overview.
    It is invalidated when any of those change, or on TTL expiry.
    """

    def __init__(self):
        self._cache_name: str | None = None
        self._cache_hash: str | None = None
        self._cache_model: str | None = None
        self._lock = threading.Lock()

    def get_or_create(
        self,
        client,          # genai.Client
        model: str,
        system_prompt: str,
        tools: list,
        dataset_overview: dict,
        ttl: str = "3600s",
    ) -> str:
        """Return the cache name, creating a new cache if needed.

        Args:
            client: Initialized genai.Client.
            model: Model name (e.g. "gemini-2.5-pro").
            system_prompt: Base system prompt (no agent state).
            tools: TOOL_DECLARATIONS list.
            dataset_overview: Output of get_dataset_overview().
            ttl: Cache time-to-live.

        Returns:
            Cache resource name string for use in generate_content.
        """
        content_hash = self._compute_hash(system_prompt, tools, dataset_overview)

        with self._lock:
            # Fast path: cache exists and content hasn't changed
            if (self._cache_name
                    and self._cache_hash == content_hash
                    and self._cache_model == model):
                try:
                    client.caches.get(name=self._cache_name)
                    logger.debug("Reusing Gemini cache: %s", self._cache_name)
                    return self._cache_name
                except Exception:
                    # Cache expired or deleted — fall through to recreate
                    self._cache_name = None

            # Create new cache
            overview_json = json.dumps(dataset_overview, separators=(",", ":"))
            cache = client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_prompt,
                    tools=tools,
                    contents=[
                        types.Content(role="user", parts=[types.Part.from_text(
                            text="Dataset overview (pre-loaded context \u2014 do not call "
                            "get_dataset_overview, this data is already available):\n"
                            + overview_json
                        )]),
                        types.Content(role="model", parts=[types.Part.from_text(
                            text="I have the dataset overview with folder structure, "
                            "cross-folder themes, and statistics. I'll call "
                            "get_folder_themes when I need intra-folder details "
                            "for a specific folder. Ready to help."
                        )]),
                    ],
                    ttl=ttl,
                ),
            )

            self._cache_name = cache.name
            self._cache_hash = content_hash
            self._cache_model = model
            logger.info("Created Gemini cache: %s", cache.name)
            return cache.name

    def invalidate(self):
        """Mark cache as stale. Next get_or_create() will build a new one."""
        with self._lock:
            if self._cache_name:
                logger.info("Invalidating cache: %s", self._cache_name)
            self._cache_name = None
            self._cache_hash = None
            self._cache_model = None

    def _compute_hash(self, system_prompt, tools, dataset_overview) -> str:
        """Stable hash of all cached content."""
        h = hashlib.sha256()
        h.update(system_prompt.encode())
        h.update(json.dumps(dataset_overview, sort_keys=True, separators=(",", ":")).encode())
        # Tools are static objects — hash their string representation
        for tool in tools:
            h.update(str(tool).encode())
        return h.hexdigest()


# Module-level singleton
cache_manager = CacheManager()
