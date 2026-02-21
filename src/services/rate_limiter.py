"""Gemini API rate limiter — sliding window enforcer for requests per minute."""

import logging
import threading
import time
from collections import deque

from src.models.settings import get_setting

logger = logging.getLogger(__name__)

# Default: 3000 requests per minute
DEFAULT_RATE_LIMIT = 3000


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Tracks timestamps of recent API calls and blocks (sleeps) when the
    configured requests-per-minute ceiling is about to be exceeded.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()
        self._window = 60.0  # 1 minute sliding window

    def _get_limit(self) -> int:
        """Read the current rate limit from settings (or fall back to default)."""
        try:
            val = get_setting("gemini_rate_limit")
            if val is not None:
                return int(val)
        except Exception:
            pass
        return DEFAULT_RATE_LIMIT

    def acquire(self):
        """Block until a request slot is available, then record the request."""
        with self._lock:
            limit = self._get_limit()
            if limit <= 0:
                # No rate limiting
                return

            now = time.monotonic()

            # Purge timestamps outside the sliding window
            while self._timestamps and self._timestamps[0] <= now - self._window:
                self._timestamps.popleft()

            if len(self._timestamps) >= limit:
                # Window is full — sleep until the oldest entry expires
                sleep_until = self._timestamps[0] + self._window
                wait = sleep_until - now
                if wait > 0:
                    logger.debug(
                        "Rate limit reached (%d/%d RPM). Sleeping %.2fs",
                        len(self._timestamps), limit, wait,
                    )
                    # Release the lock while sleeping so other threads
                    # don't deadlock — they'll just queue up behind us.
                    self._lock.release()
                    try:
                        time.sleep(wait)
                    finally:
                        self._lock.acquire()

                    # Re-purge after sleeping
                    now = time.monotonic()
                    while self._timestamps and self._timestamps[0] <= now - self._window:
                        self._timestamps.popleft()

            self._timestamps.append(time.monotonic())


# Module-level singleton shared by all Gemini callers
_limiter = RateLimiter()


def acquire():
    """Acquire a rate-limit slot (blocking if necessary)."""
    _limiter.acquire()
