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

    def acquire(self, count: int = 1, status_callback=None):
        """Block until *count* request slots are available, then record them.

        Each item in a batch embedding counts as one request toward the RPM
        quota, so callers should pass count=len(batch).
        """
        with self._lock:
            limit = self._get_limit()
            if limit <= 0:
                return

            now = time.monotonic()

            # Purge timestamps outside the sliding window
            while self._timestamps and self._timestamps[0] <= now - self._window:
                self._timestamps.popleft()

            # Wait until enough capacity exists for all `count` items
            while len(self._timestamps) + count > limit:
                # Need oldest entries to expire to free capacity
                sleep_until = self._timestamps[0] + self._window
                wait = sleep_until - now
                if wait > 0:
                    msg = f"Rate limiter: waiting {wait:.0f}s for capacity..."
                    logger.debug(
                        "Rate limit reached (%d+%d/%d RPM). Sleeping %.2fs",
                        len(self._timestamps), count, limit, wait,
                    )
                    if status_callback:
                        try:
                            status_callback(msg)
                        except Exception:
                            pass
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

            # Record `count` timestamps
            ts = time.monotonic()
            for _ in range(count):
                self._timestamps.append(ts)


# Module-level singleton shared by all Gemini callers
_limiter = RateLimiter()


def acquire(count: int = 1, status_callback=None):
    """Acquire *count* rate-limit slots (blocking if necessary)."""
    _limiter.acquire(count=count, status_callback=status_callback)
