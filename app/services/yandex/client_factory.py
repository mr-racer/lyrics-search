"""Build a yandex_music.Client (authorized or anonymous) + a shared throttle.

The unofficial Yandex API is anti-bot sensitive, so every outbound call is
paced by a process-wide :class:`Throttle` (default 1 request / 0.5 s). Download
concurrency is bounded separately by the import runner's thread pool (2 workers);
the throttle bounds *request rate* on top of that.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Spacing between Yandex requests (seconds). Spec §7.1: 0.5 s / track.
DEFAULT_MIN_INTERVAL = 0.5


class Throttle:
    """Thread-safe minimum-interval limiter shared across download workers."""

    def __init__(self, min_interval: float = DEFAULT_MIN_INTERVAL):
        self._min = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        """Block until at least ``min_interval`` has passed since the last call."""
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self._min:
                time.sleep(self._min - delta)
            self._last = time.monotonic()


def build_client(token: str | None):
    """Return an initialized ``yandex_music.Client``.

    ``token`` None → anonymous client (catalog reads / search still work, used by
    the metadata-enrichment fallback). A bad/expired token raises on ``init()``;
    callers decide whether to fall back to anonymous.
    """
    from yandex_music import Client

    client = Client(token) if token else Client()
    client.init()
    return client
