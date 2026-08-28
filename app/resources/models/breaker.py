"""A load failure that expires.

``ModelRegistry._failed`` was a set: a leg that failed to load once was dead
until the process restarted. That was right when the only way to fail was a
missing model file — re-attempting it on every query turns one slow request
into every request being slow — and it is wrong for the two failures that
actually happen now.

A load can fail because the card was full at that moment: it is shared with a
separately launched LLM whose footprint moves underneath us, and that clears on
its own. And once these weights are reachable over HTTP, the consumer's idea of
"this leg is dead" must not outlive the condition by hours.

So: still no retry storm, which was the whole point of the original set, but the
door reopens after ``ttl`` seconds.
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class CircuitBreaker:
    """Per-leg "do not retry yet", with a deadline.

    ``ttl`` is 5 minutes by default: long enough that a genuinely missing model
    is not re-downloaded on every request, short enough that a transient
    out-of-memory at startup does not cost the leg for the rest of the day.
    """

    def __init__(self, ttl: float = 300.0) -> None:
        self.ttl = ttl
        self._lock = threading.Lock()
        # leg -> (retry_at, reason)
        self._open: dict[str, tuple[float, str]] = {}

    def is_open(self, leg: str) -> bool:
        """True while ``leg`` must not be retried. Expiry is lazy — checked
        here rather than swept by a timer, because nothing else needs a timer
        and a stale entry costs nothing until somebody asks."""
        with self._lock:
            entry = self._open.get(leg)
            if entry is None:
                return False
            if time.monotonic() >= entry[0]:
                del self._open[leg]
                return False
            return True

    def trip(self, leg: str, reason: str) -> None:
        """Record that ``leg`` failed to load, and start the clock."""
        with self._lock:
            self._open[leg] = (time.monotonic() + self.ttl, reason)

    def reset(self, leg: Optional[str] = None) -> None:
        """Close the breaker for ``leg``, or for everything when omitted."""
        with self._lock:
            if leg is None:
                self._open.clear()
            else:
                self._open.pop(leg, None)

    def reason(self, leg: str) -> Optional[str]:
        """Why ``leg`` is closed off, for the message on the way out."""
        with self._lock:
            entry = self._open.get(leg)
            return entry[1] if entry else None

    def open_legs(self) -> list:
        """The legs currently closed off, oldest name first.

        Surfaced by ``retrieval_status()`` under the key ``failed``, which is
        the name it had as a set and the name the route and its test already
        read.
        """
        now = time.monotonic()
        with self._lock:
            for leg, (retry_at, _) in list(self._open.items()):
                if now >= retry_at:
                    del self._open[leg]
            return sorted(self._open)
