"""Counters for what the legs actually did, as opposed to what is loaded.

``retrieval_status()`` answered "is the stack up?" with three booleans meaning
"the weights are resident", and a leg whose every encode failed still read as
healthy — for hours, with worse answers as the only symptom. The failure and
OOM-retry counters were added for that reason and simply move here.

The new one is ``degradations``: how often a caller decided to carry on without
a leg. That decision is legitimate — ranking on two signals beats not answering
— but until now it was invisible, and an invisible degradation is exactly the
failure this change exists to surface. A run where the sparse leg was dropped
92 times is a different run from one where it was dropped none, and both used to
look identical from outside.

Incremented from worker threads (``asyncio.to_thread``), hence the lock: ``+=``
on a dict value is a read and a write, and losing counts would undermine the one
thing these numbers are for.
"""

from __future__ import annotations

import threading


class ModelStats:
    """Process-wide counters. One instance, :data:`STATS`, is the live one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._encode_failures: dict[str, int] = {}
        self._oom_retries: dict[str, int] = {}
        self._degradations: dict[str, int] = {}

    # ── recording ──

    def encode_failed(self, leg: str) -> None:
        """One run lost ``leg`` on the encode path."""
        self._bump(self._encode_failures, leg)

    def oom_retry(self, leg: str) -> None:
        """One run KEPT ``leg`` by shrinking its batch.

        A load signal rather than a fault, and counted separately for that
        reason: it says the card is tight, not that anything broke.
        """
        self._bump(self._oom_retries, leg)

    def degraded(self, leg: str, where: str) -> None:
        """A caller at ``where`` continued without ``leg``.

        Keyed by both so the counter names the DECISION, not just the leg: the
        retriever dropping a signal and a Wikipedia gate refusing to judge are
        the same cause and very different consequences.
        """
        self._bump(self._degradations, f"{leg}/{where}")

    def _bump(self, bucket: dict, key: str, n: int = 1) -> None:
        with self._lock:
            bucket[key] = bucket.get(key, 0) + n

    # ── reading ──

    def snapshot(self) -> dict:
        """A copy, safe to serialise. Empty buckets stay empty rather than
        being pre-filled with zeros for every leg — "never happened" and
        "happened zero times since the last reset" are the same thing here, and
        a short dict reads better in the one place this is surfaced."""
        with self._lock:
            return {
                "encode_failures": dict(self._encode_failures),
                "oom_retries": dict(self._oom_retries),
                "degradations": dict(self._degradations),
            }

    def reset(self) -> None:
        """Tests only."""
        with self._lock:
            self._encode_failures.clear()
            self._oom_retries.clear()
            self._degradations.clear()


#: The live counters. Everything writes here; ``GET /search/models/loaded`` reads it.
STATS = ModelStats()
