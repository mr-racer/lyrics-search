"""Progress events, in the shape the NDJSON stream already speaks.

``routes/assistant.py`` ships ``{"type": "status", "stage": ..., "human": ...,
**fields}`` lines and ``service.EventSink`` is what normalises them. This is the
producer side: the pipeline calls ``put("fetch", count=3)`` and the frame lands
in that queue, hopping back onto the event loop when the caller is a worker
thread (most of the search phase runs through ``asyncio.to_thread``).

``history`` keeps every frame. The runs are short and the frames are small, so
there is nothing to bound, and after a run it is the one place that says what
actually happened in what order.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class AgentSink:
    """Collects events and forwards them to the stream's sink, if there is one.

    ``forward`` is ``service.EventSink.on_status`` — a sync callable taking one
    dict, already safe to call from any thread. When it is None (the
    non-streaming endpoint, or a test) the frames are still recorded.
    """

    def __init__(self, forward: Optional[Callable] = None, *,
                 verbose: bool = False):
        self.forward = forward
        self.verbose = verbose
        self.history: list = []
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def put(self, stage: str, **fields: Any) -> None:
        """Record one event and pass it on. Safe from any thread."""
        frame = {"type": "status", "stage": stage, **fields}
        self.history.append(frame)
        if self.verbose:
            detail = " ".join(f"{k}={v!r}" for k, v in fields.items())
            logger.info("[%s] %s", stage, detail)
        if self.forward is None:
            return
        try:
            self.forward(frame)
        except Exception:  # noqa: BLE001 — a broken UI must not stop a run
            logger.debug("[events] forward failed", exc_info=True)

    async def emit(self, stage: str, **fields: Any) -> None:
        self.put(stage, **fields)

    def of(self, stage: str) -> list:
        return [f for f in self.history if f.get("stage") == stage]

    def summary(self) -> str:
        """One line per event — what you want in the log after a run."""
        lines = []
        for f in self.history:
            fields = {k: v for k, v in f.items() if k not in ("type", "stage")}
            lines.append(f"{f['stage']:<16} " +
                         " ".join(f"{k}={v!r}" for k, v in fields.items()))
        return "\n".join(lines)
