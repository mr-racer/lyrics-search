"""Progress events, in the shape the production stream already speaks.

``app/services/assistant/service.py`` normalises everything into
``{"type": "status", "stage": ..., "human": ..., **fields}`` and ships it as
NDJSON. Emitting the same frames here means the eventual port needs no
translation layer, and in the meantime a notebook can just print them.

Two callback styles, like in prod: ``emit`` for async callers and ``on_status``
for anything that might be running inside ``asyncio.to_thread``. The sync one
hops back onto the loop rather than touching the queue from a worker thread.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class EventSink:
    """Collects events and optionally forwards them to a callback.

    ``history`` keeps everything for post-mortem in the notebook — the runs are
    short and the frames are small, so there is nothing to bound.
    """

    def __init__(self, callback: Optional[Callable[[dict], Any]] = None,
                 *, verbose: bool = False):
        self.callback = callback
        self.verbose = verbose
        self.history: list[dict] = []
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    # ── emitting ──────────────────────────────────────────────────────────

    def put(self, stage: str, **fields: Any) -> None:
        """Record one event. Safe from the event loop thread."""
        frame = {"type": "status", "stage": stage, **fields}
        self.history.append(frame)
        if self.verbose:
            detail = " ".join(f"{k}={v!r}" for k, v in fields.items())
            logger.info("[%s] %s", stage, detail)
            print(f"· {stage}" + (f"  {detail}" if detail else ""))
        if self.callback is not None:
            try:
                self.callback(frame)
            except Exception:  # noqa: BLE001 — a broken UI must not stop a run
                logger.debug("[events] callback failed", exc_info=True)

    async def emit(self, stage: str, **fields: Any) -> None:
        self.put(stage, **fields)

    def on_status(self, stage: str, **fields: Any) -> None:
        """Sync form, safe to call from a worker thread."""
        if self._loop is None:
            self.put(stage, **fields)
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self.put(stage, **fields)
        else:
            self._loop.call_soon_threadsafe(lambda: self.put(stage, **fields))

    # ── reading back ──────────────────────────────────────────────────────

    def of(self, stage: str) -> list[dict]:
        return [f for f in self.history if f.get("stage") == stage]

    def summary(self) -> str:
        """One line per event — what you actually want after a notebook run."""
        lines = []
        for f in self.history:
            fields = {k: v for k, v in f.items() if k not in ("type", "stage")}
            lines.append(f"{f['stage']:<16} " +
                         " ".join(f"{k}={v!r}" for k, v in fields.items()))
        return "\n".join(lines)
