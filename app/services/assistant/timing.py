"""Where a run's wall clock actually went.

Pure instrumentation — nothing here changes what the assistant does. It exists
because the answer to "why did that take two minutes?" is never guessable: the
search phase is mostly a paced sleep, the fetch phase is mostly waiting on hosts
that will refuse, and the GPU work is the part that feels expensive and is not.

Spans nest and overlap freely (the structured read runs concurrently with the
open-web search), so the lanes deliberately do NOT sum to the wall clock. That is
the overlap showing up, not a bug in the report.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


class Timings:
    """Named spans, accumulated across one run."""

    def __init__(self):
        self.spans: dict = {}
        self.counts: dict = {}
        self.total: float = 0.0

    def reset(self) -> None:
        self.spans.clear()
        self.counts.clear()
        self.total = 0.0

    @contextmanager
    def measure(self):
        """Wrap a whole run: clears the previous one and records the total."""
        self.reset()
        started = time.monotonic()
        try:
            yield self
        finally:
            self.total = time.monotonic() - started

    @contextmanager
    def span(self, name: str):
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            self.spans[name] = self.spans.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    def report(self) -> str:
        if not self.spans:
            return f"total {self.total:.1f}s"
        width = max(len(n) for n in self.spans)
        lines = [f"total {self.total:.1f}s"]
        for name, seconds in sorted(self.spans.items(), key=lambda kv: -kv[1]):
            share = (100 * seconds / self.total) if self.total else 0.0
            lines.append(f"  {name:<{width}}  {seconds:6.1f}s  {share:4.0f}%  "
                         f"x{self.counts[name]}")
        return "\n".join(lines)
