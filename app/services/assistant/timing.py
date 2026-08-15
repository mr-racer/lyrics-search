"""Where a run's time actually went.

Built to answer one question — what to optimise next — so every decision here
serves that and nothing else.

**Spans are flat.** A span never contains another span. Nesting would silently
double-count the inner one and the shares would still add up to something
plausible, which is the worst kind of wrong number: it looks like a measurement.
The guard is a depth counter that names both spans in a warning, and the
``unaccounted`` row goes negative when it happens, so a mistake is visible in the
report itself rather than only in a log nobody reads.

**Spans measure wall time, not CPU time.** ``fetch`` downloading eight pages
concurrently is one span of three seconds, not eight spans of three. That is what
the user waits for, and it keeps the shares summing to 100%. The cost is that
this cannot tell you a stage is under-parallelised — for that, compare its wall
time against how many items it handled.

**Nothing hides.** ``wall`` is measured around the whole run, and the difference
between it and the sum of the spans gets its own row. An instrumentation gap
shows up as a large ``unaccounted`` instead of as a stage that looks cheaper than
it is.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from time import perf_counter

logger = logging.getLogger(__name__)

# Span-name prefix -> lane. The lane totals are the actual answer to "where do we
# optimise": a run that is 70% `llm` wants a smaller model or fewer calls, one
# that is 70% `net` wants concurrency, and they are different projects.
LANES = {
    "llm": "llm",
    "search": "net", "fetch": "net", "discography": "net",
    "rerank": "gpu", "index": "gpu", "select": "gpu", "facts": "gpu",
    "structured": "cpu", "resolve": "cpu",
}

LANE_NAMES = {"llm": "LLM", "net": "net", "gpu": "GPU", "cpu": "CPU"}


def lane_for(name: str) -> str:
    """The lane a span belongs to, from its prefix. Unknown prefixes get ''."""
    return LANES.get(name.split(".", 1)[0], "")


class Timings:
    """The spans of one run, and the report over them.

    ``clock`` is injectable so the tests can assert on exact numbers instead of
    sleeping — a timing test that sleeps is slow and flaky at once.
    """

    def __init__(self, clock=perf_counter):
        self._clock = clock
        self.spans: list = []          # (name, lane, seconds)
        self.wall = 0.0
        self._stack: list = []

    def reset(self) -> None:
        self.spans.clear()
        self.wall = 0.0
        self._stack.clear()

    @contextmanager
    def measure(self):
        """Wrap the whole run: clears the previous one and takes wall time.

        Resetting here rather than in the constructor is what makes a second
        ``run()`` report its own numbers instead of the two runs added together.
        """
        self.reset()
        started = self._clock()
        try:
            yield self
        finally:
            self.wall = self._clock() - started

    @contextmanager
    def span(self, name: str, lane: str = ""):
        if self._stack:
            logger.warning(
                "[timing] %r is nested inside %r — both will be counted and the "
                "shares will exceed the wall clock. Spans must be leaves.",
                name, self._stack[-1])
        self._stack.append(name)
        started = self._clock()
        try:
            yield
        finally:
            self._stack.pop()
            self.spans.append((name, lane or lane_for(name),
                               self._clock() - started))

    # ── reading back ──────────────────────────────────────────────────────

    def rows(self) -> list:
        """One row per distinct span name, slowest first."""
        agg: dict = {}
        for name, lane, seconds in self.spans:
            row = agg.setdefault(name, {"stage": name, "lane": lane,
                                        "calls": 0, "total": 0.0})
            row["calls"] += 1
            row["total"] += seconds
        for row in agg.values():
            row["mean"] = row["total"] / row["calls"]
            row["share"] = row["total"] / self.wall if self.wall else 0.0
        return sorted(agg.values(), key=lambda r: -r["total"])

    def by_lane(self) -> dict:
        out: dict = {}
        for _, lane, seconds in self.spans:
            if lane:
                out[lane] = out.get(lane, 0.0) + seconds
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def report(self) -> str:
        rows = self.rows()
        if not rows and not self.wall:
            return "timings empty — nothing ran"

        measured = sum(r["total"] for r in rows)
        rest = self.wall - measured
        rule = "─" * 62
        out = [f"{'stage':<22}{'calls':>8}{'total':>8}{'mean':>9}{'share':>6}"
               f"  lane", rule]
        for r in rows:
            out.append(f"{r['stage']:<22}{r['calls']:>8}{r['total']:>7.1f}s"
                       f"{r['mean']:>8.1f}s{100 * r['share']:>5.0f}%"
                       f"  {LANE_NAMES.get(r['lane'], '')}")
        # A negative remainder can only mean overlapping spans; say so rather
        # than printing a minus sign and letting it be read as rounding.
        note = "  ⚠ spans overlap" if rest < -0.05 else ""
        out.append(f"{'unaccounted':<22}{'':>8}{rest:>7.1f}s{'':>8}"
                   f"{100 * rest / self.wall if self.wall else 0:>5.0f}%{note}")
        out.append(rule)
        out.append(f"{'total':<22}{'':>8}{self.wall:>7.1f}s")

        lanes = self.by_lane()
        if lanes and self.wall:
            out.append("")
            out.append("by lane:  " + "   ".join(
                f"{LANE_NAMES.get(k, k)} {100 * v / self.wall:.0f}%"
                for k, v in lanes.items()))
        return "\n".join(out)
