"""Types owned by the retrieval stack.

They live here rather than in ``lab.agent.models`` so this subpackage keeps its
one structural promise: it imports nothing from the rest of ``lab`` and can be
lifted into ``app/`` whole. ``lab.agent.models`` re-exports both names, so the
rest of the package still has a single place to import contracts from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(slots=True)
class Ranked:
    """One scored candidate. ``index`` points into the retriever's own docs."""

    index: int
    rrf: float
    ce: Optional[float]
    ce_prob: Optional[float]
    final: float
    ranks: dict[str, Optional[int]] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class Fact:
    """A raw source fact about a song or an artist, as stored."""

    row_id: int
    kind: Literal["song", "artist"]
    slug: str
    text: str
    source: str = ""
    category: str = ""
    ce_prob: Optional[float] = None
