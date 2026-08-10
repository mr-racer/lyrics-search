"""Collapsing near-duplicate passages out of a context pack.

The failure this exists for: ask about an artist and the open web returns the
same syndicated bio on five hosts. Every copy scores alike because every copy
IS alike, so the ranked top-10 is one paragraph repeated five times, and the
model is handed five slots that say one thing.

Exact copies are already gone by the time anything gets here — the pipeline
hashes chunk bodies at index time. What survives that is the interesting case:
the same text with a citation marker stripped, a date written the other way
round, one sentence more.

**This never deletes anything.** It runs over the ranked candidates for ONE
question and decides which of them go in front of the model this time. A
passage dropped here is still in the index, still retrievable, and the next
question may well pick it. That is the whole answer to "what if it collapses
things it should not" — the blast radius is one context pack.

The second guard is the rule itself: a pair is a duplicate only when EVERY
signal that has an opinion says so, and the signals fail differently. Dense
embeddings read meaning, so two paragraphs about the same event look nearly
identical to them even when one carries a fact the other lacks. Learned-sparse
reads (expanded) terms, so it separates them again. Requiring both to agree
means neither model's characteristic mistake can drop a passage on its own.

Nothing here imports torch: the caller hands in similarity matrices as plain
lists, which is also what makes the policy testable without a GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Duplicate:
    """One candidate that did not take a slot, and the one that did.

    Positions index the CANDIDATE list handed to :func:`pick_diverse`, not the
    retriever's documents — the caller owns that mapping.
    """

    index: int
    twin: int
    sims: dict[str, float] = field(default_factory=dict)
    # True when ``index`` is the one that was already in the pack and got
    # displaced: same content, and the newcomer had more of it.
    replaced: bool = False


@dataclass(slots=True)
class Selection:
    kept: list[int] = field(default_factory=list)
    duplicates: list[Duplicate] = field(default_factory=list)


def pair_similarity(sims: Optional[dict], i: int, j: int) -> dict[str, float]:
    """Every signal's opinion about one pair, skipping the ones that have none."""
    out: dict[str, float] = {}
    for name, matrix in (sims or {}).items():
        try:
            value = matrix[i][j]
        except (IndexError, KeyError, TypeError):
            continue
        if value is None:
            continue
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def is_duplicate(pair: dict[str, float],
                 thresholds: Optional[dict[str, float]]) -> bool:
    """Unanimity among the signals that were actually computed.

    A signal with no threshold configured is ignored rather than trusted, and a
    pair nobody scored is not a duplicate — the default answer to "we could not
    tell" is to keep the passage.
    """
    scored = {n: v for n, v in pair.items() if n in (thresholds or {})}
    if not scored:
        return False
    return all(v >= thresholds[n] for n, v in scored.items())


def pick_diverse(lengths: list[int], sims: Optional[dict], *,
                 thresholds: Optional[dict[str, float]],
                 limit: int, prefer_longer: float = 1.2) -> Selection:
    """Fill ``limit`` slots with candidates that do not repeat each other.

    Candidates arrive best-first and that order is preserved: this changes WHO
    is in the pack, never the order of the ones that stay. When a later
    candidate duplicates one already picked, the longer of the two takes the
    slot — but only if it is longer by ``prefer_longer``, so a rewording does
    not outrank a better-scoring passage over three extra characters.

    Scanning continues past the point where the pack is full, because a longer
    copy of something already in it is still worth swapping in.
    """
    kept: list[int] = []
    duplicates: list[Duplicate] = []
    if limit <= 0:
        return Selection(kept=kept, duplicates=duplicates)

    for pos in range(len(lengths)):
        twin_slot = None
        twin_sims: dict[str, float] = {}
        for slot, chosen in enumerate(kept):
            pair = pair_similarity(sims, chosen, pos)
            if is_duplicate(pair, thresholds):
                twin_slot, twin_sims = slot, pair
                break

        if twin_slot is None:
            if len(kept) < limit:
                kept.append(pos)
            continue

        incumbent = kept[twin_slot]
        if lengths[pos] > lengths[incumbent] * prefer_longer:
            kept[twin_slot] = pos
            duplicates.append(Duplicate(index=incumbent, twin=pos,
                                        sims=twin_sims, replaced=True))
        else:
            duplicates.append(Duplicate(index=pos, twin=incumbent,
                                        sims=twin_sims))
    return Selection(kept=kept, duplicates=duplicates)


def duplicate_report(labels: list[str], lengths: list[int],
                     sims: Optional[dict], *,
                     thresholds: Optional[dict[str, float]] = None,
                     floor: float = 0.8, limit: int = 40) -> str:
    """Every pair any signal scored at or above ``floor``, most similar first.

    A calibration tool, not part of the pipeline. Thresholds like these cannot
    be reasoned out — the distributions depend on the corpus and on both
    models — so the way to set them is to look at what the numbers actually are
    for pairs you can read, and put the line between the ones that repeat each
    other and the ones that merely share a subject.

    ``floor`` is deliberately below any sane threshold: the pairs just under
    the cut are the informative ones.
    """
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pair = pair_similarity(sims, i, j)
            if not pair or max(pair.values()) < floor:
                continue
            rows.append((max(pair.values()), i, j, pair))
    if not rows:
        return f"no pair scored {floor:.2f} on any signal"

    rows.sort(key=lambda r: -r[0])
    out = []
    for _, i, j, pair in rows[:limit]:
        verdict = "DUPLICATE" if is_duplicate(pair, thresholds) else "kept both"
        scores = "  ".join(f"{n}={v:.3f}" for n, v in sorted(pair.items()))
        out.append(f"{verdict:<10} {scores}")
        out.append(f"           [{i}] {lengths[i]:>5}ch  {labels[i][:80]}")
        out.append(f"           [{j}] {lengths[j]:>5}ch  {labels[j][:80]}")
    return "\n".join(out)
