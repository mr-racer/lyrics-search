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


def margin(pair: dict[str, float],
           thresholds: Optional[dict[str, float]]) -> float:
    """How far a pair is from being called a duplicate. ``>= 0`` means it is.

    The distance to the CUT, not the similarity: with a rule that needs every
    signal to agree, a pair is only as close as its weakest signal, and 0.99
    dense next to 0.40 sparse is not close at all. Ranking by this is what puts
    the informative pairs — the ones just under the line — at the top of the
    report.
    """
    scored = {n: v for n, v in pair.items() if n in (thresholds or {})}
    if not scored:
        return min(pair.values()) if pair else 0.0
    return min(v - thresholds[n] for n, v in scored.items())


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1,
                       max(0, int(round(q * (len(ordered) - 1)))))]


def duplicate_report(labels: list[str], lengths: list[int],
                     sims: Optional[dict], *,
                     thresholds: Optional[dict[str, float]] = None,
                     floor: float = 0.8, top: int = 10,
                     limit: int = 40) -> str:
    """What the similarity numbers actually look like on this corpus.

    A calibration tool, not part of the pipeline. Thresholds like these cannot
    be reasoned out — the scale depends on the corpus and on both models, and
    two embedding families disagree about what "0.9" means — so the way to set
    them is to read pairs you can judge by eye next to the numbers they got.

    Three things, in order of usefulness:

    * the DISTRIBUTION per signal, which is the part that says whether a
      threshold is in the right postcode at all. A corpus whose closest pair
      scores 0.87 dense cannot produce a duplicate at 0.95, and no list of
      pairs makes that as obvious as one line of quantiles;
    * every pair that clears the thresholds;
    * the ``top`` closest pairs regardless, because when nothing clears them
      the near-misses are the entire signal. An empty report proves nothing —
      it looks the same whether the corpus has no duplicates or the threshold
      is in the wrong place.

    ``floor`` only widens the listing beyond ``top``; it never hides the
    closest pairs.
    """
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pair = pair_similarity(sims, i, j)
            if pair:
                pairs.append((margin(pair, thresholds), i, j, pair))
    if not pairs:
        return "no signal scored a single pair — are there vectors in the index?"

    out = [f"{len(pairs)} pairs over {len(labels)} passages"]
    for name in sorted({n for _, _, _, p in pairs for n in p}):
        values = [p[name] for _, _, _, p in pairs if name in p]
        cut = (thresholds or {}).get(name)
        out.append(f"  {name:<7} p50={_quantile(values, 0.5):.3f}  "
                   f"p90={_quantile(values, 0.9):.3f}  "
                   f"p99={_quantile(values, 0.99):.3f}  "
                   f"max={max(values):.3f}" +
                   (f"   порог {cut:.2f}" if cut is not None else ""))

    pairs.sort(key=lambda r: -r[0])
    duplicates = [r for r in pairs if r[0] >= 0]
    shown = duplicates or []
    for row in pairs[:max(top, len(shown))]:
        if row not in shown:
            shown.append(row)
    extra = [r for r in pairs if r not in shown
             and max(r[3].values()) >= floor][:max(0, limit - len(shown))]
    shown = (shown + extra)[:limit]

    out.append("")
    out.append(f"{len(duplicates)} pair(s) cleared every threshold"
               if duplicates else
               "nothing cleared every threshold — the closest pairs:")
    for m, i, j, pair in shown:
        verdict = "DUPLICATE" if m >= 0 else "kept both"
        scores = "  ".join(f"{n}={v:.3f}" for n, v in sorted(pair.items()))
        out.append(f"{verdict:<10} margin={m:+.3f}   {scores}")
        out.append(f"           [{i}] {lengths[i]:>5}ch  {labels[i][:80]}")
        out.append(f"           [{j}] {lengths[j]:>5}ch  {labels[j][:80]}")
    return "\n".join(out)
