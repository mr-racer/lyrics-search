"""Distractor selection — the false options in a round.

Two problems are solved here, and they pull in opposite directions.

The first is that CLAP over-weights vocals: a track's nearest neighbours in
CLAP space are overwhelmingly the same performer or the same record, so a slate
built from "the three closest" gives the answer away by elimination. Hence the
hard exclusion in :func:`shares_artist_or_album` (invariant I-3), applied
before any ranking, and distance measured in the six-dimensional sonic-axis
space rather than over the raw 512-dimensional vector. The axes are CLAP
projected onto interpretable poles, and voice identity dissolves in them —
``vocal_lead`` says "vocals lead", not "same voice".

The second is that a slate of three near-identical tracks is a coin flip. So
the slate is deliberately mixed: one genuinely confusable neighbour, one
era-and-genre match that sounds different, one familiar track. The player
should finish a round feeling they reasoned, not that they guessed.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §9.
"""
from __future__ import annotations

import math
import random as _random
from typing import Dict, List, Optional, Sequence, Set

from app.resources.clap_features import AXIS_NAMES

# How deep into the sorted neighbour list the "close" slot may reach. Rank 1 is
# too predictable across rounds; 15 keeps it confusable but varied.
NEAR_RANK_MAX = 15

# Half-width of the "same era" window, in years.
ERA_YEARS = 5


def _norm(value) -> str:
    return " ".join(str(value or "").lower().split())


def shares_artist_or_album(a: Dict, b: Dict) -> bool:
    """True when two tracks share a performer, a feature credit, or an album.

    Album equality alone counts, even across different performers. That is
    stricter than "same record" strictly requires — two unrelated artists can
    both have a *Greatest Hits* — but the cost is losing a candidate, while the
    cost of being wrong is handing the answer away. On a library of any size
    the pool can afford it.
    """
    slug_a, slug_b = _norm(a.get("primary_artist_slug")), _norm(b.get("primary_artist_slug"))
    if slug_a and slug_a == slug_b:
        return True

    artists_a = {_norm(x) for x in (a.get("artists") or []) if _norm(x)}
    artists_b = {_norm(x) for x in (b.get("artists") or []) if _norm(x)}
    if artists_a & artists_b:
        return True

    album_a, album_b = _norm(a.get("album")), _norm(b.get("album"))
    return bool(album_a) and album_a == album_b


def axis_distance(a: Dict, b: Dict, stats: Optional[Dict]) -> Optional[float]:
    """Euclidean distance in sonic-axis space, or None when either side lacks axes.

    Only the per-axis standard deviation from ``stats`` is applied: the mean
    cancels in a difference of two z-scores, so subtracting it would be busy
    work. An axis with a narrow spread therefore counts for more, which is the
    point — a small move along a tight axis is a big perceptual move.
    """
    axes_a, axes_b = a.get("sonic_axes"), b.get("sonic_axes")
    if not axes_a or not axes_b:
        return None

    std_map = (stats or {}).get("std") or {}
    total = 0.0
    for name in AXIS_NAMES:
        va, vb = axes_a.get(name), axes_b.get(name)
        if va is None or vb is None:
            continue
        delta = float(va) - float(vb)
        std = float(std_map.get(name) or 0.0)
        if std > 0.0:
            delta /= std
        total += delta * delta
    return math.sqrt(total)


def pick_distractors(
    *,
    answer: Dict,
    candidates: Sequence[Dict],
    n: int = 3,
    axis_stats: Optional[Dict] = None,
    familiar_ids: Optional[Set[str]] = None,
    rng=None,
) -> List[Dict]:
    """Build the false options for a round. Returns fewer than ``n`` if it must.

    Returning a short slate is deliberate: on a thin library the caller decides
    whether that is still a playable round, rather than this function inventing
    filler that violates I-3.
    """
    rng = rng or _random
    answer_id = answer.get("track_id")

    pool: List[Dict] = []
    seen: Set[str] = set()
    for cand in candidates:
        track_id = cand.get("track_id")
        if track_id == answer_id or track_id in seen:
            continue
        if shares_artist_or_album(answer, cand):
            continue
        seen.add(track_id)
        pool.append(cand)
    if not pool:
        return []

    chosen: List[Dict] = []
    taken: Set[str] = set()

    def take(cand: Optional[Dict]) -> None:
        if cand is not None and cand["track_id"] not in taken:
            taken.add(cand["track_id"])
            chosen.append(cand)

    def remaining() -> List[Dict]:
        # sorted() so a seeded rng reproduces the same slate regardless of the
        # order Qdrant happened to return candidates in.
        return sorted((c for c in pool if c["track_id"] not in taken),
                      key=lambda c: c["track_id"])

    # ── Slot 1: a genuinely confusable neighbour ──
    scored = [(axis_distance(answer, c, axis_stats), c) for c in remaining()]
    near = sorted(((d, c) for d, c in scored if d is not None),
                  key=lambda t: (t[0], t[1]["track_id"]))
    if near:
        take(rng.choice([c for _, c in near[:NEAR_RANK_MAX]]))

    # ── Slot 2: same era and genre, but sonically distant ──
    answer_year, answer_genre = answer.get("year"), _norm(answer.get("genre"))
    era_genre = [
        c for c in remaining()
        if (answer_year is None or c.get("year") is None
            or abs(int(c["year"]) - int(answer_year)) <= ERA_YEARS)
        and (not answer_genre or _norm(c.get("genre")) == answer_genre)
    ]
    if era_genre:
        by_distance = sorted(
            ((axis_distance(answer, c, axis_stats) or 0.0, c) for c in era_genre),
            key=lambda t: (-t[0], t[1]["track_id"]),
        )
        far_half = [c for _, c in by_distance][: max(1, len(by_distance) // 2)]
        take(rng.choice(far_half))

    # ── Slot 3: something the listener knows well ──
    rest = remaining()
    if rest:
        familiar = [c for c in rest if familiar_ids and c["track_id"] in familiar_ids]
        take(rng.choice(familiar or rest))

    # ── Fill: a thin library may not support the designed composition ──
    while len(chosen) < n:
        rest = remaining()
        if not rest:
            break
        take(rng.choice(rest))

    return chosen[:n]
