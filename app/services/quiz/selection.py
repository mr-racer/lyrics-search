"""Answer-track selection: familiarity, skill, and the adaptive band.

Difficulty adapts on how often the listener actually plays a track, not on how
obscure the track is in general. The quiz is about *your own* collection, so
your play history is the only honest difficulty knob available — and the one
the listener recognises when it moves.

Everything here is a pure function over plain dicts. Nothing touches Qdrant,
SQLite or the clock, which is what makes the difficulty curve testable rather
than something you can only judge by playing it.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §8.
"""
from __future__ import annotations

import math
import random as _random
from typing import Dict, Optional, Set, Tuple

Band = Tuple[float, float]

# Where a listener starts, and where they return when struggling: the tracks
# they play most. A first round has to be winnable.
STARTER_BAND: Band = (60.0, 100.0)

_HIGH_BAND: Band = (10.0, 45.0)   # confident — reach into the rarely played
_MID_BAND: Band = (35.0, 70.0)    # the mid-tail

# Answers required before adaptivity switches on at all.
WARMUP_ANSWERS = 5

# Below this many tracks percentiles carry no information, so the whole library
# is fair game and adaptivity is off (spec §16 R-2).
MIN_LIBRARY_FOR_ADAPTIVITY = 200

# Consecutive answers pointing at a different band before the band actually
# moves. Without it one lucky guess lurches the difficulty.
BAND_SWITCH_STREAK = 3

SKILL_ALPHA = 0.25

# Decay constant for "when did you last play this", in days.
RECENCY_TAU_DAYS = 120.0

# How far the band opens per retry when nothing inside it qualifies.
_WIDEN_STEP = 10.0

_SECONDS_PER_DAY = 86400.0


def update_skill(skill: float, correct: bool, alpha: float = SKILL_ALPHA) -> float:
    """EWMA of correctness. Stays in [0, 1] because both endpoints do."""
    target = 1.0 if correct else 0.0
    return skill + alpha * (target - skill)


def band_for_skill(skill: float, n_answered: int, library_size: int) -> Band:
    """The percentile band this skill level should be answering in.

    The three bands overlap on purpose: a listener drifting between them does
    not fall off a cliff, they just see a different slice of the same tail.
    """
    if library_size < MIN_LIBRARY_FOR_ADAPTIVITY:
        return (0.0, 100.0)
    if n_answered < WARMUP_ANSWERS:
        return STARTER_BAND
    if skill >= 0.85:
        return _HIGH_BAND
    if skill >= 0.60:
        return _MID_BAND
    return STARTER_BAND


def next_band(
    current: Band, *, skill: float, n_answered: int,
    out_of_band: int, library_size: int,
) -> Tuple[Band, int]:
    """Apply hysteresis: return the band to use next and the new streak count.

    A band change needs ``BAND_SWITCH_STREAK`` consecutive answers pointing
    elsewhere. Any answer pointing back at the current band clears the streak,
    so difficulty responds to a trend rather than to the last round.
    """
    target = band_for_skill(skill, n_answered, library_size)
    if target == current:
        return current, 0
    streak = out_of_band + 1
    if streak >= BAND_SWITCH_STREAK:
        return target, 0
    return current, streak


def familiarity_percentiles(
    plays: Dict[str, int],
    last_played: Dict[str, Optional[float]],
    now: float,
) -> Dict[str, float]:
    """Rank the library by how familiar each track is: 100 = most familiar.

    ``log1p(plays)`` because the tenth play tells you far less than the second,
    and a recency weight that bottoms out at 0.5 rather than 0 — a record you
    wore out years ago is still one you know, so it must never sink below a
    track you have played once.
    """
    raw: Dict[str, float] = {}
    for track_id, count in plays.items():
        stamp = last_played.get(track_id)
        if stamp is None:
            weight = 0.5
        else:
            days = max(0.0, (now - stamp) / _SECONDS_PER_DAY)
            weight = 0.5 + 0.5 * math.exp(-days / RECENCY_TAU_DAYS)
        raw[track_id] = math.log1p(max(0, count)) * weight

    n = len(raw)
    if n == 0:
        return {}
    if n == 1:
        return {next(iter(raw)): 100.0}

    ordered = sorted(raw.items(), key=lambda kv: kv[1])
    percentiles: Dict[str, float] = {}
    i = 0
    while i < n:
        # Tied scores share a percentile — otherwise two identically-played
        # tracks land in different difficulty bands for no reason.
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        value = 100.0 * ((i + j) / 2.0) / (n - 1)
        for k in range(i, j + 1):
            percentiles[ordered[k][0]] = value
        i = j + 1
    return percentiles


def pick_answer_track(
    *,
    percentiles: Dict[str, float],
    band: Band,
    plays: Dict[str, int],
    exclude: Set[str],
    rng=None,
) -> Optional[str]:
    """Choose the track a round will ask about, or None when nothing qualifies.

    Never returns a track with zero plays: asking about something the listener
    has never once played is not a game, it is an accusation (spec §7 M1).
    When the band holds no eligible track it widens rather than failing — an
    exhausted band is a normal state after enough rounds, not an error.
    """
    rng = rng or _random
    eligible = [
        track_id for track_id in percentiles
        if plays.get(track_id, 0) >= 1 and track_id not in exclude
    ]
    if not eligible:
        return None

    lo, hi = band
    while True:
        pool = [t for t in eligible if lo <= percentiles[t] <= hi]
        if pool:
            # sorted() so a seeded rng gives a reproducible pick regardless of
            # dict ordering.
            return rng.choice(sorted(pool))
        if lo <= 0.0 and hi >= 100.0:
            return None
        lo = max(0.0, lo - _WIDEN_STEP)
        hi = min(100.0, hi + _WIDEN_STEP)
