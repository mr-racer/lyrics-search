"""M3 «Слепой год» — name the release year from the sound.

The scoring is the design. A year sits on a line, so binary right/wrong makes
missing by one feel exactly like missing by twenty — which is both untrue and
discouraging, and it throws away the only interesting signal the mode produces.
Partial credit turns the round into a calibration reading: over a few dozen
answers you learn which decades you actually hear.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §7 M3.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.services.quiz.context import RoundContext, RoundSpec
from app.services.quiz.errors import NoRoundAvailable
from app.services.quiz.snippet import start_point

KEY = "blind_year"

HAS_AUDIO = True

# The answer is typed on a scale rather than picked from a list, so the client
# renders a year control instead of option keys.
INPUT_KIND = "year"

# Years off → points. Not a curve in code: a table, because these are product
# judgements about how forgiving the mode should feel, and they should be
# readable and arguable as such.
_SCORE_TABLE = ((0, 100.0), (1, 80.0), (2, 60.0), (4, 30.0))

# A round counts as "got it" at two years out. Wider would make the dossier's
# per-decade accuracy meaningless; narrower punishes a genuinely good ear.
CORRECT_AT = 60.0


def pool_size(ctx: RoundContext) -> int:
    """Played tracks that carry a release year — the rest cannot be asked."""
    return len(_candidates(ctx, respect_exclude=False))


def build(ctx: RoundContext, *, snippet_sec: int) -> RoundSpec:
    candidates = _candidates(ctx)
    if not candidates:
        # Fall back to ignoring anti-repeat rather than refusing: on a library
        # with few dated tracks the exclusion window empties the pool long
        # before the material is actually exhausted.
        candidates = _candidates(ctx, respect_exclude=False)
    if not candidates:
        raise NoRoundAvailable("no played track carries a release year")

    answer = ctx.rng.choice(sorted(candidates, key=lambda t: t["track_id"]))
    years = [int(t["year"]) for t in ctx.tracks if _year_of(t) is not None]

    return RoundSpec(
        mode=KEY,
        track_id=answer["track_id"],
        options=[],
        correct_option_id="",
        start_sec=start_point(answer, snippet_sec, ctx.rng),
        length_sec=float(snippet_sec),
        reveal={"year": int(answer["year"])},
        # The scale needs bounds. The library's own span is the honest choice
        # and is identical for every round, so it reveals nothing.
        meta={"year_min": min(years), "year_max": max(years)},
    )


def score(spec: Dict, answer: Dict) -> Tuple[bool, float]:
    """Graded by distance. Anything unparseable is a miss, never an error."""
    truth = (spec.get("reveal") or {}).get("year")
    guess = (answer or {}).get("year")
    try:
        distance = abs(int(truth) - int(guess))
    except (TypeError, ValueError):
        return False, 0.0

    points = 0.0
    for limit, value in _SCORE_TABLE:
        if distance <= limit:
            points = value
            break
    return points >= CORRECT_AT, points


# ── internals ────────────────────────────────────────────────────────────────

def _year_of(track: Dict) -> Optional[int]:
    try:
        year = int(track.get("year"))
    except (TypeError, ValueError):
        return None
    return year if year > 0 else None


def _candidates(ctx: RoundContext, *, respect_exclude: bool = True) -> List[Dict]:
    return [
        track for track in ctx.tracks
        if _year_of(track) is not None
        and ctx.plays.get(track.get("track_id"), 0) >= 1
        and not (respect_exclude and track.get("track_id") in ctx.exclude)
    ]
