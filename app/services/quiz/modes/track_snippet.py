"""M1 "What's playing" — the main mode.

A snippet plays; the listener picks the track from four options. This is the
only mode with adaptive difficulty, and the difficulty comes from *how often
you actually play a track*, not from how short the snippet is. Shrinking the
clip is the cheap way to make a music quiz hard and it makes it annoying
instead; reaching further down your own tail makes it hard and interesting.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §7 M1.
"""
from __future__ import annotations

import uuid
from typing import Dict, Set, Tuple

from app.services.quiz.context import RoundContext, RoundSpec
from app.services.quiz.distractors import pick_distractors
from app.services.quiz.errors import NoRoundAvailable
from app.services.quiz.selection import pick_answer_track

KEY = "track_snippet"

OPTION_COUNT = 4

# The window a snippet may start in, as a fraction of the track. Not the
# opening — intros are recognised instantly and would collapse the difficulty
# curve — and not the tail, where fades and outros carry little information.
_START_LO = 0.15
_START_HI = 0.70

# Familiarity percentile at or above which a track counts as "one you know",
# used for the third distractor slot.
_FAMILIAR_PCT = 50.0


def pool_size(ctx: RoundContext) -> int:
    """How many tracks could be asked about — the I-5 gate reads this."""
    return sum(
        1 for track_id, count in ctx.plays.items()
        if count >= 1 and ctx.by_id(track_id) is not None
    )


def build(ctx: RoundContext, *, snippet_sec: int) -> RoundSpec:
    """Assemble one round, or refuse.

    Refusing is a first-class outcome: a library where every remaining
    candidate shares the answer's artist cannot produce an honest slate, and
    inventing filler would break I-3.
    """
    band = (float(ctx.skill.get("band_lo", 60.0)),
            float(ctx.skill.get("band_hi", 100.0)))
    answer_id = pick_answer_track(
        percentiles=ctx.percentiles, band=band, plays=ctx.plays,
        exclude=ctx.exclude, rng=ctx.rng,
    )
    if answer_id is None:
        raise NoRoundAvailable("no played track is eligible right now")

    answer = ctx.by_id(answer_id)
    if answer is None:
        raise NoRoundAvailable("the chosen track is missing from the snapshot")

    distractors = pick_distractors(
        answer=answer,
        candidates=ctx.tracks,
        n=OPTION_COUNT - 1,
        axis_stats=ctx.axis_stats,
        familiar_ids=_familiar_ids(ctx),
        rng=ctx.rng,
    )
    if len(distractors) < OPTION_COUNT - 1:
        raise NoRoundAvailable("not enough candidates survive the artist filter")

    options = [_option(track) for track in [answer] + distractors]
    correct_option_id = options[0]["option_id"]
    ctx.rng.shuffle(options)

    return RoundSpec(
        mode=KEY,
        track_id=answer_id,
        options=options,
        correct_option_id=correct_option_id,
        start_sec=_start_point(answer, snippet_sec, ctx.rng),
        length_sec=float(snippet_sec),
    )


def score(spec: Dict, answer: Dict) -> Tuple[bool, float]:
    """Binary for this mode; graded modes (M3) return partial credit instead."""
    chosen = (answer or {}).get("option_id")
    correct = bool(chosen) and chosen == spec.get("correct_option_id")
    return correct, 100.0 if correct else 0.0


# ── internals ────────────────────────────────────────────────────────────────

def _familiar_ids(ctx: RoundContext) -> Set[str]:
    return {track_id for track_id, pct in ctx.percentiles.items()
            if pct >= _FAMILIAR_PCT}


def _option(track: Dict) -> Dict:
    """One rendered option.

    Carries no ``track_id``: the option list travels to the client before the
    answer is known, and an id there would let anyone look the answer up.
    """
    return {
        "option_id": uuid.uuid4().hex[:12],
        "title": track.get("title_display") or track.get("title") or "—",
        "artist": track.get("artist") or "—",
        "cover_art_path": track.get("cover_art_path"),
    }


def _start_point(track: Dict, length_sec: float, rng) -> float:
    """Pick where the snippet starts, never running past the end of the file."""
    duration = float(track.get("duration_sec") or 0.0)
    if duration <= 0.0:
        return 0.0
    low = _START_LO * duration
    # On a short track the "70%" ceiling can sit past the last playable start,
    # so clamp to whatever leaves room for the whole snippet.
    high = min(_START_HI * duration, max(low, duration - float(length_sec)))
    if high <= low:
        return round(low, 3)
    return round(rng.uniform(low, high), 3)
