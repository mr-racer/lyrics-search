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
from app.services.quiz.snippet import start_point

KEY = "track_snippet"

# This round IS a listening round; the client draws a play key for it.
HAS_AUDIO = True

OPTION_COUNT = 4


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
        start_sec=start_point(answer, snippet_sec, ctx.rng),
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
    """One rendered option — text only, and deliberately so.

    Carries no ``track_id``: the option list travels to the client before the
    answer is known, and an id there would let anyone look the answer up.

    Carries no ``cover_art_path`` either, which is the mode's whole design: in
    a music library the cover IS the answer, so this round withholds artwork
    and the reveal hands it back. Sending covers the UI is trusted not to draw
    would put the discipline in the wrong place — the question payload simply
    must not contain them. Modes that legitimately show art (M2) put it in
    their own options.
    """
    return {
        "option_id": uuid.uuid4().hex[:12],
        "title": track.get("title_display") or track.get("title") or "—",
        "artist": track.get("artist") or "—",
    }


# The window rules live in quiz.snippet — M3 listens to the same three seconds
# under the same constraints, and two copies would drift.
