"""Where a snippet starts — shared by every listening mode.

Kept in one place so the window rules cannot drift between modes: a track that
gives itself away in M1 must give itself away in M3 too, or the two modes are
measuring different things.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §7.
"""
from __future__ import annotations

from typing import Dict

# The window a snippet may start in, as a fraction of the track. Not the
# opening — intros are recognised instantly and would flatten the difficulty
# curve — and not the tail, where fades carry little information.
START_LO = 0.15
START_HI = 0.70


def track_duration(track: Dict) -> float:
    """Length in seconds, from whichever key this payload happens to use.

    The canonical payload key is ``duration`` — that is what the Qdrant light
    payload and the SQLite ``track_metadata`` mirror both carry.
    ``duration_sec`` exists only on the ``TrackMetadata`` API model, and
    reading only that one silently pinned every snippet to 0.0 once already.
    """
    return float(track.get("duration") or track.get("duration_sec") or 0.0)


# A narrow window a third of the way in, used where the point is to hear how a
# record was MADE rather than to recognise it: past the intro, inside the body
# of the arrangement, before anything winds down.
BODY_LO = 0.28
BODY_HI = 0.40


def start_point(
    track: Dict, length_sec: float, rng, *,
    lo: float = START_LO, hi: float = START_HI,
) -> float:
    """Pick where the snippet starts, never running past the end of the file."""
    duration = track_duration(track)
    if duration <= 0.0:
        return 0.0
    low = lo * duration
    # On a short track the ceiling can sit past the last playable start, so
    # clamp to whatever leaves room for the whole snippet.
    high = min(hi * duration, max(low, duration - float(length_sec)))
    if high <= low:
        return round(low, 3)
    return round(rng.uniform(low, high), 3)
