"""Stream RecSys — session-aware personalized radio («Поток»).

Design: docs/2026-06-09-stream-recsys-design.md. Stateless: every request
rebuilds taste profiles from SQLite (playback_events + track_reactions),
pulls candidates from Qdrant, scores, and assembles the next chunk.

This module is layered bottom-up:
  reward model (event → weight)  →  profiles (anchors + axis prefs)
  →  candidate pools (A/B/C)     →  scoring + chunk assembly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Reward model (design §3) ────────────────────────────────────────────────
# Event weights.
W_LIKE = 1.0            # explicit like — strongest positive
W_REPLAY = 0.9          # instant replay of a completed track
W_FULL = 0.4            # listened ≥ 85%
W_MOST = 0.25           # listened 65–85%
W_NEUTRAL = 0.0         # 30s–65% — noisy zone (user may have walked away)
W_SKIP = -0.6           # skipped early
W_DISLIKE = -1.0        # explicit dislike — also a hard filter

# Skip detection: absolute threshold for normal tracks; ratio for short ones
# (30s of a 90s track is a third of it — not a skip).
SKIP_ABS_SEC = 30.0
SKIP_SHORT_TRACK_SEC = 120.0
SKIP_SHORT_RATIO = 0.25

# Listening-completeness boundaries (fractions of total_dur).
MOST_RATIO = 0.65       # below → neutral; above → W_MOST
FULL_RATIO = 0.85       # above → W_FULL; also the replay precondition

# Time-decay: w_eff = w · exp(−Δdays / H). Explicit signals outlive implicit.
H_LIKE_DAYS = 90.0      # likes/dislikes, from track_reactions.updated_at
H_IMPLICIT_DAYS = 30.0  # playback events, from played_at

# Idle rule (design §3): after IDLE_STREAK consecutive no-interaction tracks,
# subsequent events stop moving the profile until the user acts again.
IDLE_STREAK = 5


@dataclass(frozen=True)
class PlaybackSignal:
    """One playback event, normalised for profile building.

    ``interacted=None`` means the client didn't report the flag (legacy events
    predating the column) — treated as interacted so old history still counts.
    """
    track_id: str
    played_sec: float
    total_dur: float | None
    played_at: datetime
    session_id: str
    interacted: bool | None = None


def is_skip(played_sec: float, total_dur: float | None) -> bool:
    """Design §3 skip rule: <30s, but для коротких треков — <25% длительности."""
    if total_dur and 0.0 < total_dur < SKIP_SHORT_TRACK_SEC:
        return played_sec < SKIP_SHORT_RATIO * total_dur
    return played_sec < SKIP_ABS_SEC


def base_weight(played_sec: float, total_dur: float | None) -> float:
    """Implicit weight from listening completeness (no replay/idle context).

    Without a known duration only the skip rule applies — completeness
    ratios are uncomputable, so anything ≥ 30s stays neutral.
    """
    if is_skip(played_sec, total_dur):
        return W_SKIP
    if not total_dur or total_dur <= 0.0:
        return W_NEUTRAL
    ratio = played_sec / total_dur
    if ratio >= FULL_RATIO:
        return W_FULL
    if ratio >= MOST_RATIO:
        return W_MOST
    return W_NEUTRAL


def _is_replay(prev: PlaybackSignal, cur: PlaybackSignal) -> bool:
    """Instant replay: same track right after a ≥85% listen, same session.

    The ≥85% precondition kills the false-positive of double-clicking a track
    in search results (design pitfall #2).
    """
    if prev.track_id != cur.track_id or prev.session_id != cur.session_id:
        return False
    if not prev.total_dur or prev.total_dur <= 0.0:
        return False
    return prev.played_sec / prev.total_dur >= FULL_RATIO


def weight_events(events: list[PlaybackSignal]) -> list[float]:
    """Weight a chronologically-ordered single-session event list.

    Applies, in order: base completeness weight → replay upgrade → idle rule.
    The idle rule zeroes events AFTER a streak of IDLE_STREAK passive tracks
    (the streak itself keeps its weights) until an interacted event shows up —
    background listening keeps playing and logging, it just stops moving taste.
    """
    weights: list[float] = []
    passive_streak = 0
    for i, ev in enumerate(events):
        w = base_weight(ev.played_sec, ev.total_dur)
        if i > 0 and _is_replay(events[i - 1], ev):
            w = W_REPLAY

        interacted = ev.interacted is not False  # None (legacy) counts as action
        if interacted:
            passive_streak = 0
        else:
            if passive_streak >= IDLE_STREAK:
                w = 0.0
            passive_streak += 1
        weights.append(w)
    return weights


def decayed(weight: float, age_days: float, half_life_days: float) -> float:
    """Time-decay: ``w · exp(−Δdays / H)``. Future timestamps clamp to no decay."""
    if age_days <= 0.0:
        return weight
    return weight * math.exp(-age_days / half_life_days)
