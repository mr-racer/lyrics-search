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

import numpy as np

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


# ── Taste profiles: anchors + axis preferences (design §4) ─────────────────
# A profile is NOT a centroid: it is a set of concrete liked tracks (anchors,
# raw CLAP vectors) plus a 6-dim axis-preference vector. Heterogeneous taste
# stays multi-modal instead of averaging into a meaningless middle.

M_LONG_TERM_ANCHORS = 20        # top tracks by decayed weight → anchor candidates
TOP_EFFECTIVE_ANCHORS = 5       # anchors actually queried against Qdrant
ANCHOR_MERGE_THRESHOLD = 0.85   # cos > this → same taste region, weights merge
# «≥2 скипа, decayed»: two fresh skips sum to 2.0 and fade below this after
# roughly two weeks of not touching the track.
SKIP_NEG_DECAYED_COUNT = 1.5
SESSION_SIGNALS_SATURATION = 10  # w_s = min(1, n_signals / 10)
SESSION_ANCHOR_BOOST = 2.0       # session anchor weight × (1 + BOOST·w_s)
CONFIDENCE_SATURATION = 5.0      # axis-pref confidence = min(1, Σpos_weight / this)


@dataclass(frozen=True)
class ReactionSignal:
    track_id: str
    reaction: str           # 'like' | 'dislike'
    updated_at: datetime


@dataclass
class Anchor:
    track_id: str           # representative track (highest-weight in its merge group)
    weight: float
    vector: list[float] | None = None   # raw CLAP, attached from Qdrant


def _age_days(ts: datetime, now: datetime) -> float:
    return (now - ts).total_seconds() / 86400.0


def aggregate_event_weights(
    signals: list[PlaybackSignal], now: datetime,
) -> dict[str, float]:
    """Decayed implicit weight per track over a chronological event list.

    Events are re-grouped by session (replay + idle rules are session-scoped),
    weighted, decayed with H_IMPLICIT, then summed per track.
    """
    by_session: dict[str, list[PlaybackSignal]] = {}
    for s in signals:
        by_session.setdefault(s.session_id, []).append(s)

    out: dict[str, float] = {}
    for sess_events in by_session.values():
        for ev, w in zip(sess_events, weight_events(sess_events)):
            if w == 0.0:
                continue
            w_eff = decayed(w, _age_days(ev.played_at, now), H_IMPLICIT_DAYS)
            out[ev.track_id] = out.get(ev.track_id, 0.0) + w_eff
    return out


def aggregate_reaction_weights(
    reactions: list[ReactionSignal], now: datetime,
) -> dict[str, float]:
    """±1.0 per reaction, decayed with H_LIKE from ``updated_at`` (last flip —
    reaction history is not stored, which also kills like↔dislike oscillation)."""
    out: dict[str, float] = {}
    for r in reactions:
        base = W_LIKE if r.reaction == "like" else W_DISLIKE if r.reaction == "dislike" else 0.0
        if base == 0.0:
            continue
        out[r.track_id] = out.get(r.track_id, 0.0) + decayed(
            base, _age_days(r.updated_at, now), H_LIKE_DAYS,
        )
    return out


def combine_weights(*maps: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in maps:
        for tid, w in m.items():
            out[tid] = out.get(tid, 0.0) + w
    return out


def negative_track_ids(
    signals: list[PlaybackSignal],
    reactions: list[ReactionSignal],
    now: datetime,
) -> set[str]:
    """Hard-filter set: current dislikes (absolute, no decay — «пока дизлайк
    стоит») + tracks whose decayed skip count passes the multi-skip threshold."""
    out = {r.track_id for r in reactions if r.reaction == "dislike"}
    skip_counts: dict[str, float] = {}
    for ev in signals:
        if is_skip(ev.played_sec, ev.total_dur):
            skip_counts[ev.track_id] = skip_counts.get(ev.track_id, 0.0) + decayed(
                1.0, _age_days(ev.played_at, now), H_IMPLICIT_DAYS,
            )
    out.update(tid for tid, c in skip_counts.items() if c >= SKIP_NEG_DECAYED_COUNT)
    return out


def select_positive_anchors(
    track_weights: dict[str, float], top_m: int = M_LONG_TERM_ANCHORS,
) -> list[Anchor]:
    """Top-M tracks by positive decayed weight → anchor candidates."""
    positive = [(tid, w) for tid, w in track_weights.items() if w > 0.0]
    positive.sort(key=lambda p: p[1], reverse=True)
    return [Anchor(track_id=tid, weight=w) for tid, w in positive[:top_m]]


def merge_anchors(
    anchors: list[Anchor],
    vectors: dict[str, np.ndarray],
    threshold: float = ANCHOR_MERGE_THRESHOLD,
) -> list[Anchor]:
    """Greedy merge: cos > threshold collapses near-duplicate anchors into the
    strongest one, summing weights — three liked tracks off one album become
    one strong anchor, not three Qdrant queries into the same neighborhood.

    Anchors without a CLAP vector are dropped (can't query Qdrant with them).
    """
    with_vec = [a for a in anchors if a.track_id in vectors]
    with_vec.sort(key=lambda a: a.weight, reverse=True)

    kept: list[Anchor] = []
    kept_vecs: list[np.ndarray] = []
    for a in with_vec:
        v = np.asarray(vectors[a.track_id], dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        v = v / norm
        merged = False
        for i, kv in enumerate(kept_vecs):
            if float(v @ kv) > threshold:
                kept[i].weight += a.weight
                merged = True
                break
        if not merged:
            kept.append(Anchor(track_id=a.track_id, weight=a.weight,
                               vector=vectors[a.track_id]))
            kept_vecs.append(v)
    return kept


def session_blend_weight(n_session_signals: int) -> float:
    """w_s = min(1, n/10): by ~10 session signals the session taste dominates."""
    return min(1.0, n_session_signals / SESSION_SIGNALS_SATURATION)


def count_session_signals(
    session_events: list[PlaybackSignal],
    session_reactions: list[ReactionSignal],
) -> int:
    """Signals = session events that carry information (nonzero weight after
    replay/idle processing) + reactions made during the session."""
    weights = weight_events(session_events)
    return sum(1 for w in weights if w != 0.0) + len(session_reactions)


def axis_preferences(
    track_weights: dict[str, float],
    z_by_track: dict[str, dict[str, float]],
    axis_names: tuple[str, ...],
) -> tuple[dict[str, float] | None, float]:
    """Weighted mean of z-scores over POSITIVE-weight tracks → (p ∈ R⁶, confidence).

    Confidence grows with positive signal volume and saturates at
    CONFIDENCE_SATURATION; tracks without axis data are skipped. Returns
    ``(None, 0.0)`` when nothing usable exists (axis term then drops out of
    scoring entirely).
    """
    total_w = 0.0
    acc = {a: 0.0 for a in axis_names}
    for tid, w in track_weights.items():
        if w <= 0.0:
            continue
        z = z_by_track.get(tid)
        if not z:
            continue
        for a in axis_names:
            acc[a] += w * z.get(a, 0.0)
        total_w += w
    if total_w <= 0.0:
        return None, 0.0
    p = {a: acc[a] / total_w for a in axis_names}
    confidence = min(1.0, total_w / CONFIDENCE_SATURATION)
    return p, confidence


def blend_axis_preferences(
    p_long: dict[str, float] | None,
    p_session: dict[str, float] | None,
    w_s: float,
    axis_names: tuple[str, ...],
) -> dict[str, float] | None:
    """p_итог = (1 − w_s)·p_long + w_s·p_session, with graceful one-sided falls."""
    if p_long is None and p_session is None:
        return None
    if p_long is None:
        return dict(p_session)
    if p_session is None:
        return dict(p_long)
    return {a: (1.0 - w_s) * p_long.get(a, 0.0) + w_s * p_session.get(a, 0.0)
            for a in axis_names}


def union_anchor_weights(
    long_weights: dict[str, float],
    session_weights: dict[str, float],
    w_s: float,
) -> dict[str, float]:
    """якоря = long-term (×1) ∪ session (×(1 + 2·w_s)).

    Session events are a subset of the long-term history, so for tracks present
    on both sides we take max(long, boosted-session) instead of summing —
    summing would double-count every fresh session play.
    """
    boost = 1.0 + SESSION_ANCHOR_BOOST * w_s
    out = dict(long_weights)
    for tid, w in session_weights.items():
        boosted = w * boost
        if boosted > out.get(tid, float("-inf")):
            out[tid] = boosted
    return out
