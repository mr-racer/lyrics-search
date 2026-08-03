"""Signal primitives shared by every recsys layer.

The bottom of the stack: what a playback event / reaction *is*, how a raw
listen maps to a weight, how weights decay, and the greedy anchor merge. Split
out of ``stream_service`` so ``baseline``/``session``/``pools`` can depend on
these without importing the orchestrator (which imports them back).

Nothing here knows about Qdrant, SQLite or sessions — pure functions and
dataclasses only. ``stream_service`` re-exports the whole surface, so existing
``from app.services.stream_service import is_skip`` imports keep working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from app.services._payload_coerce import coerce_float

# ── Reward model ───────────────────────────────────────────────────────────
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

# Minimum track duration for recommendation surfaces (intros/interludes filter).
MIN_TRACK_DURATION_SEC = 60.0

# cos > this → same taste region, weights merge. Raw-cosine default kept for the
# long-term profile surface; the session layer passes a percentile instead.
ANCHOR_MERGE_THRESHOLD = 0.85


@dataclass(frozen=True)
class PlaybackSignal:
    """One playback event, normalised for profile building.

    ``interacted`` is still read from the DB (the column keeps being written)
    but no longer affects weighting — the idle rule it powered was removed by
    the 2026-08-03 redesign.

    ``influence=True`` (default) means this event contributes to the taste
    profile. Hand-queued tracks set ``influence=False`` so they are kept for
    anti-repeat but excluded from the "For You" profile aggregation.
    """
    track_id: str
    played_sec: float
    total_dur: float | None
    played_at: datetime
    session_id: str
    interacted: bool | None = None
    influence: bool = True


@dataclass(frozen=True)
class ReactionSignal:
    track_id: str
    reaction: str           # 'like' | 'dislike'
    updated_at: datetime


@dataclass(frozen=True)
class FireSignal:
    """One огонёк/вода gesture from the append-only ``taste_signals`` journal."""
    track_id: str
    kind: str               # 'fire' | 'water'
    created_at: datetime


@dataclass
class StreamCandidate:
    """One candidate track flowing through scoring/assembly.

    ``payload`` is the raw Qdrant payload (title/artist/…/sonic_axes) — the
    route layer converts it to TrackMetadata at the very end. ``pool`` is one of
    ``fresh`` | ``familiar`` | ``liked`` | ``replay`` | ``anchor`` | ``axis``
    (the last two belong to the similar/axis surfaces, which score differently).
    """
    track_id: str
    payload: dict
    pool: str
    anchor_track_id: str | None = None
    max_anchor_cos: float = 0.0
    axis_match: float | None = None
    score: float = 0.0


@dataclass
class Anchor:
    track_id: str           # representative track (highest-weight in its merge group)
    weight: float
    vector: list[float] | None = None   # raw CLAP, attached from Qdrant
    # Track ids absorbed by the merge (representative included, first).
    members: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.members is None:
            self.members = [self.track_id]


def is_skip(played_sec: float, total_dur: float | None) -> bool:
    """Skip rule: <30s, but для коротких треков — <25% длительности."""
    if total_dur and 0.0 < total_dur < SKIP_SHORT_TRACK_SEC:
        return played_sec < SKIP_SHORT_RATIO * total_dur
    return played_sec < SKIP_ABS_SEC


def listen_ratio(played_sec: float, total_dur: float | None) -> float | None:
    """Fraction of the track heard, or None when the duration is unknown."""
    if not total_dur or total_dur <= 0.0:
        return None
    return played_sec / total_dur


def base_weight(played_sec: float, total_dur: float | None) -> float:
    """Implicit weight from listening completeness (no replay context).

    Without a known duration only the skip rule applies — completeness
    ratios are uncomputable, so anything ≥ 30s stays neutral.
    """
    if is_skip(played_sec, total_dur):
        return W_SKIP
    ratio = listen_ratio(played_sec, total_dur)
    if ratio is None:
        return W_NEUTRAL
    if ratio >= FULL_RATIO:
        return W_FULL
    if ratio >= MOST_RATIO:
        return W_MOST
    return W_NEUTRAL


def decayed(weight: float, age_days: float, half_life_days: float) -> float:
    """Time-decay: ``w · exp(−Δdays / H)``. Future timestamps clamp to no decay."""
    if age_days <= 0.0:
        return weight
    return weight * math.exp(-age_days / half_life_days)


def age_days(ts: datetime, now: datetime) -> float:
    return (now - ts).total_seconds() / 86400.0


# Legacy private alias — several call sites (and tests) still use it.
_age_days = age_days


def parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def clap_vector(point):
    """Extract a point's CLAP vector, tolerating both named-vector dicts
    (``{"clap": [...]}``) and the bare-list vectors older indexes stored."""
    return point.vector.get("clap") if isinstance(point.vector, dict) else point.vector


def unit(vec) -> np.ndarray | None:
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def duration_ok(payload) -> bool:
    """True unless the track's duration is KNOWN and shorter than 60s.
    Missing/zero duration is kept (avoid dropping real songs with absent tags)."""
    d = coerce_float((payload or {}).get("duration"))
    return d is None or d <= 0.0 or d >= MIN_TRACK_DURATION_SEC


def combine_weights(*maps: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in maps:
        for tid, w in m.items():
            out[tid] = out.get(tid, 0.0) + w
    return out


def latest_per_track(signals: list[FireSignal]) -> list[FireSignal]:
    """Collapse the append-only journal to the single newest signal per track.

    Implements «одна активная реакция на трек»: water pressed over a fire (or vice
    versa) supersedes it — the older opposite signal no longer counts anywhere.
    Re-pressing the same kind just refreshes the timestamp.
    """
    newest: dict[str, FireSignal] = {}
    for s in signals:
        cur = newest.get(s.track_id)
        if cur is None or s.created_at > cur.created_at:
            newest[s.track_id] = s
    return list(newest.values())


def select_positive_anchors(
    track_weights: dict[str, float], top_m: int = 20,
) -> list[Anchor]:
    """Top-M tracks by positive weight → anchor candidates."""
    positive = [(tid, w) for tid, w in track_weights.items() if w > 0.0]
    positive.sort(key=lambda p: p[1], reverse=True)
    return [Anchor(track_id=tid, weight=w) for tid, w in positive[:top_m]]


def merge_anchors(
    anchors: list[Anchor],
    vectors: dict[str, np.ndarray],
    threshold: float = ANCHOR_MERGE_THRESHOLD,
    *,
    calibration=None,
) -> list[Anchor]:
    """Greedy merge: near-duplicate anchors collapse into the strongest one,
    summing weights — three liked tracks off one album become one strong anchor,
    not three Qdrant queries into the same neighborhood.

    With ``calibration`` the threshold is read as a **percentile** of the
    collection's own cosine distribution (design §5); without it, as a raw
    cosine (the long-term profile surface still uses raw). Anchors without a
    CLAP vector are dropped — they can't query Qdrant.
    """
    with_vec = [a for a in anchors if a.track_id in vectors]
    with_vec.sort(key=lambda a: a.weight, reverse=True)

    kept: list[Anchor] = []
    kept_vecs: list[np.ndarray] = []
    for a in with_vec:
        v = unit(vectors[a.track_id])
        if v is None:
            continue
        merged = False
        for i, kv in enumerate(kept_vecs):
            cos = float(v @ kv)
            score = calibration.sim_pct(cos) if calibration is not None else cos
            if score > threshold:
                kept[i].weight += a.weight
                kept[i].members.append(a.track_id)
                merged = True
                break
        if not merged:
            kept.append(Anchor(track_id=a.track_id, weight=a.weight,
                               vector=vectors[a.track_id]))
            kept_vecs.append(v)
    return kept


# ── Sonic-axis helpers (z-space) ───────────────────────────────────────────
# Axis match: RMS z-distance mapped to [−1, 1]; dist 0 → 1, dist=SCALE → 0.
AXIS_MATCH_DIST_SCALE = 2.0


def z_scores_for_axes(
    raw_axes: dict | None,
    stats: dict | None,
    axis_names: tuple[str, ...],
) -> dict[str, float] | None:
    """Raw payload scores → z-scores via blended stats. None when unusable.

    A zero/missing std yields z=0 for that axis (axis carries no signal in
    this collection rather than exploding to ±inf).
    """
    if not raw_axes or not stats:
        return None
    mean, std = stats.get("mean") or {}, stats.get("std") or {}
    out: dict[str, float] = {}
    for a in axis_names:
        if a not in raw_axes:
            return None  # malformed payload — treat the whole dict as unusable
        s = std.get(a, 0.0)
        out[a] = (raw_axes[a] - mean.get(a, 0.0)) / s if s > 1e-9 else 0.0
    return out


def axis_match_score(
    z: dict[str, float] | None,
    p: dict[str, float] | None,
    confidence: float,
    axis_names: tuple[str, ...],
) -> float:
    """−‖z − p‖₂ normalised: RMS distance mapped to [−1, 1], × confidence.

    No axis data or no profile → 0 (the term drops out of the score).
    """
    if not z or not p or confidence <= 0.0:
        return 0.0
    sq = sum((z.get(a, 0.0) - p.get(a, 0.0)) ** 2 for a in axis_names)
    rms = math.sqrt(sq / len(axis_names))
    match = max(-1.0, min(1.0, 1.0 - rms / AXIS_MATCH_DIST_SCALE))
    return match * confidence


def centroid(member_ids: list[str], vectors: dict, weights: dict | None = None) -> np.ndarray | None:
    """Normalized (optionally weighted) mean of members' unit CLAP vectors."""
    acc = None
    for tid in member_ids:
        if tid not in vectors:
            continue
        v = unit(vectors[tid])
        if v is None:
            continue
        w = float(weights.get(tid, 1.0)) if weights else 1.0
        acc = w * v if acc is None else acc + w * v
    return unit(acc) if acc is not None else None
