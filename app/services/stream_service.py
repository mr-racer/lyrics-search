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
import random as _random_module
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from app.resources.metadata_db import MetadataDB
from app.resources.qdrant_utils import PAYLOAD_EXCLUDE_LYRICS, light_points

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


def _clap_vector(point):
    """Extract a point's CLAP vector, tolerating both named-vector dicts
    (``{"clap": [...]}``) and the bare-list vectors older indexes stored."""
    return point.vector.get("clap") if isinstance(point.vector, dict) else point.vector


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
SESSION_SIGNALS_SATURATION = 10  # w_s ramp: each signal adds 1/10
# Hard ceiling on session influence: the session never outweighs long-term
# (50/50 at most). A session drifting into a weird corner must stay escapable —
# long-term anchors always keep at least half the vote.
SESSION_BLEND_MAX = 0.5
SESSION_ANCHOR_BOOST = 2.0       # session anchor weight × (1 + BOOST·w_s), ≤ ×2 at the cap
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
    # Track ids absorbed by the cos>0.85 merge (representative included, first).
    # Powers the «вкусовые острова» view — each island shows its member covers.
    members: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.members is None:
            self.members = [self.track_id]


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
                kept[i].members.append(a.track_id)
                merged = True
                break
        if not merged:
            kept.append(Anchor(track_id=a.track_id, weight=a.weight,
                               vector=vectors[a.track_id]))
            kept_vecs.append(v)
    return kept


def session_blend_weight(n_session_signals: int) -> float:
    """w_s = min(0.5, n/10): session influence ramps up but caps at parity.

    By ~5 signals the session reaches its 50% ceiling — it colors the stream
    strongly, yet long-term taste always keeps an equal vote so a session that
    wandered somewhere strange cannot trap the user there.
    """
    return min(SESSION_BLEND_MAX, n_session_signals / SESSION_SIGNALS_SATURATION)


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


# ── Candidate pools + scoring + chunk assembly (design §5–6) ───────────────

# score(t) weights — tuned later against live sessions; keep them named.
SCORE_W_ANCHOR = 0.50     # max_cos to the CLOSEST anchor (not the mean)
SCORE_W_AXIS = 0.25       # axis match × profile confidence
SCORE_W_NOVELTY = 0.10    # low play_count boost
SCORE_W_RECENT = 0.15     # exp(−hours_since_played / 24) penalty
ARTIST_REPEAT_PENALTY = 0.05      # SMALL nudge, window = last 3 tracks
ARTIST_REPEAT_WINDOW = 3
RECENT_PENALTY_HALFLIFE_H = 24.0
# Axis match: RMS z-distance mapped to [−1, 1]; dist 0 → 1, dist=SCALE → 0.
AXIS_MATCH_DIST_SCALE = 2.0

ANCHOR_TOP_K = 30          # Qdrant neighbors fetched per effective anchor

EXPLORE_SHARE = 0.12       # ≈10–15% of non-liked slots — filter-bubble insurance
EXPLORE_MAX_PLAY_COUNT = 1  # «низкий play_count»: 0 or 1 plays
EXPLORE_POOL_SIZE = 12
EXPLORE_BIN_EDGE = 0.5     # z-bins: < −0.5 | −0.5..0.5 | > 0.5 (energy × experimental)
NEG_PROXIMITY_THRESHOLD = 0.80  # explore candidate too close to a negative anchor

LIKED_COOLDOWN_H = 8.0     # hard «не чаще раза в 8 часов»
MAX_CONSECUTIVE_LIKED = 2
MAX_CONSECUTIVE_ARTIST = 2  # autoplay_service rule, reused

# Anti-repeat floor (design 2026-06-14: «бесконечный круг»). The «жёсткий пол»
# under round replay: these stay hard-excluded even after a «круг» wraps, so the
# just-heard tracks never recur immediately. Everything older is only soft-demoted
# (relax pass + recency penalty) — «жёсткий пол + мягкий хвост».
ANTIREPEAT_FLOOR_TRACKS = 10    # last N played tracks never repeat
ANTIREPEAT_FLOOR_MINUTES = 30   # anything played within X minutes never repeats

DEFAULT_CHUNK_N = 3
DEFAULT_LIKED_SHARE = 0.30
LONG_TERM_EVENT_CAP = 2000  # newest events fed into profile building


@dataclass
class StreamCandidate:
    """One candidate track flowing through scoring/assembly.

    ``payload`` is the raw Qdrant payload (title/artist/…/sonic_axes) — the
    route layer converts it to TrackMetadata at the very end.
    """
    track_id: str
    payload: dict
    pool: str                          # 'anchor' | 'explore' | 'liked'
    anchor_track_id: str | None = None
    max_anchor_cos: float = 0.0
    axis_match: float | None = None
    score: float = 0.0


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


def pool_anchor_candidates(
    qdrant_client,
    collection_name: str,
    anchors: list[Anchor],
    excluded: set[str],
    k: int = ANCHOR_TOP_K,
) -> dict[str, StreamCandidate]:
    """Pool A: Qdrant CLAP top-K per effective anchor, deduped by track_id.

    A track found by several anchors keeps its best (max) cosine — design §6
    scores against the closest anchor, not the average.
    """
    out: dict[str, StreamCandidate] = {}
    for anchor in anchors:
        if anchor.vector is None:
            continue
        try:
            # qdrant-client >= 1.10: query_points replaced the removed .search()
            hits = qdrant_client.query_points(
                collection_name=collection_name,
                query=list(anchor.vector),
                using="clap",
                limit=k,
                with_payload=PAYLOAD_EXCLUDE_LYRICS,
            ).points
        except Exception:
            logger.exception("[stream] anchor search failed for %s", anchor.track_id)
            continue
        for h in hits:
            tid = str(h.id)
            if tid in excluded:
                continue
            cos = float(h.score or 0.0)
            existing = out.get(tid)
            if existing is None:
                out[tid] = StreamCandidate(
                    track_id=tid, payload=h.payload or {}, pool="anchor",
                    anchor_track_id=anchor.track_id, max_anchor_cos=cos,
                )
            elif cos > existing.max_anchor_cos:
                existing.max_anchor_cos = cos
                existing.anchor_track_id = anchor.track_id
    return out


def _explore_bin(z_energy: float, z_experimental: float) -> tuple[int, int]:
    def bucket(z: float) -> int:
        if z < -EXPLORE_BIN_EDGE:
            return 0
        if z > EXPLORE_BIN_EDGE:
            return 2
        return 1
    return bucket(z_energy), bucket(z_experimental)


def stratify_explore(
    eligible: list[tuple[str, dict[str, float] | None]],
    pool_size: int,
    rng,
) -> list[str]:
    """Round-robin sample across energy×experimental z-bins.

    ``eligible`` is ``[(track_id, z_axes | None)]``; tracks without axis data
    fall into one shared bin. Stratification keeps exploration spread across
    the sonic space instead of clustering around the collection's bulk.
    """
    bins: dict[tuple, list[str]] = {}
    for tid, z in eligible:
        key = _explore_bin(z["energy"], z["experimental"]) if z else ("nz",)
        bins.setdefault(key, []).append(tid)

    for members in bins.values():
        rng.shuffle(members)

    out: list[str] = []
    bin_lists = list(bins.values())
    i = 0
    while len(out) < pool_size and any(bin_lists):
        lst = bin_lists[i % len(bin_lists)]
        if lst:
            out.append(lst.pop())
        i += 1
        if i > 10_000:  # all bins drained
            break
        if all(not lst for lst in bin_lists):
            break
    return out


def pool_explore_candidates(
    qdrant_client,
    collection_name: str,
    *,
    excluded: set[str],
    reacted_ids: set[str],
    play_counts: dict[str, int],
    axis_stats: dict | None,
    negative_vectors: list[np.ndarray],
    axis_names: tuple[str, ...],
    rng,
    pool_size: int = EXPLORE_POOL_SIZE,
    scroll_cap: int = 5000,
) -> list[StreamCandidate]:
    """Pool B: low-play-count unreacted tracks, stratified over axis bins,
    not too close to negative anchors. Sonic Descriptor clusters are NOT used."""
    # 1. Light, lyrics-free payloads from the shared per-collection cache
    #    (card fields + sonic_axes only). Capped to bound the scoring loop.
    points = light_points(qdrant_client, collection_name)[:scroll_cap]

    payload_by_id: dict[str, dict] = {}
    eligible: list[tuple[str, dict[str, float] | None]] = []
    for tid, payload in points:
        if tid in excluded or tid in reacted_ids:
            continue
        if play_counts.get(tid, 0) > EXPLORE_MAX_PLAY_COUNT:
            continue
        payload_by_id[tid] = payload
        z = z_scores_for_axes(payload.get("sonic_axes"), axis_stats, axis_names)
        eligible.append((tid, z))

    # 2. Stratified sample.
    sampled = stratify_explore(eligible, pool_size, rng)
    if not sampled or not negative_vectors:
        return [StreamCandidate(track_id=t, payload=payload_by_id[t], pool="explore")
                for t in sampled]

    # 3. Negative-proximity check — only for the small sampled set.
    neg = np.stack([v / (np.linalg.norm(v) or 1.0) for v in negative_vectors])
    kept: list[StreamCandidate] = []
    try:
        pts = qdrant_client.retrieve(
            collection_name=collection_name, ids=sampled,
            with_payload=False, with_vectors=["clap"],
        )
    except Exception:
        logger.exception("[stream] explore vector retrieve failed — skipping negativity check")
        pts = []
    vec_by_id = {}
    for p in pts:
        v = _clap_vector(p)
        if v:
            vec_by_id[str(p.id)] = np.asarray(v, dtype=np.float32)
    for tid in sampled:
        v = vec_by_id.get(tid)
        if v is not None:
            v = v / (np.linalg.norm(v) or 1.0)
            if float(np.max(neg @ v)) > NEG_PROXIMITY_THRESHOLD:
                continue  # too close to something the user actively rejects
        kept.append(StreamCandidate(track_id=tid, payload=payload_by_id[tid], pool="explore"))
    return kept


def sample_liked_tracks(
    liked_weights: dict[str, float],
    recency_hours: dict[str, float],
    n_needed: int,
    rng,
    *,
    cooldown_h: float = LIKED_COOLDOWN_H,
    excluded: set[str] | frozenset = frozenset(),
) -> list[str]:
    """Pool C sampler: weight-proportional, anti-repeat, honest rotation.

    Two-pass topup (design §6.3): first pass respects the hard cooldown;
    if the quota is still unfilled (tiny liked list / slider at 100%), the
    cooldown is relaxed rather than under-filling the chunk.
    """
    def _weighted_draw(pool: dict[str, float], k: int) -> list[str]:
        chosen: list[str] = []
        pool = dict(pool)
        while pool and len(chosen) < k:
            ids = list(pool)
            weights = [max(pool[t], 1e-6) for t in ids]
            pick = rng.choices(ids, weights=weights, k=1)[0]
            chosen.append(pick)
            del pool[pick]
        return chosen

    def _adjusted(ids) -> dict[str, float]:
        out = {}
        for tid in ids:
            h = recency_hours.get(tid)
            anti_repeat = math.exp(-h / RECENT_PENALTY_HALFLIFE_H) if h is not None else 0.0
            out[tid] = liked_weights[tid] * (1.0 - 0.9 * anti_repeat)
        return out

    available = [t for t in liked_weights if t not in excluded]
    fresh = [t for t in available
             if recency_hours.get(t) is None or recency_hours[t] >= cooldown_h]

    chosen = _weighted_draw(_adjusted(fresh), n_needed)
    if len(chosen) < n_needed:  # topup: relax the cooldown, keep anti-repeat weighting
        rest = [t for t in available if t not in chosen]
        chosen += _weighted_draw(_adjusted(rest), n_needed - len(chosen))
    return chosen


def score_candidates(
    candidates: list[StreamCandidate],
    *,
    p_final: dict[str, float] | None,
    confidence: float,
    play_counts: dict[str, int],
    recency_hours: dict[str, float],
    axis_stats: dict | None,
    axis_names: tuple[str, ...],
) -> None:
    """Fill ``score`` + ``axis_match`` in place (design §6 formula, minus the
    artist-repeat term — that one is positional and applied during assembly)."""
    for c in candidates:
        z = z_scores_for_axes(c.payload.get("sonic_axes"), axis_stats, axis_names)
        c.axis_match = axis_match_score(z, p_final, confidence, axis_names)

        novelty = 1.0 / (1.0 + play_counts.get(c.track_id, 0))
        h = recency_hours.get(c.track_id)
        recent_pen = math.exp(-h / RECENT_PENALTY_HALFLIFE_H) if h is not None else 0.0

        c.score = (
            SCORE_W_ANCHOR * c.max_anchor_cos
            + SCORE_W_AXIS * c.axis_match
            + SCORE_W_NOVELTY * novelty
            - SCORE_W_RECENT * recent_pen
        )


def assemble_chunk(
    main: list[StreamCandidate],
    liked: list[StreamCandidate],
    *,
    n: int,
    liked_share: float,
    recent_artists: list[str],
) -> list[StreamCandidate]:
    """Slot-quota assembly: ``round(n · liked_share)`` slots go to pool C, the
    rest to the best-scored A∪B candidates.

    Interleaving rules: ≤2 liked подряд, ≤2 одного артиста подряд (the artist
    window seeds from the session's last plays). Artist-repeat soft penalty
    (last ARTIST_REPEAT_WINDOW tracks) is applied positionally here. When one
    side runs dry the other tops up — недобор хуже мягкого нарушения квоты —
    EXCEPT at liked_share == 0: the slider's «новое» extreme is a hard promise,
    so a dry main pool yields a short chunk rather than liked tracks beyond
    quota (in small libraries main dries constantly — liked + session-played
    are excluded — and the topup would make the zero position a no-op).
    """
    liked_quota = max(0, min(n, round(n * liked_share)))
    main_sorted = sorted(main, key=lambda c: c.score, reverse=True)
    liked_queue = list(liked)

    out: list[StreamCandidate] = []
    artist_tail: list[str] = list(recent_artists)[-ARTIST_REPEAT_WINDOW:]
    consecutive_liked = 0

    def _artist(c: StreamCandidate) -> str:
        return (c.payload.get("artist") or "").strip().lower()

    def _violates_artist_rule(c: StreamCandidate) -> bool:
        a = _artist(c)
        return (len(artist_tail) >= MAX_CONSECUTIVE_ARTIST
                and a != ""
                and all(t == a for t in artist_tail[-MAX_CONSECUTIVE_ARTIST:]))

    def _pick(pool: list[StreamCandidate], dynamic_score: bool) -> StreamCandidate | None:
        # First pass honors the artist rule; second pass (topup) bends it —
        # same rationale as autoplay_service: undersupply is the worse UX.
        for bend_rules in (False, True):
            best, best_idx, best_val = None, -1, float("-inf")
            for i, c in enumerate(pool):
                if not bend_rules and _violates_artist_rule(c):
                    continue
                val = c.score if dynamic_score else -i  # liked queue keeps sample order
                if dynamic_score and _artist(c) in artist_tail:
                    val -= ARTIST_REPEAT_PENALTY
                if val > best_val:
                    best, best_idx, best_val = c, i, val
            if best is not None:
                pool.pop(best_idx)
                return best
        return None

    for _ in range(n):
        # When every remaining slot is owed to the quota (slider near 100%),
        # the quota wins over the ≤2-consecutive rule — the slider promised
        # a share, and alternation is impossible in an all-liked chunk.
        slots_left = n - len(out)
        must_liked = liked_quota >= slots_left
        want_liked = (liked_quota > 0 and liked_queue
                      and (consecutive_liked < MAX_CONSECUTIVE_LIKED or must_liked))
        c = None
        if want_liked:
            c = _pick(liked_queue, dynamic_score=False)
            if c is not None:
                liked_quota -= 1
                consecutive_liked += 1
        if c is None:
            c = _pick(main_sorted, dynamic_score=True)
            if c is not None:
                consecutive_liked = 0
        if c is None and liked_queue and liked_share > 0:
            # main dry — topup from liked beyond quota, unless the slider sits
            # at the strict-«новое» extreme
            c = _pick(liked_queue, dynamic_score=False)
            if c is not None:
                consecutive_liked += 1
        if c is None:
            break  # both pools dry — return a short chunk
        out.append(c)
        artist_tail.append(_artist(c))
        artist_tail = artist_tail[-ARTIST_REPEAT_WINDOW:]
    return out


# ── Orchestration: GET /stream/next entry point ─────────────────────────────

def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def _retrieve_track_data(
    qdrant_client, collection_name: str, track_ids: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Batch-fetch ``{id: clap_vector}`` + ``{id: payload}`` from Qdrant."""
    if not track_ids:
        return {}, {}
    try:
        pts = qdrant_client.retrieve(
            collection_name=collection_name, ids=track_ids,
            with_payload=PAYLOAD_EXCLUDE_LYRICS, with_vectors=["clap"],
        )
    except Exception:
        logger.exception("[stream] track data retrieve failed")
        return {}, {}
    vectors: dict[str, np.ndarray] = {}
    payloads: dict[str, dict] = {}
    for p in pts:
        tid = str(p.id)
        payloads[tid] = p.payload or {}
        v = _clap_vector(p)
        if v:
            vectors[tid] = np.asarray(v, dtype=np.float32)
    return vectors, payloads


def _anti_repeat_floor(
    recency_hours: dict[str, float],
    *,
    n_tracks: int = ANTIREPEAT_FLOOR_TRACKS,
    minutes: float = ANTIREPEAT_FLOOR_MINUTES,
) -> set[str]:
    """Hard «не повторять только что услышанное»: the last ``n_tracks`` played
    (by recency) ∪ everything played within the last ``minutes``.

    ``recency_hours`` is ``{track_id: hours_since_last_play}`` (already built in
    ``next_chunk``). Global (per-collection), not session-scoped — a track heard
    minutes ago shouldn't recur even in a fresh tab/session. This is the floor
    that survives a round reset; older plays fall through to the soft relax pass.
    """
    if not recency_hours:
        return set()
    window_h = minutes / 60.0
    floor = {tid for tid, h in recency_hours.items() if h <= window_h}
    by_recent = sorted(recency_hours.items(), key=lambda kv: kv[1])
    floor.update(tid for tid, _ in by_recent[:n_tracks])
    return floor


def next_chunk(
    *,
    qdrant_client,
    collection_name: str,
    session_id: str,
    n: int = DEFAULT_CHUNK_N,
    liked_share: float | None = None,
    exclude_ids: list[str] | None = None,
    rng=None,
    now: datetime | None = None,
) -> dict:
    """Stateless «Поток»: rebuild profiles from SQLite, pull candidates from
    Qdrant, score, assemble. Returns ``{"tracks": [StreamCandidate], "diagnostics": {…}}``.

    ``exclude_ids`` covers the frontend prefetch buffer — tracks already issued
    but not yet reported as playback events (the stateless gap design §2 closes
    by re-requesting after strong signals).
    """
    from app.resources.clap_features import (
        AXIS_NAMES, blend_axis_stats, load_axis_norm_reference,
    )

    now = now or datetime.utcnow()
    rng = rng or _random_module

    # 1. Signals from SQLite.
    raw_signals = MetadataDB.get_playback_signals(collection_name, LONG_TERM_EVENT_CAP)
    signals = [PlaybackSignal(**r) for r in raw_signals]
    reactions = []
    for tid, reaction, ts in MetadataDB.get_reactions_with_updated_at(collection_name):
        dt = _parse_iso(ts)
        if dt is not None:
            reactions.append(ReactionSignal(track_id=tid, reaction=reaction, updated_at=dt))

    # 2. Session split. In-session reactions = updated after the session began.
    session_events = [s for s in signals if s.session_id == session_id]
    if session_events:
        session_start = session_events[0].played_at
        session_reactions = [r for r in reactions if r.updated_at >= session_start]
    else:
        session_reactions = []

    # 3. Profiles + blend.
    long_weights = combine_weights(
        aggregate_event_weights(signals, now),
        aggregate_reaction_weights(reactions, now),
    )
    session_weights = combine_weights(
        aggregate_event_weights(session_events, now),
        aggregate_reaction_weights(session_reactions, now),
    )
    n_session_signals = count_session_signals(session_events, session_reactions)
    w_s = session_blend_weight(n_session_signals)
    anchor_weights = union_anchor_weights(long_weights, session_weights, w_s)
    negatives = negative_track_ids(signals, reactions, now)

    # 4. Anchor candidates → vectors → merge → top effective.
    anchor_cands = select_positive_anchors(anchor_weights)
    positive_ids = [a.track_id for a in anchor_cands]
    session_positive = [tid for tid, w in session_weights.items() if w > 0.0]
    fetch_ids = list(dict.fromkeys(positive_ids + session_positive + sorted(negatives)))
    vectors, payloads = _retrieve_track_data(qdrant_client, collection_name, fetch_ids)

    merged = merge_anchors(anchor_cands, vectors)
    merged.sort(key=lambda a: a.weight, reverse=True)
    top_anchors = merged[:TOP_EFFECTIVE_ANCHORS]

    # 5. Axis preferences in z-space (shrinkage-blended collection stats).
    axis_stats = blend_axis_stats(
        MetadataDB.get_axis_norm_stats(collection_name),
        load_axis_norm_reference(),
    )
    z_by_track = {
        tid: z for tid, pl in payloads.items()
        if (z := z_scores_for_axes(pl.get("sonic_axes"), axis_stats, AXIS_NAMES))
    }
    p_long, conf_long = axis_preferences(long_weights, z_by_track, AXIS_NAMES)
    p_sess, conf_sess = axis_preferences(session_weights, z_by_track, AXIS_NAMES)
    p_final = blend_axis_preferences(p_long, p_sess, w_s, AXIS_NAMES)
    confidence = (1.0 - w_s) * conf_long + w_s * conf_sess

    # 6. Shared filter set + per-track stats.
    session_played = {s.track_id for s in session_events}
    liked_ids = {r.track_id for r in reactions if r.reaction == "like"}

    play_counts = MetadataDB.get_play_counts_by_track(collection_name)
    recency_hours: dict[str, float] = {}
    for tid, iso in MetadataDB.get_play_recency_map(collection_name).items():
        dt = _parse_iso(iso)
        if dt is not None:
            recency_hours[tid] = max(0.0, (now - dt).total_seconds() / 3600.0)

    # HARD exclusions (never served this request). session_played is deliberately
    # NOT hard: once a «круг» exhausts the library the stream must replay rather
    # than go empty (design 2026-06-14). Older session plays demote via the relax
    # pass + recency penalty; only the anti-repeat FLOOR (just-heard tracks) stays
    # hard across the round boundary — «жёсткий пол + мягкий хвост».
    floor_ids = _anti_repeat_floor(recency_hours)
    base_exclude = set(exclude_ids or [])
    hard_excluded = negatives | base_exclude | liked_ids | floor_ids
    fresh_excluded = hard_excluded | session_played   # Pass 1 strictness

    # 7. Explore + liked pools — immune to round exhaustion (explore is low-play
    # by definition; liked rotates on its own 8h cooldown), so built once.
    liked_quota = max(0, min(n, round(n * (liked_share if liked_share is not None
                                           else DEFAULT_LIKED_SHARE))))
    non_liked_slots = n - liked_quota
    frac = EXPLORE_SHARE * non_liked_slots
    explore_slots = int(frac) + (1 if rng.random() < (frac - int(frac)) else 0)
    # Cold start: no anchors at all → the whole non-liked budget is exploration.
    if not top_anchors:
        explore_slots = non_liked_slots
    explore_cands: list[StreamCandidate] = []
    if explore_slots > 0:
        negative_vectors = [vectors[t] for t in negatives if t in vectors]
        explore_cands = pool_explore_candidates(
            qdrant_client, collection_name,
            excluded=fresh_excluded, reacted_ids={r.track_id for r in reactions},
            play_counts=play_counts, axis_stats=axis_stats,
            negative_vectors=negative_vectors, axis_names=AXIS_NAMES, rng=rng,
        )

    liked_cands: list[StreamCandidate] = []
    if liked_quota > 0 and liked_ids:
        liked_weights = {
            tid: w for tid, w in aggregate_reaction_weights(
                [r for r in reactions if r.reaction == "like"], now,
            ).items() if w > 0.0
        }
        sampled = sample_liked_tracks(
            liked_weights, recency_hours, liked_quota, rng,
            excluded=base_exclude | session_played,
        )
        _, liked_payloads = _retrieve_track_data(qdrant_client, collection_name, sampled)
        # Drop liked ids that no longer resolve in Qdrant. Likes live in SQLite
        # but re-indexing mints fresh point ids, orphaning old reactions; an
        # unresolved id would otherwise ship as an empty-payload «—» track that
        # 404s on /stream and /lyrics. Pools A/B are immune (built from Qdrant).
        liked_cands = [
            StreamCandidate(track_id=t, payload=liked_payloads[t], pool="liked")
            for t in sampled
            if t in liked_payloads
        ]

    recent_artists = [
        (payloads.get(s.track_id) or {}).get("artist", "").strip().lower()
        for s in session_events[-ARTIST_REPEAT_WINDOW:]
    ]

    def _anchor_main(excluded: set[str]) -> list[StreamCandidate]:
        """Build + score the anchor pool against an exclusion set."""
        pool = pool_anchor_candidates(qdrant_client, collection_name, top_anchors, excluded)
        cands = list(pool.values())
        score_candidates(
            cands, p_final=p_final, confidence=confidence,
            play_counts=play_counts, recency_hours=recency_hours,
            axis_stats=axis_stats, axis_names=AXIS_NAMES,
        )
        return cands

    # 8. Pass 1 — fresh only: today's behaviour (anchor + explore + liked).
    main = _anchor_main(fresh_excluded)
    pool_a_size = len(main)
    explore_picks = explore_cands[:explore_slots]
    chunk = assemble_chunk(
        main, liked_cands,
        n=n - len(explore_picks), liked_share=(liked_quota / n if n else 0.0),
        recent_artists=recent_artists,
    )
    # Explore picks slot in at random positions — exploration shouldn't always
    # land at the tail where it is most likely to be cut off by a re-request.
    for c in explore_picks:
        chunk.insert(rng.randrange(len(chunk) + 1), c)
    chunk = chunk[:n]

    # 9. Pass 2 — relax: the «круг» wrapped. Replay already-heard anchors
    # (oldest-ish first via the recency penalty in the score), still honouring the
    # anti-repeat floor and dislikes. Only the anchor pool needs topping up.
    relaxed_used = False
    if len(chunk) < n:
        chosen = {c.track_id for c in chunk}
        relaxed_main = sorted(_anchor_main(hard_excluded | chosen),
                              key=lambda c: c.score, reverse=True)
        if relaxed_main:
            relaxed_used = True
            chunk.extend(relaxed_main[: n - len(chunk)])

    # 10. Pass 3 — last resort (library ≤ floor): replay least-recently-played.
    # Floor lifted, dislikes stay hard («пока дизлайк стоит»). Guarantees the
    # stream is never empty while a non-disliked track exists in the collection.
    fallback_used = False
    if len(chunk) < n:
        chosen = {c.track_id for c in chunk}
        stale = [
            tid for tid, _ in sorted(recency_hours.items(),
                                     key=lambda kv: kv[1], reverse=True)
            if tid not in negatives and tid not in base_exclude and tid not in chosen
        ]
        need = n - len(chunk)
        _, stale_payloads = _retrieve_track_data(
            qdrant_client, collection_name, stale[:need])
        for tid in stale[:need]:
            if tid in stale_payloads:
                fallback_used = True
                chunk.append(StreamCandidate(
                    track_id=tid, payload=stale_payloads[tid], pool="replay"))

    # 11. Round number — cosmetic (display only, never gates selection): how many
    # times the session has cycled the eligible library.
    round_no = 1
    try:
        total = qdrant_client.count(collection_name=collection_name).count
    except Exception:
        total = None
    if total:
        eligible_size = max(1, total - len(negatives))
        # total session plays (incl. repeats) ÷ library size = how many times the
        # session has cycled. session_played is a SET (≤ library), so it could
        # never exceed round 1 — use the event count.
        round_no = max(1, math.ceil(len(session_events) / eligible_size))

    diagnostics = {
        "n_session_signals": n_session_signals,
        "w_session": round(w_s, 3),
        "anchors": [{"track_id": a.track_id, "weight": round(a.weight, 3)}
                    for a in top_anchors],
        "n_negatives": len(negatives),
        "profile_confidence": round(confidence, 3),
        "axis_stats_source": (axis_stats or {}).get("source"),
        "pool_sizes": {"anchor": pool_a_size, "explore": len(explore_cands),
                       "liked": len(liked_cands)},
        "liked_quota": liked_quota,
        "explore_slots": len(explore_picks),
        "round": round_no,
        "n_floor": len(floor_ids),
        "relaxed": relaxed_used,
        "fallback": fallback_used,
    }
    return {"tracks": chunk, "diagnostics": diagnostics}


# ── Similar tracks: CLAP neighbors re-ranked by sonic axes ──────────────────
# Powers GET /recommend/similar (Recommend tab «похожие» + ai-playlist agent
# tool). Unlike autoplay (pure CLAP order), candidates are re-ranked by a
# blend of CLAP cosine and axis-space closeness to the seed.

SIMILAR_W_CLAP = 0.7
SIMILAR_W_AXES = 0.3
SIMILAR_FETCH_MULT = 3   # fetch limit×3 neighbors before re-ranking


def similar_tracks(
    *,
    qdrant_client,
    collection_name: str,
    seed_track_id: str,
    limit: int = 10,
    exclude_ids: list[str] | None = None,
) -> dict:
    """CLAP top-K of the seed, re-ranked by axis closeness.

    score = 0.7·cos + 0.3·axis_closeness, where axis_closeness maps the RMS
    z-distance between seed and candidate onto [−1, 1] (same scale as the
    stream's axis_match). Without usable axis stats the axis term is 0 for
    everyone — the order gracefully degrades to pure CLAP cosine.

    Returns ``{"seed_track_id", "tracks": [StreamCandidate]}`` with
    ``max_anchor_cos`` = CLAP cosine and ``axis_match`` = axis closeness;
    ``score`` is the blend. Dislikes are hard-filtered.
    """
    from app.resources.clap_features import (
        AXIS_NAMES, blend_axis_stats, load_axis_norm_reference,
    )

    excluded = set(exclude_ids or [])
    excluded.add(seed_track_id)

    # 1. Seed vector + axes.
    seed_vectors, seed_payloads = _retrieve_track_data(
        qdrant_client, collection_name, [seed_track_id],
    )
    seed_vec = seed_vectors.get(seed_track_id)
    if seed_vec is None:
        return {"seed_track_id": seed_track_id, "tracks": []}

    axis_stats = blend_axis_stats(
        MetadataDB.get_axis_norm_stats(collection_name),
        load_axis_norm_reference(),
    )
    seed_z = z_scores_for_axes(
        (seed_payloads.get(seed_track_id) or {}).get("sonic_axes"),
        axis_stats, AXIS_NAMES,
    )

    # 2. CLAP neighbors.
    k = max(limit * SIMILAR_FETCH_MULT, 30)
    try:
        # qdrant-client >= 1.10: query_points replaced the removed .search()
        hits = qdrant_client.query_points(
            collection_name=collection_name,
            query=list(seed_vec),
            using="clap",
            limit=k,
            with_payload=PAYLOAD_EXCLUDE_LYRICS,
        ).points
    except Exception:
        logger.exception("[similar] CLAP search failed for %s", seed_track_id)
        return {"seed_track_id": seed_track_id, "tracks": []}

    # 3. Dislike filter (single batched lookup).
    candidate_ids = [str(h.id) for h in hits]
    reactions = MetadataDB.get_reactions_for_tracks(collection_name, candidate_ids)
    dislikes = {tid for tid, r in reactions.items() if r == "dislike"}

    # 4. Re-rank by cos + axis closeness to the SEED (not the user profile).
    out: list[StreamCandidate] = []
    for h in hits:
        tid = str(h.id)
        if tid in excluded or tid in dislikes:
            continue
        payload = h.payload or {}
        cos = float(h.score or 0.0)
        cand_z = z_scores_for_axes(payload.get("sonic_axes"), axis_stats, AXIS_NAMES)
        axis_closeness = axis_match_score(cand_z, seed_z, 1.0, AXIS_NAMES)
        out.append(StreamCandidate(
            track_id=tid, payload=payload, pool="anchor",
            anchor_track_id=seed_track_id,
            max_anchor_cos=cos, axis_match=axis_closeness,
            score=SIMILAR_W_CLAP * cos + SIMILAR_W_AXES * axis_closeness,
        ))
    out.sort(key=lambda c: c.score, reverse=True)
    return {"seed_track_id": seed_track_id, "tracks": out[:limit]}


# ── Long-term taste profile surface (Recommend tab «центр вкуса») ──────────

ISLANDS_MAX = 6           # taste islands shown in the profile
ISLAND_MEMBERS_MAX = 8    # covers per island (representative first)


def long_term_profile(*, qdrant_client, collection_name: str, now: datetime | None = None) -> dict:
    """Explainable long-term profile: 6 axes (z + level), confidence, islands.

    Pure long-term — no session blending: this is the «кто я как слушатель»
    view, it must be stable across a listening session. Reuses the exact same
    aggregation the stream runs, so what the user sees IS what the stream uses.
    """
    from app.resources.clap_features import (
        AXIS_NAMES, blend_axis_stats, load_axis_norm_reference, z_to_level,
    )

    now = now or datetime.utcnow()

    raw_signals = MetadataDB.get_playback_signals(collection_name, LONG_TERM_EVENT_CAP)
    signals = [PlaybackSignal(**r) for r in raw_signals]
    reactions = []
    for tid, reaction, ts in MetadataDB.get_reactions_with_updated_at(collection_name):
        dt = _parse_iso(ts)
        if dt is not None:
            reactions.append(ReactionSignal(track_id=tid, reaction=reaction, updated_at=dt))

    long_weights = combine_weights(
        aggregate_event_weights(signals, now),
        aggregate_reaction_weights(reactions, now),
    )
    n_signals = len(signals) + len(reactions)

    # Anchors → islands (merge groups carry their member track ids).
    anchor_cands = select_positive_anchors(long_weights)
    member_pool = [a.track_id for a in anchor_cands]
    vectors, payloads = _retrieve_track_data(qdrant_client, collection_name, member_pool)
    merged = merge_anchors(anchor_cands, vectors)
    merged.sort(key=lambda a: a.weight, reverse=True)
    islands = []
    for a in merged[:ISLANDS_MAX]:
        members = []
        for tid in a.members[:ISLAND_MEMBERS_MAX]:
            p = payloads.get(tid) or {}
            members.append({
                "track_id": tid,
                "title": p.get("title") or "—",
                "artist": p.get("artist") or "—",
                "cover_art_path": p.get("cover_art_path"),
            })
        islands.append({
            "track_id": a.track_id,
            "weight": round(a.weight, 3),
            "tracks": members,
        })

    # Axis preferences in z-space + discrete levels.
    axis_stats = blend_axis_stats(
        MetadataDB.get_axis_norm_stats(collection_name),
        load_axis_norm_reference(),
    )
    z_by_track = {
        tid: z for tid, pl in payloads.items()
        if (z := z_scores_for_axes(pl.get("sonic_axes"), axis_stats, AXIS_NAMES))
    }
    p_long, confidence = axis_preferences(long_weights, z_by_track, AXIS_NAMES)
    axes = (
        {a: {"z": round(p_long[a], 3), "level": z_to_level(p_long[a])} for a in AXIS_NAMES}
        if p_long is not None else None
    )

    return {
        "axes": axes,
        "confidence": round(confidence, 3),
        "n_signals": n_signals,
        "islands": islands,
        "axis_stats_source": (axis_stats or {}).get("source"),
    }


# ── Axis playlist: «как обычно, но поспокойнее» (управляемые ручки) ────────

AXIS_PLAYLIST_W_MATCH = 0.85
AXIS_PLAYLIST_W_NOVELTY = 0.15
AXIS_PLAYLIST_SCROLL_CAP = 5000


def axis_playlist(
    *,
    qdrant_client,
    collection_name: str,
    axis_targets: dict[str, float],
    limit: int = 20,
) -> dict:
    """Rank the whole collection against a target z-profile (the radar knobs).

    score = 0.85·closeness(z, target) + 0.15·novelty. Tracks without axis data
    are skipped (nothing to match on); dislikes are hard-filtered. Returns
    empty + a reason when axis stats are unusable — the frontend should hide
    the knobs in that state rather than show a fake ranking.
    """
    from app.resources.clap_features import (
        AXIS_NAMES, blend_axis_stats, load_axis_norm_reference,
    )

    axis_stats = blend_axis_stats(
        MetadataDB.get_axis_norm_stats(collection_name),
        load_axis_norm_reference(),
    )
    if not axis_stats:
        return {"tracks": [], "diagnostics": {"reason": "no_axis_stats", "scanned": 0}}

    targets = {a: float(axis_targets.get(a, 0.0)) for a in AXIS_NAMES}

    # Shared per-collection cache of lyrics-free light payloads; cap to bound
    # the scoring loop (this ranks the whole library against the radar knobs).
    points = light_points(qdrant_client, collection_name)[:AXIS_PLAYLIST_SCROLL_CAP]

    candidate_ids = [tid for tid, _ in points]
    reactions = MetadataDB.get_reactions_for_tracks(collection_name, candidate_ids)
    dislikes = {tid for tid, r in reactions.items() if r == "dislike"}
    play_counts = MetadataDB.get_play_counts_by_track(collection_name)

    out: list[StreamCandidate] = []
    skipped_no_axes = 0
    for tid, payload in points:
        if tid in dislikes:
            continue
        z = z_scores_for_axes(payload.get("sonic_axes"), axis_stats, AXIS_NAMES)
        if z is None:
            skipped_no_axes += 1
            continue
        closeness = axis_match_score(z, targets, 1.0, AXIS_NAMES)
        novelty = 1.0 / (1.0 + play_counts.get(tid, 0))
        out.append(StreamCandidate(
            track_id=tid, payload=payload, pool="axis",
            axis_match=closeness,
            score=AXIS_PLAYLIST_W_MATCH * closeness + AXIS_PLAYLIST_W_NOVELTY * novelty,
        ))
    out.sort(key=lambda c: c.score, reverse=True)
    return {
        "tracks": out[:limit],
        "diagnostics": {
            "scanned": len(points),
            "skipped_no_axes": skipped_no_axes,
            "targets": targets,
            "axis_stats_source": axis_stats.get("source"),
        },
    }
