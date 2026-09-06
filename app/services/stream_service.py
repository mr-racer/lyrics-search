"""Stream RecSys — session-driven personalized radio («Поток»).

Design: docs/superpowers/specs/2026-08-03-session-recsys-design.md (session
engine), 2026-06-29-fire-wave-recsys-design.md (огонёк/вода/волна).

Stateless: every request rebuilds the listener baseline, the session profile
and the candidate pools from SQLite + Qdrant, then assembles the next chunk.

This module is the ORCHESTRATOR. The machinery lives in ``app/services/stream/``:

    signals      — event/reaction primitives, reward weights, anchor merge
    calibration  — CLAP cosine → per-collection percentile
    baseline     — the listener's own skip/completion/reaction rates → weights
    session      — reaction cutoff, skip forgiveness, ±clusters, carryover
    pools        — fresh/familiar pools, slider quotas, chunk assembly

What stays here: ``next_chunk``, the long-term profile surface (islands, axes,
«вайбики»), ``similar_tracks``, ``axis_playlist`` and the vibe album rail — the
surfaces the 2026-08-03 redesign deliberately did not touch. The whole
``stream.signals``/``stream.session`` surface is re-exported below so existing
``from app.services.stream_service import is_skip`` imports keep working.
"""

from __future__ import annotations

import logging
import math
import random as _random_module
import time
from datetime import datetime

import numpy as np

from app.resources.metadata_db import MetadataDB
from app.resources.qdrant_utils import light_map, light_points
from app.services.stream import baseline as baseline_mod
from app.services.stream import calibration as calib_mod
from app.services.stream import pools as pools_mod
from app.services.stream import session as session_mod
from app.services.stream.pools import (  # noqa: F401  (public surface)
    DEFAULT_CHUNK_N,
    DEFAULT_LIKED_SHARE,
    FRESH_WINDOW_DAYS,
    RECENT_PENALTY_HALFLIFE_H,
    W_AFFINITY,
    W_AXIS,
    W_NOVELTY,
    W_RECENT,
    W_REPULSION,
    assemble_chunk,
    is_fresh,
    score_candidate,
    stratify,
)
from app.services.stream.session import (  # noqa: F401  (public surface)
    FIRE_BASE,
    FIRE_COUNT_FULL,
    FIRE_COUNT_ZERO,
    FIRE_TIME_MAX_H,
    H_REACTION_DAYS,
    H_WATER_DAYS,
    REACTION_GRACE_SEC,
    WATER_BASE,
    WATER_MUTE_DAYS,
    WATER_TAIL_DAYS,
    aggregate_event_weights,
    aggregate_taste_anchors,
    fire_count_factor,
    fire_time_factor,
    muted_track_ids,
    reaction_contribution,
    reaction_cutoffs,
    superseded,
    water_charge,
    water_weights,
    weight_events,
)
from app.services.stream.signals import (  # noqa: F401  (public surface)
    ANCHOR_MERGE_THRESHOLD,
    AXIS_MATCH_DIST_SCALE,
    FULL_RATIO,
    H_IMPLICIT_DAYS,
    H_LIKE_DAYS,
    MIN_TRACK_DURATION_SEC,
    MOST_RATIO,
    SKIP_ABS_SEC,
    SKIP_SHORT_RATIO,
    SKIP_SHORT_TRACK_SEC,
    W_DISLIKE,
    W_FULL,
    W_LIKE,
    W_MOST,
    W_NEUTRAL,
    W_REPLAY,
    W_SKIP,
    Anchor,
    FireSignal,
    PlaybackSignal,
    ReactionSignal,
    StreamCandidate,
    _age_days,
    axis_match_score,
    base_weight,
    clap_vector,
    combine_weights,
    decayed,
    duration_ok as _duration_ok,
    is_explore_source,
    is_skip,
    latest_per_track,
    merge_anchors,
    parse_iso as _parse_iso,
    select_positive_anchors,
    unit as _unit,
    z_scores_for_axes,
)

logger = logging.getLogger(__name__)

# ── Long-term taste: острова (low learning rate) ────────────────────────────
# Islands are fed ONLY by fires (fat) + ≥85% completions (weak). Partial
# listens and skips never shape long-term taste, so it drifts slowly.
FIRE_ISLAND_DEPOSIT = 0.35
COMPLETION_DEPOSIT = 0.2
H_ISLAND_DAYS = 30.0

# «Вайбики» — ephemeral mood clusters: what the user plays RIGHT NOW. Same
# anchor-merge machinery as islands, but on a days-scale clock, with pairs
# allowed and an explicit negative loop: skips/water on tracks that sound like
# a vibe (cos ≥ VIBE_NEG_SIM to its centroid) press its net weight back down.
H_VIBE_DAYS = 2.5
VIBE_FIRE_DEPOSIT = 0.7
VIBE_FULL_DEPOSIT = 0.4
VIBE_MOST_DEPOSIT = 0.15
VIBE_POOL_SIZE = 30
VIBE_MIN_MEMBERS = 2
VIBES_MAX = 3
VIBE_NEG_SIM = 0.88
VIBE_SKIP_PENALTY = 0.35
VIBE_WATER_PENALTY = 0.6
VIBE_MIN_NET = 0.3
VIBE_SIGNAL_MAX_AGE_DAYS = 10.0

# «Favorites» — the slider's liked pool, COMPUTED (hearts removed): most
# fired (dominant) + most listened-through.
FAV_FIRE_W = 0.35
FAV_LISTEN_W = 0.3
FAV_POOL_SIZE = 60

# Fresh-session explore warmup («от островов, широко»): a cold session starts
# with a wide random slice that tapers to the steady-state floor as signals land.
WARMUP_SIGNALS = 8
WARMUP_EXPLORE_SHARE = 0.5
EXPLORE_SHARE = 0.12
IDLE_RESET_H = 30.0

# Profile aggregation.
# «≥2 скипа, decayed»: two fresh skips sum to 2.0 and fade below this after
# roughly two weeks of not touching the track.
SKIP_NEG_DECAYED_COUNT = 1.5
CONFIDENCE_SATURATION = 5.0     # axis-pref confidence = min(1, Σpos_weight / this)

# Anti-repeat floor (design 2026-06-14: «бесконечный круг»). The «жёсткий пол»
# under round replay: these stay hard-excluded even after a «круг» wraps, so the
# just-heard tracks never recur immediately. Everything older is only soft-demoted
# (relax pass + recency penalty) — «жёсткий пол + мягкий хвост».
ANTIREPEAT_FLOOR_TRACKS = 10
ANTIREPEAT_FLOOR_MINUTES = 30

LIKED_COOLDOWN_H = 8.0     # hard «не чаще раза в 8 часов» for the favorites pool
# Newest events fed into profile building. Raised from 2000 on 2026-09-06: the
# live library had already logged 3428 events, so the cap was binding and the
# «long-term» islands — the one force that pulls a session OUT of its current
# region — could see only about a month back. Islands decay on H_ISLAND_DAYS=30
# anyway, so a deeper window costs little and restores the pull.
LONG_TERM_EVENT_CAP = 6000

# Similar-tracks re-rank blend (GET /recommend/similar).
SIMILAR_W_CLAP = 0.7
SIMILAR_W_AXES = 0.3
SIMILAR_FETCH_MULT = 3


def aggregate_reaction_weights(
    reactions: list[ReactionSignal], now: datetime,
) -> dict[str, float]:
    """±1.0 per reaction, decayed with H_LIKE from ``updated_at`` (last flip —
    reaction history is not stored, which also kills like↔dislike oscillation).

    Legacy surface: the heart UI is gone, but stored dislikes still act as a
    hard filter and old rows keep their meaning.
    """
    out: dict[str, float] = {}
    for r in reactions:
        base = W_LIKE if r.reaction == "like" else W_DISLIKE if r.reaction == "dislike" else 0.0
        if base == 0.0:
            continue
        out[r.track_id] = out.get(r.track_id, 0.0) + decayed(
            base, _age_days(r.updated_at, now), H_LIKE_DAYS,
        )
    return out


def negative_track_ids(
    signals: list[PlaybackSignal],
    reactions: list[ReactionSignal],
    now: datetime,
    *,
    cutoffs: dict[str, datetime] | None = None,
) -> set[str]:
    """Hard-filter set: current dislikes (absolute, no decay — «пока дизлайк
    стоит») + tracks whose decayed skip count passes the multi-skip threshold.

    Skips already superseded by an explicit reaction don't count (§2.1): if you
    fired a track and skipped it once, the fire speaks, not the skip.

    Skips on the wave's OWN exploratory picks don't count either (2026-09-06
    §3.4). This ban is what removes a track from the stream for over a week, and
    charging it for a guess the listener never asked for is how the reachable
    library shrinks: on prod, exploratory tracks are first plays, and first
    plays are skipped no more often (16.0%) than familiar ones.
    """
    cutoffs = cutoffs or {}
    out = {r.track_id for r in reactions if r.reaction == "dislike"}
    skip_counts: dict[str, float] = {}
    for ev in signals:
        if not is_skip(ev.played_sec, ev.total_dur) or superseded(ev, cutoffs):
            continue
        if is_explore_source(getattr(ev, "source", None)):
            continue
        skip_counts[ev.track_id] = skip_counts.get(ev.track_id, 0.0) + decayed(
            1.0, _age_days(ev.played_at, now), H_IMPLICIT_DAYS,
        )
    out.update(tid for tid, c in skip_counts.items() if c >= SKIP_NEG_DECAYED_COUNT)
    return out


def count_session_signals(
    session_events: list[PlaybackSignal],
    session_reactions: list[ReactionSignal],
    *,
    cutoffs: dict[str, datetime] | None = None,
) -> int:
    """Signals = session events that carry information (nonzero weight after
    the reaction cutoff) + reactions made during the session."""
    weights = weight_events(session_events, cutoffs=cutoffs)
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


def sample_liked_tracks(
    liked_weights: dict[str, float],
    recency_hours: dict[str, float],
    n_needed: int,
    rng,
    *,
    cooldown_h: float = LIKED_COOLDOWN_H,
    excluded: set[str] | frozenset = frozenset(),
) -> list[str]:
    """Favorites sampler: weight-proportional, anti-repeat, honest rotation.

    Two-pass topup: the first pass respects the hard cooldown; if the quota is
    still unfilled (tiny liked list / slider at 100%), the cooldown is relaxed
    rather than under-filling the chunk.
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


def decayed_fire_counts(
    signals: list[FireSignal], now: datetime, *, half_life: float,
) -> dict[str, float]:
    """Decayed count of fires per track (water ignored). Powers the long-term
    island deposit and the «favorites» rank — the clearest explicit signal."""
    out: dict[str, float] = {}
    for s in signals:
        if s.kind != "fire":
            continue
        out[s.track_id] = out.get(s.track_id, 0.0) + decayed(
            1.0, _age_days(s.created_at, now), half_life,
        )
    return out


def completion_counts(
    signals: list[PlaybackSignal], now: datetime, *, half_life: float,
    cutoffs: dict[str, datetime] | None = None,
) -> dict[str, float]:
    """Decayed count of deep (≥85%) listens per track. Partial listens and skips
    are excluded — only a finished song is a clean long-term signal.

    Listens superseded by an explicit reaction (§2.1) don't count either: a
    track watered at the three-minute mark must not enter «favorites» just
    because the last minute kept playing.
    """
    out: dict[str, float] = {}
    for ev in signals:
        if not ev.total_dur or ev.total_dur <= 0.0:
            continue
        if ev.played_sec / ev.total_dur < FULL_RATIO:
            continue
        if superseded(ev, cutoffs or {}):
            continue
        out[ev.track_id] = out.get(ev.track_id, 0.0) + decayed(
            1.0, _age_days(ev.played_at, now), half_life,
        )
    return out


def island_taste_weights(
    fire_signals: list[FireSignal],
    playback_signals: list[PlaybackSignal],
    now: datetime,
    *,
    cutoffs: dict[str, datetime] | None = None,
) -> dict[str, float]:
    """Long-term taste weights for island building: fat fire deposit + weak ≥85%
    completion deposit, both on the slow island half-life. This is the whole
    «низкий learning rate» — noisy mid-listens never enter."""
    fires = {tid: FIRE_ISLAND_DEPOSIT * c
             for tid, c in decayed_fire_counts(
                 fire_signals, now, half_life=H_ISLAND_DAYS).items()}
    comps = {tid: COMPLETION_DEPOSIT * c
             for tid, c in completion_counts(
                 playback_signals, now, half_life=H_ISLAND_DAYS,
                 cutoffs=cutoffs).items()}
    return combine_weights(fires, comps)


# ── «Вайбики»: fast-clock mood clusters with a negative feedback loop ───────

def partial_counts(
    signals: list[PlaybackSignal], now: datetime, *, half_life: float,
    cutoffs: dict[str, datetime] | None = None,
) -> dict[str, float]:
    """Decayed count of partial (65–85%) listens per track. Only the vibe layer
    consumes these — its days-scale clock keeps the noise from accumulating."""
    out: dict[str, float] = {}
    for ev in signals:
        if not ev.total_dur or ev.total_dur <= 0.0:
            continue
        ratio = ev.played_sec / ev.total_dur
        if not (MOST_RATIO <= ratio < FULL_RATIO):
            continue
        if superseded(ev, cutoffs or {}):
            continue
        out[ev.track_id] = out.get(ev.track_id, 0.0) + decayed(
            1.0, _age_days(ev.played_at, now), half_life,
        )
    return out


def vibe_taste_weights(
    fire_signals: list[FireSignal],
    playback_signals: list[PlaybackSignal],
    now: datetime,
    *,
    cutoffs: dict[str, datetime] | None = None,
) -> dict[str, float]:
    """Positive vibe weights: fires + full listens + partial listens. Fires decay
    on the short H_REACTION_DAYS clock (a fire is a «вайб дня», not a taste);
    listens stay on H_VIBE_DAYS. Skips and water never deposit — they act as
    pressure against already-formed vibes instead (see current_vibes)."""
    fires = {tid: VIBE_FIRE_DEPOSIT * c
             for tid, c in decayed_fire_counts(
                 fire_signals, now, half_life=H_REACTION_DAYS).items()}
    fulls = {tid: VIBE_FULL_DEPOSIT * c
             for tid, c in completion_counts(
                 playback_signals, now, half_life=H_VIBE_DAYS,
                 cutoffs=cutoffs).items()}
    mosts = {tid: VIBE_MOST_DEPOSIT * c
             for tid, c in partial_counts(
                 playback_signals, now, half_life=H_VIBE_DAYS,
                 cutoffs=cutoffs).items()}
    return combine_weights(fires, fulls, mosts)


def _vibe_centroid(member_ids: list[str], vectors: dict) -> np.ndarray | None:
    """Normalized mean of the members' unit CLAP vectors (members already sit
    within the merge threshold of each other, so unweighted is fine)."""
    acc = None
    for tid in member_ids:
        if tid not in vectors:
            continue
        v = _unit(vectors[tid])
        if v is None:
            continue
        acc = v if acc is None else acc + v
    return _unit(acc) if acc is not None else None


def current_vibes(
    *,
    qdrant_client,
    collection_name: str,
    signals: list[PlaybackSignal],
    taste_signals: list[FireSignal],
    now: datetime,
) -> list[dict]:
    """Up to VIBES_MAX ephemeral mood clusters («вайбики»).

    Positive side: vibe_taste_weights → top-VIBE_POOL_SIZE anchors → greedy
    merge (same soft threshold as the profile mosaic), pairs allowed. Negative
    side: each recent skip/water whose track sounds like a vibe's centroid
    (cos ≥ VIBE_NEG_SIM) presses that vibe down; a vibe whose net weight falls
    below VIBE_MIN_NET dissolves — «юзер уже не хочет слушать такое».
    """
    influencing = [s for s in signals if getattr(s, "influence", True)]
    # The vibe layer's own rules are unchanged; it just consumes signals with
    # the reaction cutoff applied, so a watered track can no longer seed a vibe.
    cutoffs = reaction_cutoffs(taste_signals)
    weights = vibe_taste_weights(taste_signals, influencing, now, cutoffs=cutoffs)
    cands = select_positive_anchors(weights, top_m=VIBE_POOL_SIZE)
    if not cands:
        return []

    skip_events = [
        s for s in influencing
        if is_skip(s.played_sec, s.total_dur)
        and not superseded(s, cutoffs)
        and _age_days(s.played_at, now) <= VIBE_SIGNAL_MAX_AGE_DAYS
    ]
    waters = [
        t for t in taste_signals
        if t.kind == "water"
        and _age_days(t.created_at, now) <= VIBE_SIGNAL_MAX_AGE_DAYS
    ]
    fetch_ids = list(dict.fromkeys(
        [a.track_id for a in cands]
        + [s.track_id for s in skip_events]
        + [w.track_id for w in waters]))
    vectors = _clap_vectors(qdrant_client, collection_name, fetch_ids)
    payloads = _track_meta(qdrant_client, collection_name)

    merged = merge_anchors(cands, vectors, threshold=ISLAND_MERGE_THRESHOLD)
    merged = [a for a in merged if len(a.members) >= VIBE_MIN_MEMBERS]
    if not merged:
        return []

    # (unit vector, decayed penalty) per recent negative event.
    negatives: list[tuple[np.ndarray, float]] = []
    for ev in skip_events:
        v = _unit(vectors[ev.track_id]) if ev.track_id in vectors else None
        if v is not None:
            negatives.append((v, VIBE_SKIP_PENALTY * decayed(
                1.0, _age_days(ev.played_at, now), H_VIBE_DAYS)))
    for w in waters:
        v = _unit(vectors[w.track_id]) if w.track_id in vectors else None
        if v is not None:
            negatives.append((v, VIBE_WATER_PENALTY * decayed(
                1.0, _age_days(w.created_at, now), H_REACTION_DAYS)))

    scored: list[tuple[float, Anchor]] = []
    for a in merged:
        vibe_vec = _vibe_centroid(a.members, vectors)
        if vibe_vec is None:
            continue
        pressure = sum(pen for v, pen in negatives
                       if float(v @ vibe_vec) >= VIBE_NEG_SIM)
        net = a.weight - pressure
        if net >= VIBE_MIN_NET:
            scored.append((net, a))
    scored.sort(key=lambda p: p[0], reverse=True)

    vibes = []
    for net, a in scored[:VIBES_MAX]:
        members = []
        for tid in a.members[:ISLAND_MEMBERS_MAX]:
            p = payloads.get(tid) or {}
            members.append({
                "track_id": tid,
                "title": p.get("title") or "—",
                "artist": p.get("artist") or "—",
                "album": p.get("album"),
                "genre": p.get("genre"),
                "cover_art_path": p.get("cover_art_path"),
            })
        vibes.append({
            "track_id": a.track_id,
            "weight": round(net, 3),
            "tracks": members,
        })
    return vibes


# ── «Favorites»: computed liked pool for the слайдер (hearts removed) ───────

def favorite_weights(
    fire_counts: dict[str, float],
    listen_counts: dict[str, float],
    *,
    fire_w: float = FAV_FIRE_W,
    listen_w: float = FAV_LISTEN_W,
) -> dict[str, float]:
    """«Любимые» = most fired (dominant) + most listened-through. Replaces the
    heart-like pool that used to feed the liked/new slider."""
    out: dict[str, float] = {}
    for tid, c in fire_counts.items():
        out[tid] = out.get(tid, 0.0) + fire_w * c
    for tid, c in listen_counts.items():
        out[tid] = out.get(tid, 0.0) + listen_w * c
    return {tid: w for tid, w in out.items() if w > 0.0}


# ── Fresh-session explore warmup («от островов, широко») ────────────────────

def explore_share_for_warmth(
    warmth: float,
    *,
    base: float = EXPLORE_SHARE,
    warmup: float = WARMUP_EXPLORE_SHARE,
    saturation: int = WARMUP_SIGNALS,
) -> float:
    """Explore share as a function of session «warmth» (signal count): high for a
    fresh session, lerping down to the steady-state baseline by ``saturation``
    signals. Long-term anchors stay in play — this only widens the spread."""
    ramp = max(0.0, min(1.0, warmth / saturation)) if saturation > 0 else 1.0
    return warmup + (base - warmup) * ramp


# ── Orchestration: GET /stream/next entry point ─────────────────────────────

def _fire_signals(rows: list) -> list[FireSignal]:
    """Parse ``(track_id, kind, created_at_iso)`` journal rows into FireSignals,
    dropping any with an unparseable timestamp."""
    out: list[FireSignal] = []
    for tid, kind, ts in rows:
        dt = _parse_iso(ts)
        if dt is not None:
            out.append(FireSignal(track_id=tid, kind=kind, created_at=dt))
    return out


def _clap_vectors(
    qdrant_client, collection_name: str, track_ids: list[str],
) -> dict[str, np.ndarray]:
    """Batch-fetch ``{id: clap_vector}`` — the ONE thing this module still asks
    Qdrant for on a per-request basis.

    Explicitly ``with_payload=False``: track metadata comes from the SQLite
    mirror (``_track_meta``). Dragging payloads along a vector retrieve is what
    made a stream chunk take tens of seconds.
    """
    if not track_ids:
        return {}
    try:
        pts = qdrant_client.retrieve(
            collection_name=collection_name, ids=track_ids,
            with_payload=False, with_vectors=["clap"],
        )
    except Exception:
        logger.exception("[stream] clap vector retrieve failed")
        return {}
    vectors: dict[str, np.ndarray] = {}
    for p in pts:
        v = clap_vector(p)
        if v:
            vectors[str(p.id)] = np.asarray(v, dtype=np.float32)
    return vectors


def _track_meta(qdrant_client, collection_name: str) -> dict[str, dict]:
    """``{track_id: payload}`` for the whole collection, from SQLite.

    ``light_map`` reads the ``track_metadata`` mirror (card fields +
    ``sonic_axes``) with a 90 s cache and only falls back to a fields-projected
    Qdrant scroll when the mirror is empty. Everything the stream needs to
    render a track — title/artist/album/year/genre/duration/paths/axes — is in
    there, so no per-request payload transfer is required.
    """
    return light_map(qdrant_client, collection_name)


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
    """Stateless «Поток»: rebuild the session profile from SQLite, pull
    candidates from Qdrant, score, assemble.

    Returns ``{"tracks": [StreamCandidate], "diagnostics": {…},
    "session_adaptation": {…} | None}``.

    ``exclude_ids`` covers the frontend prefetch buffer — tracks already issued
    but not yet reported as playback events.

    The shape of a request (design §4–6):

      1. signals + reactions from SQLite
      2. listener baseline  → adaptive weights
      3. CLAP calibration   → percentile similarity
      4. session profile    → positive/negative clusters
      5. neighbourhood maps → affinity (C⁺) and repulsion (C⁻)
      6. split fresh/familiar, apply the slider quota, assemble
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
    # latest-wins: at most one active reaction per track (water cancels an older
    # fire and vice versa).
    session_taste = latest_per_track(_fire_signals(
        MetadataDB.get_taste_signals(collection_name, session_id=session_id)))
    all_taste = latest_per_track(_fire_signals(
        MetadataDB.get_taste_signals(collection_name)))

    session_events = [s for s in signals if s.session_id == session_id]
    influencing = [s for s in signals if getattr(s, "influence", True)]
    cutoffs = reaction_cutoffs(all_taste)

    # 2–3. Who this listener is, and what «similar» means in this library.
    listener = baseline_mod.compute(influencing, all_taste)
    # The whole library's metadata, from SQLite (cached) — this is what every
    # candidate is rendered from, and it doubles as the track count.
    meta = _track_meta(qdrant_client, collection_name)
    total = len(meta) or None
    calibration = calib_mod.load(qdrant_client, collection_name, n_tracks=total)

    # 4. Session profile: what we pull toward and push away from.
    long_weights = island_taste_weights(all_taste, influencing, now, cutoffs=cutoffs)

    def _fetch_vectors(ids):
        return _clap_vectors(qdrant_client, collection_name, ids)

    profile = session_mod.build(
        signals=signals,
        session_id=session_id,
        session_taste=session_taste,
        all_taste=all_taste,
        long_weights=long_weights,
        baseline=listener,
        calibration=calibration,
        fetch_vectors=_fetch_vectors,
        now=now,
    )

    negatives = negative_track_ids(signals, reactions, now, cutoffs=cutoffs)

    # 5. Per-track stats + exclusion sets.
    play_counts = MetadataDB.get_play_counts_by_track(collection_name)
    recency_hours: dict[str, float] = {}
    for tid, iso in MetadataDB.get_play_recency_map(collection_name).items():
        dt = _parse_iso(iso)
        if dt is not None:
            recency_hours[tid] = max(0.0, (now - dt).total_seconds() / 3600.0)

    floor_ids = _anti_repeat_floor(recency_hours)
    base_exclude = set(exclude_ids or [])
    # Watered tracks are muted outright for WATER_MUTE_DAYS (§3) — the debuff
    # the old 4-hour, session-scoped water penalty never actually delivered.
    hard_excluded = negatives | base_exclude | floor_ids | profile.muted
    fresh_excluded = hard_excluded | profile.session_played

    # Axis preferences in z-space (shrinkage-blended collection stats).
    axis_stats = blend_axis_stats(
        MetadataDB.get_axis_norm_stats(collection_name),
        load_axis_norm_reference(),
    )
    profile_ids = list(dict.fromkeys(
        list(profile.positive_weights) + list(long_weights)))[:200]
    z_by_track = {
        tid: z for tid in profile_ids
        if (z := z_scores_for_axes(
            (meta.get(tid) or {}).get("sonic_axes"), axis_stats, AXIS_NAMES))
    }
    p_long, conf_long = axis_preferences(long_weights, z_by_track, AXIS_NAMES)
    p_sess, conf_sess = axis_preferences(profile.positive_weights, z_by_track, AXIS_NAMES)
    w_long = profile.w_long
    p_final = blend_axis_preferences(p_sess, p_long, w_long, AXIS_NAMES)
    confidence = (1.0 - w_long) * conf_sess + w_long * conf_long

    # 6. Neighbourhood maps. One Qdrant search per cluster centroid; a candidate
    #    absent from every negative neighbourhood simply has zero repulsion.
    repulsion, _neg_owner, neg_sim = pools_mod.cluster_neighbourhood(
        qdrant_client, collection_name, profile.negative, calibration,
        limit=pools_mod.NEG_FETCH_K, use_k=True,
    )
    # One search, two zones. The near field (rank < FRESH_FETCH_K) is the pool
    # the wave has always drawn from and is left byte-identical; everything
    # between it and BAND_FETCH_K is the band, which only ever fills a reserved
    # exploration slot. Deepening the search must NOT widen the ordinary pool —
    # that would quietly change the near-field behaviour the owner likes.
    cand_rank: dict[str, int] = {}
    affinity_all, cand_owner, cand_sim = pools_mod.cluster_neighbourhood(
        qdrant_client, collection_name, profile.positive, calibration,
        limit=pools_mod.BAND_FETCH_K, excluded=fresh_excluded,
        rank_out=cand_rank,
    )
    affinity = {t: v for t, v in affinity_all.items()
                if cand_rank.get(t, 1 << 30) < pools_mod.FRESH_FETCH_K}

    # Computed over the whole search, near field and band alike — the band
    # candidates are scored by the same rules as everyone else.
    axis_match_by_id = {
        tid: axis_match_score(
            z_scores_for_axes((meta.get(tid) or {}).get("sonic_axes"),
                              axis_stats, AXIS_NAMES),
            p_final, confidence, AXIS_NAMES)
        for tid in affinity_all
    }

    def _pool_of(tid: str) -> str:
        return "fresh" if pools_mod.is_fresh(tid, recency_hours) else "familiar"

    # Slider: «РЕДКОЕ» ↔ «ЛЮБИМОЕ». liked_share is what the frontend persists,
    # so the fresh quota is its complement.
    if liked_share is None:
        liked_share = DEFAULT_LIKED_SHARE
    liked_share = max(0.0, min(1.0, liked_share))
    fresh_quota = max(0, min(n, round(n * (1.0 - liked_share))))
    fresh_is_hard = liked_share <= 0.0

    # Favorites feed the familiar side of the slider. Sampled BEFORE the
    # neighbourhood candidates are labelled, so a favorite is served as
    # pool="liked" rather than being swallowed by the generic familiar pool —
    # and then held out of that pool so it can't be served twice.
    fav_weights = favorite_weights(
        decayed_fire_counts(all_taste, now, half_life=H_LIKE_DAYS),
        completion_counts(influencing, now, half_life=H_IMPLICIT_DAYS, cutoffs=cutoffs),
    )
    fav_top = dict(sorted(fav_weights.items(), key=lambda kv: kv[1],
                          reverse=True)[:FAV_POOL_SIZE])
    liked_quota = n - fresh_quota
    liked_cands: list[StreamCandidate] = []
    if liked_quota > 0 and fav_top:
        # NOT gated by the anti-repeat floor: that floor is «the last 10 tracks
        # played», which on a small history is everything the user has ever
        # heard — and favorites are by definition heard. Their rotation is
        # governed by LIKED_COOLDOWN_H instead. Dislikes, the prefetch buffer,
        # the water mute and this session's plays still apply.
        sampled = sample_liked_tracks(
            fav_top, recency_hours, liked_quota, rng,
            excluded=negatives | base_exclude | profile.muted | profile.session_played,
        )
        # Drop favorites that no longer resolve in the library: signals live in
        # SQLite but re-indexing mints fresh point ids, orphaning old ones; an
        # unresolved id would ship as an empty-payload «—» track that 404s.
        for tid in sampled:
            pl = meta.get(tid)
            if pl is None or not _duration_ok(pl):
                continue
            liked_cands.append(StreamCandidate(
                track_id=tid, payload=pl, pool="liked",
                score=pools_mod.score_candidate(
                    affinity=1.0, repulsion=repulsion.get(tid, 0.0),
                    axis_match=0.0, play_count=play_counts.get(tid, 0),
                    recency_h=recency_hours.get(tid),
                ),
            ))

    sampled_liked = {c.track_id for c in liked_cands}
    scored = pools_mod.build_candidates(
        affinity_by_id={t: a for t, a in affinity.items() if t not in sampled_liked},
        repulsion_by_id=repulsion,
        payload_by_id=meta,
        owner_by_id=cand_owner,
        axis_match_by_id=axis_match_by_id,
        play_counts=play_counts,
        recency_hours=recency_hours,
        pool_of=_pool_of,
    )
    fresh_cands = [c for c in scored if c.pool == "fresh"]
    familiar_cands = [c for c in scored if c.pool == "familiar"] + liked_cands

    # Exploration — filter-bubble insurance. Wide on a cold session, tapering
    # to the floor as the session finds itself; a burst of skips widens it again
    # (the only survivor of the old pivot).
    stratified_share = max(
        pools_mod.FRESH_STRATIFIED_SHARE,
        explore_share_for_warmth(profile.n_signals),
    )
    if pools_mod.skip_burst(session_events, is_skip):
        stratified_share = max(stratified_share, pools_mod.SKIP_BURST_STRATIFIED_SHARE)
    # Counted over the session's own timeline, not per chunk: the old
    # per-chunk `int(round(fresh_quota * share))` rounded 0.5 to zero and served
    # no exploration at all in the steady state (design §3.2).
    n_explore = pools_mod.explore_slots(
        len(session_events), n, stratified_share * (1.0 - liked_share))
    # True cold start — no clusters at all, so «familiar» is an empty category
    # by definition. The random slice has to carry the WHOLE chunk, including
    # the slots the slider nominally owes to the liked side; otherwise a fresh
    # account with the slider on «ЛЮБИМОЕ» gets an empty stream.
    if not profile.positive:
        n_explore = max(n_explore, n)

    # The band fills the reserved slots first; the random sampler is the
    # fallback for when the band is dry (cold start, or a library too small to
    # have a band at all).
    explore_cands = pools_mod.band_candidates(
        value_by_id={t: v for t, v in affinity_all.items() if t not in sampled_liked},
        rank_by_id=cand_rank,
        sim_by_id=cand_sim,
        owner_by_id=cand_owner,
        payload_by_id=meta,
        axis_match_by_id=axis_match_by_id,
        repulsion_by_id=repulsion,
        play_counts=play_counts,
        recency_hours=recency_hours,
    )
    n_band_available = len(explore_cands)
    if n_explore > len(explore_cands):
        picks = pools_mod.stratified_fresh(
            meta,
            excluded=(fresh_excluded | {c.track_id for c in fresh_cands}
                      | {c.track_id for c in explore_cands}),
            recency_hours=recency_hours,
            repulsion_by_id=repulsion, neg_sim_by_id=neg_sim,
            axis_stats=axis_stats, axis_names=AXIS_NAMES,
            p_final=p_final, confidence=confidence,
            play_counts=play_counts, pool_size=n_explore * 3, rng=rng,
            pool_label="explore",
        )
        rng.shuffle(picks)
        explore_cands.extend(picks[: n_explore - n_band_available])

    chunk = pools_mod.assemble_chunk(
        fresh_cands, familiar_cands,
        n=n, fresh_quota=fresh_quota,
        recent_artists=[
            (meta.get(s.track_id) or {}).get("artist", "").strip().lower()
            for s in session_events[-pools_mod.ARTIST_REPEAT_WINDOW:]
        ],
        fresh_is_hard=fresh_is_hard,
        explore=explore_cands, n_explore=n_explore,
    )

    # 7. Relax + fallback — familiar slots only. Freshness is a promise, so an
    #    empty fresh pool yields a short chunk instead of quietly serving
    #    something heard last week.
    relaxed_used = False
    fallback_used = False
    if len(chunk) < n and not fresh_is_hard:
        chosen = {c.track_id for c in chunk}
        relaxed, relaxed_owner, _ = pools_mod.cluster_neighbourhood(
            qdrant_client, collection_name, profile.positive, calibration,
            limit=pools_mod.FRESH_FETCH_K, excluded=hard_excluded | chosen,
        )
        relaxed_cands = pools_mod.build_candidates(
            affinity_by_id=relaxed, repulsion_by_id=repulsion,
            payload_by_id=meta, owner_by_id=relaxed_owner,
            axis_match_by_id={
                tid: axis_match_score(
                    z_scores_for_axes((meta.get(tid) or {}).get("sonic_axes"),
                                      axis_stats, AXIS_NAMES),
                    p_final, confidence, AXIS_NAMES)
                for tid in relaxed
            },
            play_counts=play_counts,
            recency_hours=recency_hours, pool_of=lambda _t: "familiar",
        )
        relaxed_cands.sort(key=lambda c: c.score, reverse=True)
        if relaxed_cands:
            relaxed_used = True
            chunk.extend(relaxed_cands[: n - len(chunk)])

    if len(chunk) < n and not fresh_is_hard:
        # Last resort (library ≤ floor): replay the least-recently-played.
        # Dislikes and the water mute stay hard.
        chosen = {c.track_id for c in chunk}
        stale = [
            tid for tid, _ in sorted(recency_hours.items(),
                                     key=lambda kv: kv[1], reverse=True)
            if tid not in negatives and tid not in base_exclude
            and tid not in profile.muted and tid not in chosen
        ]
        need = n - len(chunk)
        for tid in stale[:need]:
            pl = meta.get(tid)
            if pl is not None and _duration_ok(pl):
                fallback_used = True
                chunk.append(StreamCandidate(
                    track_id=tid, payload=pl, pool="replay"))

    # 8. Round number — cosmetic (display only, never gates selection).
    round_no = 1
    if total:
        eligible_size = max(1, total - len(negatives))
        round_no = max(1, math.ceil(len(session_events) / eligible_size))

    n_fresh_served = sum(1 for c in chunk if c.pool == "fresh")
    diagnostics = {
        "n_session_signals": profile.n_signals,
        "w_long": round(profile.w_long, 3),
        "carryover_w": round(profile.carryover_w, 3),
        "forgiven_skips": profile.forgiven_skips,
        "baseline": listener.as_diagnostics(),
        "calibration": calibration.source,
        "clusters": {
            "positive": [{"track_id": c.track_id, "weight": round(c.weight, 3),
                          "members": len(c.members)} for c in profile.positive],
            "negative": [{"track_id": c.track_id, "kind": c.kind,
                          "weight": round(c.weight, 3),
                          "members": len(c.members)} for c in profile.negative],
        },
        "n_negatives": len(negatives),
        "n_muted": len(profile.muted),
        # Candidates the CLAP search returned but the SQLite mirror doesn't
        # know — they are dropped, so a non-zero value means the mirror needs a
        # backfill (scripts/backfill_track_metadata.py).
        "meta_missing": sum(1 for tid in affinity if tid not in meta),
        "n_meta": len(meta),
        "profile_confidence": round(confidence, 3),
        "axis_stats_source": (axis_stats or {}).get("source"),
        "pool_sizes": {"fresh": len(fresh_cands), "familiar": len(familiar_cands)},
        "fresh_quota": fresh_quota,
        "fresh_served": n_fresh_served,
        "fresh_exhausted": bool(n_fresh_served < fresh_quota),
        "stratified_share": round(stratified_share, 3),
        # Exploration, the 2026-09-06 surface: how many slots were reserved,
        # how many the band could actually fill, and how many were served. A
        # persistent explore_slots>0 with band_served=0 means the band is empty
        # (library too small, or every neighbour already heard).
        "explore_slots": n_explore,
        "band_available": n_band_available,
        "band_served": sum(1 for c in chunk if c.pool == "band"),
        "explore_served": sum(1 for c in chunk if c.pool in ("band", "explore")),
        "fav_pool": len(fav_top),
        "round": round_no,
        "n_floor": len(floor_ids),
        "relaxed": relaxed_used,
        "fallback": fallback_used,
    }

    # «Подстроились под твой вайб»: the session tracks whose contribution is
    # actually distinguishable (a fire, or an instant replay).
    fire_anchor_weights = aggregate_taste_anchors(
        session_taste, kind="fire",
        session_play_times=sorted(s.played_at for s in session_events), now=now)
    adaptation = session_adaptation(session_events, fire_anchor_weights)
    if adaptation is not None:
        adaptation = {
            "active": True,
            "tracks": [
                {
                    "track_id": tid,
                    "title": (meta.get(tid) or {}).get("title") or "—",
                    "artist": (meta.get(tid) or {}).get("artist") or "—",
                    "cover_art_path": (meta.get(tid) or {}).get("cover_art_path"),
                }
                for tid in adaptation["track_ids"]
            ],
        }

    return {"tracks": chunk[:n], "diagnostics": diagnostics,
            "session_adaptation": adaptation}


SESSION_ADAPT_TRACKS = 2   # covers shown next to «подстроились под твой вайб»


def session_adaptation(
    session_events: list[PlaybackSignal],
    fire_anchor_weights: dict[str, float],
) -> dict | None:
    """«Подстроились под твой вайб» — shown ONLY when session contributions are
    distinguishable: an active fire or an instant replay. Uniform background
    listening gives every track the same claim, so naming «виновников» would be
    arbitrary — return None and show no badge.

    Returns ``{"active": True, "track_ids": [...]}`` (fires by effective
    weight first, then most recent replays), capped at SESSION_ADAPT_TRACKS.
    """
    track_ids = [t for t, _ in sorted(fire_anchor_weights.items(),
                                      key=lambda kv: kv[1], reverse=True)]
    ordered = sorted(session_events, key=lambda s: s.played_at)
    for prev, cur in reversed(list(zip(ordered, ordered[1:]))):
        if session_mod._is_replay(prev, cur) and cur.track_id not in track_ids:
            track_ids.append(cur.track_id)
    if not track_ids:
        return None
    return {"active": True, "track_ids": track_ids[:SESSION_ADAPT_TRACKS]}


# ── Similar tracks: CLAP neighbors re-ranked by sonic axes ──────────────────
# Powers GET /recommend/similar (Recommend tab «похожие» + ai-playlist agent
# tool). Unlike autoplay (pure CLAP order), candidates are re-ranked by a
# blend of CLAP cosine and axis-space closeness to the seed.

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
    seed_vectors = _clap_vectors(qdrant_client, collection_name, [seed_track_id])
    seed_vec = seed_vectors.get(seed_track_id)
    if seed_vec is None:
        return {"seed_track_id": seed_track_id, "tracks": []}
    meta = _track_meta(qdrant_client, collection_name)

    axis_stats = blend_axis_stats(
        MetadataDB.get_axis_norm_stats(collection_name),
        load_axis_norm_reference(),
    )
    seed_z = z_scores_for_axes(
        (meta.get(seed_track_id) or {}).get("sonic_axes"),
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
            with_payload=False,
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
        payload = meta.get(tid)
        if (tid in excluded or tid in dislikes
                or payload is None or not _duration_ok(payload)):
            continue
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

ISLANDS_MAX = 10          # taste islands shown in the profile
ISLAND_MEMBERS_MAX = 8    # covers per island (representative first)
# The profile mosaic needs a much wider candidate pool than the stream's
# top-20 anchor window (which only ever feeds 5 Qdrant queries): with 20
# candidates most clusters can't reach 3 members and the mosaic freezes.
ISLAND_POOL_SIZE = 60         # anchor candidates clustered for the mosaic
ISLAND_MERGE_THRESHOLD = 0.80  # display clustering is softer than the stream's 0.85
ISLAND_MIN_MEMBERS = 3        # an island is a real cluster, not a pair


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
    all_taste = latest_per_track(_fire_signals(
        MetadataDB.get_taste_signals(collection_name)))

    # Low learning rate: islands are fed ONLY by the clearest explicit signals —
    # fires (fat) + ≥85% completions (weak). Partial listens and skips never enter,
    # so the long-term profile drifts slowly and stays trustworthy.
    influencing = [s for s in signals if getattr(s, "influence", True)]
    cutoffs = reaction_cutoffs(all_taste)
    long_weights = island_taste_weights(all_taste, influencing, now, cutoffs=cutoffs)
    n_fires = sum(1 for s in all_taste if s.kind == "fire")
    n_completions = sum(
        1 for s in influencing
        if s.total_dur and s.total_dur > 0.0
        and s.played_sec / s.total_dur >= FULL_RATIO
        and not superseded(s, cutoffs)
    )
    n_signals = n_fires + n_completions

    # Anchors → islands (merge groups carry their member track ids).
    anchor_cands = select_positive_anchors(long_weights, top_m=ISLAND_POOL_SIZE)
    member_pool = [a.track_id for a in anchor_cands]
    vectors = _clap_vectors(qdrant_client, collection_name, member_pool)
    payloads = _track_meta(qdrant_client, collection_name)
    merged = merge_anchors(anchor_cands, vectors, threshold=ISLAND_MERGE_THRESHOLD)
    # An island is a *cluster* of taste, not a lone track or a chance pair:
    # keep only merge groups of ISLAND_MIN_MEMBERS+, so no «остров» is ever
    # built from a single song.
    merged = [a for a in merged if len(a.members) >= ISLAND_MIN_MEMBERS]
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
                "album": p.get("album"),
                "genre": p.get("genre"),
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
        tid: z for tid in long_weights
        if (z := z_scores_for_axes(
            (payloads.get(tid) or {}).get("sonic_axes"), axis_stats, AXIS_NAMES))
    }
    p_long, confidence = axis_preferences(long_weights, z_by_track, AXIS_NAMES)
    axes = (
        {a: {"z": round(p_long[a], 3), "level": z_to_level(p_long[a])} for a in AXIS_NAMES}
        if p_long is not None else None
    )

    # «Вайбики» ride the same response — the fast mood layer under the islands.
    vibes = current_vibes(
        qdrant_client=qdrant_client, collection_name=collection_name,
        signals=signals, taste_signals=all_taste, now=now,
    )

    return {
        "axes": axes,
        "confidence": round(confidence, 3),
        "n_signals": n_signals,
        "islands": islands,
        "vibes": vibes,
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


# ── Vibe album suggestions («что послушать» в библиотеке) ───────────────────
# Per vibe: the library album whose mean CLAP (over its own tracks) is closest
# to the vibe's centroid, excluding albums already represented inside the vibe.
# Candidates come from a clap search around the centroid (so we only compute
# full album means for albums that plausibly match, not the whole library).

VIBE_ALBUM_SEARCH_LIMIT = 300     # top clap hits per vibe → candidate albums
VIBE_ALBUM_CANDIDATES_MAX = 24    # albums fully scored per vibe
VIBE_ALBUM_SAMPLE = 12            # tracks per album used for the album mean
VIBE_ALBUM_MIN_TRACKS = 3         # 1–2-track "albums" are singles, not подборка
VIBE_ALBUM_MAX_PER_VIBE = 2
VIBE_ALBUM_TOTAL_MAX = 6
_VIBE_ALBUM_CACHE_TTL_SEC = 6 * 3600.0
# {collection_name: (monotonic_ts, result_dict)} — vibes drift on a days scale
# (H_VIBE_DAYS=2.5), so a 6h in-process cache keeps the endpoint cheap without
# the suggestions going stale in any user-visible way.
_vibe_album_cache: dict[str, tuple[float, dict]] = {}


def _album_key(title) -> str | None:
    t = (title or "").strip().lower()
    return t or None


def vibe_album_suggestions(
    *,
    qdrant_client,
    collection_name: str,
    albums,
    now: datetime | None = None,
) -> dict:
    """Album picks for the library rail, one per current vibe first.

    ``albums`` is ``LibraryService.get_albums(...).albums`` (the route passes it
    in so this module stays free of the library-service dependency). Returns
    ``{"suggestions": [dict], "vibes": [vibe dict]}`` — the raw vibes ride along
    so the route can attach cached LLM vibe names.

    Selection is round-robin across vibes (strongest vibe first): every vibe
    places its best album before any vibe places its second, ≤2 per vibe and
    ≤6 total — diversity over depth, per the feature spec. current_vibes caps
    at VIBES_MAX=3, so 3×2 fills the rail exactly.
    """
    cached = _vibe_album_cache.get(collection_name)
    if cached and (time.monotonic() - cached[0]) < _VIBE_ALBUM_CACHE_TTL_SEC:
        return cached[1]

    now = now or datetime.utcnow()
    raw_signals = MetadataDB.get_playback_signals(collection_name, LONG_TERM_EVENT_CAP)
    signals = [PlaybackSignal(**r) for r in raw_signals]
    all_taste = _fire_signals(MetadataDB.get_taste_signals(collection_name))
    vibes = current_vibes(
        qdrant_client=qdrant_client, collection_name=collection_name,
        signals=signals, taste_signals=all_taste, now=now,
    )

    albums_by_key: dict[str, object] = {}
    for a in albums or []:
        k = _album_key(a.album_title)
        if k and a.track_count >= VIBE_ALBUM_MIN_TRACKS and a.tracks:
            albums_by_key[k] = a

    result: dict = {"suggestions": [], "vibes": vibes}
    if not vibes or not albums_by_key:
        _vibe_album_cache[collection_name] = (time.monotonic(), result)
        return result

    album_centroids: dict[str, np.ndarray | None] = {}
    ranked_per_vibe: list[list[tuple[float, str]]] = []

    for vibe in vibes:
        member_ids = [t["track_id"] for t in vibe["tracks"]]
        vectors = _clap_vectors(qdrant_client, collection_name, member_ids)
        centroid = _vibe_centroid(member_ids, vectors)
        if centroid is None:
            ranked_per_vibe.append([])
            continue
        # Albums already inside the vibe are excluded — the point is to lead
        # the user somewhere adjacent, not back to what the vibe is made of.
        vibe_album_keys = {_album_key(t.get("album")) for t in vibe["tracks"]}
        vibe_album_keys.discard(None)

        try:
            hits = qdrant_client.query_points(
                collection_name=collection_name,
                query=[float(x) for x in centroid],
                using="clap",
                limit=VIBE_ALBUM_SEARCH_LIMIT,
                with_payload=["album"],
            ).points
        except Exception:
            logger.exception("[vibe-albums] clap search failed")
            ranked_per_vibe.append([])
            continue

        cand_keys: list[str] = []
        for h in hits:
            k = _album_key((h.payload or {}).get("album"))
            if not k or k in vibe_album_keys or k not in albums_by_key:
                continue
            if k not in cand_keys:
                cand_keys.append(k)
            if len(cand_keys) >= VIBE_ALBUM_CANDIDATES_MAX:
                break

        # Album mean CLAP over an even sample of its tracklist, one batched
        # retrieve for every album this vibe still needs.
        need_ids: list[str] = []
        sample_by_key: dict[str, list[str]] = {}
        for k in cand_keys:
            if k in album_centroids:
                continue
            tracks = albums_by_key[k].tracks
            step = max(1, len(tracks) // VIBE_ALBUM_SAMPLE)
            sample = [t.track_id for t in tracks[::step]][:VIBE_ALBUM_SAMPLE]
            sample_by_key[k] = sample
            need_ids.extend(sample)
        if need_ids:
            vecs = _clap_vectors(qdrant_client, collection_name, need_ids)
            for k, sample in sample_by_key.items():
                album_centroids[k] = _vibe_centroid(sample, vecs)

        ranked = [
            (float(ac @ centroid), k)
            for k in cand_keys
            if (ac := album_centroids.get(k)) is not None
        ]
        ranked.sort(reverse=True)
        ranked_per_vibe.append(ranked)

    suggestions: list[dict] = []
    seen_keys: set[str] = set()
    iters = [iter(r) for r in ranked_per_vibe]
    for _round in range(VIBE_ALBUM_MAX_PER_VIBE):
        for vibe, it in zip(vibes, iters):
            if len(suggestions) >= VIBE_ALBUM_TOTAL_MAX:
                break
            for score, k in it:
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                a = albums_by_key[k]
                suggestions.append({
                    "album_title": a.album_title,
                    "primary_artist": a.primary_artist,
                    "primary_artist_slug": a.primary_artist_slug,
                    "cover_art_path": a.cover_art_path,
                    "track_count": a.track_count,
                    "year": a.year,
                    "score": round(score, 3),
                    "vibe_track_id": vibe["track_id"],
                })
                break
        if len(suggestions) >= VIBE_ALBUM_TOTAL_MAX:
            break

    result["suggestions"] = suggestions
    _vibe_album_cache[collection_name] = (time.monotonic(), result)
    return result
