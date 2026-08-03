"""The session profile: what this listening session is pulling toward and
pushing away from (design §2.1, §3, §4).

The old engine built anchors from long-term islands — i.e. from the tracks the
user had already played most — and let the session nudge them by at most 50%.
That is why the wave kept circling the same songs. Here the session leads:
its own listening forms **positive clusters** (притяжение) and its skips/waters
form **negative clusters** (отталкивание); the long-term profile survives only
as a cold-start seed that fades out over the first few signals.

Four rules make the signals honest:

* an explicit reaction supersedes the listen it happened on (§2.1) — pressing
  water three minutes into a four-minute track must not credit the track;
* water debuffs for days, not hours, and mutes the track outright for two (§3);
* one skip that sounds like what you're enjoying is forgiven; the second is
  not (§4.3) — disliking a song isn't disliking the genre;
* every weight is scaled by the listener's own baseline (``baseline.py``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from app.services.stream.baseline import Baseline, NEUTRAL
from app.services.stream.signals import (
    Anchor,
    FireSignal,
    PlaybackSignal,
    W_FULL,
    W_MOST,
    W_REPLAY,
    W_SKIP,
    FULL_RATIO,
    H_IMPLICIT_DAYS,
    age_days,
    base_weight,
    centroid,
    decayed,
    is_skip,
    listen_ratio,
    merge_anchors,
    select_positive_anchors,
)

logger = logging.getLogger(__name__)

# ── Reaction supersedes the listen (§2.1) ──────────────────────────────────
# The playback event is flushed on track change, i.e. AFTER the button press,
# so the cutoff needs a little slack forward in time.
REACTION_GRACE_SEC = 30.0

# ── Огонёк: ephemeral wave steer, unchanged dual decay ─────────────────────
FIRE_BASE = 1.0
FIRE_TIME_MAX_H = 4.0      # fire fades to 0 over this many hours
FIRE_COUNT_FULL = 30       # full weight until this many session tracks since the fire
FIRE_COUNT_ZERO = 50       # zero weight at/after this many session tracks

# ── Вода: its own, much longer clock (§3) ──────────────────────────────────
WATER_BASE = 1.0
H_WATER_DAYS = 1.0         # half-life of the water charge
WATER_MUTE_DAYS = 2.0      # the watered track is not served at all this long
WATER_TAIL_DAYS = 5.0      # …and its neighbourhood penalty is gone by here

# Persistent «заряд» driving the frontend meter/lock. Same 1-day half-life as
# the water charge: a fresh signal is 100%, ≤50% after a day → button unlocks.
H_REACTION_DAYS = 1.0

# ── Clustering (§4.2) ──────────────────────────────────────────────────────
# Percentiles of the collection's OWN cosine distribution, not raw cosines.
CLUSTER_MERGE_PCT = 0.985
TOP_POSITIVE_CLUSTERS = 5
TOP_NEGATIVE_CLUSTERS = 5
CLUSTER_POOL_SIZE = 30      # candidates considered before merging

# ── Skip forgiveness (§4.3) ────────────────────────────────────────────────
FORGIVE_SIM_PCT = 0.97      # top 3% of similarity — «звучит как то, что ты любишь»

# ── Carryover between sessions (§4.4) ──────────────────────────────────────
CARRYOVER_W = 0.4
CARRYOVER_TAIL = 10          # events taken from the tail of the previous session
CARRYOVER_SIGNAL_DAYS = 2.0  # fires carried over must be at most this old
CARRYOVER_MAX_AGE_D = 3.0    # older previous session → no carryover at all
CARRYOVER_FADE_SIG = 8       # own signals that fully retire the carryover

# ── Long-term seed (§4.5) ──────────────────────────────────────────────────
LONG_TERM_FLOOR = 0.15
LONG_TERM_FADE = 6

# ── Repulsion (§4.6) ───────────────────────────────────────────────────────
REPEL_K_WATER = 1.0
REPEL_K_SKIP = 0.45


# ── Fire / water decay curves ──────────────────────────────────────────────

def fire_time_factor(delta_hours: float) -> float:
    """Wall-clock decay of a fire: raised cosine 1→0 over FIRE_TIME_MAX_H,
    smoothly (no cliff), exactly 0 at and beyond the horizon."""
    if delta_hours <= 0.0:
        return 1.0
    if delta_hours >= FIRE_TIME_MAX_H:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * delta_hours / FIRE_TIME_MAX_H))


def fire_count_factor(n_tracks: int) -> float:
    """Track-count decay: full until FIRE_COUNT_FULL session tracks since the
    fire, linear down to 0 by FIRE_COUNT_ZERO."""
    if n_tracks <= FIRE_COUNT_FULL:
        return 1.0
    if n_tracks >= FIRE_COUNT_ZERO:
        return 0.0
    return (FIRE_COUNT_ZERO - n_tracks) / (FIRE_COUNT_ZERO - FIRE_COUNT_FULL)


def reaction_contribution(age_d: float) -> float:
    """Persistent «заряд» of an огонёк/вода ∈ [0,1]: 1.0 fresh, 0.5 at exactly
    H_REACTION_DAYS (the unlock boundary), decaying toward 0. Single source of
    truth for the frontend meter AND the fire's vibe weight."""
    if age_d <= 0.0:
        return 1.0
    return max(0.0, min(1.0, 0.5 ** (age_d / H_REACTION_DAYS)))


def water_charge(age_d: float) -> float:
    """Remaining strength of a вода signal ∈ [0,1] (half-life H_WATER_DAYS),
    hard-zeroed past WATER_TAIL_DAYS so an ancient water stops costing cycles."""
    if age_d >= WATER_TAIL_DAYS:
        return 0.0
    if age_d <= 0.0:
        return 1.0
    return 0.5 ** (age_d / H_WATER_DAYS)


def aggregate_taste_anchors(
    signals: list[FireSignal],
    *,
    kind: str,
    session_play_times: list[datetime],
    now: datetime,
    base: float = FIRE_BASE,
) -> dict[str, float]:
    """Ephemeral weight per track for огонёк ('fire'), dual decay.

    eff = base · time_factor(Δt) · count_factor(n), where n is the number of
    session plays after the signal. Water no longer goes through here — it has
    its own, much longer clock (``water_weights``).
    """
    out: dict[str, float] = {}
    for s in signals:
        if s.kind != kind:
            continue
        dt_h = (now - s.created_at).total_seconds() / 3600.0
        n = sum(1 for t in session_play_times if t > s.created_at)
        eff = base * fire_time_factor(dt_h) * fire_count_factor(n)
        if eff > 0.0:
            out[s.track_id] = out.get(s.track_id, 0.0) + eff
    return out


def water_weights(
    signals: list[FireSignal], now: datetime, *, base: float = WATER_BASE,
) -> dict[str, float]:
    """Effective вода weight per track — global, on the 1-day half-life."""
    out: dict[str, float] = {}
    for s in signals:
        if s.kind != "water":
            continue
        eff = base * water_charge(age_days(s.created_at, now))
        if eff > 0.0:
            out[s.track_id] = out.get(s.track_id, 0.0) + eff
    return out


def muted_track_ids(signals: list[FireSignal], now: datetime) -> set[str]:
    """Tracks under the hard water mute — not served at all for WATER_MUTE_DAYS.

    ``signals`` must already be collapsed by ``latest_per_track``, so a track
    re-fired after being watered is not muted.
    """
    return {
        s.track_id for s in signals
        if s.kind == "water" and age_days(s.created_at, now) < WATER_MUTE_DAYS
    }


# ── Reaction cutoff (§2.1) ─────────────────────────────────────────────────

def reaction_cutoffs(signals: list[FireSignal]) -> dict[str, datetime]:
    """``{track_id: newest reaction time}`` — the instant after which that
    track's earlier listens stop counting."""
    out: dict[str, datetime] = {}
    for s in signals:
        cur = out.get(s.track_id)
        if cur is None or s.created_at > cur:
            out[s.track_id] = s.created_at
    return out


def superseded(ev: PlaybackSignal, cutoffs: dict[str, datetime]) -> bool:
    """True when an explicit reaction has already spoken for this listen.

    The grace window covers the ordering reality: the button is pressed during
    playback, the event lands on the following track change.
    """
    cut = cutoffs.get(ev.track_id)
    if cut is None:
        return False
    return ev.played_at <= cut + timedelta(seconds=REACTION_GRACE_SEC)


def _is_replay(prev: PlaybackSignal, cur: PlaybackSignal) -> bool:
    """Instant replay: same track right after a ≥85% listen, same session."""
    if prev.track_id != cur.track_id or prev.session_id != cur.session_id:
        return False
    ratio = listen_ratio(prev.played_sec, prev.total_dur)
    return ratio is not None and ratio >= FULL_RATIO


def adapt(weight: float, baseline: Baseline) -> float:
    """Scale a raw reward by the listener's own baseline (§2.3)."""
    if weight == W_SKIP:
        return weight * baseline.m_skip
    if weight in (W_FULL, W_MOST):
        return weight * baseline.m_full
    return weight


def weight_events(
    events: list[PlaybackSignal],
    *,
    cutoffs: dict[str, datetime] | None = None,
    baseline: Baseline = NEUTRAL,
) -> list[float]:
    """Weight a chronologically-ordered single-session event list.

    Base completeness weight → replay upgrade → reaction cutoff → adaptive
    scaling. The idle rule that used to zero long passive streaks is gone
    (design §2.2): background listening counts like any other listening.
    """
    cutoffs = cutoffs or {}
    weights: list[float] = []
    for i, ev in enumerate(events):
        if superseded(ev, cutoffs):
            weights.append(0.0)
            continue
        w = base_weight(ev.played_sec, ev.total_dur)
        if i > 0 and _is_replay(events[i - 1], ev):
            w = W_REPLAY
        weights.append(adapt(w, baseline))
    return weights


def aggregate_event_weights(
    signals: list[PlaybackSignal],
    now: datetime,
    *,
    cutoffs: dict[str, datetime] | None = None,
    baseline: Baseline = NEUTRAL,
) -> dict[str, float]:
    """Decayed implicit weight per track over a chronological event list.

    Events are re-grouped by session (the replay rule is session-scoped),
    weighted, decayed with H_IMPLICIT, then summed per track.
    """
    by_session: dict[str, list[PlaybackSignal]] = {}
    for s in signals:
        by_session.setdefault(s.session_id, []).append(s)

    out: dict[str, float] = {}
    for sess_events in by_session.values():
        ws = weight_events(sess_events, cutoffs=cutoffs, baseline=baseline)
        for ev, w in zip(sess_events, ws):
            if w == 0.0:
                continue
            w_eff = decayed(w, age_days(ev.played_at, now), H_IMPLICIT_DAYS)
            out[ev.track_id] = out.get(ev.track_id, 0.0) + w_eff
    return out


# ── Clusters ───────────────────────────────────────────────────────────────

@dataclass
class Cluster:
    """A merged region of CLAP space the session cares about, with a sign."""
    kind: str                 # 'positive' | 'skip' | 'water'
    track_id: str             # representative
    weight: float
    members: list[str]
    vec: np.ndarray | None = None   # unit centroid

    @property
    def repel_k(self) -> float:
        return REPEL_K_WATER if self.kind == "water" else REPEL_K_SKIP


@dataclass
class SkipEvent:
    track_id: str
    weight: float             # positive magnitude
    played_at: datetime


@dataclass
class SessionProfile:
    """Everything ``pools``/``stream_service`` need to score a candidate."""
    positive: list[Cluster] = field(default_factory=list)
    negative: list[Cluster] = field(default_factory=list)
    positive_weights: dict[str, float] = field(default_factory=dict)
    session_played: set[str] = field(default_factory=set)
    muted: set[str] = field(default_factory=set)
    n_signals: int = 0
    w_long: float = 1.0
    carryover_w: float = 0.0
    forgiven_skips: int = 0

    def affinity(self, vec: np.ndarray | None, calibration) -> float:
        """max over C⁺ of ŵ⁺ · sim_pct(candidate, centroid)."""
        return _best(self.positive, vec, calibration)

    def repulsion(self, vec: np.ndarray | None, calibration) -> float:
        """max over C⁻ of ŵ⁻ · sim_pct(candidate, centroid) · k."""
        return _best(self.negative, vec, calibration, use_k=True)


def _best(clusters: list[Cluster], vec, calibration, *, use_k: bool = False) -> float:
    if vec is None or not clusters:
        return 0.0
    best = 0.0
    for c in clusters:
        if c.vec is None:
            continue
        val = c.weight * calibration.sim_pct(float(vec @ c.vec))
        if use_k:
            val *= c.repel_k
        if val > best:
            best = val
    return best


def _normalize(clusters: list[Cluster]) -> list[Cluster]:
    """Scale cluster weights so the strongest is 1.0 — scoring coefficients
    (§4.6) are then absolute, not hostage to how loud the session happens to be."""
    top = max((c.weight for c in clusters), default=0.0)
    if top <= 0.0:
        return clusters
    for c in clusters:
        c.weight /= top
    return clusters


def _cluster(
    weights: dict[str, float], vectors: dict, calibration, kind: str,
    *, top_k: int, pool: int = CLUSTER_POOL_SIZE,
) -> list[Cluster]:
    """Greedy-merge weighted tracks into clusters, strongest ``top_k`` kept."""
    cands = select_positive_anchors(weights, top_m=pool)
    merged = merge_anchors(cands, vectors, CLUSTER_MERGE_PCT, calibration=calibration)
    merged.sort(key=lambda a: a.weight, reverse=True)
    out: list[Cluster] = []
    for a in merged[:top_k]:
        # Members are within the merge threshold of each other, so an unweighted
        # centroid is fine and is more stable than the representative's vector.
        vec = centroid(a.members, vectors)
        if vec is None:
            continue
        out.append(Cluster(kind=kind, track_id=a.track_id, weight=a.weight,
                           members=list(a.members), vec=vec))
    return out


def forgive_skips(
    skips: list[SkipEvent],
    positive: list[Cluster],
    vectors: dict,
    calibration,
) -> tuple[list[SkipEvent], int]:
    """Drop the FIRST skip that sounds like each positive cluster (§4.3).

    One skip inside a region you're otherwise enjoying is «not this song».
    A second one in the same region is «not this sound» — at which point both
    count, the forgiven one retroactively included. Skips that resemble no
    positive cluster are never forgiven.
    """
    if not positive or not skips:
        return skips, 0

    # Which positive cluster (if any) each skip belongs to.
    by_cluster: dict[str, list[SkipEvent]] = {}
    unattached: list[SkipEvent] = []
    for s in sorted(skips, key=lambda e: e.played_at):
        v = vectors.get(s.track_id)
        if v is None:
            unattached.append(s)
            continue
        vu = v / (np.linalg.norm(v) or 1.0)
        best_c, best_sim = None, 0.0
        for c in positive:
            if c.vec is None:
                continue
            sim = calibration.sim_pct(float(vu @ c.vec))
            if sim >= FORGIVE_SIM_PCT and sim > best_sim:
                best_c, best_sim = c, sim
        if best_c is None:
            unattached.append(s)
        else:
            by_cluster.setdefault(best_c.track_id, []).append(s)

    kept = list(unattached)
    forgiven = 0
    for members in by_cluster.values():
        if len(members) == 1:
            forgiven += 1          # a single dissenting song — let it go
        else:
            kept.extend(members)   # a pattern — all of them count, including the first
    return kept, forgiven


# ── Carryover ──────────────────────────────────────────────────────────────

def previous_session_tail(
    signals: list[PlaybackSignal], session_id: str, now: datetime,
) -> tuple[list[PlaybackSignal], str | None]:
    """The tail of the session the user listened to *before* this one.

    «Подсовываем то, на чём остановился» — a new tab is a new session, but it
    should not start from nothing. Returns ``([], None)`` when that session is
    older than CARRYOVER_MAX_AGE_D (taste may well have moved on) or absent.
    """
    own = [s for s in signals if s.session_id == session_id]
    boundary = min((s.played_at for s in own), default=now)

    earlier = [s for s in signals if s.session_id != session_id and s.played_at < boundary]
    if not earlier:
        return [], None
    prev_id = max(earlier, key=lambda s: s.played_at).session_id
    prev = sorted((s for s in earlier if s.session_id == prev_id),
                  key=lambda s: s.played_at)
    if not prev or age_days(prev[-1].played_at, now) > CARRYOVER_MAX_AGE_D:
        return [], prev_id
    return prev[-CARRYOVER_TAIL:], prev_id


def carryover_scale(n_own_signals: int) -> float:
    """CARRYOVER_W fading linearly to 0 as the session finds its own footing."""
    if n_own_signals >= CARRYOVER_FADE_SIG:
        return 0.0
    return CARRYOVER_W * (1.0 - n_own_signals / CARRYOVER_FADE_SIG)


def long_term_weight(n_session_signals: int) -> float:
    """How much of the pull still comes from long-term islands (§4.5).

    1.0 on the first track of a session, down to LONG_TERM_FLOOR by
    LONG_TERM_FADE signals. The floor is deliberately non-zero: a session that
    wandered into a strange corner must have something to climb out on.
    """
    return max(LONG_TERM_FLOOR, 1.0 - n_session_signals / LONG_TERM_FADE)


# ── Assembly ───────────────────────────────────────────────────────────────

def build(
    *,
    signals: list[PlaybackSignal],
    session_id: str,
    session_taste: list[FireSignal],
    all_taste: list[FireSignal],
    long_weights: dict[str, float],
    baseline: Baseline,
    calibration,
    fetch_vectors,
    now: datetime,
) -> SessionProfile:
    """Build the full session profile.

    ``fetch_vectors(ids) -> {id: np.ndarray}`` keeps Qdrant access in the
    orchestrator; everything else here is deterministic given its inputs.
    """
    session_events = [s for s in signals if s.session_id == session_id]
    influencing = [s for s in session_events if getattr(s, "influence", True)]
    session_played = {s.track_id for s in session_events}
    play_times = sorted(s.played_at for s in session_events)

    cutoffs = reaction_cutoffs(all_taste)
    muted = muted_track_ids(all_taste, now)

    # 1. Own session signals.
    own_weights = aggregate_event_weights(
        influencing, now, cutoffs=cutoffs, baseline=baseline)
    fires = aggregate_taste_anchors(
        session_taste, kind="fire", session_play_times=play_times, now=now)
    fires = {t: w * baseline.m_react for t, w in fires.items()}

    positive_weights = {t: w for t, w in own_weights.items() if w > 0.0}
    for tid, w in fires.items():
        positive_weights[tid] = positive_weights.get(tid, 0.0) + w

    skips = [
        SkipEvent(track_id=ev.track_id,
                  weight=abs(W_SKIP) * baseline.m_skip * decayed(
                      1.0, age_days(ev.played_at, now), H_IMPLICIT_DAYS),
                  played_at=ev.played_at)
        for ev in influencing
        if is_skip(ev.played_sec, ev.total_dur) and not superseded(ev, cutoffs)
    ]

    n_own = len([w for w in own_weights.values() if w != 0.0]) + len(fires)

    # 2. Carryover from where the user left off last time.
    carry_scale = carryover_scale(n_own)
    carried_positive: dict[str, float] = {}
    if carry_scale > 0.0:
        tail, _prev_id = previous_session_tail(signals, session_id, now)
        tail = [s for s in tail if getattr(s, "influence", True)]
        for tid, w in aggregate_event_weights(
                tail, now, cutoffs=cutoffs, baseline=baseline).items():
            if w > 0.0:
                carried_positive[tid] = carried_positive.get(tid, 0.0) + w * carry_scale
        for s in all_taste:
            if s.kind != "fire":
                continue
            a = age_days(s.created_at, now)
            if a > CARRYOVER_SIGNAL_DAYS or s.track_id in fires:
                continue
            w = FIRE_BASE * baseline.m_react * reaction_contribution(a) * carry_scale
            if w > 0.0:
                carried_positive[s.track_id] = carried_positive.get(s.track_id, 0.0) + w
    for tid, w in carried_positive.items():
        positive_weights[tid] = positive_weights.get(tid, 0.0) + w

    # 3. Long-term islands as the fading cold-start seed.
    w_long = long_term_weight(n_own)
    pull: dict[str, float] = {t: w * (1.0 - w_long) for t, w in positive_weights.items()}
    for tid, w in long_weights.items():
        if w > 0.0:
            pull[tid] = pull.get(tid, 0.0) + w * w_long

    waters = water_weights(all_taste, now)
    waters = {t: w * baseline.m_react for t, w in waters.items()}

    # 4. Vectors for everything that will be clustered.
    need = list(dict.fromkeys(
        list(pull) + [s.track_id for s in skips] + list(waters)))
    vectors = fetch_vectors(need)

    # 5. Cluster, forgive, re-cluster the survivors.
    positive = _cluster(pull, vectors, calibration, "positive",
                        top_k=TOP_POSITIVE_CLUSTERS)
    kept_skips, n_forgiven = forgive_skips(skips, positive, vectors, calibration)

    skip_weights: dict[str, float] = {}
    for s in kept_skips:
        skip_weights[s.track_id] = skip_weights.get(s.track_id, 0.0) + s.weight

    negative = (
        _cluster(waters, vectors, calibration, "water", top_k=TOP_NEGATIVE_CLUSTERS)
        + _cluster(skip_weights, vectors, calibration, "skip", top_k=TOP_NEGATIVE_CLUSTERS)
    )

    return SessionProfile(
        positive=_normalize(positive),
        negative=_normalize(negative),
        positive_weights=pull,
        session_played=session_played,
        muted=muted,
        n_signals=n_own,
        w_long=w_long,
        carryover_w=carry_scale,
        forgiven_skips=n_forgiven,
    )
