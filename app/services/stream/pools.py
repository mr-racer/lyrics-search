"""Candidate pools, slider quotas and chunk assembly (design §4.6, §6).

Two pools replace the old anchor/explore/liked trio:

* **fresh** — tracks not played in the last FRESH_WINDOW_DAYS (or never).
  This is what the slider's «РЕДКОЕ» end actually promises, and the promise is
  now hard: at the extreme a dry fresh pool yields a SHORT chunk rather than
  quietly serving something heard last week.
* **familiar** — everything else: neighbours of the session's positive
  clusters plus the computed «favorites».

Scoring never transfers CLAP vectors *or payloads*. Affinity comes from one
Qdrant search per positive cluster centroid (the hit's own cosine), repulsion
from one search per negative centroid — a candidate absent from every negative
neighbourhood simply has zero repulsion, which is the correct answer anyway.
Those searches ask for ids + scores only (``with_payload=False``); every piece
of track metadata comes from the SQLite mirror (``qdrant_utils.light_map``),
which the caller passes in as ``meta``.
"""

from __future__ import annotations

import logging
import math

from app.services.stream.signals import (
    StreamCandidate,
    axis_match_score,
    duration_ok,
    z_scores_for_axes,
)

logger = logging.getLogger(__name__)

# ── Freshness ──────────────────────────────────────────────────────────────
FRESH_WINDOW_DAYS = 30.0
FRESH_FETCH_K = 150         # Qdrant neighbours per positive centroid
NEG_FETCH_K = 200           # …per negative centroid
FRESH_STRATIFIED_SHARE = 0.25   # floor share of fresh slots left to chance
# A stratified pick this close to a negative cluster is dropped. p97.5 is where
# the raw 0.80 this replaces landed on real libraries (p98.1 / p96.6).
EXPLORE_NEG_PCT = 0.975
EXPLORE_BIN_EDGE = 0.5      # z-bins: < −0.5 | −0.5..0.5 | > 0.5
# Ceiling on how much of the library the random sampler scans. It used to be
# 5000 against a 5959-track library, i.e. 959 tracks the explore path could
# never reach (insertion order, so always the same 959). Kept only as a sanity
# bound on pathological libraries.
STRATIFIED_SCROLL_CAP = 100_000

# ── Band candidates (design 2026-09-06 §3.3) ───────────────────────────────
# The missing middle. Top-of-kNN is what is already playing; a random library
# track is a stranger. Band candidates are the neighbours ranked past the near
# field but still recognisably in the same world — «рядом, но не то же».
BAND_FETCH_K = 400          # how deep the positive-centroid search goes
BAND_MIN_SIM_PCT = 0.85     # floor: on a small library rank 400 is already noise

# Skip burst → widen the random share (the only survivor of the old pivot).
SKIP_BURST_N = 3
SKIP_BURST_WINDOW = 5
SKIP_BURST_STRATIFIED_SHARE = 0.5

# ── Score weights (§4.6) ───────────────────────────────────────────────────
W_AFFINITY = 0.50
W_REPULSION = 0.35
W_AXIS = 0.20
W_NOVELTY = 0.10
W_RECENT = 0.15
RECENT_PENALTY_HALFLIFE_H = 24.0

# ── Assembly ───────────────────────────────────────────────────────────────
MAX_CONSECUTIVE_SAME_POOL = 2
MAX_CONSECUTIVE_ARTIST = 2
ARTIST_REPEAT_WINDOW = 3
ARTIST_REPEAT_PENALTY = 0.05

DEFAULT_CHUNK_N = 3
DEFAULT_LIKED_SHARE = 0.30


def is_fresh(track_id: str, recency_hours: dict[str, float],
             window_days: float = FRESH_WINDOW_DAYS) -> bool:
    """Never played, or last played longer ago than the freshness window."""
    h = recency_hours.get(track_id)
    return h is None or h >= window_days * 24.0


def skip_burst(session_events, is_skip_fn) -> bool:
    """≥SKIP_BURST_N skips among the last SKIP_BURST_WINDOW session events."""
    recent = sorted(session_events, key=lambda s: s.played_at)[-SKIP_BURST_WINDOW:]
    return sum(1 for s in recent if is_skip_fn(s.played_sec, s.total_dur)) >= SKIP_BURST_N


def explore_slots(n_played: int, n: int, share: float) -> int:
    """How many of this chunk's ``n`` slots are exploratory (design §3.2).

    The share used to be turned into a count with ``int(round(quota * share))``,
    which on the default chunk is ``int(round(2 * 0.25)) == int(round(0.5))``
    — and Python rounds halves to even, so the answer was **0**. Exploration was
    off in the steady state entirely; measured on prod, that covered 67.4% of
    all listening.

    Rather than raise the share, this counts honestly. ``n_played`` is how many
    tracks the session has already played, so the quota is a difference of two
    floors over the session's own timeline:

        floor((k + n)·E) − floor(k·E)

    which spends exactly ``E`` of every slot in the long run, never rounds a
    fraction away, and needs no stored accumulator — the engine is stateless and
    recomputes this from the event count on every request. At the default
    ``E = 0.175`` it yields one exploratory track roughly every six.
    """
    if share <= 0.0 or n <= 0:
        return 0
    share = min(1.0, share)
    k = max(0, n_played)
    return int(math.floor((k + n) * share) - math.floor(k * share))


def band_candidates(
    *,
    value_by_id: dict[str, float],
    rank_by_id: dict[str, int],
    sim_by_id: dict[str, float],
    owner_by_id: dict[str, str],
    payload_by_id: dict[str, dict],
    axis_match_by_id: dict[str, float],
    repulsion_by_id: dict[str, float],
    play_counts: dict[str, int],
    recency_hours: dict[str, float],
    near_k: int = FRESH_FETCH_K,
    min_sim_pct: float = BAND_MIN_SIM_PCT,
) -> list[StreamCandidate]:
    """Candidates from the similarity band just past the near field (§3.3).

    A track qualifies when its best rank across the positive centroids is at or
    beyond ``near_k`` — so it is NOT already in the pool the wave draws from —
    while its calibrated similarity still clears ``min_sim_pct``. The floor is
    what keeps this from degenerating into the random sampler on a small
    library, where rank 400 may already be half the collection.

    Scored exactly like any other candidate, but tagged ``pool="band"`` so the
    chunk assembler can reserve a slot for it and so the playback event records
    where it came from.
    """
    out: list[StreamCandidate] = []
    for tid, val in value_by_id.items():
        if rank_by_id.get(tid, -1) < near_k:
            continue                      # already part of the near field
        if sim_by_id.get(tid, 0.0) < min_sim_pct:
            continue                      # past the band — that is the stranger zone
        payload = payload_by_id.get(tid)
        if payload is None or not duration_ok(payload):
            continue
        axis = axis_match_by_id.get(tid, 0.0)
        out.append(StreamCandidate(
            track_id=tid, payload=payload, pool="band",
            anchor_track_id=owner_by_id.get(tid),
            max_anchor_cos=val,
            axis_match=axis,
            score=score_candidate(
                affinity=val, repulsion=repulsion_by_id.get(tid, 0.0),
                axis_match=axis,
                play_count=play_counts.get(tid, 0),
                recency_h=recency_hours.get(tid),
            ),
        ))
    return out


# ── Qdrant-backed neighbourhood maps ───────────────────────────────────────

def cluster_neighbourhood(
    qdrant_client, collection_name: str, clusters, calibration, *,
    limit: int, excluded: set[str] | frozenset = frozenset(),
    use_k: bool = False, rank_out: dict[str, int] | None = None,
) -> tuple[dict[str, float], dict[str, str], dict[str, float]]:
    """Search around each cluster centroid; keep the best value per track.

    Returns ``(value_by_id, cluster_by_id, sim_by_id)`` where value is
    ``ŵ · sim_pct(cos)`` (× the cluster's repel coefficient when ``use_k``) and
    ``sim_by_id`` is the bare ``sim_pct`` — the stratified sampler thresholds on
    raw similarity, not on the weighted value.

    ``rank_out``, when given, is filled in place with ``{track_id: best rank}``
    — the candidate's position in its closest cluster's hit list, counted before
    exclusions so it means «how deep into this region the track sits» and not
    «where it happened to land after filtering». It is an out-parameter rather
    than a fourth return value so existing callers keep their arity; the band
    split (§3.3) is the only consumer.

    Ids and scores only: ``with_payload=False``. Metadata is looked up in the
    SQLite mirror by the caller.
    """
    values: dict[str, float] = {}
    owner: dict[str, str] = {}
    sims: dict[str, float] = {}
    for c in clusters:
        if c.vec is None:
            continue
        try:
            hits = qdrant_client.query_points(
                collection_name=collection_name,
                query=[float(x) for x in c.vec],
                using="clap",
                limit=limit,
                with_payload=False,
            ).points
        except Exception:
            logger.exception("[stream] cluster search failed for %s", c.track_id)
            continue
        k = c.repel_k if use_k else 1.0
        for rank, h in enumerate(hits):
            tid = str(h.id)
            if tid in excluded:
                continue
            sim = calibration.sim_pct(float(h.score or 0.0))
            val = c.weight * sim * k
            if val > values.get(tid, float("-inf")):
                values[tid] = val
                owner[tid] = c.track_id
            if sim > sims.get(tid, 0.0):
                sims[tid] = sim
            if rank_out is not None and rank < rank_out.get(tid, 1 << 30):
                rank_out[tid] = rank
    return values, owner, sims


# ── Scoring ────────────────────────────────────────────────────────────────

def score_candidate(
    *,
    affinity: float,
    repulsion: float,
    axis_match: float,
    play_count: int,
    recency_h: float | None,
) -> float:
    novelty = 1.0 / (1.0 + play_count)
    recent_pen = math.exp(-recency_h / RECENT_PENALTY_HALFLIFE_H) if recency_h is not None else 0.0
    return (
        W_AFFINITY * affinity
        - W_REPULSION * repulsion
        + W_AXIS * axis_match
        + W_NOVELTY * novelty
        - W_RECENT * recent_pen
    )


def build_candidates(
    *,
    affinity_by_id: dict[str, float],
    repulsion_by_id: dict[str, float],
    payload_by_id: dict[str, dict],
    owner_by_id: dict[str, str],
    axis_match_by_id: dict[str, float],
    play_counts: dict[str, int],
    recency_hours: dict[str, float],
    pool_of,
) -> list[StreamCandidate]:
    """Turn the neighbourhood maps into scored candidates.

    ``pool_of(track_id) -> str`` labels each candidate (``fresh``/``familiar``)
    so assembly can honour the slider quota and the ≤2-in-a-row rule.

    ``payload_by_id`` is the SQLite metadata mirror. A hit that is missing from
    it is dropped rather than shipped with an empty payload — it would render
    as a «—» card whose file path 404s.
    """
    out: list[StreamCandidate] = []
    for tid, aff in affinity_by_id.items():
        payload = payload_by_id.get(tid)
        if payload is None or not duration_ok(payload):
            continue
        axis = axis_match_by_id.get(tid, 0.0)
        c = StreamCandidate(
            track_id=tid, payload=payload, pool=pool_of(tid),
            anchor_track_id=owner_by_id.get(tid),
            max_anchor_cos=aff,
            axis_match=axis,
            score=score_candidate(
                affinity=aff,
                repulsion=repulsion_by_id.get(tid, 0.0),
                axis_match=axis,
                play_count=play_counts.get(tid, 0),
                recency_h=recency_hours.get(tid),
            ),
        )
        out.append(c)
    return out


# ── Stratified fresh sampling (filter-bubble insurance) ────────────────────

def _explore_bin(z_energy: float, z_experimental: float) -> tuple[int, int]:
    def bucket(z: float) -> int:
        if z < -EXPLORE_BIN_EDGE:
            return 0
        if z > EXPLORE_BIN_EDGE:
            return 2
        return 1
    return bucket(z_energy), bucket(z_experimental)


def stratify(eligible: list[tuple[str, dict[str, float] | None]], pool_size: int, rng) -> list[str]:
    """Round-robin sample across energy×experimental z-bins.

    ``eligible`` is ``[(track_id, z_axes | None)]``; tracks without axis data
    fall into one shared bin. Stratification keeps the random slice spread
    across the sonic space instead of clustering around the library's bulk.
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
        if i > 10_000:
            break
        if all(not lst for lst in bin_lists):
            break
    return out


def stratified_fresh(
    meta: dict[str, dict], *,
    excluded: set[str],
    recency_hours: dict[str, float],
    repulsion_by_id: dict[str, float],
    neg_sim_by_id: dict[str, float],
    axis_stats: dict | None,
    axis_names: tuple[str, ...],
    p_final: dict[str, float] | None,
    confidence: float,
    play_counts: dict[str, int],
    pool_size: int,
    rng,
    pool_label: str = "fresh",
) -> list[StreamCandidate]:
    """Random-but-spread fresh picks straight from the library index.

    Deliberately NOT nearest-neighbour: this is the slice that keeps the wave
    from collapsing onto the session's own footprint. Picks sitting inside a
    negative cluster's neighbourhood (``EXPLORE_NEG_PCT``) are dropped — random
    must not lead back where the listener just walked away from.

    ``meta`` is the SQLite-backed ``{track_id: payload}`` mirror of the whole
    collection — no Qdrant traffic happens here at all.

    ``pool_label`` defaults to ``fresh`` for the historical callers. The stream
    passes ``explore`` so the pick is distinguishable downstream: ``fresh`` only
    means «not played in 30 days», which is true of most near-field candidates
    too, and the skip-forgiveness rule (§3.4) must apply to the wave's guesses
    alone, not to every unheard track.
    """
    payload_by_id: dict[str, dict] = {}
    z_by_id: dict[str, dict[str, float] | None] = {}
    eligible: list[tuple[str, dict[str, float] | None]] = []
    for tid, payload in list(meta.items())[:STRATIFIED_SCROLL_CAP]:
        if tid in excluded or not is_fresh(tid, recency_hours):
            continue
        if neg_sim_by_id.get(tid, 0.0) >= EXPLORE_NEG_PCT:
            continue
        if not duration_ok(payload):
            continue
        payload_by_id[tid] = payload
        z = z_scores_for_axes(payload.get("sonic_axes"), axis_stats, axis_names)
        z_by_id[tid] = z
        eligible.append((tid, z))

    out: list[StreamCandidate] = []
    for tid in stratify(eligible, pool_size, rng):
        payload = payload_by_id[tid]
        axis = axis_match_score(z_by_id[tid], p_final, confidence, axis_names)
        out.append(StreamCandidate(
            track_id=tid, payload=payload, pool=pool_label,
            axis_match=axis,
            score=score_candidate(
                affinity=0.0, repulsion=repulsion_by_id.get(tid, 0.0),
                axis_match=axis, play_count=play_counts.get(tid, 0),
                recency_h=None,
            ),
        ))
    return out


# ── Assembly ───────────────────────────────────────────────────────────────

def explore_positions(n: int, n_explore: int) -> set[int]:
    """Evenly spaced slot indices reserved for exploration (design §3.2).

    One exploratory track in a chunk of three lands in the middle, two in a
    chunk of six land at thirds. Spacing them keeps the widening from arriving
    as a clump.
    """
    if n_explore <= 0 or n <= 0:
        return set()
    n_explore = min(n_explore, n)
    return {int((i + 1) * n / (n_explore + 1)) for i in range(n_explore)}


def assemble_chunk(
    fresh: list[StreamCandidate],
    familiar: list[StreamCandidate],
    *,
    n: int,
    fresh_quota: int,
    recent_artists: list[str],
    fresh_is_hard: bool,
    explore: list[StreamCandidate] | None = None,
    n_explore: int = 0,
) -> list[StreamCandidate]:
    """Fill ``n`` slots honouring the slider quota and the interleaving rules.

    ``fresh_is_hard`` (slider pinned to «РЕДКОЕ») forbids topping up from the
    familiar pool: a short chunk is the honest answer when nothing unheard is
    left. In the other direction undersupply is the worse outcome, so a dry
    familiar pool does fall back to fresh.

    ``explore`` gets **reserved slots**, not a place in the fresh pool, and that
    distinction is the whole point. Exploratory picks score far below near-field
    ones by construction — the stratified sampler hard-codes ``affinity=0``, so
    its ceiling is ``W_AXIS + W_NOVELTY = 0.30`` against a typical near-field
    ``0.66`` — and this function serves the best-scoring candidate available.
    Appending them to ``fresh`` therefore buried them: they could be counted by
    the quota and still never be served. A reserved slot is what makes the quota
    mean anything. When the explore pool is dry the slot falls through to the
    normal pools rather than shortening the chunk.
    """
    fresh_sorted = sorted(fresh, key=lambda c: c.score, reverse=True)
    familiar_sorted = sorted(familiar, key=lambda c: c.score, reverse=True)
    explore_sorted = sorted(explore or [], key=lambda c: c.score, reverse=True)
    explore_left = max(0, min(n_explore, n))
    slots = explore_positions(n, explore_left)

    out: list[StreamCandidate] = []
    artist_tail: list[str] = list(recent_artists)[-ARTIST_REPEAT_WINDOW:]
    consecutive: dict[str, int] = {"fresh": 0, "familiar": 0}
    quota_left = max(0, min(n, fresh_quota))

    def _artist(c: StreamCandidate) -> str:
        return (c.payload.get("artist") or "").strip().lower()

    def _violates_artist_rule(c: StreamCandidate) -> bool:
        a = _artist(c)
        return (len(artist_tail) >= MAX_CONSECUTIVE_ARTIST
                and a != ""
                and all(t == a for t in artist_tail[-MAX_CONSECUTIVE_ARTIST:]))

    def _pick(pool: list[StreamCandidate]) -> StreamCandidate | None:
        # First pass honours the artist rule; the second bends it — same
        # rationale as autoplay_service: undersupply is the worse UX.
        for bend in (False, True):
            best, best_idx, best_val = None, -1, float("-inf")
            for i, c in enumerate(pool):
                if not bend and _violates_artist_rule(c):
                    continue
                val = c.score - (ARTIST_REPEAT_PENALTY if _artist(c) in artist_tail else 0.0)
                if val > best_val:
                    best, best_idx, best_val = c, i, val
            if best is not None:
                pool.pop(best_idx)
                return best
        return None

    for _ in range(n):
        slots_left = n - len(out)

        c = None
        # Reserved exploration slot: this position, or any position left once
        # the remaining slots are exactly the unspent explore quota.
        if explore_left > 0 and explore_sorted and (
                len(out) in slots or explore_left >= slots_left):
            c = _pick(explore_sorted)
            if c is not None:
                explore_left -= 1
                quota_left = max(0, quota_left - 1)   # an explore pick is unheard

        # Every branch below is guarded by ``c is None``, so these only matter
        # when the reserved slot did not claim this position.
        must_fresh = quota_left >= slots_left
        want_fresh = (c is None and quota_left > 0 and fresh_sorted
                      and (consecutive["fresh"] < MAX_CONSECUTIVE_SAME_POOL or must_fresh))

        if want_fresh:
            c = _pick(fresh_sorted)
            if c is not None:
                quota_left -= 1
        if c is None and not must_fresh:
            c = _pick(familiar_sorted)
        if c is None and fresh_sorted:
            # familiar dry — fresh tops up regardless of quota
            c = _pick(fresh_sorted)
            quota_left = max(0, quota_left - 1)
        if c is None and familiar_sorted and not fresh_is_hard:
            c = _pick(familiar_sorted)
        if c is None:
            break  # both pools dry — return a short chunk

        # The exploratory pools are unheard by construction, so they count as
        # fresh for the ≤2-in-a-row rule — otherwise a band pick would license
        # two more familiar ones straight after it.
        pool = "fresh" if c.pool in ("fresh", "band", "explore") else "familiar"
        for key in consecutive:
            consecutive[key] = consecutive[key] + 1 if key == pool else 0
        out.append(c)
        artist_tail.append(_artist(c))
        artist_tail = artist_tail[-ARTIST_REPEAT_WINDOW:]
    return out
