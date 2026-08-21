"""Round lifecycle: list modes, build a round, take an answer, resolve audio.

This is the only module in the package that touches storage. Modes stay pure
functions over a snapshot, which is what keeps the difficulty design testable.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §6, §12.
"""
from __future__ import annotations

import json
import logging
import random as _random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.resources.clap_features import blend_axis_stats, load_axis_norm_reference
from app.resources.metadata_db import MetadataDB
from app.services.quiz.context import RoundContext
from app.services.quiz.errors import (
    AlreadyAnswered,
    NoRoundAvailable,
    RoundNotFound,
)
from app.services.quiz.library import load_library
from app.services.quiz.modes import MODES, get_mode
from app.services.quiz.selection import (
    MIN_LIBRARY_FOR_ADAPTIVITY,
    familiarity_percentiles,
    next_band,
    update_skill,
)

# I-5: a mode stays hidden until the library can support this many rounds.
MIN_POOL = 20

# How long a round stays answerable.
ROUND_TTL_SEC = 120.0

# Anti-repeat window: whichever limit bites first.
ANTI_REPEAT_DAYS = 30
ANTI_REPEAT_ROUNDS = 50

DEFAULT_SNIPPET_SEC = 3
ALLOWED_SNIPPET_SEC = (3, 5, 10)

_SECONDS_PER_DAY = 86400.0

logger = logging.getLogger(__name__)


@dataclass
class _Snapshot:
    """Everything read from storage once per call, shared across modes."""

    tracks: List[Dict]
    plays: Dict[str, int]
    last_played: Dict[str, Optional[float]]
    percentiles: Dict[str, float]
    axis_stats: Optional[Dict]
    producers: Dict[str, Dict]
    now: float


def list_modes(*, qdrant_client, collection_name: str) -> List[Dict]:
    """Every mode with its pool size and whether it clears the I-5 floor.

    Unavailable modes are still listed: the UI says "not enough here yet" in
    plain words, which is friendlier than a menu that silently changes shape.
    """
    snapshot = _snapshot(qdrant_client, collection_name)
    out: List[Dict] = []
    for key, mode in MODES.items():
        ctx = _context_for(snapshot, collection_name, key)
        size = int(mode.pool_size(ctx))
        out.append({
            "key": key, "pool_size": size, "available": size >= MIN_POOL,
            # Whether the round has anything to listen to. The client needs it
            # up front so it does not draw a play key for a knowledge round.
            "has_audio": bool(getattr(mode, "HAS_AUDIO", True)),
        })
    return out


def build_round(
    *, qdrant_client, collection_name: str, mode: str,
    snippet_sec: int = DEFAULT_SNIPPET_SEC,
) -> Dict:
    """Create a round and return it WITHOUT the answer."""
    mode_module = get_mode(mode)
    if mode_module is None:
        raise NoRoundAvailable(f"unknown mode: {mode}")
    if snippet_sec not in ALLOWED_SNIPPET_SEC:
        snippet_sec = DEFAULT_SNIPPET_SEC

    snapshot = _snapshot(qdrant_client, collection_name)
    ctx = _context_for(snapshot, collection_name, mode)
    if mode_module.pool_size(ctx) < MIN_POOL:
        raise NoRoundAvailable("not enough material for this mode yet")

    spec = mode_module.build(ctx, snippet_sec=snippet_sec)
    round_id = uuid.uuid4().hex
    expires_at = snapshot.now + ROUND_TTL_SEC
    MetadataDB.create_quiz_round(
        round_id=round_id,
        collection_name=collection_name,
        mode=mode,
        track_id=spec.track_id,
        spec_json=json.dumps(spec.to_stored()),
        expires_at=expires_at,
        created_at=snapshot.now,
    )
    # No track_id and no correct_option_id: the client learns the answer only
    # by submitting one.
    return {
        "round_id": round_id,
        "mode": mode,
        "options": spec.options,
        "start_sec": spec.start_sec,
        "length_sec": spec.length_sec,
        "expires_at": expires_at,
        "has_audio": bool(getattr(mode_module, "HAS_AUDIO", True)),
    }


def submit_answer(
    *, qdrant_client, collection_name: str, round_id: str, answer: Dict,
) -> Dict:
    """Score a round once, then reveal the truth."""
    row = MetadataDB.get_quiz_round(round_id)
    if row is None or row["collection_name"] != collection_name:
        # Deliberately the same error as "no such round": confirming that a
        # round exists but belongs to someone else is itself a leak.
        raise RoundNotFound(round_id)
    if row["answered_at"] is not None:
        raise AlreadyAnswered(round_id)

    mode_module = get_mode(row["mode"])
    if mode_module is None:
        raise RoundNotFound(round_id)

    spec = json.loads(row["spec_json"])
    expired = time.time() > float(row["expires_at"])
    if expired:
        correct, score = False, 0.0
    else:
        correct, score = mode_module.score(spec, answer or {})

    if not MetadataDB.answer_quiz_round(
        round_id=round_id, answer_json=json.dumps(answer or {}),
        correct=correct, score=score,
    ):
        # Lost a race with a concurrent submission.
        raise AlreadyAnswered(round_id)

    snapshot = _snapshot(qdrant_client, collection_name)
    if not expired:
        # An expired round teaches nothing about skill: the listener walked
        # away, they did not fail. Scoring it would quietly lower difficulty.
        _advance_skill(collection_name, row["mode"], correct,
                       library_size=len(snapshot.tracks))

    track = next((t for t in snapshot.tracks
                  if t.get("track_id") == row["track_id"]), None) or {}
    return {
        "correct": correct,
        "score": score,
        "expired": expired,
        "correct_option_id": spec.get("correct_option_id"),
        # Only now: the producer whose three tracks those were, the year that
        # was being guessed. Withheld from the question payload entirely.
        "reveal": spec.get("reveal") or {},
        "truth": {
            "track_id": row["track_id"],
            "title": track.get("title_display") or track.get("title") or "—",
            "artist": track.get("artist") or "—",
            "album": track.get("album"),
            "year": track.get("year"),
            "cover_art_path": track.get("cover_art_path"),
        },
    }


def resolve_round_audio(
    *, collection_name: str, round_id: str,
) -> Tuple[str, float, float]:
    """``(track_id, start_sec, length_sec)`` for a round the caller owns.

    Returns the track id rather than a path so this layer stays free of file
    I/O and Qdrant; the router resolves the file exactly as the normal stream
    route does.
    """
    row = MetadataDB.get_quiz_round(round_id)
    if row is None or row["collection_name"] != collection_name or not row["track_id"]:
        raise RoundNotFound(round_id)
    spec = json.loads(row["spec_json"])
    return (
        str(row["track_id"]),
        float(spec.get("start_sec") or 0.0),
        float(spec.get("length_sec") or 0.0),
    )


# ── internals ────────────────────────────────────────────────────────────────

def _snapshot(qdrant_client, collection_name: str) -> _Snapshot:
    tracks = load_library(qdrant_client, collection_name)
    play_counts = MetadataDB.get_play_counts_by_track(collection_name)
    recency = MetadataDB.get_play_recency_map(collection_name)
    now = time.time()

    ids = [t.get("track_id") for t in tracks]
    plays = {track_id: int(play_counts.get(track_id, 0)) for track_id in ids}
    last_played = {track_id: _epoch(recency.get(track_id)) for track_id in ids}
    return _Snapshot(
        tracks=tracks,
        plays=plays,
        last_played=last_played,
        percentiles=familiarity_percentiles(plays, last_played, now),
        axis_stats=blend_axis_stats(
            MetadataDB.get_axis_norm_stats(collection_name),
            load_axis_norm_reference(),
        ),
        producers=_producer_index(collection_name),
        now=now,
    )


def _producer_index(collection_name: str) -> Dict[str, Dict]:
    """Credits are optional data: a collection with none must still play M1."""
    try:
        from app.services.track_credits_service import producer_index
        return producer_index(collection_name)
    except Exception:
        logger.warning("[quiz] producer index unavailable for %s",
                       collection_name, exc_info=True)
        return {}


def _context_for(
    snapshot: _Snapshot, collection_name: str, mode_key: str, *, rng=None,
) -> RoundContext:
    skill = MetadataDB.get_quiz_skill(collection_name, mode_key)
    if len(snapshot.tracks) < MIN_LIBRARY_FOR_ADAPTIVITY:
        # Percentiles carry no information on a small library — open the band
        # rather than pretending the tail means something (spec §16 R-2).
        skill = {**skill, "band_lo": 0.0, "band_hi": 100.0}
    return RoundContext(
        collection_name=collection_name,
        tracks=snapshot.tracks,
        plays=snapshot.plays,
        last_played=snapshot.last_played,
        percentiles=snapshot.percentiles,
        skill=skill,
        exclude=MetadataDB.recent_quiz_track_ids(
            collection_name, mode_key,
            limit=ANTI_REPEAT_ROUNDS,
            since_ts=snapshot.now - ANTI_REPEAT_DAYS * _SECONDS_PER_DAY,
        ),
        axis_stats=snapshot.axis_stats,
        producers=snapshot.producers,
        rng=rng or _random,
        now=snapshot.now,
    )


def _advance_skill(
    collection_name: str, mode_key: str, correct: bool, *, library_size: int,
) -> None:
    state = MetadataDB.get_quiz_skill(collection_name, mode_key)
    skill = update_skill(float(state["skill"]), correct)
    n_answered = int(state["n_answered"]) + 1
    band, out_of_band = next_band(
        (float(state["band_lo"]), float(state["band_hi"])),
        skill=skill,
        n_answered=n_answered,
        out_of_band=int(state["out_of_band"]),
        library_size=library_size,
    )
    MetadataDB.save_quiz_skill(
        collection_name=collection_name, mode=mode_key, skill=skill,
        n_answered=n_answered, band_lo=band[0], band_hi=band[1],
        out_of_band=out_of_band,
    )


def _epoch(value) -> Optional[float]:
    """ISO timestamp from the playback tables to epoch seconds, or None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None
