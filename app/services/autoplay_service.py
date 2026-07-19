"""Autoplay queue service — CLAP-similarity-based 'play next' candidates.

Flow:
  1. Resolve the seed track's CLAP vector from Qdrant.
  2. Query Qdrant for top-K candidate neighbors (K = limit * 3, min 30).
  3. Apply hard filters: drop seed, drop excluded ids, drop dislikes.
  4. Apply soft diversity demotion (no >2 consecutive same-artist).
  5. Return the trimmed list plus diagnostics.

The Qdrant client and reactions accessor are injected at call time so
this module stays unit-testable without a live database.
"""

from __future__ import annotations

from typing import Iterable

from app.domain.models import AutoplayQueueDiagnostics, AutoplayQueueResponse, TrackMetadata
from app.resources.metadata_db import MetadataDB
from app.resources.qdrant_utils import PAYLOAD_EXCLUDE_LYRICS
from app.services._payload_coerce import coerce_float
from app.services.artist_split import artist_refs_for_track, display_title_for_track


def _point_to_track(pt) -> TrackMetadata:
    p = pt.payload or {}
    return TrackMetadata(
        track_id=p.get("track_id") or str(pt.id),
        title=p.get("title", ""),
        title_display=display_title_for_track(p),
        artist=p.get("artist", ""),
        album=p.get("album"),
        year=p.get("year"),
        cover_art_path=p.get("cover_art_path"),
        file_path=p.get("file_path") or "",
        duration_sec=p.get("duration_sec") or 0.0,
        genre=p.get("genre"),
        artist_refs=artist_refs_for_track(p),
    )


def _apply_filters(
    candidates: Iterable,
    *,
    seed_track_id: str,
    exclude_ids: set[str],
    dislikes: set[str],
    limit: int,
) -> tuple[list[TrackMetadata], AutoplayQueueDiagnostics]:
    """Filter Qdrant candidates and return up to `limit` tracks + diagnostics."""
    candidates = list(candidates)
    fetched = len(candidates)

    dropped_excluded = 0
    dropped_disliked = 0
    accepted_main: list[TrackMetadata] = []
    demoted_tail: list[TrackMetadata] = []

    last_artists: list[str] = []  # rolling window of last 2 artists in accepted_main

    for pt in candidates:
        track = _point_to_track(pt)
        tid = track.track_id

        if tid == seed_track_id or tid in exclude_ids:
            dropped_excluded += 1
            continue
        if tid in dislikes:
            dropped_disliked += 1
            continue

        dur = coerce_float((pt.payload or {}).get("duration"))
        if dur is not None and 0.0 < dur < 60.0:
            dropped_excluded += 1
            continue

        # Diversity: if the last 2 entries share the artist with this candidate, demote.
        if (
            len(last_artists) >= 2
            and last_artists[-1] == track.artist
            and last_artists[-2] == track.artist
        ):
            demoted_tail.append(track)
            continue

        accepted_main.append(track)
        last_artists.append(track.artist)
        if len(last_artists) > 2:
            last_artists.pop(0)

        if len(accepted_main) >= limit:
            break

    out = accepted_main[:limit]
    demoted_inserted: set[int] = set()

    if len(out) < limit:
        # Top up with demoted ones — re-insert UNCONDITIONALLY, without
        # re-applying the >2-consecutive-same-artist rule.
        #
        # Rationale: when the entire candidate pool is dominated by one artist
        # (e.g. a tiny library, or a query whose neighbors happen to cluster on
        # one artist's catalog), enforcing diversity here would leave the queue
        # under-filled — a worse UX than mild rule-bending. Diversity was already
        # maximised by the main loop above; topup only fires when no
        # diversity-safe candidate remained to fill the remaining slots.
        for i, track in enumerate(demoted_tail):
            if len(out) >= limit:
                break
            out.append(track)
            demoted_inserted.add(i)

    dropped_diversity = len(demoted_tail) - len(demoted_inserted)

    diag = AutoplayQueueDiagnostics(
        candidates_fetched=fetched,
        dropped_excluded=dropped_excluded,
        dropped_disliked=dropped_disliked,
        dropped_diversity=max(0, dropped_diversity),
        returned=len(out),
    )
    return out, diag


def next_queue(
    *,
    qdrant_client,
    collection_name: str,
    seed_track_id: str,
    exclude_ids: list[str] | None = None,
    limit: int = 20,
) -> AutoplayQueueResponse:
    """Public entry point. Pulls candidates from Qdrant and runs the filter chain."""
    exclude_set = set(exclude_ids or [])
    # Hard cap to prevent runaway query strings.
    if len(exclude_set) > 200:
        exclude_set = set(list(exclude_set)[:200])

    # 1. Fetch seed CLAP vector.
    try:
        seed_pts = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[seed_track_id],
            with_payload=False,
            with_vectors=["clap"],
        )
    except Exception:
        seed_pts = []
    if not seed_pts:
        return AutoplayQueueResponse(
            seed_track_id=seed_track_id,
            tracks=[],
            diagnostics=AutoplayQueueDiagnostics(
                candidates_fetched=0, dropped_excluded=0, dropped_disliked=0,
                dropped_diversity=0, returned=0,
            ),
        )

    seed_vec = (
        seed_pts[0].vector.get("clap")
        if isinstance(seed_pts[0].vector, dict)
        else seed_pts[0].vector
    )
    # A track may exist in Qdrant without a 'clap' named vector (legacy points
    # indexed before CLAP support). In that case we can't form a neighbor
    # query — return an empty queue gracefully instead of crashing in
    # qdrant_client.query_points().
    if not seed_vec:
        return AutoplayQueueResponse(
            seed_track_id=seed_track_id,
            tracks=[],
            diagnostics=AutoplayQueueDiagnostics(
                candidates_fetched=0, dropped_excluded=0, dropped_disliked=0,
                dropped_diversity=0, returned=0,
            ),
        )

    # 2. Query top-K neighbors.
    # qdrant-client >= 1.10: query_points replaced the removed .search()
    k = max(limit * 3, 30)
    candidates = qdrant_client.query_points(
        collection_name=collection_name,
        query=seed_vec,
        using="clap",
        limit=k,
        with_payload=PAYLOAD_EXCLUDE_LYRICS,
    ).points

    # 3. Reactions lookup (single batched query).
    candidate_ids = [str(c.id) for c in candidates]
    reactions = MetadataDB.get_reactions_for_tracks(collection_name, candidate_ids)
    dislikes = {tid for tid, r in reactions.items() if r == "dislike"}

    # 4. Apply filters.
    out, diag = _apply_filters(
        candidates,
        seed_track_id=seed_track_id,
        exclude_ids=exclude_set,
        dislikes=dislikes,
        limit=limit,
    )

    return AutoplayQueueResponse(
        seed_track_id=seed_track_id,
        tracks=out,
        diagnostics=diag,
    )
