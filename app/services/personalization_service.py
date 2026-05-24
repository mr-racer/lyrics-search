"""For-You seed picker — PLACEHOLDER for real personalization (user_vector
ranking, PLATFORM_DESIGN §5.4). Today: weighted pick from likes + recents,
random fallback. Designed to be swapped without changing the endpoint."""
from __future__ import annotations

import random

from app.domain.models import ForYouSeedResponse, HomeTrack
from app.resources.metadata_db import MetadataDB
from app.services._payload_coerce import coerce_float, coerce_year


def _home_track(pt) -> HomeTrack:
    p = pt.payload or {}
    return HomeTrack(
        track_id=str(pt.id),
        title=p.get("title") or "—",
        artist=p.get("artist") or "—",
        album=p.get("album"),
        year=coerce_year(p.get("year")),
        duration=coerce_float(p.get("duration")),
        cover_art_path=p.get("cover_art_path"),
        genre=p.get("genre"),
    )


def pick_for_you_seed(*, qdrant_client, collection_name: str) -> ForYouSeedResponse:
    liked = [tid for tid, _ in MetadataDB.get_liked_track_ids_with_updated_at(collection_name)]
    recent = [tid for tid, _, _ in MetadataDB.get_recent_tracks(collection_name, limit=30)]

    pool: list[str] = []
    pool += liked * 3
    for rank, tid in enumerate(recent):
        weight = max(1, 5 - rank // 6)
        pool += [tid] * weight

    chosen: str | None = random.choice(pool) if pool else None
    # Report the source the chosen seed actually came from (likes take priority
    # when a track is both liked and recent), not merely which lists are non-empty.
    source = ("liked" if chosen in set(liked) else "recent") if chosen is not None else "random"

    if chosen is None:
        try:
            points, _ = qdrant_client.scroll(
                collection_name=collection_name, limit=64,
                with_payload=False, with_vectors=False,
            )
        except Exception:
            points = []
        if not points:
            return ForYouSeedResponse(collection_name=collection_name, source="random")
        chosen = str(random.choice(points).id)
        source = "random"

    try:
        pts = qdrant_client.retrieve(
            collection_name=collection_name, ids=[chosen],
            with_payload=True, with_vectors=False,
        )
    except Exception:
        pts = []
    if not pts:
        return ForYouSeedResponse(collection_name=collection_name, source=source)

    return ForYouSeedResponse(
        seed_track_id=str(pts[0].id), track=_home_track(pts[0]),
        source=source, collection_name=collection_name,
    )
