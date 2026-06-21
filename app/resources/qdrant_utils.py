"""Small Qdrant helpers shared across services.

`scroll_all` centralises the offset-following pagination loop that was copied
across indexing/library/similarity code. It yields **every** point in a
collection, following ``next_page_offset`` until it comes back ``None`` (or a
batch comes back empty). It does not impose a page/count cap and propagates any
client exception to the caller — callers that need a cap, an early break, or
partial-result-on-error must keep their own loop.

Two payload-projection helpers live here because the recommendation/home
surfaces were transferring the **full** Qdrant payload — including each track's
entire ``lyrics`` blob — for thousands of points on every request:

- ``PAYLOAD_EXCLUDE_LYRICS`` — a ``with_payload=`` selector that returns the
  whole payload *except* lyrics. Use for bounded retrieves/queries where the
  caller reads many small metadata fields but never the lyrics.
- ``light_points`` — a per-collection cached, lyrics-free full-collection
  scroll (card fields + ``sonic_axes`` only). Collapses the repeated
  whole-library scans the home page + post-like stream refresh did on every
  request into one shared scroll.

The player still shows lyrics for the *playing* track: the frontend fetches
them on demand via ``GET /search/tracks/{id}/lyrics`` whenever the inline field
is absent, so stripping lyrics from these payloads is transparent.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Iterator

from qdrant_client import models

logger = logging.getLogger(__name__)

# Return the whole payload EXCEPT the heavy lyrics blob. The single big field;
# everything else (producer, label, samples, track/disc numbers, …) stays.
PAYLOAD_EXCLUDE_LYRICS = models.PayloadSelectorExclude(exclude=["lyrics"])

# Canonical "light" payload for recsys/home/library full-collection scans: card
# fields for display + ``sonic_axes`` for scoring + the artist-split fields the
# distinct-artist enumeration needs + ``duration_range`` for the stats
# histogram. NO lyrics. One cached scroll per collection serves the explore
# pool, axis playlist, "artist of the day", the library albums grid and the
# landing-stats panel.
LIGHT_PAYLOAD_FIELDS = [
    "title", "artist", "artists", "artist_slugs", "primary_artist_slug",
    "album", "year", "genre", "duration", "duration_range",
    "file_path", "cover_art_path", "sonic_axes",
]

# Cache: collection_name -> (monotonic_ts, point_count, [(id, payload), ...]).
_LIGHT_CACHE: dict[str, tuple[float, int, list[tuple[str, dict]]]] = {}
_LIGHT_CACHE_LOCK = threading.Lock()
LIGHT_CACHE_TTL = 90.0  # seconds a cached scroll is trusted without re-checking

# Parallel cache for the SQLite-backed fast path (track_metadata mirror). Kept
# separate from the Qdrant scroll cache so the two sources never cross-pollute;
# both are cleared together by invalidate_light_cache().
_SQLITE_LIGHT_CACHE: dict[str, tuple[float, int, list[tuple[str, dict]]]] = {}
_SQLITE_LIGHT_LOCK = threading.Lock()


def light_points(client, collection_name: str, *, batch_size: int = 512) -> list[tuple[str, dict]]:
    """Full-collection ``[(id, light_payload)]`` (lyrics stripped), cached per
    collection.

    Within ``LIGHT_CACHE_TTL`` the cached list is returned untouched. After the
    TTL lapses the cache is revalidated against ``client.count`` (cheap — no
    payload transfer): an unchanged point count keeps the cache for another
    window, a changed count (tracks added/removed/reindexed) forces a re-scroll.
    On a scroll error the last good cache is served if present, else ``[]``.

    Fast path: when the SQLite ``track_metadata`` mirror is populated for this
    collection (post-backfill / normal indexing), the light payloads are served
    from there with NO Qdrant I/O — this is what collapses the 20+ scroll burst
    on the stream/home/library/search surfaces. The Qdrant scroll below is the
    fallback for pre-backfill collections (empty mirror).
    """
    sqlite_pts = _sqlite_light_points(collection_name)
    if sqlite_pts:
        return sqlite_pts

    now = time.monotonic()
    with _LIGHT_CACHE_LOCK:
        entry = _LIGHT_CACHE.get(collection_name)
    if entry is not None:
        ts, count, pts = entry
        if now - ts < LIGHT_CACHE_TTL:
            return pts
        cur = _safe_count(client, collection_name)
        if cur is not None and cur == count:
            with _LIGHT_CACHE_LOCK:
                _LIGHT_CACHE[collection_name] = (now, count, pts)
            return pts

    fresh = _scroll_light(client, collection_name, batch_size=batch_size)
    if fresh is None:  # scroll failed — don't poison the cache, serve stale.
        return entry[2] if entry is not None else []
    count = _safe_count(client, collection_name)
    with _LIGHT_CACHE_LOCK:
        _LIGHT_CACHE[collection_name] = (
            time.monotonic(), count if count is not None else len(fresh), fresh,
        )
    return fresh


def _sqlite_light_points(collection_name: str) -> list[tuple[str, dict]]:
    """Cached read of the SQLite ``track_metadata`` mirror (card fields +
    ``sonic_axes``). Returns ``[]`` when the mirror is empty so ``light_points``
    falls back to a Qdrant scroll.

    Same TTL + count-revalidation contract as the Qdrant cache, but the cheap
    revalidation count comes from SQLite (``COUNT(*)``) — so a warm collection
    never touches Qdrant here.
    """
    now = time.monotonic()
    with _SQLITE_LIGHT_LOCK:
        entry = _SQLITE_LIGHT_CACHE.get(collection_name)
    if entry is not None:
        ts, count, pts = entry
        if now - ts < LIGHT_CACHE_TTL:
            return pts
        cur = _sqlite_count(collection_name)
        if cur is not None and cur == count:
            with _SQLITE_LIGHT_LOCK:
                _SQLITE_LIGHT_CACHE[collection_name] = (now, count, pts)
            return pts

    try:
        from app.resources.metadata_db import MetadataDB
        pts = MetadataDB.get_light_points(collection_name)
    except Exception:
        logger.exception("[sqlite] light points read failed for %s", collection_name)
        return entry[2] if entry is not None else []

    with _SQLITE_LIGHT_LOCK:
        _SQLITE_LIGHT_CACHE[collection_name] = (time.monotonic(), len(pts), pts)
    return pts


def _sqlite_count(collection_name: str) -> int | None:
    try:
        from app.resources.metadata_db import MetadataDB
        return MetadataDB.get_track_count_for_collection(collection_name)
    except Exception:
        return None


def _scroll_light(client, collection_name: str, *, batch_size: int) -> list[tuple[str, dict]] | None:
    out: list[tuple[str, dict]] = []
    offset = None
    try:
        while True:
            batch, offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                with_payload=LIGHT_PAYLOAD_FIELDS,
                with_vectors=False,
                offset=offset,
            )
            out.extend((str(p.id), p.payload or {}) for p in batch)
            if offset is None or not batch:
                break
    except Exception:
        logger.exception("[qdrant] light scroll failed for %s", collection_name)
        return None
    return out


def _safe_count(client, collection_name: str) -> int | None:
    try:
        return client.count(collection_name=collection_name, exact=True).count
    except Exception:
        return None


def invalidate_light_cache(collection_name: str | None = None) -> None:
    """Drop the cached light index for one collection (``None`` = all).

    Call after a (re)index so new tracks surface immediately instead of waiting
    out the TTL; also used by tests to keep the module-level cache from leaking
    between cases.
    """
    with _LIGHT_CACHE_LOCK:
        if collection_name is None:
            _LIGHT_CACHE.clear()
        else:
            _LIGHT_CACHE.pop(collection_name, None)
    with _SQLITE_LIGHT_LOCK:
        if collection_name is None:
            _SQLITE_LIGHT_CACHE.clear()
        else:
            _SQLITE_LIGHT_CACHE.pop(collection_name, None)


def scroll_all(client, collection_name: str, *, batch_size: int = 256, **scroll_kwargs) -> Iterator:
    """Yield each point of ``collection_name``, paging via Qdrant scroll.

    Parameters mirror ``client.scroll`` — extra keyword args (``with_payload``,
    ``with_vectors``, ``scroll_filter`` …) are forwarded verbatim. Stops when
    the returned offset is ``None`` or a page is empty, matching the
    ``if offset is None or not points: break`` idiom the call sites used.
    """
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=batch_size,
            **scroll_kwargs,
        )
        for point in points:
            yield point
        if offset is None or not points:
            break
