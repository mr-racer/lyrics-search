"""Feature #2: fill EMPTY metadata fields from the Yandex catalog.

Given a scan ``meta`` dict that already has ``artist`` + ``title`` (the pipeline's
hard requirement), search the Yandex catalog and backfill only the fields that are
missing — never overwrite values already read from the file's tags.

Credentials (spec decision): the account's token when linked, otherwise an
anonymous client. Failures are always swallowed — enrichment must never break
indexing.
"""

from __future__ import annotations

import logging
import os
import threading

from app.services.yandex.client_factory import Throttle, build_client

logger = logging.getLogger(__name__)

# Fields we attempt to backfill (cover is handled separately by the pipeline's
# embedded-art extraction). Only filled when currently empty.
_ENRICHABLE = ("album", "year", "genre")

# Duration sanity window (seconds) to reject a wrong search match.
_DURATION_TOLERANCE_S = 5

_throttle = Throttle()

# Lazily-built anonymous client, shared across enrichment calls. ``False`` is the
# "tried and failed" sentinel so we don't re-attempt a dead client per track.
_anon_lock = threading.Lock()
_anon_client = None  # None = not tried, False = failed, else a Client


def _enabled() -> bool:
    return os.environ.get("MUSIX_YM_ENRICH", "1").strip().lower() not in ("0", "false", "no")


def get_anonymous_client():
    """Return a cached anonymous Yandex client, or None if it can't be built."""
    global _anon_client
    with _anon_lock:
        if _anon_client is None:
            try:
                _anon_client = build_client(None)
            except Exception:
                logger.warning("[yandex/enrichment] anonymous client init failed", exc_info=True)
                _anon_client = False
        return _anon_client or None


def client_for_account(account_id: str | None):
    """Best client for enrichment: account token if linked, else anonymous.

    Built once per import/index job by the caller and threaded into the scan, so
    we don't pay a client init per file.
    """
    if account_id:
        try:
            from app.services.yandex import token_store

            blob = token_store.load_token(account_id)
            if blob and blob.get("access_token"):
                return build_client(blob["access_token"])
        except Exception:
            logger.debug("[yandex/enrichment] account client init failed, falling back",
                         exc_info=True)
    return get_anonymous_client()


def _missing_fields(meta: dict) -> list[str]:
    return [f for f in _ENRICHABLE if not meta.get(f)]


def _search_track(client, artist: str, title: str):
    """Return the best matching Yandex Track for 'artist title', or None."""
    res = client.search(f"{artist} {title}")
    if res is None:
        return None
    best = getattr(res, "best", None)
    if best is not None and getattr(best, "type", None) == "track" and best.result:
        return best.result
    tracks = getattr(res, "tracks", None)
    if tracks is not None and tracks.results:
        return tracks.results[0]
    return None


def _duration_matches(meta: dict, ytrack) -> bool:
    """True unless both durations are known AND differ by more than the tolerance."""
    want = meta.get("duration")
    got_ms = getattr(ytrack, "duration_ms", None)
    if not want or not got_ms:
        return True  # can't compare → don't block enrichment
    return abs(want - got_ms / 1000.0) <= _DURATION_TOLERANCE_S


def apply_yandex_track(meta: dict, ytrack) -> dict:
    """Backfill empty fields of ``meta`` from a Yandex ``Track`` (pure)."""
    album = ytrack.albums[0] if getattr(ytrack, "albums", None) else None
    values = {
        "album": getattr(album, "title", None) if album else None,
        "year": getattr(album, "year", None) if album else None,
        "genre": getattr(album, "genre", None) if album else None,
    }
    for field in _ENRICHABLE:
        if not meta.get(field) and values.get(field):
            meta[field] = values[field]
    return meta


def enrich_metadata(meta: dict, client=None) -> dict:
    """Fill empty album/year/genre from Yandex. Returns ``meta`` (mutated in place).

    No-op when disabled, when nothing is missing, when artist/title are absent, or
    on any Yandex/network error (logged, never raised).
    """
    if not _enabled():
        return meta
    if not meta.get("artist") or not meta.get("title"):
        return meta
    if not _missing_fields(meta):
        return meta

    client = client or get_anonymous_client()
    if client is None:
        return meta

    try:
        _throttle.wait()
        ytrack = _search_track(client, meta["artist"], meta["title"])
        if ytrack is None:
            return meta
        if not _duration_matches(meta, ytrack):
            logger.debug("[yandex/enrichment] duration mismatch, skipping %s — %s",
                         meta["artist"], meta["title"])
            return meta
        return apply_yandex_track(meta, ytrack)
    except Exception:
        logger.debug("[yandex/enrichment] enrichment failed for %s — %s",
                     meta.get("artist"), meta.get("title"), exc_info=True)
        return meta
