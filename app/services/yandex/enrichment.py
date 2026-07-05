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


def _clean_str(v) -> str | None:
    """Normalise a Yandex catalog string (title/genre) to a non-empty stripped
    value, or None. Guards against stray whitespace and None/empty."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def apply_yandex_track(meta: dict, ytrack) -> dict:
    """Backfill empty ``album``/``year``/``genre`` of ``meta`` from a Yandex
    ``Track`` (pure). Never overwrites a value the file's tags already carry.

    Types are kept consistent with the tag-read pipeline: ``album``/``genre`` are
    clean strings, and ``year`` is validated through the SAME ``validate_year``
    the readers use — so an out-of-range catalog year is rejected and a good one
    lands as an ``int`` (not the raw catalog value), matching tag-read metadata.
    """
    album = ytrack.albums[0] if getattr(ytrack, "albums", None) else None
    year_val = None
    if album is not None:
        raw_year = getattr(album, "year", None)
        if raw_year is not None:
            from app.indexing.metadata_readers import validate_year
            year_val = validate_year(str(raw_year))
    values = {
        "album": _clean_str(getattr(album, "title", None)) if album else None,
        "year": year_val,
        "genre": _clean_str(getattr(album, "genre", None)) if album else None,
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


def fetch_cover_bytes(meta: dict, client=None) -> bytes | None:
    """Best-effort album cover for a manual upload with NO embedded art.

    Searches the Yandex catalog for ``meta['artist'] + meta['title']`` and, on a
    duration-sane match, downloads the album cover (600×600 JPEG bytes). Returns
    the raw bytes or None. Only called by the cover-save pass when the file itself
    carries no embedded picture, so the common (tagged) case never hits the
    network. Never raises — cover art must never break indexing.

    Credentials mirror :func:`enrich_metadata`: the account token when linked
    (higher-quality catalog access), otherwise an anonymous client.
    """
    if not _enabled():
        return None
    if not meta.get("artist") or not meta.get("title"):
        return None

    client = client or get_anonymous_client()
    if client is None:
        return None

    try:
        _throttle.wait()
        ytrack = _search_track(client, meta["artist"], meta["title"])
        if ytrack is None or not _duration_matches(meta, ytrack):
            return None
        if not getattr(ytrack, "cover_uri", None):
            return None
        raw = ytrack.download_cover_bytes(size="600x600")
        return raw or None
    except Exception:
        logger.debug("[yandex/enrichment] cover fetch failed for %s — %s",
                     meta.get("artist"), meta.get("title"), exc_info=True)
        return None
