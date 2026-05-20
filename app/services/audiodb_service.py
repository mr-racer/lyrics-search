"""AudioDB (theaudiodb.com) artist enrichment.

Pulls bio + mood + country + label + 2 PNGs per artist during the FACTS stage
of indexing. Mandatory: runs always, fails gracefully on network errors.

Sibling of artist_facts_service.py — same sequential-fetch-with-progress pattern.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import Callable

import requests

from app.resources.metadata_db import MetadataDB
from app.services.artist_facts_service import _slugify as _slugify_artist

logger = logging.getLogger(__name__)


_FEAT_RE = re.compile(
    r"\s+(?:feat\.?|ft\.?|featuring|with)\s+.*$",
    re.IGNORECASE,
)


ARTIST_COVERS_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "covers" / "artists"
IMAGE_TIMEOUT_SEC = 3.0


def _canonical_artist_name(artist: str) -> str:
    """Strip trailing 'feat. X' / 'ft. X' / 'featuring X' / 'with X' from artist names.

    'Dua Lipa feat. Angele' -> 'Dua Lipa'
    'Tyler, the Creator' -> 'Tyler, the Creator'  (no change — no trailing feat)
    """
    return _FEAT_RE.sub("", artist).strip()


def _audiodb_slug(canonical_artist: str) -> str:
    """Lowercase + drop URL-unsafe punctuation + join words with '+'.

    Caller is expected to pass the canonical name (apply _canonical_artist_name first).
    """
    s = canonical_artist.lower()
    # Drop punctuation. `+` is dropped because we use it as the separator below.
    s = re.sub(r"[,.'`\"!?\\/&()+]", "", s)
    # Collapse whitespace + dashes + underscores into single space, then join with +.
    s = re.sub(r"[\s\-_]+", " ", s).strip()
    return "+".join(s.split())


JSON_TIMEOUT_SEC = 0.4


def _sync_get_json(url: str) -> dict | None:
    """Blocking GET that raises_for_status and parses JSON."""
    r = requests.get(url, timeout=JSON_TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()


async def _http_get_json(url: str) -> dict | None:
    """Async wrapper around _sync_get_json with single retry on ConnectionError/Timeout."""
    try:
        return await asyncio.to_thread(_sync_get_json, url)
    except (requests.ConnectionError, requests.Timeout):
        await asyncio.sleep(1.0)
        try:
            return await asyncio.to_thread(_sync_get_json, url)
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning("[AudioDB] retry failed for %s: %s", url, e)
            return None
        except Exception as e:
            logger.warning("[AudioDB] error during retry %s: %s", url, e)
            return None
    except Exception as e:
        logger.warning("[AudioDB] error %s: %s", url, e)
        return None


async def _download_image(url: str | None) -> str | None:
    """Download an image to /covers/artists/<sha256:16>.<ext>; return relative URL or None.

    No retry — best-effort. PNG / JPEG detected by magic bytes; default to .png.
    Content-addressed: identical bytes share a single file regardless of source URL."""
    if not url:
        return None
    try:
        r = await asyncio.to_thread(
            lambda: requests.get(url, timeout=IMAGE_TIMEOUT_SEC),
        )
        r.raise_for_status()
        data = r.content
        if not data:
            return None
        if data[:8].startswith(b"\x89PNG\r\n\x1a\n"):
            ext = "png"
        elif data[:3] == b"\xff\xd8\xff":
            ext = "jpg"
        else:
            ext = "png"
        ARTIST_COVERS_DIR.mkdir(parents=True, exist_ok=True)
        content_hash = hashlib.sha256(data).hexdigest()[:16]
        dest = ARTIST_COVERS_DIR / f"{content_hash}.{ext}"
        if not dest.exists():
            dest.write_bytes(data)
        return f"/covers/artists/{content_hash}.{ext}"
    except Exception as e:
        logger.warning("[AudioDB] image download failed for %s: %s", url, e)
        return None


AUDIODB_BASE_URL = "https://www.theaudiodb.com/api/v1/json/123/search.php"


async def fetch_audiodb_for_artist(artist: str, collection_name: str) -> None:
    """Fetch + persist AudioDB data for one artist. Idempotent.

    Skip-if-already-fetched: examines audiodb_fetched_at, which is set on every
    persisted write (including empty 'artist not found' rows). Total-network-failure
    does NOT set the timestamp, so the next indexing run retries.
    """
    canonical = _canonical_artist_name(artist)
    slug = _slugify_artist(canonical)
    existing = MetadataDB.get_artist_audiodb(slug, collection_name)
    if existing and existing.get("audiodb_fetched_at"):
        return

    url = f"{AUDIODB_BASE_URL}?s={_audiodb_slug(canonical)}"
    data = await _http_get_json(url)

    # Total fetch failure → don't write a row at all so we retry next time.
    if data is None:
        return

    artists_list = data.get("artists")
    if not artists_list:
        # AudioDB knows this is a miss — mark fetched-but-empty so we don't re-fetch.
        MetadataDB.upsert_artist_audiodb(
            slug=slug, collection_name=collection_name,
            audiodb_bio=None, mood=None,
            country_code=None, country=None, label=None,
            cutout_path=None, thumb_path=None, audiodb_mbid=None,
        )
        return

    a = artists_list[0]
    cutout_path = await _download_image(a.get("strArtistCutout"))
    thumb_path = await _download_image(a.get("strArtistThumb"))

    MetadataDB.upsert_artist_audiodb(
        slug=slug,
        collection_name=collection_name,
        audiodb_bio=a.get("strBiographyEN") or a.get("strBiography"),
        mood=a.get("strMood"),
        country_code=a.get("strCountryCode"),
        country=a.get("strCountry"),
        label=a.get("strLabel"),
        cutout_path=cutout_path,
        thumb_path=thumb_path,
        audiodb_mbid=a.get("strMusicBrainzID"),
    )


async def fetch_audiodb_for_artists(
    artists: list[str],
    collection_name: str,
    delay: float = 0.3,
    progress_callback: Callable[[int, int, str, bool], None] | None = None,
) -> dict[str, bool]:
    """Sequential fetch matching the fetch_facts_for_artists pattern.

    Dedups by canonical name so 'Dua Lipa' and 'Dua Lipa feat. Angele' hit
    AudioDB once. Returns {canonical_artist: enriched_with_data_bool}.
    Progress callback receives (idx, total, label, found_bool); label is the
    first-seen original artist name (e.g. 'Dua Lipa feat. Angele' rather than
    the canonical 'Dua Lipa') so the UI shows what the user has."""
    canonical_to_first_seen: dict[str, str] = {}
    for a in artists:
        canon = _canonical_artist_name(a)
        canonical_to_first_seen.setdefault(canon, a)
    unique_canonicals = list(canonical_to_first_seen.keys())

    results: dict[str, bool] = {}
    total = len(unique_canonicals)
    for idx, canonical in enumerate(unique_canonicals, 1):
        progress_label = canonical_to_first_seen[canonical]
        try:
            await fetch_audiodb_for_artist(canonical, collection_name)
            slug = _slugify_artist(canonical)
            row = MetadataDB.get_artist_audiodb(slug, collection_name) or {}
            found = bool(
                row.get("audiodb_bio")
                or row.get("mood")
                or row.get("country_code"),
            )
            results[canonical] = found
            if progress_callback:
                progress_callback(idx, total, progress_label, found)
        except Exception as e:
            logger.warning("[AudioDB] unhandled error for %s: %s", canonical, e)
            results[canonical] = False
            if progress_callback:
                progress_callback(idx, total, progress_label, False)
        await asyncio.sleep(delay)
    return results
