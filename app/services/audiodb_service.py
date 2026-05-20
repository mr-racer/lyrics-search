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

import requests

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
