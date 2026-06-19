"""Lyrics fetchers: lyrics.ovh + syncedlyrics fallback chain.

Extracted from legacy file_processor/utils.py during Refactor 3.
"""

from __future__ import annotations

import json
import logging
import re
import time

import requests
import syncedlyrics

from app.services.proxy_config import get_proxy

logger = logging.getLogger(__name__)

PROVIDERS = ["Musixmatch", "Lrclib", "NetEase", "Megalobiz"]
TIME_BETWEEN_REQUESTS_STANDARD = 0.15
TIME_BETWEEN_REQUESTS_OVH = 0.4
TIME_BETWEEN_REQUESTS_ENHANCED_LYRICS = 3


def fetch_lyrics_ovh(artist: str, title: str) -> str | None:
    """Fetch lyrics from lyrics.ovh API. Returns plain lyrics string or None."""
    try:
        url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
        logger.info("[Lyrics-ovh] Fetching lyrics for '%s — %s' (%s)", artist, title, url)
        resp = requests.get(url, timeout=5, proxies=get_proxy())
        logger.info("[Lyrics-ovh] '%s — %s' → HTTP %s", artist, title, resp.status_code)
        if resp.status_code == 200:
            data = json.loads(resp.text)
            lyrics = data["lyrics"]
            # Convert literal \n escape sequences to real newlines
            lyrics = lyrics.replace("\\n", "\n")
            logger.info("[Lyrics-ovh] got %d chars for '%s — %s'", len(lyrics), artist, title)
            return lyrics.strip()
        elif resp.status_code >= 400:
            logger.warning("[Lyrics-ovh] HTTP %s for '%s — %s'", resp.status_code, artist, title)
        else:
            logger.info("[Lyrics-ovh] no lyrics (HTTP %s) for '%s — %s'", resp.status_code, artist, title)
    except requests.RequestException as e:
        logger.warning("[Lyrics-ovh] request failed for '%s — %s': %s", artist, title, e)
    except Exception as e:
        logger.warning("[Lyrics-ovh] error for '%s — %s': %s", artist, title, e)
    return None


def get_lyrics(title: str, artist: str, better_lyrics_quality: bool) -> str | None:
    # Primary: try lyrics.ovh
    lyrics = fetch_lyrics_ovh(artist, title)
    if lyrics:
        return lyrics

    # Fallback: syncedlyrics
    if better_lyrics_quality:
        providers = PROVIDERS
        time_to_sleep = TIME_BETWEEN_REQUESTS_ENHANCED_LYRICS
    else:
        providers = [x for x in PROVIDERS if x != "Musixmatch"]
        time_to_sleep = TIME_BETWEEN_REQUESTS_STANDARD

    logger.info("[Lyrics-syncedlyrics] Falling back to syncedlyrics for '%s — %s' (providers: %s)",
                artist, title, ", ".join(providers))
    try:
        lyrics = syncedlyrics.search(
            f"{title} {artist}",
            providers=providers,
            plain_only=True,
        )
        time.sleep(time_to_sleep)

        if not lyrics:
            logger.info("[Lyrics-syncedlyrics] no lyrics found for '%s — %s'", artist, title)
            return None
        lyrics = re.sub(r'\[.*?\]', '', lyrics)
        logger.info("[Lyrics-syncedlyrics] got %d chars for '%s — %s'", len(lyrics), artist, title)

    except Exception as e:
        logger.warning("[Lyrics-syncedlyrics] error for '%s — %s': %s", artist, title, e)
        return None

    return lyrics
