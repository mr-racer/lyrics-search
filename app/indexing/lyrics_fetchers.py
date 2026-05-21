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

logger = logging.getLogger(__name__)

PROVIDERS = ["Musixmatch", "Lrclib", "NetEase", "Megalobiz"]
TIME_BETWEEN_REQUESTS_STANDARD = 0.15
TIME_BETWEEN_REQUESTS_OVH = 0.4
TIME_BETWEEN_REQUESTS_ENHANCED_LYRICS = 3


def fetch_lyrics_ovh(artist: str, title: str) -> str | None:
    """Fetch lyrics from lyrics.ovh API. Returns plain lyrics string or None."""
    try:
        url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
        resp = requests.get(url, timeout=5)
        time.sleep(TIME_BETWEEN_REQUESTS_OVH)
        if resp.status_code == 200:
            data = json.loads(resp.text)
            lyrics = data["lyrics"]
            # Convert literal \n escape sequences to real newlines
            lyrics = lyrics.replace("\\n", "\n")
            return lyrics.strip()
    except Exception:
        pass
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

    try:
        lyrics = syncedlyrics.search(
            f"{title} {artist}",
            providers=providers,
            plain_only=True,
        )
        time.sleep(time_to_sleep)

        if not lyrics:
            return None
        lyrics = re.sub(r'\[.*?\]', '', lyrics)

    except Exception:
        return None

    return lyrics


def research_with_musixmatch(song: str) -> str:
    try:
        info = song.split(' — ', maxsplit=1)
        new_text = syncedlyrics.search(
            f"{info[0]} {info[1]}",
            providers=['Musixmatch'],
            plain_only=True,
        )
        time.sleep(TIME_BETWEEN_REQUESTS_ENHANCED_LYRICS)

        if new_text:
            new_text = re.sub(r'\[.*?\]', '', new_text)
            return new_text
    except Exception as e:
        logger.warning("[lyrics] musixmatch fetch failed: %s", e)
        return None
