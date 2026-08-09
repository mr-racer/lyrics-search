"""Getting track references off a page: structure first, model last.

Three paths, chosen by what the page actually is:

1. **Apple Music** — ``parse_apple_playlist`` reads the page's own embedded
   JSON. Positions, artists, albums, durations, ids. No model, no ambiguity.
   Only editorial playlists are accepted: an Apple-curated list is a
   professionally assembled answer to a request like "2000s club hits", a
   stranger's public playlist is noise with the same URL shape.
2. **Wikipedia / Fandom** — markdown tables, read by column header
   (``tables.py``). A soundtrack page is two hundred rows and a model asked to
   transcribe it silently keeps twenty.
3. **Everything else** — prose and listicles, where there is no structure to
   read and judgement is genuinely needed. That is the model's job, and its
   output is checked against the library before it counts for anything.

The rule that makes path 3 safe: a title the model produced is a CLAIM. It
becomes a track only after ``LibraryCatalog`` matches it. A hallucinated title
matches nothing and disappears.
"""

from __future__ import annotations

import logging
from typing import Optional

from lab.agent.llm import LLMClient, as_int, as_str
from lab.agent.models import Page, TrackRef
from lab.agent.prompts import EXTRACT_TRACKS_SYSTEM
from lab.agent.tables import tracks_from_markdown
from lab.agent.urls import source_for_url

logger = logging.getLogger(__name__)

# Editorial playlists are credited to "Apple Music" plus a genre desk —
# "Apple Music Alternative", "Apple Music Hip-Hop", or plain "Apple Music".
# Anything else is somebody's personal playlist wearing the same URL shape.
_EDITORIAL_AUTHOR = "apple music"
# User-created playlist ids carry a `u-` marker. A hard reject on top of the
# author check: it needs no parsing to have succeeded, so it still holds when
# the page shape changes and the author comes back empty.
_USER_PLAYLIST_PREFIX = "pl.u-"


def is_editorial_playlist(playlist) -> bool:
    """Whether an Apple playlist is curated by Apple rather than by a listener.

    Both signals must agree, and they fail in opposite directions: the id
    catches a user playlist whose author string is missing, the author catches
    a curator that is neither Apple nor a listener (a label, a brand).
    """
    playlist_id = (getattr(playlist, "playlist_id", "") or "").lower()
    if playlist_id.startswith(_USER_PLAYLIST_PREFIX):
        return False
    author = (getattr(playlist, "author", "") or "").strip().lower()
    return author.startswith(_EDITORIAL_AUTHOR)


def apple_tracks(page: Page) -> list[TrackRef]:
    """Tracks from an Apple Music playlist page, or [] if it is not editorial.

    Re-fetches the raw HTML: the pipeline's markdown extraction throws away the
    embedded JSON this parser needs, and Apple pages render almost nothing as
    text anyway.
    """
    from lab import websearch_lab as L
    from lab.apple_music_playlist import parse_apple_playlist

    try:
        html = L.http_get_text(page.url)
        playlist = parse_apple_playlist(html, url=page.url)
    except Exception:
        logger.info("[extract] apple parse failed for %s", page.url, exc_info=True)
        return []

    if not is_editorial_playlist(playlist):
        # Logged with the author, because "no tracks from Apple" and "the
        # parser stopped finding the author" look identical from outside.
        logger.info("[extract] %s: skipped — author=%r id=%r is not editorial",
                    page.url, playlist.author, playlist.playlist_id)
        return []

    out: list[TrackRef] = []
    for track in playlist.tracks:
        if not track.title:
            continue
        out.append(TrackRef(title=track.title,
                            artist=(track.artists[0] if track.artists else None),
                            year=None, source="apple", source_url=page.url))
    logger.info("[extract] apple %r by %r: %d tracks",
                playlist.title, playlist.author, len(out))
    return out


def structured_tracks(page: Page, *,
                      default_artist: Optional[str] = None) -> list[TrackRef]:
    """Whatever the page yields without asking a model. May be empty.

    Dispatched on the HOST, not on ``page.source``. The source label records
    which search stream found the page, and a Wikipedia article found by the
    open-web search used to be labelled "web" — so the table parser skipped a
    900-row discography without a word.
    """
    kind = source_for_url(page.url, fallback=page.source)
    if kind == "apple":
        return apple_tracks(page)
    if kind in ("wikipedia", "fandom"):
        return tracks_from_markdown(page.markdown, source=kind,
                                    source_url=page.url,
                                    page_title=page.title,
                                    default_artist=default_artist)
    return []


class TrackExtractor:
    """The model's leg: titles out of prose, one call for a batch of passages."""

    def __init__(self, llm: LLMClient, config=None, sink=None):
        from lab.agent.config import AgentConfig

        self.llm = llm
        self.cfg = config or AgentConfig()
        self.sink = sink

    async def from_passages(self, passages: list[tuple[str, str]], *,
                            request: str) -> list[TrackRef]:
        """``passages`` is ``[(url, text), ...]``. Returns claims, not tracks.

        One call for all of them rather than one per passage: the model needs
        to see the request beside the material, and repeating that framing five
        times costs five times as much for no more accuracy.
        """
        passages = [(u, t) for u, t in passages if (t or "").strip()]
        if not passages:
            return []

        blocks = "\n\n".join(f"[{i + 1}] ({url})\n{text}"
                             for i, (url, text) in enumerate(passages))
        raw = await self.llm.ask_json([
            {"role": "system", "content": EXTRACT_TRACKS_SYSTEM},
            {"role": "user", "content": f"Request: {request}\n\nPassages:\n{blocks}"},
        ], required=("tracks",))
        if raw is None:
            return []

        items = raw.get("tracks")
        if not isinstance(items, list):
            return []

        primary_url = passages[0][0]
        out: list[TrackRef] = []
        seen: set[tuple[str, str]] = set()
        for item in items[:120]:
            if not isinstance(item, dict):
                continue
            title = as_str(item.get("title"), 200)
            if not title:
                continue
            artist = as_str(item.get("artist"), 160) or None
            key = (title.lower(), (artist or "").lower())
            if key in seen:
                continue
            seen.add(key)
            year = as_int(item.get("year"))
            if year is not None and not (1900 <= year <= 2100):
                year = None
            out.append(TrackRef(title=title, artist=artist, year=year,
                                source="web", source_url=primary_url))

        if self.sink is not None:
            self.sink.put("extract", passages=len(passages), claims=len(out))
        logger.info("[extract] model produced %d claims from %d passages",
                    len(out), len(passages))
        return out
