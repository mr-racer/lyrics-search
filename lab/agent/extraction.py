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

logger = logging.getLogger(__name__)

# An Apple playlist curated by anyone but Apple is a user playlist.
_APPLE_CURATORS = ("apple music", "apple ")


def apple_tracks(page: Page) -> list[TrackRef]:
    """Tracks from an Apple Music playlist page, or [] if it is not a verified one.

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

    author = (playlist.author or "").lower()
    if not any(author.startswith(c) for c in _APPLE_CURATORS):
        logger.info("[extract] %s: skipping playlist by %r — not editorial",
                    page.url, playlist.author)
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
    """Whatever the page yields without asking a model. May be empty."""
    if page.source == "apple":
        return apple_tracks(page)
    if page.source in ("wikipedia", "fandom"):
        return tracks_from_markdown(page.markdown, source=page.source,
                                    source_url=page.url,
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
