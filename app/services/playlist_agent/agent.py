"""Playlist-builder pydantic-ai agent.

Handles two prompt shapes with one instruction set + three tools:
  * "artist hits" — collect an artist's best-known songs;
  * "film/game soundtrack" — collect a title's soundtrack.

Tools:
  * ``fetch_filters(artist_query)`` — map how the user named an artist to the
    real names in the library (difflib over the collection's artist values), so
    "Канье" resolves to "Kanye West" before searching.
  * ``web_search(query)`` — SearXNG/DDG web search (capped at
    ``_MAX_WEB_SEARCHES`` per run via a closure counter).
  * ``get_songs(items)`` — resolve proposed titles against the library
    (exact → fuzzy), returning per-item match info so the model can react.

The agent output is a :class:`PlaylistDraft`; ``get_songs`` also records every
resolved track into the shared ``state`` dict so the route can build a preview
(and drop any track_id the model might hallucinate) without trusting the LLM to
echo IDs perfectly.

Uses ``instructions=`` (not ``system_prompt=``) per the repo convention: a
``system_prompt`` is dropped once ``message_history`` is passed.
"""
from __future__ import annotations

import asyncio
import logging
from difflib import SequenceMatcher
from typing import Optional

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from app.domain.models import PlaylistDraft
from app.services.agents import _create_pydantic_model
from app.services.llm_web_search import smart_web_search
from app.services.playlist_agent.resolver import resolve_songs

logger = logging.getLogger(__name__)

_MAX_WEB_SEARCHES = 4
_MAX_LLM_REQUESTS = 15

_INSTRUCTIONS = """You build a music playlist from the user's request by finding real songs and matching them against THIS user's music library. Two request shapes:

1) ARTIST HITS ("собери хиты X", "best of X"):
   - FIRST call fetch_filters with the artist name as the user wrote it, to learn how the artist is actually spelled in the library (e.g. "Канье" -> "Kanye West"). Use the returned real name from then on.
   - Then call web_search 1-2 times ("<artist> greatest hits", "<artist> most popular songs") to get a list of the artist's best-known song titles.
   - Then call get_songs with those {title, artist} pairs.

2) FILM / GAME SOUNDTRACK ("саундтрек к <фильм/игра>"):
   - FIRST call web_search once to confirm the EXACT official title of the film/game in the right language (titles are often localized or abbreviated).
   - Then call web_search 1-2 times for its soundtrack / featured songs to get real song titles and their artists.
   - Then call get_songs with those {title, artist} pairs.

Rules:
- Only put a track into the playlist if get_songs returned it with match "exact" or "fuzzy" (never "none"). Use the track_id from get_songs verbatim.
- If fewer than 3 tracks were found in the library, say so honestly in the comment; still return whatever was found.
- List songs you wanted but that were NOT in the library (match "none") in the "missing" field, as "Title — Artist".
- web_search is limited; don't waste calls. Do not call get_songs before you have real song titles.
- Write the playlist "title" and "comment" in the user's language (given as [lang=..] at the start of the prompt).
- Output strictly the PlaylistDraft structure."""


class SongQuery(BaseModel):
    title: str
    artist: Optional[str] = None


def create_playlist_agent(model, deps, catalog, state):
    """Construct the agent with its three tools bound to ``deps``/``catalog``.

    ``state`` accumulates cross-tool data: ``web`` (search counter),
    ``resolved`` (track_id -> {title, artist, match}), ``missing`` (list of
    "Title — Artist" not in the library), and an optional ``on_status`` sync
    callback for SSE progress.
    """
    agent = Agent(model, output_type=PlaylistDraft, instructions=_INSTRUCTIONS)

    def _emit(stage: str) -> None:
        cb = state.get("on_status")
        if cb is None:
            return
        try:
            cb(stage)
        except Exception:  # pragma: no cover — a broken callback must not break the agent
            logger.debug("[playlist_agent] on_status failed", exc_info=True)

    @agent.tool_plain
    async def fetch_filters(artist_query: str) -> list:
        """Resolve how the user named an artist to the real artist names present
        in the library. Returns up to 5 candidates with a similarity score."""
        _emit("thinking")
        try:
            values = await deps.resolve_filter_values("artist", artist_query)
        except Exception as exc:
            logger.warning("[playlist_agent] fetch_filters failed: %s", exc)
            return []
        scored = sorted(
            ((SequenceMatcher(None, artist_query.lower(), v.lower()).ratio(), v) for v in values),
            reverse=True,
        )[:5]
        return [{"artist": v, "score": round(s, 2)} for s, v in scored]

    @agent.tool_plain
    async def web_search(query: str) -> str:
        """Search the web for real song titles (artist hits, soundtrack tracklists)."""
        if state["web"] >= _MAX_WEB_SEARCHES:
            return (
                f"(web search limit reached — {_MAX_WEB_SEARCHES} searches already used. "
                "Build the playlist from the titles you already have.)"
            )
        state["web"] += 1
        _emit("web_search")
        try:
            return await asyncio.to_thread(smart_web_search, query, False, 5) or "(no web results)"
        except Exception as exc:
            logger.warning("[playlist_agent] web_search failed: %s", exc)
            return "(web search unavailable)"

    @agent.tool_plain
    async def get_songs(items: list[SongQuery]) -> list:
        """Match proposed {title, artist} songs against the user's library.

        Returns one entry per item with ``match`` ("exact"|"fuzzy"|"none") and,
        when found, the library ``track_id``/``title``/``artist``."""
        _emit("matching")
        payload = [i.model_dump() for i in items]
        try:
            resolved = await asyncio.to_thread(resolve_songs, payload, catalog)
        except Exception as exc:
            logger.warning("[playlist_agent] get_songs failed: %s", exc)
            return [{"query_title": i.get("title", ""), "match": "none"} for i in payload]
        for r in resolved:
            if r["match"] != "none" and r.get("track_id"):
                state["resolved"][r["track_id"]] = {
                    "title": r.get("title"), "artist": r.get("artist"), "match": r["match"],
                }
            elif r["match"] == "none" and r.get("query_title"):
                state["missing"].append(r["query_title"])
        return resolved

    return agent


async def run_playlist_agent(prompt, lang, deps, catalog, state,
                             llm_base_url=None, llm_model=None, on_status=None):
    """Run the playlist agent to completion and return its :class:`PlaylistDraft`.

    ``state`` (caller-provided dict) is populated with ``resolved``/``missing``
    so the route can build a track preview and filter hallucinated ids.
    """
    state.setdefault("web", 0)
    state.setdefault("resolved", {})
    state.setdefault("missing", [])
    state["on_status"] = on_status

    model = _create_pydantic_model(llm_base_url, llm_model)
    agent = create_playlist_agent(model, deps, catalog, state)
    result = await agent.run(
        f"[lang={lang}] {prompt}",
        usage_limits=UsageLimits(request_limit=_MAX_LLM_REQUESTS),
    )
    return result.output
