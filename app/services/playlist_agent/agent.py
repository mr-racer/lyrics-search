"""Playlist-builder pydantic-ai agent.

Handles two prompt shapes with one instruction set + three tools:
  * "artist hits" — collect an artist's best-known songs;
  * "film/game soundtrack" — collect a title's soundtrack.

Tools:
  * ``fetch_filters(artist_query)`` — map how the user named an artist to the
    real names in the library (transliteration-aware difflib over the
    collection's artist values), so "Канье" resolves to "Kanye West" before
    searching.
  * ``web_search(query)`` — SearXNG/DDG web search (capped at
    ``_MAX_WEB_SEARCHES`` per run via a closure counter).
  * ``get_songs(items)`` — resolve proposed titles against the library
    (exact → fuzzy), returning per-item match info so the model can react.

The agent output is a :class:`PlaylistDraft`; ``get_songs`` also records every
resolved track into the shared ``state`` dict so the caller (the recsys
``web_hits`` delegation) can build the track list (and drop any track_id the
model might hallucinate) without trusting the LLM to echo IDs perfectly.

Progress: every tool emits a structured event through ``state["on_status"]``
(``{"stage": ..., "query"/"count"/"found"/...}``) so the streaming route can
show the user a human-readable chain of what the agent is doing — which tool,
with which query, and what it found.

Uses ``instructions=`` (not ``system_prompt=``) per the repo convention: a
``system_prompt`` is dropped once ``message_history`` is passed.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from app.domain.models import PlaylistDraft
from app.services.agents import _create_pydantic_model
from app.services.llm_web_search import smart_web_search
from app.services.name_match import score_names
from app.services.playlist_agent.resolver import resolve_songs

logger = logging.getLogger(__name__)

_MAX_WEB_SEARCHES = 4
_MAX_LLM_REQUESTS = 15

_INSTRUCTIONS = """You build a music playlist from the user's request by finding real songs and matching them against THIS user's music library. Two request shapes:

1) ARTIST HITS ("собери хиты X", "best of X", "Kanye hits"):
   - FIRST call fetch_filters with the artist name as the user wrote it, to learn how the artist is actually spelled in the library (e.g. "Канье" -> "Kanye West"; it transliterates across alphabets). Use the returned real name from then on.
   - If fetch_filters returned nothing, the artist may still be in the library under the canonical name — do NOT give up and do NOT stop: continue with the canonical name you know (or confirm it via web_search).
   - Then ALWAYS call web_search 1-2 times ("<artist> greatest hits", "<artist> most popular songs") to get a list of the artist's best-known song titles.
   - Then call get_songs with those {title, artist} pairs.

2) FILM / GAME SOUNDTRACK ("саундтрек к <фильм/игра>"):
   - FIRST call web_search once to confirm the EXACT official title of the film/game in the right language (titles are often localized or abbreviated).
   - Then call web_search 1-2 times for its soundtrack / featured songs to get real song titles and their artists.
   - Then call get_songs with those {title, artist} pairs.

Rules:
- Song titles MUST come from web_search results — never from your own memory, and never by reinterpreting the user's request text as a song title or a vibe/mood to search for. get_songs refuses to run until web_search has been called.
- Only put a track into the playlist if get_songs returned it with match "exact" or "fuzzy" (never "none"). Use the track_id from get_songs verbatim.
- If fewer than 3 tracks were found in the library, say so honestly in the comment; still return whatever was found.
- List songs you wanted but that were NOT in the library (match "none") in the "missing" field, as "Title — Artist".
- web_search is limited; don't waste calls. Do not call get_songs before you have real song titles.
- Write the playlist "title" and "comment" in the user's language (given as [lang=..] at the start of the prompt).
- Output strictly the PlaylistDraft structure."""


def _score_artists(query: str, values) -> list[dict]:
    """Cross-script artist candidates via the shared ``name_match`` scoring
    ("канье" finds "Kanye West"). Up to 5 with score >= 0.5, best first."""
    scored = score_names(query, values)
    return [{"artist": v, "score": round(s, 2)} for s, v in scored[:5] if s >= 0.5]


class SongQuery(BaseModel):
    title: str
    artist: Optional[str] = None


def create_playlist_agent(model, deps, catalog, state):
    """Construct the agent with its three tools bound to ``deps``/``catalog``.

    ``state`` accumulates cross-tool data: ``web`` (search counter),
    ``resolved`` (track_id -> {title, artist, match}), ``missing`` (list of
    "Title — Artist" not in the library), and an optional ``on_status`` sync
    callback for SSE progress. The callback receives event dicts —
    ``{"stage": str, ...}`` with human-relevant details (the query a tool ran
    with, how many songs matched) so the UI can render a progress chain.
    """
    agent = Agent(model, output_type=PlaylistDraft, instructions=_INSTRUCTIONS)

    def _emit(stage: str, **info) -> None:
        cb = state.get("on_status")
        if cb is None:
            return
        try:
            cb({"stage": stage, **info})
        except Exception:  # pragma: no cover — a broken callback must not break the agent
            logger.debug("[playlist_agent] on_status failed", exc_info=True)

    @agent.tool_plain
    async def fetch_filters(artist_query: str) -> list:
        """Resolve how the user named an artist to the real artist names present
        in the library (transliteration-aware, e.g. "Канье" -> "Kanye West").
        Returns up to 5 candidates with a similarity score."""
        _emit("filters", query=artist_query)
        try:
            values = await deps.resolve_filter_values("artist", artist_query)
        except Exception as exc:
            logger.warning("[playlist_agent] fetch_filters failed: %s", exc)
            _emit("filters_done", query=artist_query, best=None)
            return []
        scored = _score_artists(artist_query, values)
        logger.info("[playlist_agent] fetch_filters query=%r scanned=%d artists -> candidates=%s",
                    artist_query, len(values), scored)
        _emit("filters_done", query=artist_query,
              best=scored[0]["artist"] if scored else None)
        return scored

    @agent.tool_plain
    async def web_search(query: str) -> str:
        """Search the web for real song titles (artist hits, soundtrack tracklists)."""
        if state["web"] >= _MAX_WEB_SEARCHES:
            return (
                f"(web search limit reached — {_MAX_WEB_SEARCHES} searches already used. "
                "Build the playlist from the titles you already have.)"
            )
        state["web"] += 1
        _emit("web_search", query=query)
        logger.info("[playlist_agent] web_search #%d query=%r", state["web"], query)
        try:
            res = await asyncio.to_thread(smart_web_search, query, False, 5) or "(no web results)"
        except Exception as exc:
            logger.warning("[playlist_agent] web_search failed: %s", exc)
            return "(web search unavailable)"
        logger.info("[playlist_agent] web_search #%d -> %d chars: %.300s",
                    state["web"], len(res), res.replace("\n", " · "))
        return res

    @agent.tool_plain
    async def get_songs(items: list[SongQuery]) -> list | str:
        """Match proposed {title, artist} songs against the user's library.

        Returns one entry per item with ``match`` ("exact"|"fuzzy"|"none") and,
        when found, the library ``track_id``/``title``/``artist``."""
        if state["web"] == 0:
            # Web-first is the whole point: without it the model pattern-matches
            # the request text against the library ("semantic" garbage).
            logger.info("[playlist_agent] get_songs refused (no web_search yet), items=%s",
                        [f"{i.title} — {i.artist}" for i in items])
            return (
                "(refused: call web_search first to get REAL song titles from "
                "the internet, then call get_songs with those titles.)"
            )
        _emit("matching", count=len(items))
        logger.info("[playlist_agent] get_songs items=%s",
                    [f"{i.title} — {i.artist}" for i in items])
        payload = [i.model_dump() for i in items]
        try:
            resolved = await asyncio.to_thread(resolve_songs, payload, catalog)
        except Exception as exc:
            logger.warning("[playlist_agent] get_songs failed: %s", exc)
            return [{"query_title": i.get("title", ""), "match": "none"} for i in payload]
        logger.info("[playlist_agent] get_songs results=%s",
                    [(r["query_title"], r["match"],
                      f'{r.get("title")} — {r.get("artist")}' if r["match"] != "none" else None)
                     for r in resolved])
        for r in resolved:
            if r["match"] != "none" and r.get("track_id"):
                state["resolved"][r["track_id"]] = {
                    "title": r.get("title"), "artist": r.get("artist"), "match": r["match"],
                }
            elif r["match"] == "none" and r.get("query_title"):
                state["missing"].append(r["query_title"])
        found = sum(1 for r in resolved if r["match"] != "none")
        _emit("matching_done", found=found, total=len(resolved))
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

    logger.info("[playlist_agent] start prompt=%r lang=%s model=%s base_url=%s",
                prompt, lang, llm_model or "(default)", llm_base_url or "(default)")
    model = _create_pydantic_model(llm_base_url, llm_model)
    agent = create_playlist_agent(model, deps, catalog, state)
    result = await agent.run(
        f"[lang={lang}] {prompt}",
        usage_limits=UsageLimits(request_limit=_MAX_LLM_REQUESTS),
    )
    draft = result.output
    logger.info(
        "[playlist_agent] done title=%r draft_track_ids=%s (resolved_in_library=%d, "
        "missing=%d, web_searches=%d)",
        draft.title, draft.track_ids, len(state["resolved"]),
        len(state["missing"]), state["web"],
    )
    return draft
