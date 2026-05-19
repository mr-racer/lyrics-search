"""Track Chat Service — pydantic-ai agent for chatting about a single track,
with optional web_search tool fallback.

Used by POST /chat/track-chat. Two modes:
- 'song' — multi-turn drawer chat
- 'lyric_explain' — single-shot explanation of a specific lyric line

Backend resolves raw song_facts server-side (NOT refined) so the agent always
sees the original facts regardless of AI-Indexing state.

Schema notes (metadata_db.py verified):
- song_facts table: (id PK, song_slug FK→songs.slug, lang, fact TEXT, ...)
  Column is `fact`, NOT `notes`. One row per fact (not a single blob).
- song_slug = get_song_facts_key(artist, song)  ← from song_facts_service
  which returns  "{artist_slug}-{title_slug}"  (same as songs.slug).
- MetadataDB.get_song_facts(slug, collection_name) returns List[str], needs
  collection_name — we use a wildcard JOIN approach via direct SQL on song_slug
  only, since at chat time we may not have the collection_name handy.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.domain.models import TrackChatContext

logger = logging.getLogger(__name__)

# ─── Prompts ──────────────────────────────────────────────────────────────────

TRACK_CHAT_PROMPT = """
You are a music expert helping a listener understand a track they're playing.

You have full lyrics, metadata, and curated raw facts about the song below.
Use the `web_search` tool ONLY when the user's question cannot be answered from this context — e.g., samples in the track, production trivia, controversy, chart history.

If `web_search` returns nothing useful, say so honestly. Don't invent facts.

When discussing the lyrics, quote the exact phrase the user is asking about. Be specific, not generic.

Reply in the language of the user's message (Russian or English).

TRACK CONTEXT:
{track_context_block}
""".strip()


LYRIC_EXPLAIN_PROMPT = """
You are explaining a single lyric line from a song the listener is hearing.

Focus on that line. Refer to surrounding lines only when essential.
Use the `web_search` tool only when the line references something concrete (a place, person, event) that isn't in the provided facts.

Reply in the language of the user's message (Russian or English), in 2-4 sentences.

TRACK CONTEXT:
{track_context_block}

SELECTED LINE:
{selected_line}
""".strip()


# ─── Helpers ──────────────────────────────────────────────────────────────────


def resolve_song_facts(title: str, artist: str) -> str:
    """Look up raw song_facts.fact rows for (artist, title). Returns joined text or "".

    Reads from the `song_facts` table directly — NOT the refined-facts variant.

    Real schema (verified from metadata_db.py):
    - song_facts.song_slug FK → songs.slug
    - song_facts.fact TEXT  (individual fact per row, not a blob)
    - song_slug = get_song_facts_key(artist, song) = "{artist_slug}-{title_slug}"
    We query by song_slug directly (no collection_name filter) so we get facts
    regardless of which collection the track belongs to.
    """
    if not title or not artist:
        return ""
    try:
        from app.resources.metadata_db import MetadataDB
        from app.services.song_facts_service import get_song_facts_key

        song_slug = get_song_facts_key(artist, title)
        conn = MetadataDB._connect()
        rows = conn.execute(
            "SELECT fact FROM song_facts WHERE song_slug = ? AND lang = 'en' ORDER BY id",
            (song_slug,),
        ).fetchall()
        if rows:
            return "\n\n".join(r[0] for r in rows if r[0])
    except Exception as exc:
        logger.warning("[track_chat] resolve_song_facts failed: %s", exc)
    return ""


def build_track_context_block(
    context: TrackChatContext, song_facts: str
) -> str:
    """Render the agent-system-prompt's TRACK CONTEXT block."""
    lines = [
        f"Title: {context.title}",
        f"Artist: {context.artist}",
    ]
    if context.album:
        lines.append(f"Album: {context.album}")
    if context.year:
        lines.append(f"Year: {context.year}")
    if context.genre:
        lines.append(f"Genre: {context.genre}")
    if song_facts:
        lines.append("")
        lines.append("Facts:")
        lines.append(song_facts.strip())
    if context.full_lyrics:
        lines.append("")
        lines.append("Lyrics:")
        lines.append(context.full_lyrics.strip())
    return "\n".join(lines)


# ─── Agent factory + orchestrator ─────────────────────────────────────────────

from pydantic_ai import Agent  # noqa: E402


async def _run_agent(agent, message: str, system_prompt: str, history: list):
    """Run a pydantic-ai agent. Extracted as a function for ease of mocking in tests."""
    return await agent.run(message)


def create_track_chat_agent(
    llm_base_url: Optional[str],
    llm_model: Optional[str],
    system_prompt: str,
):
    """Build a pydantic-ai Agent with the web_search tool registered.

    Each request gets a fresh agent (cheap to construct), so the per-request
    system_prompt is baked in at construction time. The agent returns plain
    string output (free-form text reply).

    Returns:
        (agent, state) — state dict contains 'web_search_calls' counter.
    """
    from app.services.agents import _create_pydantic_model
    from app.services.llm_web_search import smart_web_search

    model = _create_pydantic_model(llm_base_url, llm_model)
    agent = Agent(model, output_type=str, system_prompt=system_prompt)

    # Track tool invocations so we can report web_search_used
    state: dict = {"web_search_calls": 0}
    agent._test_state = state  # exposed for tests; safe — pydantic-ai ignores unknown attrs

    @agent.tool_plain
    async def web_search(query: str) -> str:
        """Search the web for facts not in the provided track context.

        Use ONLY when the user's question cannot be answered from the track
        context (lyrics + facts + meta). Examples of when to call:
        - "What songs does this sample?"
        - "What was the controversy around this song?"
        - "Chart position when released?"

        Args:
            query: A focused web search query (3-8 words).

        Returns:
            Formatted search results (titles, snippets, URLs).
        """
        state["web_search_calls"] += 1
        try:
            from anyio import to_thread

            # smart_web_search is sync; run in a worker thread so we don't block the loop
            result = await to_thread.run_sync(smart_web_search, query, True, 3)
            return result or "(no web results)"
        except Exception as exc:
            logger.warning("[track_chat] web_search failed: %s", exc)
            return "(web search unavailable)"

    return agent, state


async def answer_track_chat(req):
    """Orchestrate: validate, build context block, run agent, return response."""
    from app.domain.models import TrackChatResponse

    if req.mode == "lyric_explain" and not req.selected_line:
        raise ValueError("selected_line is required for mode='lyric_explain'")

    # Resolve raw facts and build the context block
    song_facts = resolve_song_facts(req.track_context.title, req.track_context.artist)
    block = build_track_context_block(req.track_context, song_facts)

    # Choose prompt and fill placeholders
    if req.mode == "song":
        system_prompt = TRACK_CHAT_PROMPT.format(track_context_block=block)
    else:
        system_prompt = LYRIC_EXPLAIN_PROMPT.format(
            track_context_block=block,
            selected_line=req.selected_line,
        )

    # Build agent (per-request — simpler than thread-safety analysis)
    agent, state = create_track_chat_agent(req.llm_base_url, req.llm_model, system_prompt)

    # First-pass: ignore req.history (multi-turn agent history is a follow-up).
    history: list = []

    result = await _run_agent(agent, req.message, system_prompt, history)
    message = getattr(result, "output", "") or ""
    web_search_used = state["web_search_calls"] > 0
    return TrackChatResponse(message=message, web_search_used=web_search_used)
