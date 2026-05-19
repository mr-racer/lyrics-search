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
