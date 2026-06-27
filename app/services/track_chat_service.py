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
import os
from typing import Optional

from app.domain.models import TrackChatContext

logger = logging.getLogger(__name__)

_LANG_NAMES = {"ru": "Russian", "en": "English"}

# ─── Context budget (song-facts trimming) ─────────────────────────────────────
# The whole player-chat prompt is capped at ~_max_prompt_tokens(). Song facts are
# the single elastic component: lyrics + chat history go in full, facts are packed
# shortest-first into whatever budget remains (see _pack_facts / answer_track_chat).
# Tunable via env so a deployment can raise/lower it without code changes.
_DEFAULT_MAX_PROMPT_TOKENS = 32_000
CHARS_PER_TOKEN = 4  # crude estimate — no tokenizer dependency


def _max_prompt_tokens() -> int:
    """Prompt token budget, read from env at call time (deploy/test friendly)."""
    try:
        return int(os.getenv("TRACK_CHAT_MAX_PROMPT_TOKENS", str(_DEFAULT_MAX_PROMPT_TOKENS)))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_PROMPT_TOKENS


def _est_tokens(text: str) -> int:
    """Rough token estimate (chars / CHARS_PER_TOKEN). Good enough for trimming."""
    return (len(text or "") + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _reply_lang_directive(lang: Optional[str]) -> str:
    """Explicit reply-language instruction from the UI language.

    Falls back to the message-language heuristic when ``lang`` is unset/unknown
    (e.g. free-form song chat where matching the user's own message is desirable).
    """
    name = _LANG_NAMES.get((lang or "").strip().lower())
    if name:
        return f"Reply ONLY in {name}, regardless of the language of the user's message."
    return "Reply in the language of the user's message (Russian or English)."


# ─── Prompts ──────────────────────────────────────────────────────────────────

TRACK_CHAT_PROMPT = """
You are talking with someone who's listening to a specific track and wants to understand it better. Think of yourself as a sharp, well-read friend at a listening session — not an encyclopedia, not an essay writer.

THREE SOURCES OF TRUTH — keep them strictly separate:
1. The lyrics below — your basis for INTERPRETATION (what the song seems to be about, how an image works, the tone). Interpretation is yours to offer freely, grounded in specific lines.
2. The curated facts below — your basis for REAL-WORLD claims already provided (who made it, when, samples, chart history, etc.).
3. `web_search` — for any real-world claim NOT already in the facts.

GROUNDING (most important rule):
- Interpreting the lyrics ≠ stating facts about the real world. "The narrator sounds like he's at rock bottom" is interpretation. "The artist wrote this after rehab" is a factual claim and needs a source.
- NEVER reconstruct a creation story, recording history, artist biography, or real-life event from the lyrics. If a question needs a real-world fact you don't have, and search turns up nothing, just say you don't have reliable info on that — that is a complete, acceptable answer. Do NOT fill the gap with plausible-sounding invention.
- Never invent dates, numbers, names, or sources.

WHEN TO SEARCH:
- DEFAULT to `web_search` for factual questions beyond plain lyric interpretation: production and recording, samples and interpolations, chart history, controversy, collaborators, or the real meaning of a place / person / event named in the song — whenever the answer isn't already in the facts below.
- Do NOT search for purely interpretive questions answerable from the lyrics in front of you.
- If `web_search` returns nothing useful, say so plainly and answer only what you're confident about from the lyrics and the facts.

HOW TO ANSWER:
- Lead with the actual answer. No warm-up thesis ("This song is a deeply personal, confessional work exploring…"). Just say what it's about.
- Match length to the question. "What's this about?" deserves a few tight sentences, not three paragraphs. Most answers are short.
- Say each point once. No recap / "In summary" paragraph that repeats what you just said.
- Don't default to a numbered list of themes. Use a list only if the song genuinely has several distinct threads worth separating — and keep it lean even then.
- When the user asks about a specific line, answer about THAT line: quote it, explain it, stop. Don't re-analyze the whole song.
- Be concrete. Tie claims to specific words in the lyrics, not generic talk about identity, struggle, and inner demons.
- Sound like a person talking, not a report being generated.

{lang_directive}

TRACK CONTEXT:
{track_context_block}
""".strip()


LYRIC_EXPLAIN_PROMPT = """
You are explaining one lyric line to a listener who tapped on it. Your job is NOT to always find deep meaning — it's to explain what's actually there, and nothing more.

FIRST, decide which case this line is:

CASE 1 — the line has a concrete anchor: a reference to a real place / person / event / work, a piece of wordplay, a double meaning, slang, or an image that depends on outside context to land.
→ Explain that anchor. Lead with the non-obvious part — the thing the listener wouldn't get just from reading the words. If the anchor is a real-world fact not in the facts below, use `web_search` before explaining rather than guessing.

CASE 2 — the line is straightforward: it says what it says, with no hidden reference or wordplay.
→ Just give a short, natural rendering of its meaning (a translation / paraphrase). One or two sentences. Do NOT manufacture a deeper layer, symbolism, or significance that isn't there. "This line just literally means X" is a correct and complete answer.

HARD RULES:
- When unsure which case you're in, treat it as Case 2. A plain translation is always safer than an invented interpretation.
- Never invent a reference, backstory, symbolism, date, or source. If `web_search` returns nothing useful, say so and explain only what you're confident about.
- Don't name the literary device. Don't say "this is a hyperbole / metaphor / irony." Explain the effect in plain words instead, or skip it.
- Focus on the selected line. Bring in surrounding lines only if the selected line is meaningless without them.
- Length follows the line: a rich line gets a few sentences, a plain line gets one. Never pad.

EXAMPLES (style only — do not reuse this content):

Line: "cover your mouth up like you got SARS"
→ The point isn't just that her breath is bad. SARS made everyone picture covering your mouth to stop spreading something contagious — he's putting bad breath in that same frame, like it's something to be quarantined. The image is what makes the insult land.

Line: "I woke up this morning, poured myself a drink"
→ This one's literal: he wakes up and pours a drink to start the day. Nothing hidden underneath — it's just setting the scene.

{lang_directive}

TRACK CONTEXT:
{track_context_block}

SELECTED LINE:
{selected_line}
""".strip()


# ─── Helpers ──────────────────────────────────────────────────────────────────


def resolve_song_facts_list(title: str, artist: str) -> list[str]:
    """Look up raw song_facts.fact rows for (artist, title). Returns a list of
    individual facts (one per row), or [].

    Reads from the `song_facts` table directly — NOT the refined-facts variant.

    Real schema (verified from metadata_db.py):
    - song_facts.song_slug FK → songs.slug
    - song_facts.fact TEXT  (individual fact per row, not a blob)
    - song_slug = get_song_facts_key(artist, song) = "{artist_slug}-{title_slug}"
    We query by song_slug directly (no collection_name filter) so we get facts
    regardless of which collection the track belongs to.
    """
    if not title or not artist:
        return []
    try:
        from app.resources.metadata_db import MetadataDB
        from app.services.song_facts_service import get_song_facts_key

        song_slug = get_song_facts_key(artist, title)
        conn = MetadataDB._connect()
        rows = conn.execute(
            "SELECT fact FROM song_facts WHERE song_slug = ? AND lang = 'en' ORDER BY id",
            (song_slug,),
        ).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception as exc:
        logger.warning("[track_chat] resolve_song_facts failed: %s", exc)
    return []


def resolve_song_facts(title: str, artist: str) -> str:
    """Raw song facts joined into one block (back-compat string form)."""
    return "\n\n".join(resolve_song_facts_list(title, artist))


def _select_facts(facts: list[str], token_budget: int) -> list[str]:
    """Pick facts shortest-first until ``token_budget`` is exhausted; the fattest
    facts are dropped. Returns the kept facts in their ORIGINAL order so the
    block still reads naturally.

    Rationale: a single bloated fact can eat the whole window — preferring the
    leanest facts keeps the *most* facts inside the budget.
    """
    if token_budget <= 0:
        return []
    indexed = [(i, f) for i, f in enumerate(facts) if f and f.strip()]
    kept_idx: list[int] = []
    used = 0
    SEP_TOKENS = 2  # overhead of the "\n\n" separator between facts
    for i, f in sorted(indexed, key=lambda p: len(p[1])):  # shortest first
        cost = _est_tokens(f) + SEP_TOKENS
        if used + cost > token_budget:
            break  # sorted ascending → everything after is at least as long
        kept_idx.append(i)
        used += cost
    return [facts[i] for i in sorted(kept_idx)]


def _pack_facts(facts: list[str], token_budget: int) -> str:
    """String form of :func:`_select_facts` — facts joined into one block."""
    return "\n\n".join(_select_facts(facts, token_budget))


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


def _to_message_history(history) -> list:
    """Convert ChatMessage[] (role/content) into pydantic-ai message_history.

    user → ModelRequest(UserPromptPart); assistant → ModelResponse(TextPart).
    Empty / content-less turns are skipped. System parts are intentionally
    omitted — the agent's own system_prompt (facts + lyrics) applies to the run.
    """
    from pydantic_ai.messages import (
        ModelRequest, ModelResponse, UserPromptPart, TextPart,
    )
    out: list = []
    for m in history or []:
        content = (getattr(m, "content", None) or "").strip()
        if not content:
            continue
        if getattr(m, "role", None) == "assistant":
            out.append(ModelResponse(parts=[TextPart(content=content)]))
        else:
            out.append(ModelRequest(parts=[UserPromptPart(content=content)]))
    return out


async def _run_agent(agent, message: str, system_prompt: str, history: list):
    """Run a pydantic-ai agent. Extracted as a function for ease of mocking in tests."""
    return await agent.run(message, message_history=history or None)


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
        """Search the web for facts about the track that aren't in the provided context.

        Call this BY DEFAULT for any factual question that isn't pure lyric
        interpretation — production trivia, samples, chart positions, controversy,
        history, collaborators, or the real-world meaning of a place / person /
        event named in the song. When in doubt about a fact, search instead of
        guessing. Examples:
        - "What songs does this sample?"
        - "What was the controversy around this song?"
        - "Chart position when released?"

        Args:
            query: A focused web search query (3-8 words).

        Returns:
            Formatted search results (titles, snippets, URLs), or a 'no results' marker.
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

    ctx = req.track_context
    directive = _reply_lang_directive(getattr(req, "lang", None))

    def _compose(facts_str: str) -> str:
        """Fill the mode's prompt template with the given (already-packed) facts."""
        block = build_track_context_block(ctx, facts_str)
        if req.mode == "song":
            return TRACK_CHAT_PROMPT.format(
                track_context_block=block, lang_directive=directive,
            )
        return LYRIC_EXPLAIN_PROMPT.format(
            track_context_block=block,
            selected_line=req.selected_line,
            lang_directive=directive,
        )

    # History goes in full (user's choice); facts are the single elastic
    # component — packed shortest-first into whatever budget is left over.
    facts_list = resolve_song_facts_list(ctx.title, ctx.artist)
    history = _to_message_history(req.history)

    history_tokens = sum(_est_tokens(getattr(m, "content", "")) for m in (req.history or []))
    fixed_tokens = _est_tokens(_compose("")) + _est_tokens(req.message) + history_tokens
    facts_budget = max(0, _max_prompt_tokens() - fixed_tokens)
    kept_facts = _select_facts(facts_list, facts_budget)
    packed_facts = "\n\n".join(kept_facts)

    if facts_list:
        logger.info(
            "[track_chat] facts: kept %d/%d (~%d tok, budget %d; prompt ~%d, hist ~%d)",
            len(kept_facts), len(facts_list), _est_tokens(packed_facts), facts_budget,
            fixed_tokens - history_tokens, history_tokens,
        )

    system_prompt = _compose(packed_facts)

    # Build agent (per-request — simpler than thread-safety analysis)
    agent, state = create_track_chat_agent(req.llm_base_url, req.llm_model, system_prompt)

    result = await _run_agent(agent, req.message, system_prompt, history)
    message = getattr(result, "output", "") or ""
    web_search_used = state["web_search_calls"] > 0
    return TrackChatResponse(message=message, web_search_used=web_search_used)
