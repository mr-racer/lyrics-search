"""Chat endpoint — two-call agentic LLM-driven music search.

Flow per user message
─────────────────────
Call 1  (classification)
    System: CLASSIFICATION_SYSTEM_PROMPT
    User:   req.message
    → dict  (passed as `classification` in the response; also available to
             extend DEVELOPER_PROMPT if you want)

Call 2…N  (agentic search loop, up to NUM_ATTEMPTS)
    System: DEVELOPER_PROMPT filled with {query, context, previous_queries,
                                          attempt, max_attempts}
    User:   req.message
    → {"action": "search"|"answer", ...}

    • "search" → run the LLM's queries against the library,
                 accumulate context, repeat.
    • "answer" → return the LLM's message + all retrieved hits.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.domain.models import ChatRequest, TrackHit
from app.services.agents import PLANNER_PROMPT
from app.services.llm_client import ask_llm
import traceback

router = APIRouter(prefix="/chat", tags=["Chat"])

# ─── Prompts ──────────────────────────────────────────────────────────────────

# Classification prompt — asks LLM to determine the search type for the user's query.
CLASSIFICATION_SYSTEM_PROMPT: str = """
You are a query classifier for a music search system. Analyze the user's query and classify it into ONE of three types:

1. **"text"** — User asks about concrete details that should literally appear in lyrics (specific words, phrases, themes, storylines).
2. **"audio"** — User describes feelings, vibe, vocals, production, atmosphere, mood — not specific words.
3. **"hybrid"** — Mix of both, or unclear which dominates.

Return ONLY a JSON object with this shape:
{{
  "type": "text" | "audio" | "hybrid",
  "reasoning": "one short sentence explaining why"
}}

No prose before or after the JSON.
""".strip()

# CLAP rephrasing prompt — transforms mood-based queries into 3 optimized
# English prompts for CLAP text-to-audio retrieval.
CLAP_REPHRASE_SYSTEM_PROMPT: str = """
# ROLE & OBJECTIVE
You are an expert audio retrieval prompt engineer specializing in the CLAP model. Transform Russian mood-based user queries into 3 optimized English prompts for text-to-audio retrieval.

# CORE RULES
1. TEMPLATE: Every prompt must start exactly with: "This song is a "
2. SEMANTIC LOCK: Preserve the exact core intent of the original query. Do NOT change genre, primary instrument, or fundamental mood. Vary ONLY acoustic/production parameters.
3. ACOUSTIC MAPPING: Replace abstract emotions with concrete proxies:
   - Tempo: slow/medium/fast, steady/driving, relaxed/upbeat
   - Timbre: bright/warm/clean/distorted/muffled/electronic/acoustic
   - Dynamics: soft/medium/loud, intimate/voluminous
   - Texture: sparse/dense, rhythmic/pad-heavy, atmospheric
4. STRUCTURE: [Genre/Style] + [Instrument] + [Tempo] + [1-2 Acoustic Details]
5. VARIATION STRATEGY: Output exactly 3 prompts that are semantically identical but differ in acoustic focus:
   - Variant 1: Tempo & Dynamics focus
   - Variant 2: Timbre & Texture focus
   - Variant 3: Key & Production style focus
6. EXCLUSIONS: Strip artist names, titles, lyrics, and subjective adjectives (epic, dreamy, cinematic, nostalgic, chill, sad). Replace strictly with acoustic equivalents.
7. CONSTRAINTS: English only. 8–15 words per prompt. Strict JSON output only.
8. QUERY COMPOSITION: Use only that sound information which was clearly provided by user. If it is unclear from user's query, it is possible to add 1-2 sound profile characteristics from artist name (if it is provided) based on your knowledge.

# OUTPUT FORMAT
Return ONLY a raw JSON array of 3 strings. No markdown, no code blocks, no explanations.
Example:
["This song is a slow acoustic guitar piece with soft dynamics", "This song is a warm timbre fingerpicking guitar track", "This song is a relaxed acoustic guitar song with sparse atmospheric texture"]

# USER QUERY
{user_query}
""".strip()

# Audio answer prompt — generates a conversational reply about the best match.
AUDIO_ANSWER_PROMPT: str = """
You are a music search assistant. The user described a song by mood or vibe,
and the system found the best audio match in their local library.

<user_query>{user_query}</user_query>

<best_match>
  Title: {title}
  Artist: {artist}
  Album: {album}
  Year: {year}
</best_match>

Respond naturally in the user's language (match the language of <user_query>).
Briefly paraphrase what the user was looking for, then name the best match.
Example: "По вашему описанию — песня с приятным женским голосом и динамичным припевом — наиболее
подходящий трек «{title}» от {artist}."

Keep it under 50 words. Return ONLY a JSON object:
{{"message": "your reply here"}}
""".strip()

# DEVELOPER_PROMPT: str = """
# You are a music search assistant. You find songs from descriptions, moods,
# vague memories, or remembered lyric fragments. You have access to a lyrics
# database via retrieval.

# You run in a loop. On each turn you EITHER answer the user from retrieved
# context, OR issue new search queries for another retrieval round.

# ═══════════════════════════════════════════════════════════════
# INPUTS
# ═══════════════════════════════════════════════════════════════

# <user_query>
# {query}
# </user_query>

# <context>
# {context}
# </context>
# Lyrics + metadata (title, artist, album, year) from previous retrieval rounds.
# Empty on the first attempt.

# <previous_queries>
# {previous_queries}
# </previous_queries>
# Queries already tried. Never repeat them. Always rephrase or shift angle.

# <attempt>
# {attempt} of {max_attempts}
# </attempt>

# ═══════════════════════════════════════════════════════════════
# STEP 1 — SCORE THE CONTEXT
# ═══════════════════════════════════════════════════════════════

# For each candidate song in <context>, check:
#   1. Do the lyrics match specific details from the user query
#      (imagery, story, fragments)?
#   2. Do metadata signals (era, artist, genre, language) match?

# Then assign ONE confidence label:
#   HIGH    — lyrics clearly match specific details the user gave.
#   MEDIUM  — plausible match, but a key detail is missing OR multiple
#             candidates tie.
#   LOW     — nothing in context fits, or context is empty.

# ═══════════════════════════════════════════════════════════════
# STEP 2 — PICK THE ACTION
# ═══════════════════════════════════════════════════════════════

# Apply these rules in order. Stop at the first match.

#   IF confidence == HIGH                        → action = "answer"
#   IF confidence == MEDIUM AND attempt <  max   → action = "search"
#   IF confidence == MEDIUM AND attempt == max   → action = "answer" (best guess, state uncertainty)
#   IF confidence == LOW    AND attempt <  max   → action = "search"
#   IF confidence == LOW    AND attempt == max   → action = "answer" (admit no match, ask ONE clarifying question)

# ═══════════════════════════════════════════════════════════════
# STEP 3 — IF action == "search", BUILD QUERIES
# ═══════════════════════════════════════════════════════════════

# 3A. Classify the user query into ONE type:

#   "text"   — User asks about a concrete detail that should literally appear
#              in lyrics.
#              Example: "Which songs mention luxury cars?"

#   "audio"  — User describes feelings, vibe, vocals, production, atmosphere,
#              not specific words. No concrete lyric fragment given.
#              Example: "Epic Lana Del Rey style song with male vocals."

#   "hybrid" — Mix of both, or unclear which one dominates.
#              Example: "A sad 80s song about driving alone at night."

# 3B. Generate queries based on type:

#   IF type == "audio":
#       Output exactly ONE query.
#       Pack it with audio/feeling vocabulary from the user's message
#       (vocal type, instruments, era, mood, production style).

#   IF type == "text" OR type == "hybrid":
#       Output 2-3 queries. Each query must:
#         - Be 3-10 words.
#         - Use words likely to appear in real lyrics
#           (concrete nouns, imagery, emotional phrases).
#         - Cover a DIFFERENT angle from the others
#           (e.g. one imagery, one emotion, one storyline).
#         - Differ from every entry in <previous_queries>.
#         - Never be empty. If unsure, paraphrase the user's query.

# 3C. Pick rephrasing mode based on attempt number:

#   Let half = floor(max_attempts / 2).

#   IF attempt <= half  → CONSERVATIVE mode:
#       Stay close to the user's wording.
#       Swap synonyms, reorder phrases. Goal: precision.

#   IF attempt >  half  → AGGRESSIVE mode:
#       Earlier searches failed. Close paraphrases will fail too.
#       Drop literal wording. Each of the 2-3 queries explores a
#       DIFFERENT direction from this list:
#         - Replace abstract feelings with concrete lyrical imagery.
#           "feeling lost" → "wandering empty streets", "no map no plan"
#         - Flip the perspective. A breakup song may be written from
#           either side. A "sad" song may have ironic upbeat lyrics.
#         - Guess a likely hook or chorus phrase, not a description.
#         - Try adjacent themes:
#           loneliness ↔ nostalgia, anger ↔ heartbreak, love ↔ obsession.
#         - Shift era/genre vocabulary if the user gave hints.
#       Do NOT output three variants of the same idea.

#   Note: when attempt == max_attempts, action is always "answer", so this
#   step does not run on the final attempt.

# ═══════════════════════════════════════════════════════════════
# OUTPUT — RETURN ONE VALID JSON OBJECT. NO TEXT BEFORE OR AFTER.
# ═══════════════════════════════════════════════════════════════

# When action == "search":
# {{
#   "action": "search",
#   "confidence": "low" | "medium",
#   "reasoning": "one short sentence on why context is insufficient",
#   "queries": [
#     {{"query": "query text 1", "type": "text" | "audio" | "hybrid"}},
#     {{"query": "query text 2", "type": "text" | "audio" | "hybrid"}},
#     {{"query": "query text 3", "type": "text" | "audio" | "hybrid"}}
#   ]
# }}

# When action == "answer":
# {{
#   "action": "answer",
#   "confidence": "high" | "medium" | "low",
#   "song": "Title" or null,
#   "artist": "Artist" or null,
#   "message": "conversational reply to the user"
# }}

# Rules for "message":
#   - HIGH    → state title + artist naturally. Optionally say why it matches.
#   - MEDIUM  → "This sounds like it might be X by Y — does that ring a bell?"
#   - LOW (final attempt) → admit no match. Briefly say what you searched.
#                           Ask ONE focused clarifying question (era,
#                           language, a remembered lyric fragment, or
#                           a specific emotion).

# ═══════════════════════════════════════════════════════════════
# HARD CONSTRAINTS — NEVER VIOLATE
# ═══════════════════════════════════════════════════════════════

#   - NEVER invent songs, artists, or lyrics that are not in <context>.
#   - NEVER answer from your own training knowledge. Only from <context>.
#   - NEVER return an empty queries list when action == "search".
#   - NEVER write any text outside the single JSON object.
#   - NEVER repeat a query from <previous_queries>.

# ═══════════════════════════════════════════════════════════════
# EXAMPLES
# ═══════════════════════════════════════════════════════════════

# Example 1 — empty context, first attempt, hybrid query:
#   user_query: "sad song about driving alone at night, 80s vibe"
#   context: (empty)
#   attempt: 1 of 4
#   →
#   {{
#     "action": "search",
#     "confidence": "low",
#     "reasoning": "Context is empty, nothing to match against.",
#     "queries": [
#       {{"query": "driving alone at night highway", "type": "text"}},
#       {{"query": "lonely night drive synth 1980s", "type": "hybrid"}},
#       {{"query": "headlights empty road sadness", "type": "text"}}
#     ]
#   }}

# Example 2 — strong match in context:
#   user_query: "song with the line about dancing with somebody who loves me"
#   context: [Whitney Houston — "I Wanna Dance with Somebody (Who Loves Me)",
#             1987, lyrics include "I wanna dance with somebody,
#             with somebody who loves me"]
#   →
#   {{
#     "action": "answer",
#     "confidence": "high",
#     "song": "I Wanna Dance with Somebody (Who Loves Me)",
#     "artist": "Whitney Houston",
#     "message": "That's I Wanna Dance with Somebody by Whitney Houston, from 1987."
#   }}

# Example 3 — audio-type query, first attempt:
#   user_query: "epic Lana Del Rey style song with male vocals, cinematic"
#   context: (empty)
#   attempt: 1 of 4
#   →
#   {{
#     "action": "search",
#     "confidence": "low",
#     "reasoning": "Context is empty, query is audio-based.",
#     "queries": [
#       {{"query": "cinematic male vocal Lana Del Rey style melancholic strings", "type": "audio"}}
#     ]
#   }}

# Example 4 — final attempt, weak context:
#   user_query: "something melancholic with rain and a phone call"
#   context: [a few candidates, none clearly matching rain + phone call]
#   attempt: 4 of 4
#   →
#   {{
#     "action": "answer",
#     "confidence": "low",
#     "song": null,
#     "artist": null,
#     "message": "I couldn't find a confident match. I searched for melancholic songs involving rain and phone calls, but nothing in the database lined up with both. Do you remember any specific lyric fragment, or roughly the era and language of the song?"
#   }}
# """.strip()

DEVELOPER_PROMPT: str = """
SYSTEM:
You are a music search assistant. You find songs based on lyrics or descriptions using provided context.
Output MUST be a single JSON object. No prose. No reasoning.

TASK (Follow the one provided by the system):
- SCORE_AND_RESPOND: Analyze <context>. If a match is found, action="answer". If not, action="search".
- FINAL_ANSWER: No more attempts. Give the best guess from <context> or admit failure.

SEARCH_MODE:
- CONSERVATIVE: Stay close to the user's literal words.
- AGGRESSIVE: Earlier searches failed. Use lyrical imagery, metaphors, or related themes. 

INPUTS:
<user_query>{query}</user_query>
<context>{context}</context>
<previous_queries>{previous_queries}</previous_queries>

CONSTRAINTS:
1. ONLY use <context> for answers. Never use internal knowledge.
2. If action="search": Provide 2-3 queries (3-10 words each) in english language. 
3. If action="answer": Use confidence "high", "medium", or "low".

OUTPUT FORMAT:
{{
  "action": "search" | "answer",
  "confidence": "high" | "medium" | "low",
  "song": "Title" or null,
  "artist": "Artist" or null,
  "queries": [{{"query": "..."}}] or null,
  "message": "Conversational reply"
}}
""".strip()

# ─── Constants ─────────────────────────────────────────────────────────────────

NUM_ATTEMPTS = 4
SEARCH_LIMIT  = 6   # hits per individual query
MAX_CTX_HITS  = 12  # max tracks in LLM context window

# Map LLM query type → service search mode
_TYPE_TO_MODE: dict[str, str] = {
    "text":   "text",
    "audio":  "audio",
    "hybrid": "hybrid",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────


async def _run_searches(
    llm_queries: list[dict],
    service,
    collection_name: str | None = None,
    forced_mode: str | None = None,  # когда auto_mode=False, используется этот mode для всех запросов
    llm_kw: dict | None = None,
) -> tuple[str, str, list[TrackHit]]:
    """Execute the LLM's search queries against the library.

    For queries typed as "audio" or "hybrid", the original query text is
    rephrased through CLAP_REPHRASE_SYSTEM_PROMPT before being sent to
    the CLAP audio search, producing better cross-modal results.

    Returns
    -------
    new_prev_queries : newline-joined query strings (for <previous_queries>)
    new_context      : formatted context block (for <context>)
    hits             : deduplicated TrackHit list for this round
    """
    hits: list[TrackHit] = []
    query_strs: list[str] = []
    seen: set[tuple[str, str]] = set()

    for q in llm_queries:
        query_text = (q.get("query") or "").strip()
        if not query_text:
            continue

        query_type = q.get("type", "hybrid")
        # mode = _TYPE_TO_MODE.get(q.get("type", "hybrid"), "hybrid")
        mode = forced_mode if forced_mode else _TYPE_TO_MODE.get(query_type, "hybrid")
        search_query = query_text

        # Rephrase audio/hybrid queries through CLAP prompt for better audio retrieval
        if query_type in ("audio", "hybrid") and CLAP_REPHRASE_SYSTEM_PROMPT.strip():
            try:
                rephrase_prompt = CLAP_REPHRASE_SYSTEM_PROMPT.format(user_query=query_text)
                rephrased = await ask_llm(
                    query_text,
                    system_prompt=rephrase_prompt,
                    parse_json=True,
                    **(llm_kw or {}),
                )
                if isinstance(rephrased, list) and rephrased:
                    search_query = rephrased[0]
            except Exception as exc:
                print(f"[chat] CLAP rephrasing in agentic loop (non-fatal): {exc}")

        query_strs.append(query_text)

        try:
            round_hits = await service.search(
                query=search_query, mode=mode, limit=SEARCH_LIMIT,
                collection_name=collection_name,
            )
            for hit in round_hits:
                key = (hit.track.title.lower(), hit.track.artist.lower())
                if key not in seen:
                    seen.add(key)
                    hits.append(hit)
        except Exception as exc:
            print(f"[chat] search error for '{query_text}': {exc}")

    # Build context string
    ctx_parts: list[str] = []
    for hit in hits[:MAX_CTX_HITS]:
        t = hit.track
        header = f"• {t.title} — {t.artist}"
        if t.album:
            header += f" [{t.album}]"
        if t.year:
            header += f" ({t.year})"
        lines = [header]
        if t.genre:
            lines.append(f"  Genre: {t.genre}")
        lyric_text = hit.lyrics or ""
        if lyric_text:
            lines.append(f"  Lyrics: {lyric_text}")
        ctx_parts.append("\n".join(lines))

    return (
        "\n".join(query_strs),
        "\n\n".join(ctx_parts),
        hits,
    )


def _merge_hits(
    existing: list[TrackHit],
    new_hits: list[TrackHit],
) -> list[TrackHit]:
    """Append new hits, skipping duplicates already in existing."""
    seen = {(h.track.title.lower(), h.track.artist.lower()) for h in existing}
    for h in new_hits:
        key = (h.track.title.lower(), h.track.artist.lower())
        if key not in seen:
            seen.add(key)
            existing.append(h)
    return existing


# ─── Endpoint ─────────────────────────────────────────────────────────────────


@router.post("/")
async def chat(req: ChatRequest, request: Request) -> dict:
    """Agentic LLM-driven music search.

    Response shape
    --------------
    {
      "message":        str,           # LLM's conversational reply
      "song":           str | null,    # identified song title (if confident)
      "artist":         str | null,    # identified artist
      "confidence":     "high"|"medium"|"low",
      "hits":           [TrackHit…],   # all retrieved tracks (for the UI)
      "attempts":       int,           # how many LLM calls were made
      "classification": dict,          # result of call-1 (empty if prompt unset)
    }
    """
    service = request.app.state.search_service
    if service is None:
        return {
            "message":        "Search service unavailable — is Qdrant running?",
            "song":           None,
            "artist":         None,
            "confidence":     "low",
            "hits":           [],
            "attempts":       0,
            "classification": {},
        }

    # Common kwargs forwarded to every ask_llm call
    llm_kw: dict[str, Any] = {
        "base_url":   (req.llm_base_url or "").strip() or None,
        "model":      (req.llm_model or "").strip() or None,
        "extra_body": {"enable_thinking": False},
        "temperature": 0.3,
    }

    # ── Planner path (Phase 2: PydanticAI-based) ──────────────────────────
    # When planner_enabled=True, skip old classification and use the
    # PydanticAI PlannerAgent to classify, extract filters, and generate
    # initial search queries. Falls back to old behavior on any error.
    planner_queries: list[dict] | None = None
    planner_filters: dict | None = None
    planner_classification: dict = {}

    if req.planner_enabled:
        try:
            from app.services.agent_deps import SearchDeps
            from app.services.agents import create_planner_agent

            deps = SearchDeps(
                service=service,
                collection_name=req.collection_name,
                llm_base_url=llm_kw.get("base_url"),
                llm_model=llm_kw.get("model"),
            )
            planner = create_planner_agent(deps)

            # Format resolved_filters for the prompt
            resolved_filters_str = "{}"
            search_filter_query_str = ""
            previous_queries_str = "(none)"

            # First call to Planner
            filled_prompt = PLANNER_PROMPT.format(
                query=req.message,
                previous_queries=previous_queries_str,
                resolved_filters=resolved_filters_str,
                search_filter_query=search_filter_query_str,
            )
            plan_result = await planner.run(req.message, system_prompt=filled_prompt)
            plan = plan_result.data

            planner_classification = {
                "type": plan.query_type,
                "reasoning": "planner",
            }

            # Resolve filters if action == "request_filter"
            if plan.action == "request_filter" and plan.filter_lookup:
                resolved = await deps.resolve_filters(plan.filter_lookup)
                if resolved:
                    # Re-run Planner with resolved filters
                    resolved_filters_str = str(resolved)
                    search_filter_query_str = str(plan.filter_lookup)
                    filled_prompt = PLANNER_PROMPT.format(
                        query=req.message,
                        previous_queries=previous_queries_str,
                        resolved_filters=resolved_filters_str,
                        search_filter_query=search_filter_query_str,
                    )
                    plan_result = await planner.run(req.message, system_prompt=filled_prompt)
                    plan = plan_result.data

            # Extract queries for the agentic loop
            if plan.queries:
                planner_queries = [{"query": q.query, "type": q.type} for q in plan.queries]
                planner_filters = plan.filters.model_dump() if plan.filters else None

        except Exception as exc:
            print(f"[chat] Planner error (falling back to old behavior): {exc}")
            print(traceback.format_exc())
            planner_queries = None
            planner_filters = None

    # ── Call 1: classification (skipped when prompt is empty OR auto_mode=False) ──
    classification: dict = planner_classification if req.planner_enabled else {}
    if not req.planner_enabled and req.auto_mode and CLASSIFICATION_SYSTEM_PROMPT.strip():
        try:
            classification = await ask_llm(
                req.message,
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
                parse_json=True,
                **llm_kw,
            )
        except Exception as exc:
            print(f"[chat] classification error (non-fatal): {exc}")
            print(traceback.format_exc())

    # Determine effective search mode
    if req.planner_enabled and planner_classification:
        detected_type = planner_classification.get("type", "hybrid")
    elif req.auto_mode:
        detected_type = classification.get("type", "hybrid")
    else:
        detected_type = None
    effective_mode = req.mode if not req.auto_mode and not req.planner_enabled else (detected_type or "hybrid")

    # ── Audio fast path: rephrase → 3× CLAP search → answer ─────────────────
    # When the query is classified as "audio" (or forced to "audio"), skip the
    # agentic loop. Instead:
    #   1. Ask LLM to rephrase user's mood/vibe into 3 CLAP-friendly prompts
    #   2. Run each through CLAP audio search (10 results each)
    #   3. Merge, pick top 5, send #1 to LLM for conversational answer
    if effective_mode == "audio" or (not req.auto_mode and req.mode == "audio"):
        audio_rephrased_queries: list[str] = []

        if CLAP_REPHRASE_SYSTEM_PROMPT.strip():
            try:
                rephrase_prompt = CLAP_REPHRASE_SYSTEM_PROMPT.format(
                    user_query=req.message,
                )
                rephrase_result = await ask_llm(
                    req.message,
                    system_prompt=rephrase_prompt,
                    parse_json=True,
                    **llm_kw,
                )
                if isinstance(rephrase_result, list) and rephrase_result:
                    audio_rephrased_queries = rephrase_result
            except Exception as exc:
                print(f"[chat] CLAP rephrasing error (non-fatal): {exc}")

        # Fallback: if rephrasing produced nothing, use the original query
        if not audio_rephrased_queries:
            audio_rephrased_queries = [req.message]

        # Run each rephrased query through CLAP audio search (10 results each)
        all_hits: list[TrackHit] = []
        for rq in audio_rephrased_queries:
            try:
                round_hits = await service.search(
                    query=rq, mode="audio", limit=10,
                    collection_name=req.collection_name,
                )
                all_hits = _merge_hits(all_hits, round_hits)
            except Exception as exc:
                print(f"[chat] audio search error for '{rq}': {exc}")

        # Sort by score, pick top 5
        all_hits.sort(key=lambda h: h.score, reverse=True)
        top5 = all_hits[:5]

        if top5:
            best = top5[0]

            # Generate conversational answer via LLM
            try:
                answer_prompt = AUDIO_ANSWER_PROMPT.format(
                    user_query=req.message,
                    title=best.track.title,
                    artist=best.track.artist,
                    album=best.track.album or "—",
                    year=best.track.year or "—",
                )
                answer_result = await ask_llm(
                    req.message,
                    system_prompt=answer_prompt,
                    parse_json=True,
                    **llm_kw,
                )
                if isinstance(answer_result, dict):
                    audio_message = answer_result.get("message", "")
                else:
                    audio_message = ""
            except Exception as exc:
                print(f"[chat] audio-answer LLM error (non-fatal): {exc}")
                audio_message = ""

            # Fallback message if LLM didn't produce one
            if not audio_message:
                album_part = f" [{best.track.album}]" if best.track.album else ""
                audio_message = (
                    f"По звучанию ближе всего — «{best.track.title}» "
                    f"({best.track.artist}{album_part})."
                )

            return {
                "message":    audio_message,
                "song":       best.track.title,
                "artist":     best.track.artist,
                "confidence": "medium",
                "best_hit":   best.model_dump(),
                "hits":       [h.model_dump() for h in top5],
                "attempts":   1,
                "classification": classification,
            }
        else:
            return {
                "message":    "Не удалось найти треков по описанию звука. Попробуй уточнить настроение, инструменты, или стиль вокала.",
                "song":       None,
                "artist":     None,
                "confidence": "low",
                "best_hit":   None,
                "hits":       [],
                "attempts":   1,
                "classification": classification,
            }

    # ── Calls 2…N: agentic search loop (text / hybrid) ──────────────────────
    previous_queries = ""
    context          = ""
    all_hits: list[TrackHit] = []
    final_result: dict       = {}
    attempts_done            = 0

    for attempt in range(1, NUM_ATTEMPTS + 1):
        # Use Planner's queries on the first attempt (skip LLM call)
        if attempt == 1 and planner_queries:
            queries = planner_queries
            action = "search"
        else:
            action = None

        attempts_done = attempt

        # Only call LLM if Planner didn't provide queries
        if action is None:
            filled = DEVELOPER_PROMPT.format(
                query=req.message,
                context=context          or "(empty — no results yet)",
                previous_queries=previous_queries or "(none)",
                # attempt=attempt,
                # max_attempts=NUM_ATTEMPTS,
            )

            try:
                result: dict = await ask_llm(
                    req.message,
                    system_prompt=filled,
                    parse_json=True,
                    **llm_kw,
                )
            except Exception as exc:
                print(f"[chat] LLM error on attempt {attempt}: {exc}")
                final_result = {
                    "action":     "answer",
                    "confidence": "low",
                    "song":       None,
                    "artist":     None,
                    "message":    f"LLM error on attempt {attempt}: {exc}",
                }
                break

            action = result.get("action", "answer")

        if action == "search":
            # If LLM generated the action, extract queries from result
            if 'result' in locals() and isinstance(result, dict):
                queries = result.get("queries") or []
            forced_mode = req.mode if not req.auto_mode else None
            new_pq, new_ctx, new_hits = await _run_searches(queries, service, collection_name=req.collection_name, forced_mode=forced_mode, llm_kw=llm_kw)

            if new_pq:
                previous_queries = (
                    (previous_queries + "\n" + new_pq).strip()
                )
            if new_ctx:
                context = (context + "\n\n" + new_ctx).strip()

            all_hits = _merge_hits(all_hits, new_hits)

            # Last attempt and still "search" — force exit
            if attempt == NUM_ATTEMPTS:
                final_result = {
                    "action":     "answer",
                    "confidence": "low",
                    "song":       None,
                    "artist":     None,
                    "message": (
                        "Не нашёл подходящего трека после нескольких попыток поиска. "
                        "Попробуй уточнить: язык, примерная эпоха или фрагмент текста."
                    ),
                }

        elif action == "answer":
            final_result = result
            break

    # Sort retrieved hits by score descending
    all_hits.sort(key=lambda h: h.score, reverse=True)

    return {
        "message":        final_result.get("message", ""),
        "song":           final_result.get("song"),
        "artist":         final_result.get("artist"),
        "confidence":     final_result.get("confidence", "low"),
        "hits":           [h.model_dump() for h in all_hits[:10]],
        "attempts":       attempts_done,
        "classification": classification,
    }
