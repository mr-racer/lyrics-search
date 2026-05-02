"""Chat endpoint — two-call agentic LLM-driven music search.

Flow per user message
─────────────────────
Call 1  (classification)
    System: CLASSIFICATION_SYSTEM_PROMPT   ← fill in yourself
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

import textwrap
from typing import Any

from fastapi import APIRouter, Request

from app.domain.models import ChatRequest, TrackHit
from app.services.llm_client import ask_llm
import traceback

router = APIRouter(prefix="/chat", tags=["Chat"])

# ─── Prompts ──────────────────────────────────────────────────────────────────

# TODO: provide your classification system prompt here.
# The call is skipped entirely when this string is empty.
CLASSIFICATION_SYSTEM_PROMPT: str = "You need to classify the user query into one of the following types: text, sound, hybrid"

DEVELOPER_PROMPT: str = """
You are a music search assistant. You find songs from descriptions, moods,
vague memories, or remembered lyric fragments. You have access to a lyrics
database via retrieval.

You run in a loop. On each turn you EITHER answer the user from retrieved
context, OR issue new search queries for another retrieval round.

═══════════════════════════════════════════════════════════════
INPUTS
═══════════════════════════════════════════════════════════════

<user_query>
{query}
</user_query>

<context>
{context}
</context>
Lyrics + metadata (title, artist, album, year) from previous retrieval rounds.
Empty on the first attempt.

<previous_queries>
{previous_queries}
</previous_queries>
Queries already tried. Never repeat them. Always rephrase or shift angle.

<attempt>
{attempt} of {max_attempts}
</attempt>

═══════════════════════════════════════════════════════════════
STEP 1 — SCORE THE CONTEXT
═══════════════════════════════════════════════════════════════

For each candidate song in <context>, check:
  1. Do the lyrics match specific details from the user query
     (imagery, story, fragments)?
  2. Do metadata signals (era, artist, genre, language) match?

Then assign ONE confidence label:
  HIGH    — lyrics clearly match specific details the user gave.
  MEDIUM  — plausible match, but a key detail is missing OR multiple
            candidates tie.
  LOW     — nothing in context fits, or context is empty.

═══════════════════════════════════════════════════════════════
STEP 2 — PICK THE ACTION
═══════════════════════════════════════════════════════════════

Apply these rules in order. Stop at the first match.

  IF confidence == HIGH                        → action = "answer"
  IF confidence == MEDIUM AND attempt <  max   → action = "search"
  IF confidence == MEDIUM AND attempt == max   → action = "answer" (best guess, state uncertainty)
  IF confidence == LOW    AND attempt <  max   → action = "search"
  IF confidence == LOW    AND attempt == max   → action = "answer" (admit no match, ask ONE clarifying question)

═══════════════════════════════════════════════════════════════
STEP 3 — IF action == "search", BUILD QUERIES
═══════════════════════════════════════════════════════════════

3A. Classify the user query into ONE type:

  "text"   — User asks about a concrete detail that should literally appear
             in lyrics.
             Example: "Which songs mention luxury cars?"

  "sound"  — User describes feelings, vibe, vocals, production, atmosphere,
             not specific words. No concrete lyric fragment given.
             Example: "Epic Lana Del Rey style song with male vocals."

  "hybrid" — Mix of both, or unclear which one dominates.
             Example: "A sad 80s song about driving alone at night."

3B. Generate queries based on type:

  IF type == "sound":
      Output exactly ONE query.
      Pack it with sound/feeling vocabulary from the user's message
      (vocal type, instruments, era, mood, production style).

  IF type == "text" OR type == "hybrid":
      Output 2-3 queries. Each query must:
        - Be 3-10 words.
        - Use words likely to appear in real lyrics
          (concrete nouns, imagery, emotional phrases).
        - Cover a DIFFERENT angle from the others
          (e.g. one imagery, one emotion, one storyline).
        - Differ from every entry in <previous_queries>.
        - Never be empty. If unsure, paraphrase the user's query.

3C. Pick rephrasing mode based on attempt number:

  Let half = floor(max_attempts / 2).

  IF attempt <= half  → CONSERVATIVE mode:
      Stay close to the user's wording.
      Swap synonyms, reorder phrases. Goal: precision.

  IF attempt >  half  → AGGRESSIVE mode:
      Earlier searches failed. Close paraphrases will fail too.
      Drop literal wording. Each of the 2-3 queries explores a
      DIFFERENT direction from this list:
        - Replace abstract feelings with concrete lyrical imagery.
          "feeling lost" → "wandering empty streets", "no map no plan"
        - Flip the perspective. A breakup song may be written from
          either side. A "sad" song may have ironic upbeat lyrics.
        - Guess a likely hook or chorus phrase, not a description.
        - Try adjacent themes:
          loneliness ↔ nostalgia, anger ↔ heartbreak, love ↔ obsession.
        - Shift era/genre vocabulary if the user gave hints.
      Do NOT output three variants of the same idea.

  Note: when attempt == max_attempts, action is always "answer", so this
  step does not run on the final attempt.

═══════════════════════════════════════════════════════════════
OUTPUT — RETURN ONE VALID JSON OBJECT. NO TEXT BEFORE OR AFTER.
═══════════════════════════════════════════════════════════════

When action == "search":
{{
  "action": "search",
  "confidence": "low" | "medium",
  "reasoning": "one short sentence on why context is insufficient",
  "queries": [
    {{"query": "query text 1", "type": "text" | "sound" | "hybrid"}},
    {{"query": "query text 2", "type": "text" | "sound" | "hybrid"}},
    {{"query": "query text 3", "type": "text" | "sound" | "hybrid"}}
  ]
}}

When action == "answer":
{{
  "action": "answer",
  "confidence": "high" | "medium" | "low",
  "song": "Title" or null,
  "artist": "Artist" or null,
  "message": "conversational reply to the user"
}}

Rules for "message":
  - HIGH    → state title + artist naturally. Optionally say why it matches.
  - MEDIUM  → "This sounds like it might be X by Y — does that ring a bell?"
  - LOW (final attempt) → admit no match. Briefly say what you searched.
                          Ask ONE focused clarifying question (era,
                          language, a remembered lyric fragment, or
                          a specific emotion).

═══════════════════════════════════════════════════════════════
HARD CONSTRAINTS — NEVER VIOLATE
═══════════════════════════════════════════════════════════════

  - NEVER invent songs, artists, or lyrics that are not in <context>.
  - NEVER answer from your own training knowledge. Only from <context>.
  - NEVER return an empty queries list when action == "search".
  - NEVER write any text outside the single JSON object.
  - NEVER repeat a query from <previous_queries>.

═══════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════

Example 1 — empty context, first attempt, hybrid query:
  user_query: "sad song about driving alone at night, 80s vibe"
  context: (empty)
  attempt: 1 of 4
  →
  {{
    "action": "search",
    "confidence": "low",
    "reasoning": "Context is empty, nothing to match against.",
    "queries": [
      {{"query": "driving alone at night highway", "type": "text"}},
      {{"query": "lonely night drive synth 1980s", "type": "hybrid"}},
      {{"query": "headlights empty road sadness", "type": "text"}}
    ]
  }}

Example 2 — strong match in context:
  user_query: "song with the line about dancing with somebody who loves me"
  context: [Whitney Houston — "I Wanna Dance with Somebody (Who Loves Me)",
            1987, lyrics include "I wanna dance with somebody,
            with somebody who loves me"]
  →
  {{
    "action": "answer",
    "confidence": "high",
    "song": "I Wanna Dance with Somebody (Who Loves Me)",
    "artist": "Whitney Houston",
    "message": "That's I Wanna Dance with Somebody by Whitney Houston, from 1987."
  }}

Example 3 — sound-type query, first attempt:
  user_query: "epic Lana Del Rey style song with male vocals, cinematic"
  context: (empty)
  attempt: 1 of 4
  →
  {{
    "action": "search",
    "confidence": "low",
    "reasoning": "Context is empty, query is sound-based.",
    "queries": [
      {{"query": "cinematic male vocal Lana Del Rey style melancholic strings", "type": "sound"}}
    ]
  }}

Example 4 — final attempt, weak context:
  user_query: "something melancholic with rain and a phone call"
  context: [a few candidates, none clearly matching rain + phone call]
  attempt: 4 of 4
  →
  {{
    "action": "answer",
    "confidence": "low",
    "song": null,
    "artist": null,
    "message": "I couldn't find a confident match. I searched for melancholic songs involving rain and phone calls, but nothing in the database lined up with both. Do you remember any specific lyric fragment, or roughly the era and language of the song?"
  }}
""".strip()

# ─── Constants ─────────────────────────────────────────────────────────────────

NUM_ATTEMPTS = 4
SEARCH_LIMIT  = 6   # hits per individual query
MAX_CTX_HITS  = 12  # max tracks in LLM context window

# Map LLM query type → service search mode
_TYPE_TO_MODE: dict[str, str] = {
    "text":   "text",
    "sound":  "audio",
    "hybrid": "hybrid",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────


async def _run_searches(
    llm_queries: list[dict],
    service,
) -> tuple[str, str, list[TrackHit]]:
    """Execute the LLM's search queries against the library.

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

        mode = _TYPE_TO_MODE.get(q.get("type", "hybrid"), "hybrid")
        query_strs.append(query_text)

        try:
            round_hits = await service.search(
                query=query_text, mode=mode, limit=SEARCH_LIMIT
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

    # ── Call 1: classification (skipped when prompt is empty) ─────────────────
    classification: dict = {}
    if CLASSIFICATION_SYSTEM_PROMPT.strip():
        try:
            classification = await ask_llm(
                req.message,
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
                parse_json=False,
                **llm_kw,
            )
        except Exception as exc:
            print(f"[chat] classification error (non-fatal): {exc}")
            print(traceback.format_exc())

    # ── Calls 2…N: agentic search loop ───────────────────────────────────────
    previous_queries = ""
    context          = ""
    all_hits: list[TrackHit] = []
    final_result: dict       = {}
    attempts_done            = 0

    for attempt in range(1, NUM_ATTEMPTS + 1):
        attempts_done = attempt

        filled = DEVELOPER_PROMPT.format(
            query=req.message,
            context=context          or "(empty — no results yet)",
            previous_queries=previous_queries or "(none)",
            attempt=attempt,
            max_attempts=NUM_ATTEMPTS,
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
            queries = result.get("queries") or []
            new_pq, new_ctx, new_hits = await _run_searches(queries, service)

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
