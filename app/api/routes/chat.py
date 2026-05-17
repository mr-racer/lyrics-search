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
from app.services.agents import (
    PLANNER_PROMPT,
    SCORER_PROMPT,
    _AUDIO_ANSWER_PROMPT,
    _CLAP_REPHRASE_SYSTEM_PROMPT,
    create_scorer_agent,
)
from app.services.agent_deps import SearchDeps
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
        if query_type in ("audio", "hybrid") and _CLAP_REPHRASE_SYSTEM_PROMPT.strip():
            try:
                rephrase_prompt = _CLAP_REPHRASE_SYSTEM_PROMPT.format(user_query=query_text)
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

    # ── Audio fast path: AudioAgent (Phase 4) ────────────────────────────────
    # When the query is classified as "audio", use AudioAgent which:
    #   1. Rephrases user's mood/vibe into 3 CLAP-friendly prompts
    #   2. Runs each through CLAP audio search (10 results each)
    #   3. Returns the best match with a conversational answer
    # Falls back to old inline behavior on any error.
    if effective_mode == "audio" or (not req.auto_mode and req.mode == "audio"):
        audio_used_agent = False
        audio_response: dict | None = None

        if req.planner_enabled:
            # Try AudioAgent first
            try:
                from app.services.agents import create_audio_agent

                audio_deps = SearchDeps(
                    service=service,
                    collection_name=req.collection_name,
                    llm_base_url=llm_kw.get("base_url"),
                    llm_model=llm_kw.get("model"),
                )
                audio_agent = create_audio_agent(audio_deps)
                agent_result = await audio_agent.run(req.message)
                answer = agent_result.output

                # Retrieve cached hits from the agent's search_db tool
                cached_hits: list[TrackHit] = getattr(
                    audio_agent, "_audio_hits_cache", []
                )
                cached_hits.sort(key=lambda h: h.score, reverse=True)
                top5 = cached_hits[:5]

                # Extract best_hit from answer or from cached hits
                best_hit_dump = answer.best_hit
                if not best_hit_dump and top5:
                    best_hit_dump = top5[0].model_dump()

                audio_response = {
                    "message":    answer.message,
                    "song":       best_hit_dump.get("title") if best_hit_dump else None,
                    "artist":     best_hit_dump.get("artist") if best_hit_dump else None,
                    "confidence": "medium",
                    "best_hit":   best_hit_dump,
                    "hits":       [h.model_dump() for h in top5],
                    "attempts":   1,
                    "classification": classification,
                }
                audio_used_agent = True

            except Exception as exc:
                print(f"[chat] AudioAgent error (falling back to inline): {exc}")
                print(traceback.format_exc())

        # Old inline behavior (fallback or when planner_enabled=False)
        if not audio_used_agent:
            audio_rephrased_queries: list[str] = []

            if _CLAP_REPHRASE_SYSTEM_PROMPT.strip():
                try:
                    rephrase_prompt = _CLAP_REPHRASE_SYSTEM_PROMPT.format(
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
            audio_all_hits: list[TrackHit] = []
            for rq in audio_rephrased_queries:
                try:
                    round_hits = await service.search(
                        query=rq, mode="audio", limit=10,
                        collection_name=req.collection_name,
                    )
                    audio_all_hits = _merge_hits(audio_all_hits, round_hits)
                except Exception as exc:
                    print(f"[chat] audio search error for '{rq}': {exc}")

            # Sort by score, pick top 5
            audio_all_hits.sort(key=lambda h: h.score, reverse=True)
            audio_top5 = audio_all_hits[:5]

            if audio_top5:
                audio_best = audio_top5[0]

                # Generate conversational answer via LLM
                try:
                    answer_prompt = _AUDIO_ANSWER_PROMPT.format(
                        user_query=req.message,
                        title=audio_best.track.title,
                        artist=audio_best.track.artist,
                        album=audio_best.track.album or "—",
                        year=audio_best.track.year or "—",
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
                    album_part = f" [{audio_best.track.album}]" if audio_best.track.album else ""
                    audio_message = (
                        f"По звучанию ближе всего — «{audio_best.track.title}» "
                        f"({audio_best.track.artist}{album_part})."
                    )

                audio_response = {
                    "message":    audio_message,
                    "song":       audio_best.track.title,
                    "artist":     audio_best.track.artist,
                    "confidence": "medium",
                    "best_hit":   audio_best.model_dump(),
                    "hits":       [h.model_dump() for h in audio_top5],
                    "attempts":   1,
                    "classification": classification,
                }
            else:
                audio_response = {
                    "message":    "Не удалось найти треков по описанию звука. Попробуй уточнить настроение, инструменты, или стиль вокала.",
                    "song":       None,
                    "artist":     None,
                    "confidence": "low",
                    "best_hit":   None,
                    "hits":       [],
                    "attempts":   1,
                    "classification": classification,
                }

        return audio_response

    # ── Calls 2…N: agentic search loop (text / hybrid) ──────────────────────
    # When planner_enabled, use ScorerAgent for context evaluation.
    # Otherwise, fall back to the old DEVELOPER_PROMPT + ask_llm pattern.
    previous_queries = ""
    context          = ""
    all_hits: list[TrackHit] = []
    final_result: dict       = {}
    attempts_done            = 0

    # Build SearchDeps for ScorerAgent (only when planner_enabled)
    scorer_deps: SearchDeps | None = None
    if req.planner_enabled:
        scorer_deps = SearchDeps(
            service=service,
            collection_name=req.collection_name,
            llm_base_url=llm_kw.get("base_url"),
            llm_model=llm_kw.get("model"),
        )

    for attempt in range(1, NUM_ATTEMPTS + 1):
        attempts_done = attempt
        action = None
        queries: list[dict] | None = None

        # ── First attempt: use Planner's queries (skip LLM call) ──
        if attempt == 1 and planner_queries:
            queries = planner_queries
            action = "search"

        # ── Subsequent attempts: ScorerAgent or old LLM ──
        if action is None and scorer_deps is not None:
            # ScorerAgent evaluates context and decides search/answer
            try:
                scorer = create_scorer_agent(scorer_deps)
                filled = SCORER_PROMPT.format(
                    query=req.message,
                    context=context or "(empty — no results yet)",
                    previous_queries=previous_queries or "(none)",
                    active_filters=str(planner_filters) if planner_filters else "(none)",
                    attempt_number=attempt,
                )
                score_result = await scorer.run(req.message, system_prompt=filled)
                score = score_result.data

                action = score.action
                if action == "search" and score.queries:
                    queries = [{"query": q.query, "type": q.type} for q in score.queries]
                elif action in ("answer", "final_answer"):
                    final_result = {
                        "action":     action,
                        "confidence": score.confidence,
                        "song":       score.song,
                        "artist":     score.artist,
                        "message":    score.message,
                    }
            except Exception as exc:
                print(f"[chat] ScorerAgent error on attempt {attempt}: {exc}")
                # Fallback to old behavior
                scorer_deps = None  # prevent retry

        if action is None:
            # Old behavior: DEVELOPER_PROMPT + ask_llm
            filled = DEVELOPER_PROMPT.format(
                query=req.message,
                context=context or "(empty — no results yet)",
                previous_queries=previous_queries or "(none)",
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

        # ── Execute search ──
        if action == "search" and queries:
            forced_mode = req.mode if not req.auto_mode else None
            new_pq, new_ctx, new_hits = await _run_searches(
                queries, service,
                collection_name=req.collection_name,
                forced_mode=forced_mode,
                llm_kw=llm_kw,
            )
            if new_pq:
                previous_queries = (previous_queries + "\n" + new_pq).strip()
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

        elif action in ("answer", "final_answer"):
            if not final_result:
                final_result = result if 'result' in locals() else {}
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
