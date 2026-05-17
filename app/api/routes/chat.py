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

from app.domain.models import ChatRequest, SearchFilters, TrackHit
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
import logging

logger = logging.getLogger(__name__)

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
4. If <context> has a confident match → action="answer". Otherwise → action="search".

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

# ─── Helpers ──────────────────────────────────────────────────────────────────


async def _run_searches(
    llm_queries: list[dict],
    service,
    collection_name: str | None = None,
    forced_mode: str | None = None,  # когда auto_mode=False, используется этот mode для всех запросов
    llm_kw: dict | None = None,
    skip_rephrase: bool = False,  # True когда запросы уже CLAP-оптимизированы (от Planner)
) -> tuple[str, str, list[TrackHit]]:
    """Execute the LLM's search queries against the library.

    The search mode is determined entirely by forced_mode (= effective_mode from the
    initial classifier). Queries themselves carry no type — mode is fixed once at
    classification time and never re-decided here.

    For "audio" or "hybrid" forced_mode, the query text is rephrased through
    CLAP_REPHRASE_SYSTEM_PROMPT before CLAP search unless skip_rephrase=True.

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

        # Mode is fixed once by the classifier — forced_mode always carries effective_mode.
        mode = forced_mode or "hybrid"
        search_query = query_text

        # Rephrase audio/hybrid queries through CLAP prompt for better audio retrieval.
        # Skip when queries are already CLAP-optimised (e.g. from Planner on attempt 1).
        if not skip_rephrase and mode in ("audio", "hybrid") and _CLAP_REPHRASE_SYSTEM_PROMPT.strip():
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


def _rrf_merge(ranked_lists: list[list[TrackHit]], k: int = 60) -> list[TrackHit]:
    """Reciprocal Rank Fusion across multiple ranked result lists.

    Tracks appearing in multiple lists and ranked highly in each get the
    highest combined score: RRF(d) = Σ 1 / (k + rank_i).
    """
    rrf_scores: dict[tuple[str, str], float] = {}
    best_hit: dict[tuple[str, str], TrackHit] = {}

    for ranked_list in ranked_lists:
        for rank, hit in enumerate(ranked_list, start=1):
            key = (hit.track.title.lower(), hit.track.artist.lower())
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in best_hit or hit.score > best_hit[key].score:
                best_hit[key] = hit

    sorted_keys = sorted(rrf_scores, key=lambda dk: rrf_scores[dk], reverse=True)
    return [best_hit[dk] for dk in sorted_keys]


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

            # Extract queries for the agentic loop — type comes from effective_mode, not per-query
            if plan.queries:
                planner_queries = [{"query": q.query} for q in plan.queries]
                planner_filters = plan.filters.model_dump() if plan.filters else None

            resolved_filters_log = {k: v for k, v in (planner_filters or {}).items() if v}
            logger.info(
                "[chat/planner] type=%s  queries=%s  filters=%s",
                plan.query_type,
                [q["query"] for q in (planner_queries or [])],
                resolved_filters_log or "(none)",
            )

        except Exception as exc:
            print(f"[chat] Planner error (falling back to old behavior): {exc}")
            print(traceback.format_exc())
            planner_queries = None
            planner_filters = None

    # ── Call 1: classification ────────────────────────────────────────────────
    # Use planner classification when available; fall back to old classifier when:
    # - planner is disabled, or
    # - planner failed (planner_classification is empty) and auto_mode is on
    classification: dict = planner_classification if req.planner_enabled else {}
    needs_old_classifier = (
        not req.planner_enabled or not planner_classification
    ) and req.auto_mode and CLASSIFICATION_SYSTEM_PROMPT.strip()
    if needs_old_classifier:
        try:
            classification = await ask_llm(
                req.message,
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
                parse_json=True,
                **llm_kw,
            )
        except Exception as exc:
            logger.error(f"[chat] classification error (non-fatal): {exc}", exc_info=True)

    # Determine effective search mode
    if req.planner_enabled and planner_classification:
        detected_type = planner_classification.get("type", "hybrid")
    elif req.auto_mode:
        detected_type = classification.get("type", "hybrid")
    else:
        detected_type = None
    effective_mode = req.mode if not req.auto_mode and not req.planner_enabled else (detected_type or "hybrid")

    # ── Log classification outcome ────────────────────────────────────────────
    active_filters = {k: v for k, v in (planner_filters or {}).items() if v}
    logger.info(
        "[chat] query=%r  mode=%s  filters=%s",
        req.message[:80],
        effective_mode,
        active_filters or "(none)",
    )

    # ── Audio fast path: rephrase → 3× CLAP search → RRF → answer ──────────
    # 1. LLM rephrases user's mood/vibe into 3 CLAP-optimised English prompts
    # 2. Each prompt runs through CLAP audio search (10 results each)
    # 3. Three ranked lists are merged with Reciprocal Rank Fusion — the track
    #    that appears most often AND highest across all lists wins
    # 4. Best hit is passed to LLM to produce a conversational reply
    if effective_mode == "audio" or (not req.auto_mode and req.mode == "audio"):
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
                logger.warning("[chat] CLAP rephrasing error (non-fatal): %s", exc)

        if not audio_rephrased_queries:
            audio_rephrased_queries = [req.message]

        # Run each rephrased query through CLAP audio search
        audio_filters = SearchFilters(**planner_filters) if planner_filters else None
        per_query_hits: list[list[TrackHit]] = []
        for rq in audio_rephrased_queries:
            try:
                round_hits = await service.search(
                    query=rq, mode="audio", limit=10,
                    collection_name=req.collection_name,
                    filters=audio_filters,
                )
                per_query_hits.append(round_hits)
            except Exception as exc:
                logger.warning("[chat] audio search error for %r: %s", rq, exc)

        # RRF merge: tracks appearing in multiple lists and ranked high win
        merged = _rrf_merge(per_query_hits) if per_query_hits else []
        top5 = merged[:5]

        logger.info(
            "[chat/audio] found %d hits → top: %s",
            len(merged),
            [(h.track.artist, h.track.title) for h in top5],
        )

        if top5:
            best = top5[0]

            audio_message = ""
            try:
                answer_prompt = _AUDIO_ANSWER_PROMPT.format(
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
            except Exception as exc:
                logger.warning("[chat] audio-answer LLM error (non-fatal): %s", exc)

            if not audio_message:
                album_part = f" [{best.track.album}]" if best.track.album else ""
                audio_message = (
                    f"По звучанию ближе всего — «{best.track.title}» "
                    f"({best.track.artist}{album_part})."
                )

            return {
                "message":        audio_message,
                "song":           best.track.title,
                "artist":         best.track.artist,
                "confidence":     "medium",
                "hits":           [h.model_dump() for h in top5],
                "attempts":       1,
                "classification": classification,
            }
        else:
            return {
                "message":        "Не удалось найти треков по описанию звука. Попробуй уточнить настроение, инструменты, или стиль вокала.",
                "song":           None,
                "artist":         None,
                "confidence":     "low",
                "hits":           [],
                "attempts":       1,
                "classification": classification,
            }

    # ── Agentic search loop ────────────────────────────────────────────────
    # Flow per attempt:
    #   1. Decide action (scorer/LLM evaluates context → search or answer)
    #   2. If search → execute, update context, repeat
    #   3. If answer → break and return
    previous_queries = ""
    context          = ""
    all_hits: list[TrackHit] = []
    final_result: dict       = {}
    attempts_done            = 0

    # Build ScorerAgent callable once (only when planner_enabled).
    # create_scorer_agent() returns an async callable, not an Agent — it creates
    # a fresh Agent per invocation so the formatted system_prompt is injected
    # correctly (PydanticAI doesn't support per-call system_prompt overrides).
    scorer_fn = None
    if req.planner_enabled:
        scorer_deps = SearchDeps(
            service=service,
            collection_name=req.collection_name,
            llm_base_url=llm_kw.get("base_url"),
            llm_model=llm_kw.get("model"),
        )
        scorer_fn = create_scorer_agent(scorer_deps)

    for attempt in range(1, NUM_ATTEMPTS + 1):
        attempts_done = attempt
        action = None
        queries: list[dict] | None = None

        # ── Step 1: Decide action ──────────────────────────────────────

        # First attempt with planner queries: skip LLM, use planner output
        if attempt == 1 and planner_queries:
            queries = planner_queries
            action = "search"
            logger.info("[chat] Attempt %d: using planner queries", attempt)

        # ScorerAgent evaluates accumulated context
        elif scorer_fn is not None:
            try:
                filled = SCORER_PROMPT.format(
                    query=req.message,
                    context=context or "(empty — no results yet)",
                    previous_queries=previous_queries or "(none)",
                    active_filters=str(planner_filters) if planner_filters else "(none)",
                    attempt_number=attempt,
                )
                logger.debug("[chat] Scorer prompt (attempt %d): context has %d chars",
                             attempt, len(context))
                score = await scorer_fn(req.message, filled)

                action = score.action
                if action == "search" and score.queries:
                    queries = [{"query": q.query} for q in score.queries]
                    logger.info("[chat] Scorer (attempt %d): action=search, queries=%s",
                                attempt, [q["query"] for q in queries])
                elif action in ("answer", "final_answer"):
                    final_result = {
                        "action":     action,
                        "confidence": score.confidence,
                        "song":       score.song,
                        "artist":     score.artist,
                        "message":    score.message,
                    }
                    logger.info("[chat] Scorer (attempt %d): action=%s", attempt, action)
            except Exception as exc:
                logger.error(f"[chat] ScorerAgent error on attempt {attempt}: {exc}",
                             exc_info=True)
                scorer_fn = None  # disable scorer for subsequent attempts, fall back to old LLM

        # Old behavior: DEVELOPER_PROMPT + ask_llm
        if action is None:
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
                action = result.get("action", "answer")
                logger.info("[chat] LLM (attempt %d): action=%s", attempt, action)

                # Extract queries if action is search
                if action == "search" and result.get("queries"):
                    queries = result["queries"]

            except Exception as exc:
                logger.error(f"[chat] LLM error on attempt {attempt}: {exc}", exc_info=True)
                final_result = {
                    "action":     "answer",
                    "confidence": "low",
                    "song":       None,
                    "artist":     None,
                    "message":    f"LLM error on attempt {attempt}: {exc}",
                }
                break  # LLM is broken, no point continuing

        # ── Step 2: Execute decision ───────────────────────────────────

        if action == "search" and queries:
            # In auto_mode the initial classifier already fixed the search type —
            # propagate effective_mode so ScorerAgent's per-query types cannot
            # override the top-level classification.
            forced_mode = req.mode if not req.auto_mode else effective_mode
            # Planner queries on attempt 1 are already CLAP-optimised — skip re-rephrase
            use_skip_rephrase = (attempt == 1 and bool(planner_queries))
            new_pq, new_ctx, new_hits = await _run_searches(
                queries, service,
                collection_name=req.collection_name,
                forced_mode=forced_mode,
                llm_kw=llm_kw,
                skip_rephrase=use_skip_rephrase,
            )

            logger.info(
                "[chat/loop] attempt=%d  mode=%s  queries=%s  hits=%d  tracks=%s",
                attempt,
                forced_mode,
                [q["query"] for q in queries],
                len(new_hits),
                [(h.track.artist, h.track.title) for h in new_hits],
            )

            if new_pq:
                previous_queries = (previous_queries + "\n" + new_pq).strip()
            if new_ctx:
                context = (context + "\n\n" + new_ctx).strip()
            if new_hits:
                all_hits = _merge_hits(all_hits, new_hits)

            # Last attempt and still "search" — force exit with fallback message
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
                break

        elif action in ("answer", "final_answer"):
            # scorer path → final_result already populated (lines above)
            # old LLM path → result dict is in scope, copy it explicitly
            if not final_result:
                try:
                    final_result = result  # defined in old LLM branch above  # noqa: F821
                except NameError:
                    final_result = {}
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
