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

import asyncio
import json
import re
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.domain.models import (
    ChatRequest, PlaylistBuilderRequest, SearchFilters,
    TrackChatRequest, TrackChatResponse, TrackHit, User,
)
from app.api.dependencies import get_current_user
from app.api.helpers import derive_collection_for_user
from app.api.sse_utils import sse_data
from app.services.agents import (
    PLANNER_PROMPT,
    SCORER_PROMPT,
    VALIDATOR_PROMPT,
    _AUDIO_ANSWER_PROMPT,
    _CLAP_REPHRASE_SYSTEM_PROMPT,
    create_scorer_agent,
    create_validator_agent,
)
from app.services.agent_deps import SearchDeps
from app.services.llm_client import ask_llm
from app.services.playlist_agent.agent import run_playlist_agent
from app.services.playlist_agent.resolver import CatalogAdapter
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
5. Any artist name in "artist" or "message" must appear EXACTLY as given in <context> — never translated, transliterated, localized, or grammatically declined.

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


def _extract_lyrics_for_song(context: str, song: str, artist: str) -> str:
    """Extract the lyrics excerpt for a specific song from the accumulated context string.

    Context blocks are formatted as:
        • Title — Artist [Album] (Year)
          Genre: ...
          Lyrics: ...

    Returns up to 300 chars of the Lyrics line for the matched song, or "".
    """
    song_l = song.lower()
    artist_l = artist.lower()
    lines = context.splitlines()
    in_block = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("•"):
            header = stripped[1:].strip().lower()
            in_block = song_l in header and artist_l in header
        if in_block and stripped.startswith("Lyrics:"):
            excerpt = stripped[len("Lyrics:"):].strip()
            return excerpt[:300]
    return ""


def _match_best_hit(
    hits: list[TrackHit], song: str | None, artist: str | None,
) -> TrackHit | None:
    """Resolve the LLM's named song (free-text title + artist) to a real hit.

    The scorer copies ``song``/``artist`` out of the <context> block, which is
    built from these same hits — so a normalized title (+ artist) comparison
    maps the LLM's pick back to a concrete, playable TrackHit. Returns None when
    no song was named or nothing matches, so the caller falls back to the
    top-N retrieval list.
    """
    if not song:
        return None

    def _norm(s: str | None) -> str:
        return " ".join((s or "").strip().casefold().split())

    target_song = _norm(song)
    target_artist = _norm(artist)
    if not target_song:
        return None

    # Pass 1: title AND artist both match — most specific.
    for h in hits:
        if _norm(h.track.title) == target_song and (
            not target_artist or _norm(h.track.artist) == target_artist
        ):
            return h
    # Pass 2: title matches exactly — artist may be formatted differently
    # (feat. credits, "&" vs "and", a different romanization).
    for h in hits:
        if _norm(h.track.title) == target_song:
            return h
    # Pass 3: one title contains the other — tolerates "(Remastered)" / "- Live"
    # style suffixes the LLM may have dropped or added.
    for h in hits:
        ht = _norm(h.track.title)
        if ht and (ht in target_song or target_song in ht):
            return h
    return None


def _quoted_fragments(message: str) -> list[str]:
    """Lyric quotes the LLM embedded in its answer, longest first.

    The answer message usually cites the matched line in «…», “…” or "…" —
    that citation is the ground truth for the highlight, unlike the executed
    search query the initial matched_line heuristic used. Fragments shorter
    than 3 words are dropped (song titles, single-word emphasis).
    """
    frags = re.findall(r"[«“\"]([^«»“”\"]{10,200})[»”\"]", message or "")
    frags = [f.strip() for f in frags if len(f.split()) >= 3]
    frags.sort(key=len, reverse=True)
    return frags


def _pick_matched_line(lyrics: str, query: str) -> str | None:
    """Pick the lyric line with the most word overlap with the executed query.

    The server knows the *executed* queries (planner-generated), which the
    client never sees — so this beats the frontend's user-message heuristic.
    Pure python, no LLM/embedding calls. Returns None when nothing overlaps.
    """
    terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}
    if not terms or not lyrics:
        return None

    def candidates():
        for line in lyrics.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) <= 160:
                yield stripped
            else:
                # Excerpt with mangled/absent line breaks — a whole-blob
                # "line" would be useless for highlighting; score sliding
                # ~12-word windows instead.
                words = stripped.split()
                for i in range(0, len(words), 6):
                    chunk = " ".join(words[i:i + 12])
                    if chunk:
                        yield chunk

    best: str | None = None
    best_score = 0
    for cand in candidates():
        words = set(re.findall(r"\w+", cand.lower()))
        score = len(terms & words)
        if score > best_score:
            best, best_score = cand, score
    return best


async def _run_searches(
    llm_queries: list[dict],
    service,
    collection_name: str | None = None,
    forced_mode: str | None = None,
    filters: SearchFilters | None = None,
    text_model: str | None = None,
    llm_kw: dict | None = None,
    skip_rephrase: bool = False,
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
    active_filters_log = {k: v for k, v in (filters.model_dump() if filters else {}).items() if v}

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
        logger.info(
            "[chat/search] query=%r  mode=%s  filters=%s",
            query_text, mode, active_filters_log or "(none)",
        )

        try:
            round_hits = await service.search(
                query=search_query, mode=mode, limit=SEARCH_LIMIT,
                collection_name=collection_name,
                filters=filters,
                text_model=text_model,
            )
            for hit in round_hits:
                key = (hit.track.title.lower(), hit.track.artist.lower())
                if key not in seen:
                    seen.add(key)
                    if mode in ("text", "hybrid") and hit.lyrics and hit.matched_line is None:
                        hit.matched_line = _pick_matched_line(hit.lyrics, query_text)
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


_ANSWER_LANG_NAMES = {"ru": "Russian", "en": "English"}


def _answer_lang_directive(lang: str | None) -> str:
    """Prefix that forces the user-facing answer language; '' when lang unknown
    (then existing prompt behaviour — match the query language — is preserved)."""
    name = _ANSWER_LANG_NAMES.get((lang or "").strip().lower())
    if not name:
        return ""
    return (
        f'LANGUAGE REQUIREMENT: Write the user-facing "message" strictly in {name}, '
        f"regardless of the language of the query, context, or facts.\n\n"
    )


def _history_preamble(history) -> str:
    """Compact transcript of the prior conversation, prepended to the prompts that
    interpret the user's intent (planner / scorer / answer). Lets follow-ups like
    "а побыстрее?" or "у того же артиста" resolve against the previous turn.
    Returns '' when there is no prior history."""
    if not history:
        return ""
    lines: list[str] = []
    for m in history:
        who = "Assistant" if getattr(m, "role", "") == "assistant" else "User"
        text = (getattr(m, "content", None) or "").strip()
        if text:
            lines.append(f"{who}: {text}")
    if not lines:
        return ""
    return (
        "PRIOR CONVERSATION (oldest first; the user's new message continues it):\n"
        + "\n".join(lines)
        + "\n\n"
    )


# ─── Agent step events (SSE) ──────────────────────────────────────────────────
# The streaming endpoint drains these as the agentic loop progresses. Each event
# carries a ready-to-render `human` string in the request language, so the UI does
# no phrasing of its own — it just animates the label. `emit` is None for the
# non-streaming endpoint, which makes every _emit() call a cheap no-op.

EmitFn = Callable[[dict], Awaitable[None]]

_MODE_LABEL_RU = {"text": "по тексту", "audio": "по звучанию", "hybrid": "по тексту и звуку"}
_MODE_LABEL_EN = {"text": "by lyrics", "audio": "by sound", "hybrid": "by lyrics and sound"}


def _plural_ru(n: int) -> str:
    """Russian plural suffix for «совпадени·е/·я/·й»."""
    r = abs(n) % 100
    if 11 <= r <= 14:
        return "й"
    d = r % 10
    if d == 1:
        return "е"
    if 2 <= d <= 4:
        return "я"
    return "й"


def _human(event_type: str, lang: str | None, **kw) -> str:
    """Plain-language, per-language label for a step event."""
    ru = (lang or "en").startswith("ru")
    if event_type == "classify":
        mode = kw.get("mode", "hybrid")
        return (f"Понял запрос — ищу {_MODE_LABEL_RU.get(mode, 'по тексту и звуку')}"
                if ru else f"Got it — searching {_MODE_LABEL_EN.get(mode, 'by lyrics and sound')}")
    if event_type == "plan":
        return "Составил план поиска" if ru else "Planned the search"
    if event_type == "search":
        found = kw.get("found", 0)
        if ru:
            return f"Ищу в библиотеке… нашёл {found} совпадени{_plural_ru(found)}"
        return f"Searching the library… found {found} match{'' if found == 1 else 'es'}"
    if event_type == "validate":
        if kw.get("valid", True):
            return "Проверил — трек подходит" if ru else "Checked — the track fits"
        return "Проверяю точность совпадения" if ru else "Double-checking the match"
    if event_type == "retry":
        return "Уточняю запрос и пробую снова" if ru else "Refining the query and retrying"
    if event_type == "answer":
        return "Готовлю ответ" if ru else "Preparing the answer"
    return ""


async def _emit(emit: EmitFn | None, event_type: str, lang: str | None, **fields) -> None:
    if emit is None:
        return
    await emit({"type": event_type, "human": _human(event_type, lang, **fields), **fields})


async def _run_chat_core(
    req: ChatRequest,
    request: Request,
    current_user: User,
    emit: EmitFn | None = None,
) -> dict:
    """Agentic LLM-driven music search.

    Response shape
    --------------
    {
      "message":        str,           # LLM's conversational reply
      "song":           str | null,    # identified song title (if confident)
      "artist":         str | null,    # identified artist
      "confidence":     "high"|"medium"|"low",
      "best_hit":       TrackHit|null,  # the track the LLM named, resolved to a real hit
      "hits":           [TrackHit…],   # all retrieved tracks (fallback list for the UI)
      "attempts":       int,           # how many LLM calls were made
      "classification": dict,          # result of call-1 (empty if prompt unset)
    }

    When ``best_hit`` is set the UI shows just that one track (the LLM's pick)
    instead of the top-N retrieval list; it falls back to ``hits`` otherwise.
    """
    service = request.app.state.search_service
    if service is None:
        return {
            "message":        "Search service unavailable — is Qdrant running?",
            "song":           None,
            "artist":         None,
            "confidence":     "low",
            "best_hit":       None,
            "hits":           [],
            "attempts":       0,
            "classification": {},
        }

    # Phase D-soft: derive collection from JWT user; ignore client-supplied value.
    derived = derive_collection_for_user(current_user)

    # Resolve which text model this collection was indexed with, so chat's
    # searches hit the matching Qdrant vector_name (a collection indexed with
    # qwen must not be queried with the default jina vectors).
    #
    # Phase B: resolve the NAME only and thread it through as `text_model=` to
    # service.search (which forwards it to the now-stateless engine). We MUST NOT
    # mutate db.lyrics_db._model / _vector_name / _vector_dim here — that global
    # mutation is exactly the cross-account race Phase B removed from search.py.
    # The agent/planner paths pass collection_name, and
    # SearchService._resolve_model_name falls back to the collection's pinned
    # model when text_model is None, so they resolve correctly without mutation.
    resolved_text_model: str | None = None
    if derived:
        from app.resources.metadata_db import MetadataDB
        try:
            MetadataDB.init()
            persisted = MetadataDB.get_collection_text_model(derived)
            if persisted:
                resolved_text_model = persisted
                logger.info("[chat] text model resolved: %s", persisted)
        except Exception as exc:
            logger.warning("[chat] text model lookup failed (non-fatal): %s", exc)

    # Common kwargs forwarded to every ask_llm call
    llm_kw: dict[str, Any] = {
        "base_url":   (req.llm_base_url or "").strip() or None,
        "model":      (req.llm_model or "").strip() or None,
        "extra_body": {"enable_thinking": False},
        "temperature": 0.3,
    }

    # Prior conversation, prepended to intent-interpreting prompts so follow-ups
    # ("а побыстрее?", "у того же артиста") resolve against the previous turn.
    pre = _history_preamble(req.history)

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
                collection_name=derived,
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
            plan = await planner(req.message, pre + filled_prompt)

            planner_classification = {
                "type": plan.query_type,
                "reasoning": "planner",
            }

            logger.info(
                "[chat/planner/1] action=%s  type=%s  filter_lookup=%s  queries=%s",
                plan.action,
                plan.query_type,
                plan.filter_lookup or "(none)",
                [q.query for q in plan.queries],
            )

            # Resolve filters if action == "request_filter"
            if plan.action == "request_filter" and plan.filter_lookup:
                logger.info("[chat/planner/resolve] resolving filter_lookup=%s", plan.filter_lookup)
                resolved = await deps.resolve_filters(plan.filter_lookup)
                logger.info("[chat/planner/resolve] result=%s", resolved or "(no matches in DB)")

                # Always re-run so planner knows the resolution outcome
                # (even when resolved is empty — planner can then omit filters gracefully)
                resolved_filters_str = json.dumps(resolved, ensure_ascii=False) if resolved else "{}"
                search_filter_query_str = json.dumps(
                    {k: v for k, v in plan.filter_lookup.items() if v},
                    ensure_ascii=False,
                )
                filled_prompt2 = PLANNER_PROMPT.format(
                    query=req.message,
                    previous_queries=previous_queries_str,
                    resolved_filters=resolved_filters_str,
                    search_filter_query=search_filter_query_str,
                )
                plan = await planner(req.message, pre + filled_prompt2)

                logger.info(
                    "[chat/planner/2] action=%s  filters=%s  queries=%s",
                    plan.action,
                    plan.filters.model_dump(exclude_none=True) if plan.filters else "(none)",
                    [q.query for q in plan.queries],
                )

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
            logger.error("[chat] Planner error (falling back to old behavior): %s", exc, exc_info=True)
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

    await _emit(emit, "classify", req.lang, mode=effective_mode)
    if planner_queries:
        await _emit(emit, "plan", req.lang, queries=[q["query"] for q in planner_queries])

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
                    collection_name=derived,
                    filters=audio_filters,
                    text_model=resolved_text_model,
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

        await _emit(emit, "search", req.lang, attempt=1, queries=[], found=len(merged))

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
                    system_prompt=_answer_lang_directive(req.lang) + answer_prompt,
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
                "best_hit":       best.model_dump(),
                "hits":           [h.model_dump() for h in top5],
                "attempts":       1,
                "classification": classification,
            }
        else:
            return {
                "message":        "Не удалось найти треков по описанию звука. Попробуй уточнить настроение, инструменты, или стиль вокала.",
                "song":           None,
                "artist":         None,
                "best_hit":       None,
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

    # Build ScorerAgent and ValidatorAgent callables once (only when planner_enabled).
    # Both use create_*_agent() which returns an async callable that creates a fresh
    # Agent per invocation so formatted system_prompt is injected correctly.
    scorer_fn = None
    validator_fn = None
    if req.planner_enabled:
        scorer_deps = SearchDeps(
            service=service,
            collection_name=derived,
            llm_base_url=llm_kw.get("base_url"),
            llm_model=llm_kw.get("model"),
        )
        scorer_fn = create_scorer_agent(scorer_deps)
        validator_fn = create_validator_agent(scorer_deps)
        logger.info("[chat] planner+scorer+validator enabled")

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
                score = await scorer_fn(req.message, _answer_lang_directive(req.lang) + pre + filled)

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
                    system_prompt=_answer_lang_directive(req.lang) + pre + filled,
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
            filters_obj = SearchFilters(**planner_filters) if planner_filters else None
            new_pq, new_ctx, new_hits = await _run_searches(
                queries, service,
                collection_name=derived,
                forced_mode=forced_mode,
                filters=filters_obj,
                text_model=resolved_text_model,
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

            await _emit(
                emit, "search", req.lang,
                attempt=attempt,
                queries=[q["query"] for q in queries],
                found=len(new_hits),
            )

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

            # ── Validator (only with Planner, only when a song is proposed) ──
            if validator_fn and final_result.get("song"):
                proposed_song = final_result.get("song", "")
                proposed_artist = final_result.get("artist", "")
                # Lyrics for the proposed song: resolve against the actual hits
                # first (fuzzy title match — the LLM often reformats the title,
                # so the strict header parse of the context string misses and
                # the validator would wrongly see "(no lyrics available)").
                lyrics_excerpt = ""
                val_hit = _match_best_hit(all_hits, proposed_song, proposed_artist)
                if val_hit is not None and val_hit.lyrics:
                    lyrics_excerpt = val_hit.lyrics[:300]
                if not lyrics_excerpt:
                    lyrics_excerpt = _extract_lyrics_for_song(context, proposed_song, proposed_artist)
                filled_val = VALIDATOR_PROMPT.format(
                    query=req.message,
                    song=proposed_song,
                    artist=proposed_artist,
                    lyrics_excerpt=lyrics_excerpt or "(no lyrics available)",
                    previous_queries=previous_queries or "(none)",
                    active_filters=str(planner_filters) if planner_filters else "(none)",
                )
                try:
                    val = await validator_fn(req.message, filled_val)
                    logger.info(
                        "[chat/validator] valid=%s  reason=%r  new_queries=%s",
                        val.valid, val.reason,
                        [q.query for q in (val.queries or [])],
                    )

                    await _emit(emit, "validate", req.lang, valid=val.valid)

                    if not val.valid and val.queries and attempt < NUM_ATTEMPTS:
                        # Rejected — inject new queries and continue the loop
                        queries = [{"query": q.query} for q in val.queries]
                        action = "search"
                        final_result = {}
                        await _emit(emit, "retry", req.lang, attempt=attempt + 1)
                        continue

                    if not val.valid and attempt >= NUM_ATTEMPTS:
                        # Last attempt — accept the rejected answer with low confidence
                        final_result["confidence"] = "low"
                        logger.info("[chat/validator] last attempt — accepting rejected answer with low confidence")

                except Exception as exc:
                    logger.warning("[chat/validator] error (non-fatal, accepting answer): %s", exc)

            break

    # Sort retrieved hits by score descending
    all_hits.sort(key=lambda h: h.score, reverse=True)

    # Resolve the LLM's named song back to a concrete hit so the UI can show
    # exactly that track instead of the top-N retrieval list.
    best_hit = _match_best_hit(
        all_hits, final_result.get("song"), final_result.get("artist"),
    )

    # Re-anchor the highlight on the line the LLM actually cited in its answer.
    # matched_line was picked at retrieval time from the *executed query*, which
    # regularly disagrees with the line the answer quotes.
    if best_hit is not None and best_hit.lyrics:
        for frag in _quoted_fragments(final_result.get("message", "")):
            line = _pick_matched_line(best_hit.lyrics, frag)
            if line:
                best_hit.matched_line = line
                break

    # No song named by the LLM = nothing found. Ship an empty hit list instead
    # of the raw top-N retrieval tail, so the UI doesn't present near-misses as
    # an answer.
    found = bool(final_result.get("song"))

    return {
        "message":        final_result.get("message", ""),
        "song":           final_result.get("song"),
        "artist":         final_result.get("artist"),
        "confidence":     final_result.get("confidence", "low"),
        "best_hit":       best_hit.model_dump() if best_hit else None,
        "hits":           [h.model_dump() for h in all_hits[:10]] if found else [],
        "attempts":       attempts_done,
        "classification": classification,
    }


@router.post("/")
async def chat(
    req: ChatRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Non-streaming agentic search — runs the loop to completion and returns
    the final payload. Kept for backward compatibility and as the client's
    fallback when the SSE stream is unavailable. Contract unchanged."""
    return await _run_chat_core(req, request, current_user, emit=None)


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Streaming variant of ``POST /chat/``.

    Runs the same agentic loop, emitting typed step events (classify → plan →
    search → validate → retry) over SSE as the agent works, then a final
    ``answer`` event carrying the exact payload the non-streaming endpoint
    returns. The frontend animates each step's ``human`` label in real time.
    """

    async def event_source() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        DONE = {"__done__": True}

        async def emit(event: dict) -> None:
            queue.put_nowait(event)

        async def run() -> None:
            try:
                result = await _run_chat_core(req, request, current_user, emit=emit)
                queue.put_nowait({"type": "answer", "human": _human("answer", req.lang), **result})
            except Exception as exc:  # surface a terminal error frame, never hang
                logger.error("[chat/stream] core error: %s", exc, exc_info=True)
                queue.put_nowait({
                    "type": "error",
                    "human": ("Ошибка" if (req.lang or "en").startswith("ru") else "Error"),
                    "message": str(exc),
                })
            finally:
                queue.put_nowait(DONE)

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # keep the connection warm during long LLM waits
                    continue
                if item is DONE:
                    break
                yield sse_data(item)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Track Chat ───────────────────────────────────────────────────────────────
# Single-track conversational chat with optional web-search fallback.
# Powers the AIChatDrawer + InlineLyricExplain UIs in PlayerSection.
# Separate from the agentic search loop above — no Planner/Scorer/Validator.


@router.post("/track-chat", response_model=TrackChatResponse)
async def track_chat(
    req: TrackChatRequest,
    current_user: User = Depends(get_current_user),
) -> TrackChatResponse:
    """Single-track conversational chat (drawer or per-line explain)."""
    derived = derive_collection_for_user(current_user)
    req = req.model_copy(update={"collection_name": derived})

    if req.mode == "lyric_explain" and not req.selected_line:
        raise HTTPException(
            status_code=400,
            detail="selected_line is required for mode='lyric_explain'",
        )
    try:
        from app.services.track_chat_service import answer_track_chat
        return await answer_track_chat(req)
    except ValueError as exc:
        # answer_track_chat re-raises with same message — defensive
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("[track_chat] error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI error: {str(exc)[:200]}")


@router.post("/track-chat/stream")
async def track_chat_stream(
    req: TrackChatRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Streaming variant of ``POST /chat/track-chat``.

    Same orchestration, but emits humanized status frames over SSE while the
    agent works — ``{"type":"status","stage":"thinking"|"web_search"|"reading"}``
    (web_search frames carry the ``query``) — then a terminal ``answer`` frame
    with the exact TrackChatResponse payload. The drawer's activity ticker
    animates these in real time; the client falls back to the non-streaming
    endpoint if the stream can't open.
    """
    derived = derive_collection_for_user(current_user)
    req = req.model_copy(update={"collection_name": derived})

    if req.mode == "lyric_explain" and not req.selected_line:
        raise HTTPException(
            status_code=400,
            detail="selected_line is required for mode='lyric_explain'",
        )

    async def event_source() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        DONE = {"__done__": True}

        def emit(event: dict) -> None:
            queue.put_nowait(event)

        async def run() -> None:
            try:
                from app.services.track_chat_service import answer_track_chat
                emit({"type": "status", "stage": "thinking"})
                res = await answer_track_chat(req, on_event=emit)
                queue.put_nowait({
                    "type": "answer",
                    "message": res.message,
                    "web_search_used": res.web_search_used,
                })
            except Exception as exc:  # surface a terminal error frame, never hang
                logger.error("[track_chat/stream] error: %s", exc, exc_info=True)
                queue.put_nowait({"type": "error", "message": str(exc)[:200]})
            finally:
                queue.put_nowait(DONE)

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # keep the connection warm during long LLM waits
                    continue
                if item is DONE:
                    break
                yield sse_data(item)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Playlist Builder ─────────────────────────────────────────────────────────
# Agentic playlist drafting from a natural-language prompt (artist hits or a
# film/game soundtrack): web_search + fetch_filters + get_songs, then a
# PlaylistDraft. The agent only PROPOSES — the client persists via the normal
# playlists CRUD after the user confirms.


def _build_playlist_deps(request: Request, current_user: User):
    """Build (deps, catalog) for the playlist agent, or None if the search
    service is unavailable."""
    service = request.app.state.search_service
    if service is None:
        return None
    derived = derive_collection_for_user(current_user)
    deps = SearchDeps(service=service, collection_name=derived)
    catalog = CatalogAdapter(service.lyrics_db.qdrant_client, derived)
    return deps, catalog


def _tracks_for(draft, state) -> list:
    """Assemble the resolved-track preview from the agent's ``state``, keeping
    only ids the agent actually matched (drops any hallucinated track_id)."""
    resolved = state.get("resolved", {})
    return [
        {"track_id": tid, **resolved[tid]}
        for tid in draft.track_ids
        if tid in resolved
    ]


@router.post("/playlist-builder")
async def playlist_builder(
    req: PlaylistBuilderRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Draft a playlist from a prompt (non-streaming)."""
    built = _build_playlist_deps(request, current_user)
    if built is None:
        raise HTTPException(status_code=503, detail="Search service unavailable — is Qdrant running?")
    deps, catalog = built
    state: dict = {}
    try:
        draft = await run_playlist_agent(
            req.prompt, req.lang, deps, catalog, state,
            llm_base_url=req.llm_base_url, llm_model=req.llm_model,
        )
    except Exception as exc:
        logger.error("[playlist_builder] error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI error: {str(exc)[:200]}")
    return {"draft": draft.model_dump(), "tracks": _tracks_for(draft, state)}


@router.post("/playlist-builder/stream")
async def playlist_builder_stream(
    req: PlaylistBuilderRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Streaming variant: emits ``{"type":"status","stage":...}`` frames
    ("thinking"|"web_search"|"matching") while the agent works, then a terminal
    ``{"type":"result","draft":...,"tracks":...}`` frame."""
    built = _build_playlist_deps(request, current_user)
    if built is None:
        raise HTTPException(status_code=503, detail="Search service unavailable — is Qdrant running?")
    deps, catalog = built

    async def event_source() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        DONE = {"__done__": True}
        state: dict = {}

        def on_status(stage: str) -> None:
            queue.put_nowait({"type": "status", "stage": stage})

        async def run() -> None:
            try:
                draft = await run_playlist_agent(
                    req.prompt, req.lang, deps, catalog, state,
                    llm_base_url=req.llm_base_url, llm_model=req.llm_model,
                    on_status=on_status,
                )
                queue.put_nowait({
                    "type": "result",
                    "draft": draft.model_dump(),
                    "tracks": _tracks_for(draft, state),
                })
            except Exception as exc:
                logger.error("[playlist_builder/stream] error: %s", exc, exc_info=True)
                queue.put_nowait({"type": "error", "message": str(exc)[:200]})
            finally:
                queue.put_nowait(DONE)

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if item is DONE:
                    break
                yield sse_data(item)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
