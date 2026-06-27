"""PydanticAI agents for music search.

Three agents, each with a single responsibility:

- **PlannerAgent** — classifies the user's query, extracts filters
  (artist/album/genre/era), and generates 2-3 search queries.

- **ScorerAgent** — evaluates search results (context) against the
  user's query and decides whether to answer or search again.

- **AudioAgent** — fast-path for audio/vibe queries: CLAP rephrase
  → 3× audio search → conversational answer.

All agents are created via factory functions that accept a
``SearchDeps`` instance for dependency injection.
"""

from __future__ import annotations

import logging

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel as OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.domain.models import BaseQueryItem, ScoreResult, SearchPlan, ValidatorResult
from app.services.agent_deps import SearchDeps
from app.services.llm_client import _get_client, resolve_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (from _WIP_llm_agents.py, kept as-is)
# ---------------------------------------------------------------------------

PLANNER_PROMPT: str = """
You are a music search planner. Your job is to analyze the user's query and prepare a search plan.

AVAILABLE FILTER KEYS:
- artist: performer name
- album: album name
- genre: music genre
- year_range: decade range (e.g. "1990-1999", "2000-2009")

INPUTS:
<user_query>{query}</user_query>
<previous_queries>{previous_queries}</previous_queries>

<resolved_filters>{resolved_filters}</resolved_filters>
<search_filter_query>{search_filter_query}</search_filter_query>

RULES:
1. If the user explicitly mentions an artist, album, genre, or time period — extract it as a potential filter. Always use ENGLISH as a filter query.
2. If <resolved_filters> is empty AND filters are needed → set action="request_filter" to look up valid DB values.
3. If <resolved_filters> is NOT empty → select the best matching filter value from it (fuzzy match allowed, e.g. "канье уест" → "Kanye West"). If no match — set filters=null.
4. You may only request filters ONCE per session. If <search_filter_query> already has content — skip request_filter.
5. Generate 2–3 search queries in English (3–10 words each), ordered from most to least specific. IMPORTANT — these queries search the user's LOCAL LIBRARY OF SONG LYRICS, NOT a web search engine. So:
   - Write the actual words, themes, images, or mood the lyrics themselves would use — e.g. "broken heart leaving in the rain", "city lights lonely highway at night". Not a description of a search.
   - NEVER include meta words like "lyrics", "song", "track", "music", "find", "search", or "about" — they never appear inside lyrics and only weaken the match.
   - If an artist, album, or genre is already set in <resolved_filters>, do NOT repeat it in the query text — the filter already narrows the results. Put ONLY the relevant content of the user's request (the key words, themes, or mood they described) into the query.

OUTPUT FORMAT:
{{
  "action": "request_filter" | "search",
  "query_type": "text" | "audio" | "hybrid",
  "filters": {{
    "artist": "..." | null,
    "album": "..." | null,
    "genre": "..." | null,
    "year_range": "YYYY-YYYY" | null
  }} | null,
  "filter_lookup": {{
    "artist": "raw user input to resolve" | null,
    "album": "..." | null,
    "genre": "..." | null,
    "year_range": "YYYY-YYYY" | null
  }} | null,
  "queries": [{{"query": "..."}}],
  "search_mode": "CONSERVATIVE" | "AGGRESSIVE"
}}

NOTE: query_type is the single classification for all queries. Do NOT add a per-query "type" field — it is determined once here and used throughout the session.

NOTES:
- filter_lookup is only used when action="request_filter". It contains the raw unresolved user terms to look up in DB.
- filters contains only resolved, confirmed values from <resolved_filters> or null.
- search_mode: use CONSERVATIVE on first attempt, AGGRESSIVE if previous_queries is not empty.
""".strip()

SCORER_PROMPT: str = """
You are a music search assistant. Evaluate search results and answer the user.

INPUTS:
<user_query>{query}</user_query>
<context>{context}</context>
<previous_queries>{previous_queries}</previous_queries>
<active_filters>{active_filters}</active_filters>
<attempt>{attempt_number}</attempt>

TASK:
- Evaluate <context> for a match to <user_query>.
- If active_filters are set, ensure the result satisfies them.
- If a confident match is found → action="answer".
- If no match and attempt < 3 → action="search" with new queries (AGGRESSIVE mode).
- If attempt >= 3 → action="final_answer" (best guess or admit failure).

SEARCH MODES:
- CONSERVATIVE (attempt 1): close to user's literal words
- AGGRESSIVE (attempt 2+): use imagery, metaphors, synonyms, related themes

OUTPUT FORMAT:
{{
  "action": "answer" | "search" | "final_answer",
  "confidence": "high" | "medium" | "low",
  "song": "Title" | null,
  "artist": "Artist" | null,
  "filters": {{active filters, pass-through unchanged}} | null,
  "queries": [{{"query": "..."}}] | null,
  "message": "Conversational reply to user"
}}

NOTE: Do NOT include a "type" field in queries — the search mode is already fixed from the initial classification and must not change during the session.

CONSTRAINTS:
1. ONLY use <context> for answers. Never use internal knowledge.
2. message is ALWAYS present and human-friendly.
3. On final_answer: give best guess even if confidence is low; explain uncertainty in message.
""".strip()

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _create_pydantic_model(
    base_url: str | None = None,
    model_name: str | None = None,
) -> OpenAIModel:
    """Create a pydantic_ai OpenAIModel from the project's client cache."""
    resolved_model = resolve_model(model_name)
    openai_client = _get_client(base_url)
    provider = OpenAIProvider(openai_client=openai_client)
    return OpenAIModel(resolved_model, provider=provider)


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------


def create_planner_agent(deps: SearchDeps):
    """Return async callable: (query, filled_prompt) -> SearchPlan.

    Creates a fresh Agent per call so the formatted system_prompt is injected
    correctly — same pattern as create_scorer_agent and create_validator_agent.
    Agent.run() no longer accepts system_prompt (renamed to instructions in
    PydanticAI 1.x), so we pass it through the Agent constructor instead.

    Usage::
        planner = create_planner_agent(deps)
        plan: SearchPlan = await planner(query, filled_prompt)
    """
    model = _create_pydantic_model(deps.llm_base_url, deps.llm_model)

    async def run_planner(query: str, filled_prompt: str) -> SearchPlan:
        agent = Agent(model, output_type=SearchPlan, system_prompt=filled_prompt)
        result = await agent.run(query)
        return result.output

    return run_planner


# ---------------------------------------------------------------------------
# ScorerAgent
# ---------------------------------------------------------------------------


def create_scorer_agent(deps: SearchDeps):
    """Return an async callable that runs the ScorerAgent with a formatted prompt.

    Uses ask_llm + parse_json instead of PydanticAI Agent to avoid tool-calling
    overhead — local LLMs (Gemma, Qwen, etc.) often produce malformed tool-call
    syntax that PydanticAI cannot parse. Plain JSON generation is far more reliable.

    Usage::
        scorer = create_scorer_agent(deps)
        score: ScoreResult = await scorer(query, filled_prompt)
    """
    async def run_scorer(query: str, filled_prompt: str) -> ScoreResult:
        from app.services.llm_client import ask_llm

        raw = await ask_llm(
            query,
            system_prompt=filled_prompt,
            parse_json=True,
            base_url=deps.llm_base_url,
            model=deps.llm_model,
            extra_body={"enable_thinking": False},
            temperature=0.3,
        )
        if not isinstance(raw, dict):
            return ScoreResult(action="search", confidence="medium", message="")

        raw_queries = raw.get("queries") or []
        queries: list[BaseQueryItem] | None = None
        if isinstance(raw_queries, list) and raw_queries:
            queries = [
                BaseQueryItem(query=q["query"])
                for q in raw_queries
                if isinstance(q, dict) and q.get("query")
            ]
        return ScoreResult(
            action=raw.get("action") or "search",
            confidence=raw.get("confidence") or "medium",
            song=raw.get("song"),
            artist=raw.get("artist"),
            filters=raw.get("filters"),
            queries=queries,
            message=raw.get("message") or "",
        )

    return run_scorer


# ---------------------------------------------------------------------------
# AudioAgent helpers
# ---------------------------------------------------------------------------

# Re-exported from chat.py so the audio path can use them without circular imports.
_AUDIO_ANSWER_PROMPT: str = """
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

_CLAP_REPHRASE_SYSTEM_PROMPT: str = """
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


# ---------------------------------------------------------------------------
# ValidatorAgent
# ---------------------------------------------------------------------------

VALIDATOR_PROMPT: str = """
Does "{song}" by {artist} match the user's query? Reply with JSON only.

User: {query}
Lyrics: {lyrics_excerpt}
Previous searches: {previous_queries}
Filters: {active_filters}

valid=true if lyrics contain the requested words/themes or match the described mood.
valid=false if excerpt is empty, unrelated, or clearly wrong song.
If previous_queries are already exhaustive — set valid=true to avoid endless loop.

{{"valid": true|false, "reason": "one sentence", "queries": [{{"query": "..."}}]|null}}
queries only when valid=false: 1-2 new English queries not in previous searches.
""".strip()


def create_validator_agent(deps: SearchDeps):
    """Return async callable: (query, filled_prompt) -> ValidatorResult.

    Uses ask_llm + parse_json instead of PydanticAI Agent to avoid tool-calling
    overhead — local LLMs handle plain JSON much faster than the tool-call schema.
    """
    async def run_validator(query: str, filled_prompt: str) -> ValidatorResult:
        from app.services.llm_client import ask_llm

        raw = await ask_llm(
            query,
            system_prompt=filled_prompt,
            parse_json=True,
            base_url=deps.llm_base_url,
            model=deps.llm_model,
            extra_body={"enable_thinking": False},
            temperature=0.3,
        )
        if not isinstance(raw, dict):
            return ValidatorResult(valid=True, reason="parse error — accepting answer")

        valid = bool(raw.get("valid", True))
        reason = str(raw.get("reason", ""))
        raw_queries = raw.get("queries") or []
        queries: list[BaseQueryItem] | None = None
        if isinstance(raw_queries, list) and raw_queries:
            queries = [
                BaseQueryItem(query=q["query"])
                for q in raw_queries
                if isinstance(q, dict) and q.get("query")
            ]
        return ValidatorResult(valid=valid, reason=reason, queries=queries or None)

    return run_validator


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "create_planner_agent",
    "create_scorer_agent",
    "create_validator_agent",
    "PLANNER_PROMPT",
    "SCORER_PROMPT",
    "VALIDATOR_PROMPT",
    "_AUDIO_ANSWER_PROMPT",
    "_CLAP_REPHRASE_SYSTEM_PROMPT",
]
