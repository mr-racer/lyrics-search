"""LLM leg of the gems pipeline: candidate verification + artist aliases.

Small local model (~12B) — so both prompts are maximally rigid: strict JSON,
"not sure -> no", never invent. The verifier is a pydantic_ai agent with a
web_search tool hard-capped at 2 calls per candidate (closure counter, same
scheme as ``llm_web_search._create_agent``; ``UsageLimits`` backstops a model
that ignores the refusal). Verdicts are cached forever by the caller
(``gem_resolution_cache``) — one LLM price per unique candidate.

Every public function swallows its errors: LLM being down must degrade the
pipeline to gazetteer-only, never crash the indexing task.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

_MAX_WEB_SEARCHES = 2
_MAX_LLM_REQUESTS = 6  # backstop only; the graceful path never gets near it

_VERIFY_INSTRUCTIONS = """You verify ONE candidate discovery in song lyrics for a music player.
You get: a quote line from the lyrics, the song's artist / title / year, the
candidate string, its claimed type, and (for namedrop/songref) the target it
supposedly refers to.

Claim types:
- namedrop: the quote genuinely mentions the real music artist <target> as a
  person. NOT a coincidental common word, NOT a section header like "[Verse: X]",
  NOT the performer of this very song.
- popculture: the quote references the fictional character or movie <candidate>.
- songref: the quote references the song or album <target> as a musical work.
- capsule: the quote uses <candidate> as the literal obsolete device/service
  (pager, payphone, MySpace...), not a metaphor or another sense of the word.

Rules:
1. Judge ONLY from the quote and facts you are certain about. Not sure -> "no".
   A wrong "yes" pollutes the app forever; a wrong "no" costs nothing.
2. Never invent context that is not in the quote. Never guess what the artist
   "probably meant".
3. web_search: at most 2 calls, ONLY if the quote alone cannot settle it AND a
   search realistically can (e.g. whether "Timbo" is a known alias of Timbaland).
   If after 2 searches you are still unsure -> "no".
4. Common words in their ordinary meaning are "no": "the game" is not The Game,
   "bizarre" is not Bizarre, "cars" is not the band Cars.
5. Reply with STRICT JSON and NOTHING else — no markdown, no reasoning text:
   {"verdict":"yes"|"no","canonical":"<proper name or empty>","why":"<one short sentence>"}
   If verdict is "no", canonical MUST be ""."""

_ALIAS_SYSTEM = """List widely known aliases of the music artist named by the user:
nicknames, real name, former stage names. Only aliases that are commonly used
to refer to THIS artist and are verifiable common knowledge. If you are not
sure the artist has well-known aliases, return an empty list. Never invent,
never include other artists, never include the name itself.
Reply with STRICT JSON and nothing else: {"aliases":["...","..."]}"""


def parse_strict_json(raw: str) -> Optional[dict]:
    """First ``{...}`` block of ``raw`` as a dict, else None.

    A 12B model occasionally wraps the JSON in prose or a code fence despite
    instructions — tolerate that, but nothing fancier."""
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _no(reason: str) -> dict:
    return {"verdict": "no", "canonical": "", "why": reason}


async def verify_candidate(
    *,
    kind: str,
    candidate: str,
    quote: str,
    track_artist: str,
    track_title: str,
    year: Optional[int] = None,
    target: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict:
    """One verification verdict; any failure is a "no" (silence over noise)."""
    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic_ai.usage import UsageLimits

        from app.services.llm_client import _get_client, resolve_model
        from app.services.llm_web_search import smart_web_search

        provider = OpenAIProvider(openai_client=_get_client(base_url))
        pydantic_model = OpenAIModel(resolve_model(model_name), provider=provider)
        # instructions=, not system_prompt= — survives any run shape.
        agent: Agent = Agent(pydantic_model, instructions=_VERIFY_INSTRUCTIONS)

        search_calls = 0

        @agent.tool_plain
        def web_search(query: str) -> str:  # noqa: F841
            """Search the web. HARD LIMIT: at most 2 calls per candidate.

            Args:
                query: Search query in English.
            """
            nonlocal search_calls
            if search_calls >= _MAX_WEB_SEARCHES:
                return (
                    "SEARCH LIMIT REACHED: do NOT call web_search again. "
                    "Answer NOW from what you have; if unsure, verdict is no."
                )
            search_calls += 1
            logger.info(
                "[gems-verify] web_search (%d/%d): %r",
                search_calls, _MAX_WEB_SEARCHES, query,
            )
            return smart_web_search(query, fetch_content=False)

        parts = [
            f"Claim type: {kind}",
            f"Candidate: {candidate}",
        ]
        if target:
            parts.append(f"Target it supposedly refers to: {target}")
        parts += [
            f'Song: "{track_title}" by {track_artist}' + (f" ({year})" if year else ""),
            f"Quote line from the lyrics:\n{quote}",
            "",
            "JSON:",
        ]
        result = await agent.run(
            "\n".join(parts),
            usage_limits=UsageLimits(request_limit=_MAX_LLM_REQUESTS),
        )
        parsed = parse_strict_json(result.output or "")
        if not parsed or parsed.get("verdict") not in ("yes", "no"):
            return _no("unparseable LLM reply")
        if parsed["verdict"] == "no":
            return _no(str(parsed.get("why") or ""))
        canonical = str(parsed.get("canonical") or "").strip()
        if not canonical:
            return _no("yes without canonical — treated as no")
        return {
            "verdict": "yes",
            "canonical": canonical,
            "why": str(parsed.get("why") or ""),
        }
    except Exception as e:
        logger.warning("[gems-verify] agent error for %r (%s): %s", candidate, kind, e)
        return _no(f"agent error: {e}")


async def fetch_artist_aliases(
    artist_name: str,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Optional[List[str]]:
    """Well-known aliases of ``artist_name``.

    Returns ``[]`` for a genuine "no known aliases" (the caller may cache the
    marker forever) but ``None`` when the LLM call itself failed — so a down
    LLM never permanently buries an artist's aliases behind the marker."""
    try:
        from app.services.llm_client import ask_llm

        raw = await ask_llm(
            f'Artist: "{artist_name}"\nJSON:',
            system_prompt=_ALIAS_SYSTEM,
            temperature=0.0,
            base_url=base_url,
            model=model_name,
        )
        parsed = parse_strict_json(raw or "")
        if not parsed:
            return []
        aliases = parsed.get("aliases")
        if not isinstance(aliases, list):
            return []
        out = []
        seen = set()
        for a in aliases:
            if not isinstance(a, str):
                continue
            a = a.strip()
            key = a.lower()
            # the model sometimes echoes the artist's own name — drop it
            if not a or key == artist_name.lower() or key in seen or len(a) < 2:
                continue
            seen.add(key)
            out.append(a)
        return out[:10]
    except Exception as e:
        logger.warning("[gems-aliases] LLM error for %r: %s", artist_name, e)
        return None
