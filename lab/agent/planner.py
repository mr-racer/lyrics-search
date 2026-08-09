"""Turning one LLM call into a plan code is willing to execute.

The model reads the sentence — that is the one thing it is better at than any
rule we could write. Everything it hands back is then checked, and anything
that cannot be traced to the user's own words is dropped.

The rules, and what each is defending against:

* **intent** must be one of two words, or the run stops and asks. A wrong
  branch costs the user a whole pipeline; a question costs a second.
* **era** is parsed into a pair of integers and sanity-bounded. "1990-1999" is
  a filter; "the nineties" is a string that silently matches nothing.
* **style** must appear in the user's own sentence, folded. This is the rule
  that makes "extract AS IS, do not invent" checkable instead of hopeful — the
  model reliably offers "energetic" for a request that never mentioned mood.
* **work / artist / song** are trimmed and length-capped; ``work`` additionally
  drives the abbreviation flow.
* **queries** are deduplicated and capped. A model that returns the same query
  twice would otherwise spend two searches on one.
* **allowed_tools** is written by CODE from the intent. The model never gets a
  say in what it is allowed to reach for.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Optional

from lab.agent.llm import LLMClient, as_int, as_str, as_str_list
from lab.agent.models import Abbreviation, Filters, Plan
from lab.agent.prompts import NEXT_QUERIES_SYSTEM, PLAN_SYSTEM
from lab.websearch_lab import fold

logger = logging.getLogger(__name__)

INTENTS = ("general", "playlist")
# Tools each branch is allowed to use. Assigned by code — a general question
# has no business resolving artist names against the library, and a playlist
# request must not be answered out of the fact store alone.
TOOLS = {
    "general": ["web_search", "wikipedia", "library_facts"],
    "playlist": ["web_search", "wikipedia", "apple_music", "fandom",
                 "library_catalog"],
}

EARLIEST_YEAR = 1900
_YEAR_RE = re.compile(r"(1[89]\d{2}|20\d{2})")
_DECADE_RE = re.compile(r"\b((?:19|20)\d)0\s*(?:s|х|е|ые|годов|годы)\b", re.I)


def _current_year() -> int:
    return datetime.date.today().year


def parse_era(raw, *, user_text: str = "") -> Optional[tuple[int, int]]:
    """A year range out of whatever the model returned.

    Accepts "1990-1999", "2000s", a bare year, a two-element list, or a dict
    with from/to. Anything outside 1900..next year is thrown away: a model that
    answers "0-9999" is not describing a decade.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        lo, hi = as_int(raw.get("from") or raw.get("start")), \
                 as_int(raw.get("to") or raw.get("end"))
        return _bounded(lo, hi)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return _bounded(as_int(raw[0]), as_int(raw[1]))

    text = as_str(raw, 40)
    if not text:
        return None

    years = _YEAR_RE.findall(text)
    if len(years) >= 2:
        return _bounded(int(years[0]), int(years[1]))

    decade = _DECADE_RE.search(text)
    if decade:
        base = int(decade.group(1)) * 10
        return _bounded(base, base + 9)

    if len(years) == 1:
        year = int(years[0])
        # A single year with an open-ended word around it ("after 2020") is a
        # boundary, not a point. The wording lives in the USER's sentence, not
        # in the model's answer, so that is where it is looked for.
        low = fold(user_text)
        if any(w in low for w in ("after", "since", "posle", "nachinaya",
                                  "начиная", "после")):
            return _bounded(year, _current_year() + 1)
        if any(w in low for w in ("before", "until", "до ", "ранее")):
            return _bounded(EARLIEST_YEAR, year)
        return _bounded(year, year)
    return None


def _bounded(lo: Optional[int], hi: Optional[int]) -> Optional[tuple[int, int]]:
    if lo is None and hi is None:
        return None
    lo = lo if lo is not None else EARLIEST_YEAR
    hi = hi if hi is not None else _current_year() + 1
    if lo > hi:
        lo, hi = hi, lo
    if hi < EARLIEST_YEAR or lo > _current_year() + 1:
        return None
    return max(lo, EARLIEST_YEAR), min(hi, _current_year() + 1)


def validate_style(raw, user_text: str) -> Optional[str]:
    """Keep the style only if the user actually wrote it.

    Word-level containment on the folded text, so "спокойные хиты" yields
    "спокойные" but a helpfully invented "energetic" yields nothing. Checking
    word by word rather than as one string lets "спокойные, мелодичные" survive
    when the user wrote both words separately.
    """
    text = as_str(raw, 80)
    if not text:
        return None
    haystack = set(fold(user_text).split())
    # Comparison is on the folded form, but what comes back is the ORIGINAL
    # word. fold() decomposes and strips combining marks, so "спокойные" folds
    # to "спокоиные" — returning that would hand the CLAP branch a misspelling.
    words = [w for w in text.split() if fold(w)]
    if not words:
        return None
    kept = [w for w in words if all(f in haystack for f in fold(w).split())]
    if len(kept) == len(words):
        return text
    if kept:
        logger.info("[planner] style %r trimmed to %r — the rest is not in the "
                    "user's words", text, " ".join(kept))
        return " ".join(kept)
    logger.info("[planner] style %r dropped — not in the user's words", text)
    return None


def _dedupe_queries(queries: list[str], *, limit: int = 2) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        norm = " ".join(q.lower().split())
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(q.strip())
        if len(out) >= limit:
            break
    return out


def quote_work(query: str, work: Optional[str]) -> str:
    """Pin the film/game title inside a query, in quotes.

    Code does this, not the model: a query that lost the title comes back with
    the artist's whole discography, and the model forgets the quotes roughly
    one time in three.
    """
    if not work:
        return query
    if f'"{work}"' in query:
        return query
    if work.lower() in query.lower():
        # Present but unquoted — quote it in place so the engine treats it as
        # one phrase.
        pattern = re.compile(re.escape(work), re.I)
        return pattern.sub(f'"{work}"', query, count=1)
    return f'"{work}" {query}'.strip()


def parse_abbreviation(raw) -> Optional[Abbreviation]:
    if not isinstance(raw, dict):
        return None
    original = as_str(raw.get("raw"), 80)
    expansion = as_str(raw.get("expansion"), 160)
    if not original or not expansion or fold(original) == fold(expansion):
        return None
    try:
        confidence = float(raw.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return Abbreviation(raw=original, expansion=expansion,
                        confidence=max(0.0, min(1.0, confidence)))


class Planner:
    def __init__(self, llm: LLMClient, config=None, sink=None):
        from lab.agent.config import AgentConfig

        self.llm = llm
        self.cfg = config or AgentConfig()
        self.sink = sink
        # Why the last plan attempt produced nothing, in words a human can act
        # on. Read by the caller so the run's notes say "the LLM is
        # unreachable" instead of "nothing usable".
        self.last_failure: Optional[str] = None

    async def plan(self, message: str) -> Optional[Plan]:
        """One LLM call, then every field checked. None means "ask the user"."""
        self.last_failure = None
        raw = await self.llm.ask_json(
            [{"role": "system", "content": PLAN_SYSTEM},
             {"role": "user", "content": message}],
            required=("intent",))
        if raw is None:
            transport = getattr(self.llm, "last_error", None)
            self.last_failure = (
                f"the LLM did not answer — {transport}" if transport else
                "the LLM answered, but not with a JSON object carrying "
                f"\"intent\" (last reply: {getattr(self.llm, 'last_raw', '')[:200]!r})")
            logger.warning("[planner] %s", self.last_failure)
            if self.sink is not None:
                self.sink.put("plan_failed", why=self.last_failure)
            return None

        plan = self.validate(raw, message)
        if plan is None:
            self.last_failure = (
                f"the plan had no usable intent (model said "
                f"{as_str(raw.get('intent'), 40)!r}, expected one of {INTENTS})")
            logger.warning("[planner] %s", self.last_failure)
            if self.sink is not None:
                self.sink.put("plan_failed", why=self.last_failure)
        return plan

    def validate(self, raw: dict, message: str) -> Optional[Plan]:
        intent = as_str(raw.get("intent"), 20).lower()
        if intent not in INTENTS:
            logger.warning("[planner] unusable intent %r", intent)
            return None

        work = as_str(raw.get("work"), 160) or None
        abbreviation = parse_abbreviation(raw.get("abbreviation"))
        filters = Filters(
            era=parse_era(raw.get("era"), user_text=message),
            style=validate_style(raw.get("style"), message),
            work=work,
            artist=as_str(raw.get("artist"), 120) or None,
            song=as_str(raw.get("song"), 160) or None,
            count=_valid_count(as_int(raw.get("count"))),
        )

        queries = _dedupe_queries(as_str_list(raw.get("web_queries"), limit=4))
        if not queries:
            # Nothing usable came back, but the user's sentence is always a
            # workable query — an empty search list would end the run for a
            # formatting slip.
            queries = [message.strip()]
        queries = [quote_work(q, work) for q in queries]

        ce_query = as_str(raw.get("ce_query"), 400) or message.strip()

        plan = Plan(intent=intent, filters=filters, web_queries=queries,
                    ce_query=ce_query, allowed_tools=list(TOOLS[intent]),
                    abbreviation=abbreviation,
                    rationale=as_str(raw.get("rationale"), 200))
        if self.sink is not None:
            self.sink.put("plan", intent=intent, era=filters.era,
                          style=filters.style, work=filters.work,
                          queries=plan.web_queries)
        logger.info("[planner] intent=%s era=%s style=%r work=%r queries=%s",
                    intent, filters.era, filters.style, filters.work, queries)
        return plan

    async def next_queries(self, *, message: str, context: str,
                           used: list[str]) -> tuple[list[str], str, str]:
        """Queries for another iteration: ``(web_queries, ce_query, missing)``."""
        raw = await self.llm.ask_json([
            {"role": "system", "content": NEXT_QUERIES_SYSTEM},
            {"role": "user", "content":
                f"Original request: {message}\n\n"
                f"Queries already used (never repeat these):\n"
                + "\n".join(f"- {q}" for q in used)
                + f"\n\nPassages found so far:\n{context}"},
        ], required=("web_queries",))
        if raw is None:
            return [], "", ""
        queries = _dedupe_queries(as_str_list(raw.get("web_queries"), limit=4))
        used_norm = {" ".join(q.lower().split()) for q in used}
        queries = [q for q in queries
                   if " ".join(q.lower().split()) not in used_norm]
        return queries, as_str(raw.get("ce_query"), 400), as_str(raw.get("missing"), 200)


def _valid_count(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return value if 1 <= value <= 100 else None
