"""Turning one LLM call into a plan code is willing to execute.

The model reads the sentence — that is the one thing it is better at than any
rule we could write. Everything it hands back is then checked, and anything that
cannot be traced to the user's own words is dropped.

The rules, and what each is defending against:

* **intent** must be one of four words, or the run stops and asks. A wrong branch
  costs the user a whole pipeline; a question costs a second.
* **audio_search needs a real description of sound.** The model reaches for that
  branch on "лучшие песни Sade" too, and CLAP has no vector for "best". When the
  style does not survive validation the intent is downgraded to ``playlist`` by
  code, not by asking again.
* **era** is parsed into a pair of integers and sanity-bounded. "1990-1999" is a
  filter; "the nineties" is a string that silently matches nothing.
* **style** must appear in the user's own sentence, folded. This is the rule that
  makes "extract AS IS, do not invent" checkable instead of hopeful — the model
  reliably offers "energetic" for a request that never mentioned mood.
* **the artist never enters ``ce_query`` on a lyrics search.** A name in the
  cross-encoder pair pulls the score towards every text by that artist instead of
  towards the line, and the model puts it there about half the time despite being
  told not to. Stripped in code.
* **queries** are deduplicated and capped. A model that returns the same query
  twice would otherwise spend two searches on one.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Optional

from app.services.assistant.config import AgentConfig
from app.services.assistant.contracts import Abbreviation, Filters, Plan
from app.services.assistant.llm import LLMClient, as_int, as_str, as_str_list
from app.services.assistant.prompts import NEXT_QUERIES_SYSTEM, PLAN_SYSTEM
from app.services.text_normalize import fold

logger = logging.getLogger(__name__)

INTENTS = ("general", "playlist", "lyrics_search", "audio_search")
# Branches that read the web. The other two search the user's own library, so a
# missing web_queries list is not a problem there.
WEB_INTENTS = ("general", "playlist")

EARLIEST_YEAR = 1900
_YEAR_RE = re.compile(r"(1[89]\d{2}|20\d{2})")
_DECADE_RE = re.compile(r"\b((?:19|20)\d)0\s*(?:s|х|е|ые|годов|годы)\b", re.I)


def _current_year() -> int:
    return datetime.date.today().year


def parse_era(raw, *, user_text: str = "") -> Optional[tuple]:
    """A year range out of whatever the model returned.

    Accepts "1990-1999", "2000s", a bare year, a two-element list, or a dict with
    from/to. Anything outside 1900..next year is thrown away: a model that answers
    "0-9999" is not describing a decade.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        lo, hi = (as_int(raw.get("from") or raw.get("start")),
                  as_int(raw.get("to") or raw.get("end")))
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
        # boundary, not a point. The wording lives in the USER's sentence, not in
        # the model's answer, so that is where it is looked for.
        low = fold(user_text)
        if any(w in low for w in ("after", "since", "posle", "nachinaya",
                                  "начиная", "после")):
            return _bounded(year, _current_year() + 1)
        if any(w in low for w in ("before", "until", "до ", "ранее")):
            return _bounded(EARLIEST_YEAR, year)
        return _bounded(year, year)
    return None


def _bounded(lo: Optional[int], hi: Optional[int]) -> Optional[tuple]:
    if lo is None and hi is None:
        return None
    lo = lo if lo is not None else EARLIEST_YEAR
    hi = hi if hi is not None else _current_year() + 1
    if lo > hi:
        lo, hi = hi, lo
    if hi < EARLIEST_YEAR or lo > _current_year() + 1:
        return None
    return max(lo, EARLIEST_YEAR), min(hi, _current_year() + 1)


# Words about how well known a song is, not about how it sounds. They pass the
# "did the user write it?" test and mean nothing to CLAP: there is no audio
# signature of being popular, so a style of "популярные" would filter the library
# against noise. Russian is matched by stem because of the cases; English by
# whole word.
_NOT_SOUND_STEMS = ("популярн", "известн", "знаменит", "прославлен", "раскруч",
                    "хит", "лучш", "топов", "главн", "недооцен", "малоизвестн",
                    "редк", "культов")
_NOT_SOUND_WORDS = frozenset({
    "popular", "famous", "best", "greatest", "top", "hit", "hits", "biggest",
    "essential", "iconic", "renowned", "underrated", "obscure", "rare", "known",
    "wellknown", "bestselling", "charting",
})


def _is_sound_word(folded_word: str) -> bool:
    """False for words describing fame or chart position rather than sound."""
    if folded_word in _NOT_SOUND_WORDS:
        return False
    return not any(folded_word.startswith(stem) for stem in _NOT_SOUND_STEMS)


def validate_style(raw, user_text: str) -> Optional[str]:
    """Keep the style only if the user wrote it AND it describes SOUND.

    Two independent gates, guarding two different mistakes:

    * the model inventing a mood nobody asked for — caught by requiring every word
      to appear in the user's own sentence;
    * the model answering with popularity — "популярные", "хиты", "best" — which
      does pass the first gate, because the user really did type it. It is still
      not a style: this field is handed to CLAP, and being popular has no sound.

    Word-level, so "популярные клубные хиты" keeps "клубные" and drops the rest.
    """
    text = as_str(raw, 80)
    if not text:
        return None
    haystack = set(fold(user_text).split())
    # Comparison is on the folded form, but what comes back is the ORIGINAL word.
    # fold() decomposes and strips combining marks, so "спокойные" folds to
    # "спокоиные" — returning that would hand CLAP a misspelling.
    words = [w for w in text.split() if fold(w)]
    if not words:
        return None

    def usable(word: str) -> bool:
        parts = fold(word).split()
        return (bool(parts)
                and all(p in haystack for p in parts)       # the user wrote it
                and all(_is_sound_word(p) for p in parts))  # …about the sound

    kept = [w for w in words if usable(w)]
    if len(kept) == len(words):
        return text
    if kept:
        logger.info("[planner] style %r trimmed to %r — the rest is either not "
                    "the user's words or not about sound", text, " ".join(kept))
        return " ".join(kept)
    logger.info("[planner] style %r dropped — not a description of sound the "
                "user gave", text)
    return None


def strip_artist(text: str, artist: Optional[str]) -> str:
    """Remove the artist's name from a cross-encoder query.

    The prompt asks for this and the model complies about half the time. It
    matters more than it looks: the pair the reranker scores is (this sentence,
    a lyric), so a name in it makes every text by that artist score highly and
    the actual line stops standing out. The name is doing its work as a FILTER,
    where it is exact and free.
    """
    if not text or not artist:
        return text
    out = text
    for token in {artist, *artist.split()}:
        if len(fold(token)) < 3:
            continue           # "of", "MF" — too short to remove safely
        out = re.sub(re.escape(token), " ", out, flags=re.I)
    out = " ".join(out.split())
    if out != text:
        logger.info("[planner] removed %r from the rerank query", artist)
    return out or text


def _dedupe_queries(queries: list, *, limit: int = 2) -> list:
    out: list = []
    seen: set = set()
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

    Code does this, not the model: a query that lost the title comes back with the
    artist's whole discography, and the model forgets the quotes roughly one time
    in three.
    """
    if not work:
        return query
    if f'"{work}"' in query:
        return query
    if work.lower() in query.lower():
        # Present but unquoted — quote it in place so the engine treats it as one
        # phrase.
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
    def __init__(self, llm: LLMClient, config: Optional[AgentConfig] = None,
                 sink=None, *, audio_available: bool = True):
        self.llm = llm
        self.cfg = config or AgentConfig()
        self.sink = sink
        # CLAP can be switched off per instance. When it is, the audio branch is
        # not merely worse — it cannot run — so the intent is rewritten to the
        # nearest branch that can.
        self.audio_available = audio_available
        # Why the last plan attempt produced nothing, in words a human can act on.
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
        artist = as_str(raw.get("artist"), 120) or None
        filters = Filters(
            era=parse_era(raw.get("era"), user_text=message),
            style=validate_style(raw.get("style"), message),
            work=work,
            artist=artist,
            song=as_str(raw.get("song"), 160) or None,
            count=_valid_count(as_int(raw.get("count"))),
        )

        intent = self.settle_intent(intent, filters)

        queries = _dedupe_queries(as_str_list(raw.get("web_queries"), limit=4))
        if not queries and intent in WEB_INTENTS:
            # Nothing usable came back, but the user's sentence is always a
            # workable query — an empty search list would end the run over a
            # formatting slip.
            queries = [message.strip()]
        queries = [quote_work(q, work) for q in queries]

        ce_query = as_str(raw.get("ce_query"), 400) or message.strip()
        lyrics_query = as_str(raw.get("lyrics_query"), 400)
        if intent == "lyrics_search":
            lyrics_query = lyrics_query or message.strip()
            ce_query = strip_artist(ce_query, filters.artist)

        plan = Plan(intent=intent, filters=filters, web_queries=queries,
                    ce_query=ce_query, lyrics_query=lyrics_query,
                    abbreviation=abbreviation,
                    rationale=as_str(raw.get("rationale"), 200))
        if self.sink is not None:
            self.sink.put("plan", intent=intent, era=filters.era,
                          style=filters.style, work=filters.work,
                          artist=filters.artist, queries=plan.web_queries)
        logger.info("[planner] intent=%s era=%s style=%r work=%r artist=%r "
                    "queries=%s", intent, filters.era, filters.style,
                    filters.work, filters.artist, queries)
        return plan

    def settle_intent(self, intent: str, filters: Filters) -> str:
        """Rewrite an intent the run cannot actually serve.

        Two cases, both decided by code because both are about what exists rather
        than about what the user meant:

        * ``audio_search`` with no surviving style. The model picks that branch
          for "лучшие песни Sade" as readily as for "спокойные песни Sade", and
          the style validator is what tells them apart — CLAP has no vector for
          "best". Without a sound description there is nothing to search by.
        * ``audio_search`` while CLAP is unavailable.

        Both fall back to ``playlist``, which answers the same request off the
        web: worse for the user, but an answer.
        """
        if intent != "audio_search":
            return intent
        if not self.audio_available:
            logger.info("[planner] audio search is unavailable on this instance "
                        "— answering %r from the web instead", filters.artist)
            return "playlist"
        if not filters.style:
            logger.info("[planner] audio_search without a description of sound "
                        "— treating it as a playlist request")
            return "playlist"
        return intent

    async def next_queries(self, *, message: str, context: str,
                           used: list) -> tuple:
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
        return (queries, as_str(raw.get("ce_query"), 400),
                as_str(raw.get("missing"), 200))


def plan_for_focus(fact: str, subject, message: str = "") -> Plan:
    """The plan for "explain THIS statement", built without asking the model.

    Nothing here is a judgement. The intent is known (it is an explanation), the
    subject is pinned by id, and the statement itself is the best possible
    cross-encoder query — it is the exact text a useful passage has to be about.

    The web queries are assembled by CODE from the artist, the title and the
    statement. Letting the model phrase them loses the performer about half the
    time, and an explanation of the wrong artist's sample reads exactly like an
    explanation of the right one.
    """
    artist = (getattr(subject, "artist_name", None) or "").strip()
    song = (getattr(subject, "song_title", None) or "").strip()
    head = " ".join(p for p in (artist, song) if p)
    snippet = " ".join((fact or "").split())[:160]

    queries = []
    if head:
        queries.append(f"{head} {snippet}".strip())
        queries.append(f"{head} meaning explained")
    else:
        queries.append(snippet)
    queries = _dedupe_queries(queries)

    return Plan(intent="general",
                filters=Filters(artist=artist or None, song=song or None),
                web_queries=queries, ce_query=snippet or message.strip(),
                rationale=f"Объясняю факт про {head}" if head else "Объясняю факт")


def plan_for_followup(question: str, subject) -> Plan:
    """The plan for a follow-up chip, built without asking the model.

    Nothing here is a judgement either. The branch is known (the chip carries
    it), the subject is pinned by id, and the question itself is the best
    cross-encoder query available — it was WRITTEN by a model reading the very
    material this run starts from, so it is specific in a way a rephrasing
    would not be.

    Spending a planner call here bought nothing: the intent it returned was
    overwritten by the chip's own a few lines later, and its filters were
    re-derived from a sentence that has no era, no style and no work in it.
    """
    artist = (getattr(subject, "artist_name", None) or "").strip()
    song = (getattr(subject, "song_title", None) or "").strip()
    head = " ".join(p for p in (artist, song) if p)
    text = " ".join((question or "").split())[:160]

    queries = [f"{head} {text}".strip() if head else text]
    if head:
        queries.append(f"{head} {' '.join(text.split()[:6])}".strip())
    queries = _dedupe_queries([q for q in queries if q])

    return Plan(intent="general",
                filters=Filters(artist=artist or None, song=song or None),
                web_queries=queries, ce_query=text or (question or "").strip(),
                rationale=f"Уточняю про {head}" if head else "Уточняю")


def _valid_count(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return value if 1 <= value <= 100 else None
