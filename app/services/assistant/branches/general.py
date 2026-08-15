"""Questions about an artist, a song or an incident — and "explain THIS line".

Control flow is fixed. The model is called at three known points — plan (before
this branch), answer, next queries — and everything between those calls is code.
There is no loop the model can steer and no place where "should I keep going?" is
settled by prose.

The one judgement genuinely left to the model is whether the question came out
answered. Code cannot read a passage and know that it explains the origin of a
stage name. But code CAN see the cross-encoder probabilities the model never
sees, and it holds the veto in both directions:

* model says "enough" while the best chunk scored below the threshold → another
  iteration runs anyway;
* model says "not enough" while the budget is spent, or the last round brought no
  new material → the run ends and answers with what it has.

Grounding: a numbered pack, an answer that must cite numbers, and code that
throws the answer away when the citations do not check out. An unnumbered
paragraph physically cannot reach the caller.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.services.assistant.branches.base import WebBranch
from app.services.assistant.contracts import (Evidence, GeneralResult, Plan,
                                              Subject)
from app.services.assistant.llm import as_str, as_str_list
from app.services.assistant.planner import quote_work
from app.services.assistant.prompts import ANSWER_SYSTEM, EXPLAIN_SYSTEM

logger = logging.getLogger(__name__)


class GeneralBranch(WebBranch):
    """Library facts + the open web, answered with verified citations."""

    async def run(self, message: str, plan: Plan, *,
                  focus_fact: Optional[str] = None,
                  subject: Optional[Subject] = None) -> GeneralResult:
        facts = await self.agent.subject_facts(plan, message, plan.ce_query,
                                               subject=subject)
        subject = subject or self.agent.last_subject

        queries, ce_query = plan.web_queries, plan.ce_query
        answer, used, evidence = "", [], []
        follow_ups: list = []
        iterations = 0
        notes: list = []
        parachuted = False

        while iterations < self.cfg.general_max_iterations:
            iterations += 1
            self.sink.put("iteration", n=iterations, queries=queries)

            structured_pages, prose_pages = await self.gather(
                plan, queries, ce_query, structured=False)
            new_chunks = self.index(structured_pages + prose_pages)
            chunks = self.best_chunks(ce_query)

            evidence = _pack(facts, chunks)

            # The parachute. Deployed only while falling: the open web has been
            # searched, its pages read, and not one passage cleared the bar. Once
            # per run — a second deployment would be another minute spent on the
            # same failure.
            if not evidence and self.cfg.search_reddit and not parachuted:
                parachuted = True
                if await self._reddit_rescue(ce_query):
                    chunks = self.best_chunks(ce_query)
                    evidence = _pack(facts, chunks)
                notes.append(f"reddit parachute: {len(evidence)} in the pack")

            if evidence:
                with self.timings.span("llm.answer"):
                    answer, used, sufficient, missing, follow_ups = await self._ask(
                        message, evidence, focus_fact=focus_fact, subject=subject)
                best_prob = max((e.ce_prob or 0.0 for e in evidence), default=0.0)
            else:
                # Nothing cleared the threshold. That is a reason to search AGAIN,
                # not to stop: the pages were wrong, the question was not. Stopping
                # here is what a high chunk threshold turns into if the empty case
                # is treated as an answer — one unlucky pair of queries and the run
                # is over.
                answer, used, sufficient = "", [], False
                missing = "nothing on those pages was about the question"
                best_prob = 0.0
                notes.append(f"iteration {iterations}: nothing cleared the chunk "
                             f"threshold")

            stop, why = self._should_stop(iterations=iterations,
                                          sufficient=sufficient,
                                          best_prob=best_prob,
                                          new_chunks=new_chunks)
            self.sink.put("verdict", sufficient=sufficient,
                          best_prob=round(best_prob, 3), stop=stop, why=why)
            notes.append(why)
            if stop:
                break

            with self.timings.span("llm.next_queries"):
                queries, next_ce, model_missing = \
                    await self.agent.planner.next_queries(
                        message=message,
                        context=(_render(evidence, limit=6) if evidence
                                 else "(nothing found yet)"),
                        used=self.used_queries)
            if not queries:
                notes.append("no fresh queries — stopping")
                break
            queries = [quote_work(q, plan.filters.work) for q in queries]
            ce_query = next_ce or (missing or model_missing or ce_query)

        grounded = bool(answer and used)
        if not grounded and evidence and not focus_fact:
            # In focus mode an ungrounded answer is not replaced by a listing:
            # "here is everything known about this artist" is not an explanation
            # of the line the user tapped, and printing it would look like one.
            answer = _fallback_answer(evidence, self.cfg.lang)
            notes.append("ungrounded answer discarded — showing the sources")
        self.sink.put("answer", grounded=grounded, evidence=len(evidence),
                      used=used)
        return GeneralResult(
            answer=answer, evidence=evidence, used=used, grounded=grounded,
            iterations=iterations, subject=subject, focus_fact=focus_fact,
            explained=(bool(used) if focus_fact else None),
            follow_ups=follow_ups, notes=notes)

    async def _reddit_rescue(self, ce_query: str) -> int:
        """Read Reddit threads about the question. Returns new chunks indexed.

        The last resort of this branch: code decides WHEN, the model is never
        asked whether to try. The query is the cross-encoder query rather than a
        web query, because by this point the web queries are the ones that already
        failed.

        Failure here is ordinary and expected — a blocked IP gets a challenge page
        the fetcher recognises, and a cooldown that has not elapsed skips the read
        entirely — so this returns a count and never raises.
        """
        self.sink.put("reddit_rescue", query=ce_query)
        with self.timings.span("search.reddit"):
            hits = await asyncio.to_thread(self.sources.reddit, ce_query)
        if not hits:
            logger.info("[general] reddit parachute: nothing found for %r", ce_query)
            return 0
        with self.timings.span("fetch"):
            pages = await self.fetcher.fetch_many(
                hits, limit=self.cfg.reddit_max_pages)
        fresh = self.index(pages)
        logger.info("[general] reddit parachute: %d hits, %d pages read, "
                    "%d new chunks", len(hits), len(pages), fresh)
        self.sink.put("reddit_rescue_done", hits=len(hits), pages=len(pages),
                      chunks=fresh)
        return fresh

    def _should_stop(self, *, iterations: int, sufficient: bool,
                     best_prob: float, new_chunks: int) -> tuple:
        """The veto. Code decides the budget, the model decides the meaning."""
        if iterations >= self.cfg.general_max_iterations:
            return True, "iteration budget spent"
        if new_chunks == 0 and iterations > 1:
            return True, "last round brought nothing new"
        if sufficient and best_prob >= self.cfg.weak_context_prob:
            return True, "model says answered and the context is strong"
        if sufficient:
            return False, (f"model says answered but the best chunk is only "
                           f"p={best_prob:.2f} — searching again")
        return False, "model says the question is not covered"

    async def _ask(self, message: str, evidence: list, *,
                   focus_fact: Optional[str],
                   subject: Optional[Subject]) -> tuple:
        """One answer call. Returns ``(answer, used, sufficient, missing, follow_ups)``."""
        lang = _lang_name(self.cfg.lang)
        if focus_fact:
            who = _subject_line(subject) or "this track"
            system = EXPLAIN_SYSTEM.format(lang=lang, subject=who)
            user = (f"Statement: {focus_fact}\n\n"
                    f"Material:\n{_render(evidence)}")
            required = ("answer",)
        else:
            system = ANSWER_SYSTEM.format(lang=lang)
            user = f"Question: {message}\n\nMaterial:\n{_render(evidence)}"
            required = ("answer",)

        raw = await self.agent.llm.ask_json(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}], required=required)
        if raw is None:
            return "", [], False, "", []

        answer = as_str(raw.get("answer"), 4000)
        used = _valid_citations(raw.get("used"), len(evidence))
        follow_ups = as_str_list(raw.get("follow_ups"), limit=3, item_limit=120)
        if focus_fact:
            # An explanation is one shot: there is no "search more" for a
            # statement the user tapped, so whatever came back is the verdict.
            sufficient, missing = True, ""
        else:
            sufficient = bool(raw.get("sufficient", False))
            missing = as_str(raw.get("missing"), 200)

        # The citation gate. An answer with no usable numbers is thrown away whole
        # — a paragraph nobody can trace is worse than no paragraph.
        if answer and not used:
            logger.info("[general] answer discarded: no valid citations")
            return "", [], sufficient, missing, follow_ups
        return answer, used, sufficient, missing, follow_ups


# ── helpers ──────────────────────────────────────────────────────────────────


def _subject_line(subject: Optional[Subject]) -> str:
    if subject is None:
        return ""
    parts = [p for p in (getattr(subject, "artist_name", None),
                         getattr(subject, "song_title", None)) if p]
    return " — ".join(parts)


def _pack(facts: list, chunks: list) -> list:
    """The numbered grounding pack: library facts first, then web passages."""
    pack: list = []
    for fact in facts:
        pack.append(Evidence(n=len(pack) + 1, text=fact.text, kind="fact",
                             source=fact.source, ce_prob=fact.ce_prob))
    for chunk, prob in chunks:
        pack.append(Evidence(n=len(pack) + 1, text=chunk.text, kind="chunk",
                             source=chunk.title or chunk.url, url=chunk.url,
                             ce_prob=prob))
    return pack


def _render(evidence: list, limit: Optional[int] = None) -> str:
    items = evidence[:limit] if limit else evidence
    return "\n\n".join(f"[{e.n}] {e.text}" for e in items)


def _valid_citations(raw, count: int) -> list:
    """Citation numbers that actually point at something in the pack."""
    if not isinstance(raw, (list, tuple)):
        return []
    out = []
    for value in raw:
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= count and n not in out:
            out.append(n)
    return out


def _fallback_answer(evidence: list, lang: str) -> str:
    """What the user sees when the model's answer failed the citation gate."""
    head = ("Не могу это пересказать своими словами, но вот что нашлось:"
            if (lang or "").lower().startswith("ru") else
            "I can't summarise this reliably, but here is what I found:")
    body = "\n\n".join(f"• {e.text}" for e in evidence[:6])
    return f"{head}\n\n{body}"


def _lang_name(lang: str) -> str:
    return "Russian" if (lang or "").lower().startswith("ru") else "English"
