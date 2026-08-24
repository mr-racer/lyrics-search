"""Questions about an artist, a song or an incident — and the tap-through modes.

Control flow is fixed. The model is called at known points — plan (before this
branch), answer, next queries — and everything between those calls is code.
There is no loop the model can steer and no place where "should I keep going?"
is settled by prose.

**The library answers first.** Iteration 0 builds a pack out of SQLite alone
(``local_pack``): the subject's facts, the facts of the songs it is
structurally tied to, its sample links, credits and gems — plus any passages a
previous turn already downloaded. If the model says that answers the question,
the run ends there and nothing is fetched. The web is what happens when the
library came up short, not the first thing tried.

The one judgement genuinely left to the model is whether the question came out
answered. Code cannot read a passage and know that it explains the origin of a
stage name. But code CAN see the cross-encoder probabilities the model never
sees, and it holds the veto in both directions:

* model says "enough" while the best chunk scored below the threshold → another
  iteration runs anyway;
* model says "not enough" while the budget is spent, or the last round brought no
  new material → the run ends and answers with what it has.

The local iteration is judged against its own threshold. Fact probabilities live
on the ``ce_threshold_facts`` scale, not the chunk one, so measuring them against
``weak_context_prob`` would send every local iteration to the web and quietly
undo the whole thing. Structural items carry no probability at all and are judged
on ``sufficient`` alone — they are facts ABOUT the subject, not candidates FOR
it, and scoring them would answer a question nobody asked.

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
from app.services.assistant.prompts import (ANSWER_SYSTEM, EXPLAIN_SYSTEM,
                                            SAMPLES_SYSTEM)

logger = logging.getLogger(__name__)


class GeneralBranch(WebBranch):
    """Library facts + the open web, answered with verified citations."""

    async def run(self, message: str, plan: Plan, *,
                  focus_fact: Optional[str] = None,
                  subject: Optional[Subject] = None,
                  focus_kind: Optional[str] = None,
                  allow_web: Optional[bool] = None,
                  context=None) -> GeneralResult:
        self._related: list = []
        subject = await self.agent.resolve_subject(plan, message, subject=subject)
        local = await self.agent.local_material(subject, plan.ce_query)

        # Passages a previous turn already paid for. Reranked against THIS
        # question — carried over as material, never as a ranking.
        if context is not None and getattr(context, "chunks", None):
            self.seed(context.chunks, getattr(context, "used_queries", None))

        web_allowed = self._web_allowed(focus_kind, allow_web)
        # Only the samples mode offers tracks: an answer about what a record is
        # built from invites playing those records, and under any other answer a
        # track list is the filler this branch exists to avoid.
        if focus_kind == "samples" and local.links:
            from app.services.assistant import local_pack

            self._related = await asyncio.to_thread(
                local_pack.resolve_links, self.agent.collection_name,
                local.links,
                exclude_track_id=(subject.track_id if subject else None))
        queries, ce_query = plan.web_queries, plan.ce_query
        answer, used, evidence = "", [], []
        follow_ups: list = []
        iterations = 0
        notes: list = []
        parachuted = False

        # ── iteration 0: the library, and whatever the last turn read ──────
        if self.cfg.local_first:
            evidence = _pack(local.items, self.best_chunks(ce_query))
            if evidence:
                with self.timings.span("llm.answer"):
                    answer, used, sufficient, missing, follow_ups = await self._ask(
                        message, evidence, focus_fact=focus_fact,
                        focus_kind=focus_kind, subject=subject,
                        single_shot=not web_allowed)
                stop, why = self._stop_local(sufficient=sufficient,
                                             best_prob=_best_prob(evidence),
                                             web_allowed=web_allowed)
                self.sink.put("verdict", local=True, sufficient=sufficient,
                              stop=stop, why=why)
                notes.append(why)
                if stop:
                    return self._result(answer, evidence, used, 0, subject,
                                        focus_fact, follow_ups, notes,
                                        focus_kind)
                # The model's own words for what is absent are a better search
                # query than anything code could compose from the question.
                ce_query = missing or ce_query
            elif not web_allowed:
                notes.append("nothing in the library and the web is off")
                return self._result("", [], [], 0, subject, focus_fact, [], notes,
                                    focus_kind)
            else:
                notes.append("the library had nothing — searching")

        if not web_allowed:
            # local_first off and the web forbidden: answer from the pack or say
            # nothing. Reaching the loop below would search anyway.
            evidence = evidence or _pack(local.items, [])
            if evidence and not answer:
                with self.timings.span("llm.answer"):
                    answer, used, _, _, follow_ups = await self._ask(
                        message, evidence, focus_fact=focus_fact,
                        focus_kind=focus_kind, subject=subject, single_shot=True)
            return self._result(answer, evidence, used, 0, subject, focus_fact,
                                follow_ups, notes, focus_kind)

        # ── the web ────────────────────────────────────────────────────────
        while iterations < self.cfg.general_max_iterations:
            iterations += 1
            self.sink.put("iteration", n=iterations, queries=queries)

            structured_pages, prose_pages = await self.gather(
                plan, queries, ce_query, structured=False)
            new_chunks = self.index(structured_pages + prose_pages)
            chunks = self.best_chunks(ce_query)

            evidence = _pack(local.items, chunks)

            # The parachute. Deployed only while falling: the open web has been
            # searched, its pages read, and not one passage cleared the bar. Once
            # per run — a second deployment would be another minute spent on the
            # same failure.
            if not evidence and self.cfg.search_reddit and not parachuted:
                parachuted = True
                if await self._reddit_rescue(ce_query):
                    chunks = self.best_chunks(ce_query)
                    evidence = _pack(local.items, chunks)
                notes.append(f"reddit parachute: {len(evidence)} in the pack")

            if evidence:
                with self.timings.span("llm.answer"):
                    answer, used, sufficient, missing, follow_ups = await self._ask(
                        message, evidence, focus_fact=focus_fact,
                        focus_kind=focus_kind, subject=subject)
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

        return self._result(answer, evidence, used, iterations, subject,
                            focus_fact, follow_ups, notes, focus_kind)

    # ── decisions ─────────────────────────────────────────────────────────

    def _web_allowed(self, focus_kind: Optional[str],
                     allow_web: Optional[bool]) -> bool:
        """Three-valued on purpose: the caller may override, or defer.

        ``None`` means "the mode decides". The samples mode decides no — its
        material is a verified list out of the user's own database, and going to
        the web to describe it invites the model to reconcile the two, which is
        how a correct list turns into a hedged one.
        """
        if allow_web is not None:
            return bool(allow_web)
        return focus_kind != "samples"

    def _stop_local(self, *, sufficient: bool, best_prob: Optional[float],
                    web_allowed: bool) -> tuple:
        """The iteration-0 veto. Same shape as the web one, its own threshold."""
        if not web_allowed:
            return True, "answered from the library; the web is off for this turn"
        if not sufficient:
            return False, "the library did not cover it — searching"
        if best_prob is None:
            # A pack of pure structure — links, credits, a release line. There is
            # nothing to threshold and nothing to doubt: these are not candidates
            # that might be about something else, they are records.
            return True, "answered from the library's own records"
        if best_prob >= self.cfg.weak_local_prob:
            return True, "answered from the library"
        return False, (f"model says answered but the best fact is only "
                       f"p={best_prob:.2f} — searching anyway")

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

    def _result(self, answer: str, evidence: list, used: list, iterations: int,
                subject: Optional[Subject], focus_fact: Optional[str],
                follow_ups: list, notes: list,
                focus_kind: Optional[str] = None) -> GeneralResult:
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
            follow_ups=follow_ups, notes=notes,
            related_tracks=getattr(self, "_related", []),
            focus_kind=focus_kind)

    # ── the web's last resort ─────────────────────────────────────────────

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

    # ── the answer call ───────────────────────────────────────────────────

    async def _ask(self, message: str, evidence: list, *,
                   focus_fact: Optional[str],
                   subject: Optional[Subject],
                   focus_kind: Optional[str] = None,
                   single_shot: bool = False) -> tuple:
        """One answer call. Returns ``(answer, used, sufficient, missing, follow_ups)``."""
        lang = _lang_name(self.cfg.lang)
        who = _subject_line(subject)
        if focus_kind == "samples":
            system = SAMPLES_SYSTEM.format(lang=lang, subject=who or "this track")
            user = f"Material:\n{_render(evidence)}"
        elif focus_fact:
            system = EXPLAIN_SYSTEM.format(lang=lang, subject=who or "this track")
            user = (f"Statement: {focus_fact}\n\n"
                    f"Material:\n{_render(evidence)}")
        else:
            system = ANSWER_SYSTEM.format(lang=lang)
            user = f"Question: {message}\n\nMaterial:\n{_render(evidence)}"

        raw = await self.agent.llm.ask_json(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}], required=("answer",))
        if raw is None:
            return "", [], False, "", []

        answer = as_str(raw.get("answer"), 4000)
        used = _valid_citations(raw.get("used"), len(evidence))
        follow_ups = as_str_list(raw.get("follow_ups"), limit=3, item_limit=120)
        if single_shot:
            # The web is off for this turn, so there is no "search more" to
            # decide about and whatever came back is the verdict. This is the
            # ONLY place code overrides the judgement — and it overrides it
            # because the alternative is not available, not because code thinks
            # it knows better.
            sufficient, missing = True, ""
        else:
            # Every mode asks, including the focus ones. Whether the material
            # actually explains the tapped statement is exactly the judgement a
            # model can make and code cannot, and hard-coding it to "yes" is how
            # a thin pack turns into a confident answer that explains nothing.
            if raw.get("sufficient") is None:
                # Worth saying out loud rather than defaulting quietly: a model
                # that never returns the field makes every local iteration fall
                # through to the web, which looks exactly like the local-first
                # path being off. Say which, so it can be fixed in the prompt.
                logger.info("[general] the model returned no 'sufficient' — "
                            "treating the material as incomplete")
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


def _pack(local_items: list, chunks: list) -> list:
    """The numbered grounding pack: the library first, then web passages.

    Renumbered here rather than carried: ``local_pack`` numbers its own items so
    it can be read on its own, and the chunks have to continue that sequence
    without a gap or the citation gate rejects perfectly good numbers.
    """
    pack: list = []
    for item in local_items:
        pack.append(Evidence(n=len(pack) + 1, text=item.text, kind=item.kind,
                             source=item.source, url=item.url,
                             ce_prob=item.ce_prob))
    for chunk, prob in chunks:
        pack.append(Evidence(n=len(pack) + 1, text=chunk.text, kind="chunk",
                             source=chunk.title or chunk.url, url=chunk.url,
                             ce_prob=prob))
    return pack


def _best_prob(evidence: list) -> Optional[float]:
    """The best SCORED item, or None when nothing in the pack was scored.

    None is not zero. Zero would mean "the cross-encoder read this and thought
    little of it"; None means there was nothing for it to read — a pack of
    verified records, which no threshold applies to.
    """
    scored = [e.ce_prob for e in evidence if e.ce_prob is not None]
    return max(scored) if scored else None


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
