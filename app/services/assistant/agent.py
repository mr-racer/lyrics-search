"""The assistant itself: one plan, four branches, every stop decided by code.

The model is called at known points — plan, answer/extract, triage, curate — and
everything between those calls is code. There is no loop the model can steer, no
tool it can decide to reach for, and no place where "should I keep going?" is
settled by prose.

This module owns construction and dispatch and nothing else. Each branch is
readable on its own; what they share lives in ``branches/base.py``.

One instance per user message. The budgets (searches, iterations, pages) live on
the instance, so reusing one across turns would silently hand the second question
a spent budget.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.services.assistant.chunking import MarkdownChunker
from app.services.assistant.clarify import AbbreviationResolver
from app.services.assistant.config import AgentConfig
from app.services.assistant.contracts import GeneralResult, Plan, Subject
from app.services.assistant.events import AgentSink
from app.services.assistant.facts_source import (FactsRetriever,
                                                 MetadataFactSource)
from app.services.assistant.fetcher import PageFetcher
from app.services.assistant.llm import LLMClient, as_str
from app.services.assistant.planner import Planner, plan_for_focus, quote_work
from app.services.assistant.timing import Timings
from app.services.assistant.web_sources import SearchSources
from app.services.retrieval import DEFAULT_HUB
from app.services.text_normalize import fold

logger = logging.getLogger(__name__)


def audio_search_available() -> bool:
    """Whether the CLAP branch can run at all on this instance."""
    try:
        from app.resources.model_registry import ModelRegistry
        from app.services.settings_service import settings_service

        return bool(settings_service.clap_enabled()
                    and ModelRegistry.is_clap_available())
    except Exception:  # noqa: BLE001 — an unreadable setting is not a crash
        logger.warning("[assistant] could not resolve CLAP availability",
                       exc_info=True)
        return False


class Assistant:
    """One instance per user message; one :meth:`run` per instance."""

    def __init__(self, collection_name: str, *,
                 config: Optional[AgentConfig] = None,
                 sink: Optional[AgentSink] = None,
                 search_service=None, on_clarify=None):
        self.cfg = config or AgentConfig()
        self.collection_name = collection_name
        self.sink = sink or AgentSink()
        self.timings = Timings()
        self.hub = DEFAULT_HUB
        self.llm = LLMClient(self.cfg)
        self.search_service = search_service
        self.on_clarify = on_clarify
        self.planner = Planner(self.llm, self.cfg, self.sink,
                               audio_available=audio_search_available())
        self.chunker = MarkdownChunker(self.cfg)
        # The subject the last run settled on, so the route can echo it back to
        # the card without resolving it a second time.
        self.last_subject: Optional[Subject] = None

        self.catalog = None
        try:
            from app.services.library_catalog import get_catalog

            self.catalog = get_catalog(collection_name)
        except Exception:  # noqa: BLE001 — the web branch works without it
            logger.warning("[assistant] catalog unavailable for %s",
                           collection_name, exc_info=True)

        self.facts = FactsRetriever(
            MetadataFactSource(collection_name,
                               use_refined=self.cfg.facts_use_refined,
                               lang=self.cfg.lang),
            hub=self.hub, config=self.cfg)

    # ── entry point ───────────────────────────────────────────────────────

    async def run(self, message: str, *, focus_fact: Optional[str] = None,
                  subject_track_id: Optional[str] = None,
                  subject_artist_slug: Optional[str] = None,
                  forced_intent: Optional[str] = None):
        """Answer ``message``. Returns one of the four result contracts."""
        with self.timings.measure():
            result = await self._run(message, focus_fact=focus_fact,
                                     subject_track_id=subject_track_id,
                                     subject_artist_slug=subject_artist_slug,
                                     forced_intent=forced_intent)
        logger.info("[assistant] %s", self.timings.report())
        return result

    async def _run(self, message: str, *, focus_fact: Optional[str],
                   subject_track_id: Optional[str],
                   subject_artist_slug: Optional[str],
                   forced_intent: Optional[str] = None):
        self.sink.put("start", message=message)

        pinned = self._pinned_subject(subject_track_id, subject_artist_slug)

        if focus_fact:
            # "Explain THIS statement" is not a question to classify. The intent
            # is known, the subject is pinned by id, and the statement itself is
            # the best cross-encoder query there is — it is the exact text a
            # useful passage has to be about. Asking the planner here would spend
            # a call to re-derive facts the caller already supplied, and would
            # occasionally get them wrong.
            plan = plan_for_focus(focus_fact, pinned, message)
            self.sink.put("plan", intent="general", focus=True,
                          queries=plan.web_queries)
            return await self._general(message, plan, focus_fact=focus_fact,
                                       subject=pinned)

        with self.timings.span("llm.plan"):
            plan = await self.planner.plan(message)
        if plan is None:
            why = self.planner.last_failure or "the planner returned nothing usable"
            return GeneralResult(answer="", evidence=[], used=[], grounded=False,
                                 iterations=0, notes=[f"no plan: {why}"])

        if forced_intent and forced_intent != plan.intent:
            # The caller already knows the branch: a follow-up chip, a discovery
            # card, or a button the user tapped. Their choice outranks the
            # planner's reading of the sentence — but it still goes through the
            # same settle step, so an audio request on an instance without CLAP
            # falls back like any other.
            logger.info("[assistant] intent forced to %r (planner said %r)",
                        forced_intent, plan.intent)
            plan.intent = self.planner.settle_intent(forced_intent, plan.filters)

        if plan.intent == "lyrics_search":
            from app.services.assistant.branches.lyrics import LyricsBranch

            return await LyricsBranch(self).run(message, plan, subject=pinned)
        if plan.intent == "audio_search":
            from app.services.assistant.branches.audio import AudioBranch

            return await AudioBranch(self).run(message, plan)

        sources = SearchSources(self.cfg, self.sink)
        if plan.abbreviation is not None:
            resolver = AbbreviationResolver(sources, self.cfg, self.sink,
                                            self.on_clarify)
            expansion, clarify = await resolver.resolve(plan.abbreviation)
            if clarify is not None:
                return GeneralResult(answer="", evidence=[], used=[],
                                     grounded=False, iterations=0,
                                     clarify=clarify)
            if expansion:
                plan.filters.work = expansion
                plan.web_queries = [
                    quote_work(_strip(q, plan.abbreviation.raw), expansion)
                    for q in plan.web_queries]

        fetcher = PageFetcher(self.cfg, self.sink)
        if plan.intent == "playlist":
            from app.services.assistant.branches.playlist import PlaylistBranch

            return await PlaylistBranch(self, sources, fetcher).run(message, plan)
        return await self._general(message, plan, subject=pinned,
                                   sources=sources, fetcher=fetcher)

    async def _general(self, message: str, plan: Plan, *,
                       focus_fact: Optional[str] = None,
                       subject: Optional[Subject] = None,
                       sources=None, fetcher=None):
        from app.services.assistant.branches.general import GeneralBranch

        branch = GeneralBranch(self, sources or SearchSources(self.cfg, self.sink),
                               fetcher or PageFetcher(self.cfg, self.sink))
        return await branch.run(message, plan, focus_fact=focus_fact,
                                subject=subject)

    # ── shared machinery ──────────────────────────────────────────────────

    def chunks_of(self, pages: list, start_id: int) -> list:
        out: list = []
        for page in pages:
            out.extend(self.chunker.split_page(page, start_id=start_id + len(out)))
        return out

    def library_artist(self, raw: Optional[str]) -> Optional[str]:
        """The artist as the LIBRARY spells it, best effort.

        A best guess is acceptable here, unlike when choosing whose FACTS to read:
        the name is used as a FILTER, so a wrong guess yields no tracks rather
        than the wrong ones.
        """
        if not raw or self.catalog is None:
            return raw
        subject = self.catalog.resolve_subject(artist=raw)
        best = subject.candidates[0]["artist"] if subject.candidates else None
        return subject.artist_name or best or raw

    def _pinned_subject(self, track_id: Optional[str],
                        artist_slug: Optional[str]) -> Optional[Subject]:
        """The subject the CALLER identified, resolved structurally.

        No name matching happens and none may: the id came from a card the user
        tapped. Re-deriving the artist from a string is what once resolved
        "Amerie" to "Fergie" and loaded a stranger's biography.
        """
        if self.catalog is None:
            return None
        subject = None
        if track_id:
            subject = self.catalog.subject_for_track(track_id)
        if subject is None and artist_slug:
            subject = self.catalog.subject_for_artist(artist_slug)
        if subject is not None:
            self.last_subject = subject
            self.sink.put("subject", how=subject.how, artist=subject.artist_slug,
                          song=subject.song_slug)
        return subject

    async def subject_facts(self, plan: Plan, message: str, query: str, *,
                            subject: Optional[Subject] = None) -> list:
        """The subject's own facts, ranked. Empty when the subject is unclear.

        Empty is a perfectly good outcome. The web branch answers either way, and
        loading the wrong artist's biography is a worse failure than loading none:
        it is invisible, and it makes the answer confidently about someone else.
        """
        if self.catalog is None:
            return []

        if subject is None:
            artist, song = plan.filters.artist, plan.filters.song
            if not artist and not song:
                return []
            subject = self.catalog.resolve_subject(song=song, artist=artist)
            if subject.how == "shortlist":
                subject = await self._pick_from_shortlist(subject, message)
            self.sink.put("subject", how=subject.how, artist=subject.artist_slug,
                          song=subject.song_slug)

        self.last_subject = subject
        if not subject.resolved:
            logger.info("[assistant] no confident subject — answering from the "
                        "web alone")
            return []

        with self.timings.span("facts.retrieve"):
            found = await asyncio.to_thread(
                self.facts.retrieve, query, song_slug=subject.song_slug,
                artist_slug=subject.artist_slug)
        self.sink.put("facts_done", kept=len(found))
        return found

    async def _pick_from_shortlist(self, subject: Subject,
                                   message: str) -> Subject:
        """Let the model choose between candidates code could not separate.

        The same division of labour as everywhere else: code builds the list (the
        model never sees the catalog and so cannot invent a name), the model
        judges, and code checks the answer against the list before acting on it.
        The model is worth asking here because it has what the similarity score
        structurally lacks — it knows that "1 Thing" is an Amerie song, and no
        spelling distance ever will.
        """
        from app.services.assistant.prompts import PICK_ARTIST_SYSTEM

        listing = [{"artist": c["artist"], "score": c.get("score")}
                   for c in subject.candidates]
        self.sink.put("disambiguate", candidates=[c["artist"] for c in listing])
        with self.timings.span("llm.pick_artist"):
            raw = await self.llm.ask_json([
                {"role": "system", "content": PICK_ARTIST_SYSTEM},
                {"role": "user",
                 "content": f"Question: {message}\nCandidates: {listing}"},
            ], required=("artist",))

        picked = as_str((raw or {}).get("artist"), 160)
        chosen = next((c for c in subject.candidates
                       if fold(c["artist"]) == fold(picked)), None) if picked else None
        if chosen is None:
            logger.info("[assistant] shortlist unresolved (model said %r)", picked)
            return Subject(how="none", candidates=subject.candidates)
        logger.info("[assistant] shortlist resolved to %r: %s", chosen["artist"],
                    as_str((raw or {}).get("why"), 120))
        return Subject(song_slug=chosen.get("song_slug"),
                       artist_slug=chosen.get("slug") or None,
                       artist_name=chosen["artist"], how="model-pick")


def _strip(text: str, token: str) -> str:
    import re

    if not token:
        return text
    return " ".join(re.sub(re.escape(token), " ", text, flags=re.I).split())
