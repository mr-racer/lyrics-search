"""The agent itself: two branches over one set of parts.

Control flow is fixed. The model is called at four known points — plan, next
queries, answer (or extract), curate — and everything between those calls is
code. There is no loop the model can steer, no tool it can decide to reach for,
and no place where "should I keep going?" is settled by prose.

The one judgement genuinely left to the model is whether a question came out
answered. Code cannot read a passage and know that it explains the origin of a
stage name. But code CAN see the cross-encoder probabilities the model never
sees, and it holds the veto in both directions:

* model says "enough" while the best chunk scored below the threshold →
  another iteration runs anyway;
* model says "not enough" while the budget is spent, or the last round brought
  no new material → the run ends and answers with what it has.

Grounding works the way it does in production: a numbered pack, an answer that
must cite numbers, and code that throws the answer away when the citations do
not check out. An unnumbered paragraph physically cannot reach the caller.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Optional

from lab.agent.catalog import LibraryCatalog
from lab.agent.chunking import MarkdownChunker
from lab.agent.clarify import AbbreviationResolver, ClarifyCallback
from lab.agent.events import EventSink
from lab.agent.extraction import TrackExtractor, structured_tracks
from lab.agent.fetch import PageFetcher
from lab.agent.llm import LLMClient, as_str
from lab.agent.models import (Chunk, Evidence, GeneralResult, Page, Plan,
                              PlaylistResult, ResolvedTrack, TrackRef)
from lab.agent.planner import Planner, quote_work
from lab.agent.prompts import ANSWER_SYSTEM
from lab.agent.retrieval import HybridRetriever, ModelHub
from lab.agent.retrieval.facts import FactsRetriever, SqliteFactSource
from lab.agent.selection import curate_tracks, select_tracks, triage_tracks
from lab.agent.sources import SearchSources, rerank_hits
from lab.agent.urls import canonical_url, dedupe_by_url
from lab.websearch_lab import fold

logger = logging.getLogger(__name__)

# Below this, whatever the retriever returned is not worth calling context.
# Used only for the iterate/stop veto, never to drop material.
WEAK_CONTEXT_PROB = 0.45

# Titles to try for the discography rescue, in order, until one yields rows.
#
# There is no single naming convention and no way to know which an artist has.
# "Kanye West discography" is a disambiguation stub — three links, no table —
# while the songs live under "singles discography"; JAY-Z's are under "albums
# discography"; Sade's are on the plain "discography" page. A page that yields
# nothing is almost always the stub, so the next spelling is simply tried.
#
# Order is by usefulness for a PLAYLIST: singles and song lists are tracks,
# an albums table is albums. The albums page is last and its rows arrive
# labelled with their section, so the triage pass can see what they are.
DISCOGRAPHY_TITLES = (
    "{artist} singles discography",
    "List of songs recorded by {artist}",
    "{artist} discography",
    "{artist} albums discography",
)


class Assistant:
    """One instance per conversation; one :meth:`run` per user message."""

    def __init__(self, config=None, *, hub: Optional[ModelHub] = None,
                 catalog: Optional[LibraryCatalog] = None,
                 on_clarify: Optional[ClarifyCallback] = None,
                 on_event=None, verbose: bool = False):
        from lab.agent.config import AgentConfig

        self.cfg = config or AgentConfig()
        self.sink = EventSink(on_event, verbose=verbose)
        self.hub = hub or ModelHub(self.cfg)
        self.llm = LLMClient(self.cfg)
        self.planner = Planner(self.llm, self.cfg, self.sink)
        self.chunker = MarkdownChunker(self.cfg)
        self.on_clarify = on_clarify
        # The branch of the last run, for inspection from a notebook.
        self.branch: Optional[_WebBranch] = None

        self.catalog = catalog
        if self.catalog is None and self.cfg.db_path:
            try:
                self.catalog = LibraryCatalog(self.cfg.db_path,
                                              self.cfg.collection_name)
            except Exception:
                logger.warning("[assistant] catalog unavailable", exc_info=True)

        self.facts: Optional[FactsRetriever] = None
        if self.cfg.db_path:
            self.facts = FactsRetriever(SqliteFactSource(self.cfg.db_path),
                                        hub=self.hub, config=self.cfg)

    # ── entry point ───────────────────────────────────────────────────────

    async def run(self, message: str):
        """Answer ``message``. Returns a GeneralResult or a PlaylistResult."""
        self.sink.put("start", message=message)
        plan = await self.planner.plan(message)
        if plan is None:
            why = self.planner.last_failure or "the planner returned nothing usable"
            return GeneralResult(answer="", evidence=[], used=[], grounded=False,
                                 iterations=0, notes=[f"no plan: {why}"])

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
                plan.web_queries = [quote_work(_strip(q, plan.abbreviation.raw),
                                               expansion)
                                    for q in plan.web_queries]

        fetcher = PageFetcher(self.cfg, self.sink)
        if plan.intent == "playlist":
            branch = PlaylistBranch(self, sources, fetcher)
        else:
            branch = GeneralBranch(self, sources, fetcher)
        # Kept so a notebook can look at what the run actually read — the
        # chunks and their vectors are the only place the duplicate thresholds
        # can be calibrated against real numbers instead of guessed at.
        self.branch = branch
        return await branch.run(message, plan)

    # ── shared machinery ──────────────────────────────────────────────────

    def chunks_of(self, pages: list[Page], start_id: int) -> list[Chunk]:
        out: list[Chunk] = []
        for page in pages:
            out.extend(self.chunker.split_page(page, start_id=start_id + len(out)))
        return out

    async def subject_facts(self, plan: Plan, message: str, query: str) -> list:
        """The subject's own facts, ranked. Empty when the subject is unclear.

        Empty is a perfectly good outcome. The web branch answers either way,
        and loading the wrong artist's biography is a worse failure than
        loading none: it is invisible, and it makes the answer confidently
        about someone else.
        """
        if self.facts is None or self.catalog is None:
            return []
        artist, song = plan.filters.artist, plan.filters.song
        if not artist and not song:
            return []

        subject = self.catalog.resolve_subject(song=song, artist=artist)
        if subject.how == "shortlist":
            subject = await self._pick_from_shortlist(subject, message)
        self.sink.put("subject", how=subject.how, artist=subject.artist_slug,
                      song=subject.song_slug)
        if not subject.resolved:
            logger.info("[assistant] no confident subject for artist=%r song=%r "
                        "— answering from the web alone", artist, song)
            return []

        found = await asyncio.to_thread(
            self.facts.retrieve, query,
            song_slug=subject.song_slug, artist_slug=subject.artist_slug)
        self.sink.put("facts_done", kept=len(found))
        return found

    async def _pick_from_shortlist(self, subject, message: str):
        """Let the model choose between candidates code could not separate.

        The same division of labour as everywhere else: code builds the list
        (the model never sees the catalog and so cannot invent a name), the
        model judges, and code checks the answer against the list before
        acting on it. The model is worth asking here because it has what the
        similarity score structurally lacks — it knows that "1 Thing" is an
        Amerie song, and no spelling distance ever will.
        """
        from lab.agent.catalog import Subject
        from lab.agent.prompts import PICK_ARTIST_SYSTEM

        listing = [{"artist": c["artist"], "score": c.get("score")}
                   for c in subject.candidates]
        self.sink.put("disambiguate", candidates=[c["artist"] for c in listing])
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


class _WebBranch:
    """What both branches share: searching, fetching, chunking, ranking."""

    def __init__(self, agent: Assistant, sources: SearchSources,
                 fetcher: PageFetcher):
        self.agent = agent
        self.cfg = agent.cfg
        self.sink = agent.sink
        self.sources = sources
        self.fetcher = fetcher
        self.chunks: list[Chunk] = []
        self.retriever: Optional[HybridRetriever] = None
        self.used_queries: list[str] = []
        # Content hashes of everything indexed. URL deduplication catches the
        # same page twice; this catches the same TEXT arriving from two pages,
        # which listicles syndicating each other do constantly.
        self._chunk_hashes: set[str] = set()

    async def gather(self, plan: Plan, queries: list[str], ce_query: str,
                     *, structured: bool) -> tuple[list[Page], list[Page]]:
        """Run one round of search + fetch.

        Returns ``(structured_pages, prose_pages)``. Structured sources skip
        the relevance pass: pinning the host has already done that work, and
        an Apple playlist page has almost no text for a reranker to read.
        """
        structured_hits, web_hits = [], []
        for i, query in enumerate(queries):
            self.used_queries.append(query)
            # Host-pinned sources on the first query only by default: a
            # rephrasing rarely finds a different Apple playlist, and every
            # extra call is another turn of the burst that gets the good
            # engines rate-limited.
            first_only = self.cfg.structured_first_query_only
            if structured and (i == 0 or not first_only):
                structured_hits += await asyncio.to_thread(
                    self.sources.apple_music, query)
                if plan.filters.work:
                    structured_hits += await asyncio.to_thread(
                        self.sources.fandom, query)
            structured_hits += await asyncio.to_thread(
                self.sources.wikipedia, query)
            web_hits += await asyncio.to_thread(self.sources.web, query)

        kept: list = []
        if web_hits:
            kept = await asyncio.to_thread(
                rerank_hits, web_hits, ce_query,
                hub=self.agent.hub, threshold=self.cfg.ce_threshold_docs)
        self.sink.put("rerank", candidates=len(web_hits), kept=len(kept),
                      threshold=self.cfg.ce_threshold_docs)

        # Deduplicated ACROSS the two lists, not within each: a Wikipedia
        # article reached by the pinned-host stream and again by the open-web
        # stream is one page. Structured goes first so the shared page keeps
        # its structured treatment (tables parsed, not chunked as prose).
        structured_hits = dedupe_by_url(structured_hits)
        structured_keys = {canonical_url(h.url) for h in structured_hits}
        prose_hits = [h for h in dedupe_by_url(kept)
                      if canonical_url(h.url) not in structured_keys]

        structured_pages = await self.fetcher.fetch_many(
            structured_hits, limit=self.cfg.max_pages_per_iteration)
        prose_pages = await self.fetcher.fetch_many(
            prose_hits, limit=self.cfg.max_pages_per_iteration)
        return structured_pages, prose_pages

    def index(self, pages: list[Page]) -> int:
        """Chunk ``pages`` and add them to the retriever. Returns new chunks."""
        candidates = self.agent.chunks_of(pages, len(self.chunks))
        fresh: list[Chunk] = []
        for chunk in candidates:
            digest = _text_hash(chunk.body)
            if digest in self._chunk_hashes:
                continue
            self._chunk_hashes.add(digest)
            chunk.id = len(self.chunks) + len(fresh)
            fresh.append(chunk)
        duplicates = len(candidates) - len(fresh)
        if duplicates:
            logger.info("[pipeline] dropped %d duplicate chunks", duplicates)
        if not fresh:
            return 0
        texts = [c.text for c in fresh]
        if self.retriever is None:
            self.retriever = HybridRetriever(texts, hub=self.agent.hub,
                                             config=self.cfg)
        else:
            self.retriever.extend(texts)
        self.chunks.extend(fresh)
        self.sink.put("index", new_chunks=len(fresh), total=len(self.chunks))
        return len(fresh)

    def best_chunks(self, ce_query: str) -> list[tuple[Chunk, float]]:
        """Top chunks across every page read so far, above the threshold."""
        return select_pack(self.retriever, self.chunks, ce_query,
                           config=self.cfg, sink=self.sink)


class GeneralBranch(_WebBranch):
    """Questions about an artist, a song or an incident."""

    async def run(self, message: str, plan: Plan) -> GeneralResult:
        facts = await self.agent.subject_facts(plan, message, plan.ce_query)

        queries, ce_query = plan.web_queries, plan.ce_query
        answer, used, evidence = "", [], []
        iterations = 0
        notes: list[str] = []

        while iterations < self.cfg.general_max_iterations:
            iterations += 1
            self.sink.put("iteration", n=iterations, queries=queries)

            structured_pages, prose_pages = await self.gather(
                plan, queries, ce_query, structured=False)
            new_chunks = self.index(structured_pages + prose_pages)
            chunks = self.best_chunks(ce_query)

            evidence = _pack(facts, chunks)
            if evidence:
                answer, used, sufficient, missing = await self._ask(message, evidence)
                best_prob = max((e.ce_prob or 0.0 for e in evidence), default=0.0)
            else:
                # Nothing cleared the threshold. That is a reason to search
                # AGAIN, not to stop: the pages were wrong, the question was
                # not. Stopping here is what a high chunk threshold turns into
                # if the empty case is treated as an answer — one unlucky pair
                # of queries and the run is over.
                answer, used, sufficient = "", [], False
                missing = "nothing on those pages was about the question"
                best_prob = 0.0
                notes.append(f"iteration {iterations}: nothing cleared the "
                             f"chunk threshold")

            stop, why = self._should_stop(
                iterations=iterations, sufficient=sufficient,
                best_prob=best_prob, new_chunks=new_chunks)
            self.sink.put("verdict", sufficient=sufficient,
                          best_prob=round(best_prob, 3), stop=stop, why=why)
            notes.append(why)
            if stop:
                break

            queries, next_ce, model_missing = await self.agent.planner.next_queries(
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
        if not grounded and evidence:
            answer = _fallback_answer(evidence, self.cfg.lang)
            notes.append("ungrounded answer discarded — showing the sources")
        self.sink.put("answer", grounded=grounded, evidence=len(evidence),
                      used=used)
        return GeneralResult(answer=answer, evidence=evidence, used=used,
                             grounded=grounded, iterations=iterations,
                             notes=notes)

    def _should_stop(self, *, iterations: int, sufficient: bool,
                     best_prob: float, new_chunks: int) -> tuple[bool, str]:
        """The veto. Code decides the budget, the model decides the meaning."""
        if iterations >= self.cfg.general_max_iterations:
            return True, "iteration budget spent"
        if new_chunks == 0 and iterations > 1:
            return True, "last round brought nothing new"
        if sufficient and best_prob >= WEAK_CONTEXT_PROB:
            return True, "model says answered and the context is strong"
        if sufficient:
            return False, (f"model says answered but the best chunk is only "
                           f"p={best_prob:.2f} — searching again")
        return False, "model says the question is not covered"

    async def _ask(self, message: str,
                   evidence: list[Evidence]) -> tuple[str, list[int], bool, str]:
        raw = await self.agent.llm.ask_json([
            {"role": "system",
             "content": ANSWER_SYSTEM.format(lang=_lang_name(self.cfg.lang))},
            {"role": "user",
             "content": f"Question: {message}\n\nMaterial:\n{_render(evidence)}"},
        ], required=("answer",))
        if raw is None:
            return "", [], False, ""

        answer = as_str(raw.get("answer"), 4000)
        used = _valid_citations(raw.get("used"), len(evidence))
        sufficient = bool(raw.get("sufficient", False))
        missing = as_str(raw.get("missing"), 200)

        # The citation gate. An answer with no usable numbers is thrown away
        # whole — a paragraph nobody can trace is worse than no paragraph.
        if answer and not used:
            logger.info("[general] answer discarded: no valid citations")
            return "", [], sufficient, missing
        return answer, used, sufficient, missing


class PlaylistBranch(_WebBranch):
    """Requests for songs to play."""

    async def run(self, message: str, plan: Plan) -> PlaylistResult:
        target = plan.filters.count or self.cfg.default_target_count
        # Resolved once: table rows that name no artist are tagged with it, and
        # the user's spelling («Канье») matches nothing in the library.
        artist = self._library_artist(plan)
        queries, ce_query = list(plan.web_queries), plan.ce_query
        claims: list[TrackRef] = []
        notes: list[str] = []
        iterations = relaxations = 0

        while iterations < self.cfg.playlist_max_iterations:
            iterations += 1
            self.sink.put("iteration", n=iterations, queries=queries)

            structured_pages, prose_pages = await self.gather(
                plan, queries, ce_query, structured=True)

            for page in structured_pages:
                found = await asyncio.to_thread(
                    structured_tracks, page, default_artist=artist)
                claims.extend(found)
                if found:
                    self.sink.put("structured", url=page.url, tracks=len(found))

            self.index(prose_pages)
            chunks = self.best_chunks(ce_query)
            if chunks:
                extractor = TrackExtractor(self.agent.llm, self.cfg, self.sink)
                claims += await extractor.from_passages(
                    [(c.url, c.text) for c, _ in chunks], request=message)

            resolved, _ = self._resolve(claims, plan)
            self.sink.put("matched", claims=len(claims), resolved=len(resolved),
                          target=target)

            if len(resolved) >= target:
                break

            enough = len(resolved) >= target * self.cfg.min_yield_ratio
            budget_left = iterations < self.cfg.playlist_max_iterations
            if enough or not budget_left:
                break

            # Not enough after filtering. Take a constraint OUT of the query
            # text and let the code-side filter do that work instead — the
            # phrase "спокойные хиты 80х" narrows the web result set far more
            # than it narrows the truth.
            relaxed = self._relax(plan, queries, relaxations)
            if relaxed is None:
                notes.append("nothing left to relax")
                break
            relaxations += 1
            queries = relaxed
            notes.append(f"relaxed the query (round {relaxations})")

        resolved, missing = self._resolve(claims, plan)
        if len(resolved) < self.cfg.discography_min_tracks:
            rescued = await self._discography(plan, found=len(resolved))
            if rescued:
                claims += rescued
                resolved, missing = self._resolve(claims, plan)
                notes.append(f"discography rescue added "
                             f"{len(rescued)} claims")

        resolved = await triage_tracks(self.agent.llm, message, resolved,
                                       config=self.cfg, sink=self.sink)
        resolved = resolved[:target]
        title, comment, resolved = await curate_tracks(
            self.agent.llm, message, resolved, config=self.cfg, sink=self.sink)
        self.sink.put("result", tracks=len(resolved), missing=len(missing))
        return PlaylistResult(title=title, comment=comment, tracks=resolved,
                              missing=missing[:40], iterations=iterations,
                              relaxations=relaxations, notes=notes)

    def _library_artist(self, plan: Plan) -> Optional[str]:
        """The artist as the LIBRARY spells it, best effort.

        Used for two things: the rows of a table that names no artist, and the
        discography search. Both want "Kanye West" where the user typed
        «Канье» — a table row tagged with the Cyrillic spelling matches nothing
        in the library, and a Cyrillic query finds nothing on the English
        Wikipedia.

        A best guess is acceptable here, unlike when choosing whose FACTS to
        read: every title is still matched against the library under this name,
        so a wrong guess yields no tracks rather than the wrong ones.
        """
        raw = plan.filters.artist
        if not raw or self.agent.catalog is None:
            return raw
        subject = self.agent.catalog.resolve_subject(artist=raw)
        best = subject.candidates[0]["artist"] if subject.candidates else None
        return subject.artist_name or best or raw

    async def _discography(self, plan: Plan, *, found: int) -> list[TrackRef]:
        """Last resort: go and read the artist's discography article.

        The failure this exists for is specific and common. "Хиты Канье после
        2020" has no page — nobody writes a listicle per artist per era — so
        the searches come back with charts, interviews and news, and the
        deterministic stage matches four tracks. Meanwhile Wikipedia has the
        whole singles discography in a table, and the era filter can do the
        rest itself.

        The query is built by CODE, not by the model: the artist is already
        resolved against the library, and "<artist> discography" is not a
        judgement call. Wikipedia's engine searches article titles, so it is
        also very likely to land exactly right.

        Runs at most twice, and only when the playlist would otherwise be thin.
        """
        artist = plan.filters.artist
        if not artist:
            return []
        artist = self._library_artist(plan) or artist

        self.sink.put("discography", artist=artist, found=found)
        logger.info("[playlist] only %d tracks — reading %s's discography",
                    found, artist)

        refs: list[TrackRef] = []
        tried = 0
        for template in DISCOGRAPHY_TITLES:
            if tried >= self.cfg.discography_max_queries:
                break
            tried += 1
            query = template.format(artist=artist)
            hits = await asyncio.to_thread(
                self.sources.wikipedia, query, 2, True)
            pages = await self.fetcher.fetch_many(dedupe_by_url(hits), limit=1)
            for page in pages:
                got = await asyncio.to_thread(structured_tracks, page,
                                              default_artist=artist)
                logger.info("[playlist] discography %r -> %s -> %d rows",
                            query, page.url, len(got))
                refs += got
            if refs:
                break
            # Nothing on that page. Almost always the disambiguation stub —
            # "Kanye West discography" is three links and no table — so the
            # next spelling is tried rather than giving up.
            logger.info("[playlist] %r yielded no rows, trying the next title",
                        query)

        self.sink.put("discography_done", queries=tried, claims=len(refs))
        return refs

    # ── selection, all of it code ─────────────────────────────────────────

    def _resolve(self, claims: list[TrackRef],
                 plan: Plan) -> tuple[list[ResolvedTrack], list[TrackRef]]:
        """Claims → library tracks, weighted, era-filtered, best first.

        A thin wrapper over ``selection.select_tracks`` so the notebook and the
        agent run the same selection — reimplementing the weighting in a cell
        is how a lab result stops describing the thing it measures.
        """
        tracks, missing = select_tracks(
            self.agent.catalog, claims, era=plan.filters.era,
            source_weights=self.cfg.source_weights)
        if plan.filters.era:
            self.sink.put("era_filter", range=plan.filters.era, kept=len(tracks))
        return tracks, missing

    def _relax(self, plan: Plan, queries: list[str],
               done: int) -> Optional[list[str]]:
        """Drop one constraint from the query TEXT (never from the filters).

        Order matters: the style word is the one that most distorts a web
        search ("спокойные хиты 80х" finds listicles about calm music, not
        the decade's hits), the era is second. Both keep filtering results
        afterwards — this only changes what we ask the internet for.
        """
        if done >= self.cfg.max_relaxations:
            return None
        drop = [plan.filters.style] if done == 0 else []
        if done == 1 and plan.filters.era:
            drop = [str(plan.filters.era[0]), str(plan.filters.era[1]),
                    f"{plan.filters.era[0] // 10 * 10}s"]
        drop = [d for d in drop if d]
        if not drop:
            return None
        out = []
        for query in queries:
            relaxed = query
            for token in drop:
                relaxed = _strip(relaxed, token)
            relaxed = " ".join(relaxed.split())
            if relaxed and relaxed not in out:
                out.append(relaxed)
        return out or None

# ── the context pack ─────────────────────────────────────────────────────────


def select_pack(retriever, chunks: list[Chunk], ce_query: str, *,
                config=None, sink=None) -> list[tuple[Chunk, float]]:
    """The passages that go in front of the model, and their probabilities.

    A free function, not a method, for the same reason ``selection.py`` is:
    a notebook driving the stages by hand must be able to build the SAME pack
    the agent builds. Reimplementing the pool depth and the duplicate rule in a
    cell is how a lab result stops describing the thing it measures.
    """
    from lab.agent.config import AgentConfig

    cfg = config or AgentConfig()
    if retriever is None or not chunks:
        return []

    threshold = cfg.ce_threshold_chunks
    limit = cfg.max_chunks_in_context
    # A deeper pool than the pack needs, so that dropping a copy FREES a slot
    # rather than shrinking the pack: the next distinct passage moves up into
    # it. With dedup off this is exactly the old behaviour.
    pool = limit * max(1, cfg.dedup_pool_factor) if cfg.dedup_chunks else limit
    ranked = retriever.search(ce_query, min_prob=threshold, limit=pool)
    ranked = _diverse(retriever, chunks, ranked, limit=limit, cfg=cfg, sink=sink)
    out = [(chunks[r.index], r.ce_prob or 0.0) for r in ranked]

    if not out:
        # A high chunk threshold is the right default, but "nothing passed" and
        # "nothing was found" look identical from the outside. Say which, and
        # say how close it came, so the number can be calibrated instead of
        # guessed at.
        unfiltered = retriever.search(ce_query, limit=1)
        best = unfiltered[0].ce_prob if unfiltered else None
        logger.info("[pipeline] no chunk cleared p>=%.2f over %d chunks "
                    "(best was %s)", threshold, len(chunks),
                    f"{best:.3f}" if best is not None else "unscored")
        if sink is not None:
            sink.put("chunks", selected=0, threshold=threshold,
                     best=round(best, 3) if best is not None else None)
        return []
    if sink is not None:
        sink.put("chunks", selected=len(out), threshold=threshold,
                 best=round(max(p for _, p in out), 3))
    return out


def _diverse(retriever, chunks: list[Chunk], ranked: list, *, limit: int,
             cfg, sink=None) -> list:
    """Take the top ``limit`` DISTINCT passages out of the ranked pool.

    Five hosts carrying the same syndicated bio produce five chunks that score
    alike, and the pack ends up saying one thing five times. This spends those
    slots on the next-best passage that says something else.

    Nothing is removed from the index — see
    :mod:`lab.agent.retrieval.diversity` for why that distinction is the safety
    property here.
    """
    if not cfg.dedup_chunks or len(ranked) <= 1:
        return ranked[:limit]

    from lab.agent.retrieval.diversity import pick_diverse

    sims = retriever.similarity_matrix([r.index for r in ranked])
    if not sims:
        logger.info("[pipeline] no document vectors to compare — keeping the "
                    "top %d as ranked", limit)
        return ranked[:limit]
    missing = [n for n in cfg.dedup_thresholds if n not in sims]
    if missing:
        # Worth saying out loud: the two-signal agreement rule is what keeps
        # this from collapsing passages that merely share a topic, and with one
        # signal it is not being applied.
        logger.info("[pipeline] duplicate check running on %s alone (%s "
                    "unavailable) — a weaker guard than intended",
                    ", ".join(sorted(sims)), ", ".join(missing))

    picked = pick_diverse(
        [len(chunks[r.index].text) for r in ranked], sims,
        thresholds=cfg.dedup_thresholds, limit=limit,
        prefer_longer=cfg.dedup_prefer_longer)
    for dup in picked.duplicates:
        scores = " ".join(f"{n}={v:.3f}" for n, v in sorted(dup.sims.items()))
        logger.info("[pipeline] %s chunk %d (%s) — same as chunk %d (%s) [%s]",
                    "displaced" if dup.replaced else "skipped",
                    chunks[ranked[dup.index].index].id,
                    chunks[ranked[dup.index].index].url,
                    chunks[ranked[dup.twin].index].id,
                    chunks[ranked[dup.twin].index].url, scores)
    if picked.duplicates and sink is not None:
        sink.put("dedup", pool=len(ranked), selected=len(picked.kept),
                 duplicates=len(picked.duplicates), signals=sorted(sims))
    return [ranked[i] for i in picked.kept]


# ── helpers ──────────────────────────────────────────────────────────────────


def _text_hash(text: str) -> str:
    """Content key for a chunk, insensitive to whitespace and case."""
    return hashlib.sha1(" ".join((text or "").lower().split()).encode("utf-8")
                        ).hexdigest()


def _pack(facts: list, chunks: list[tuple[Chunk, float]]) -> list[Evidence]:
    """The numbered grounding pack: library facts first, then web passages."""
    pack: list[Evidence] = []
    for fact in facts:
        pack.append(Evidence(n=len(pack) + 1, text=fact.text, kind="fact",
                             source=fact.source, ce_prob=fact.ce_prob))
    for chunk, prob in chunks:
        pack.append(Evidence(n=len(pack) + 1, text=chunk.text, kind="chunk",
                             source=chunk.title or chunk.url, url=chunk.url,
                             ce_prob=prob))
    return pack


def _render(evidence: list[Evidence], limit: Optional[int] = None) -> str:
    items = evidence[:limit] if limit else evidence
    return "\n\n".join(f"[{e.n}] {e.text}" for e in items)


def _valid_citations(raw, count: int) -> list[int]:
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


def _fallback_answer(evidence: list[Evidence], lang: str) -> str:
    """What the user sees when the model's answer failed the citation gate."""
    head = ("Не могу это пересказать своими словами, но вот что нашлось:"
            if (lang or "").lower().startswith("ru") else
            "I can't summarise this reliably, but here is what I found:")
    body = "\n\n".join(f"• {e.text}" for e in evidence[:6])
    return f"{head}\n\n{body}"


def _lang_name(lang: str) -> str:
    return "Russian" if (lang or "").lower().startswith("ru") else "English"


def _strip(text: str, token: str) -> str:
    """Remove ``token`` from ``text``, case-insensitively, as a whole phrase."""
    if not token:
        return text
    return " ".join(re.sub(re.escape(token), " ", text, flags=re.I).split())
