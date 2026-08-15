"""Requests for songs to play, answered from the web and matched to the library.

Everything that decides WHICH tracks come back is code: the tables are parsed,
the claims are matched against the library, the sources are weighted and the era
filters. The model gets two jobs and neither can add a track — triage (which of
these was the page actually offering?) and curation (order them and say why).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from app.services.assistant.branches.base import WebBranch
from app.services.assistant.contracts import Plan, PlaylistResult
from app.services.assistant.selection import (curate_tracks, select_tracks,
                                              triage_tracks)
from app.services.assistant.tracklists import (TrackExtractor,
                                               has_structured_parser,
                                               structured_tracks)
from app.services.assistant.web_urls import dedupe_by_url

logger = logging.getLogger(__name__)

# Titles to try for the discography rescue, in order, until one yields rows.
#
# There is no single naming convention and no way to know which an artist has.
# "Kanye West discography" is a disambiguation stub — three links, no table —
# while the songs live under "singles discography"; JAY-Z's are under "albums
# discography"; Sade's are on the plain "discography" page. A page that yields
# nothing is almost always the stub, so the next spelling is simply tried.
#
# Order is by usefulness for a PLAYLIST: singles and song lists are tracks, an
# albums table is albums. The albums page is last and its rows arrive labelled
# with their section, so the triage pass can see what they are.
DISCOGRAPHY_TITLES = (
    "{artist} singles discography",
    "List of songs recorded by {artist}",
    "{artist} discography",
    "{artist} albums discography",
)


class PlaylistBranch(WebBranch):
    async def run(self, message: str, plan: Plan) -> PlaylistResult:
        # Two numbers, because one was doing two jobs that only agree by accident.
        # ``target`` is when to STOP SEARCHING; ``cap`` is how many tracks to
        # RETURN. They diverge exactly on a soundtrack: the whole tracklist
        # arrives from a single page, so the run has everything it will ever get
        # the moment it has fifteen — and truncating there discards the other
        # thirty it already matched against the library.
        target = plan.filters.count or self.cfg.default_target_count
        cap = plan.filters.count or (self.cfg.work_target_count
                                     if plan.filters.work
                                     else self.cfg.default_target_count)
        # Resolved once: table rows that name no artist are tagged with it, and
        # the user's spelling («Канье») matches nothing in the library.
        artist = self._library_artist(plan)
        queries, ce_query = list(plan.web_queries), plan.ce_query
        claims: list = []
        notes: list = []
        iterations = relaxations = 0

        while iterations < self.cfg.playlist_max_iterations:
            iterations += 1
            self.sink.put("iteration", n=iterations, queries=queries)

            structured_pages, prose_pages = await self.gather(
                plan, queries, ce_query, structured=True)

            parsed, prose = await self._harvest(structured_pages + prose_pages,
                                                artist=artist)
            claims += parsed

            self.index(prose)
            chunks = self.best_chunks(ce_query)
            if chunks:
                extractor = TrackExtractor(self.agent.llm, self.cfg, self.sink)
                with self.timings.span("llm.extract"):
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

            # Not enough after filtering. Take a constraint OUT of the query text
            # and let the code-side filter do that work instead — the phrase
            # "спокойные хиты 80х" narrows the web result set far more than it
            # narrows the truth.
            relaxed = self._relax(plan, queries, relaxations)
            if relaxed is None:
                notes.append("nothing left to relax")
                break
            relaxations += 1
            queries = relaxed
            notes.append(f"relaxed the query (round {relaxations})")

        resolved, missing = self._resolve(claims, plan)
        if len(resolved) < self.cfg.discography_min_tracks:
            with self.timings.span("discography"):
                rescued = await self._discography(plan, found=len(resolved))
            if rescued:
                claims += rescued
                resolved, missing = self._resolve(claims, plan)
                notes.append(f"discography rescue added {len(rescued)} claims")

        with self.timings.span("llm.triage"):
            resolved = await triage_tracks(self.agent.llm, message, resolved,
                                           config=self.cfg, sink=self.sink)
        resolved = resolved[:cap]
        with self.timings.span("llm.curate"):
            title, comment, resolved = await curate_tracks(
                self.agent.llm, message, resolved, config=self.cfg,
                sink=self.sink)
        self.sink.put("result", tracks=len(resolved), missing=len(missing))
        return PlaylistResult(title=title, comment=comment, tracks=resolved,
                              missing=missing[:40], iterations=iterations,
                              relaxations=relaxations, notes=notes)

    async def _harvest(self, pages: list, *, artist: Optional[str]) -> tuple:
        """``(claims, prose)`` — parse what has a parser, hand back the rest.

        Routing is by HOST, and it used to be by search stream, which is not the
        same thing: ``gather`` splits its result by which stream found each page,
        so a Wikipedia discography that only the open-web search surfaced went
        into the prose lane and was read by the MODEL. That is the most expensive
        way to read a table. Extraction emits one JSON object per track, so its
        output — and therefore its wall time, the largest single span in a
        playlist run — grows with the length of the list it is retyping, and it
        was retyping rows that parse in milliseconds.

        A parser that comes back empty hands the page to the prose lane rather
        than dropping it. ``tracks_from_markdown`` reads markdown TABLES and
        nothing else, so a Fandom soundtrack written as a bulleted list is a page
        full of real tracks with no rows in it — and refusing it on its host alone
        would lose the whole page instead of just the cheap way of reading it.
        """
        claims: list = []
        prose: list = []
        for page in pages:
            if not has_structured_parser(page):
                prose.append(page)
                continue
            with self.timings.span("structured.tables"):
                found = await asyncio.to_thread(structured_tracks, page,
                                                default_artist=artist)
            if found:
                claims += found
                self.sink.put("structured", url=page.url, tracks=len(found))
                continue
            logger.info("[playlist] %s: a parseable host with nothing to parse "
                        "— reading it as prose", page.url)
            prose.append(page)
        return claims, prose

    def _library_artist(self, plan: Plan) -> Optional[str]:
        """The artist as the LIBRARY spells it, best effort.

        Used for two things: the rows of a table that names no artist, and the
        discography search. Both want "Kanye West" where the user typed «Канье» —
        a table row tagged with the Cyrillic spelling matches nothing in the
        library, and a Cyrillic query finds nothing on the English Wikipedia.

        A best guess is acceptable here, unlike when choosing whose FACTS to read:
        every title is still matched against the library under this name, so a
        wrong guess yields no tracks rather than the wrong ones.
        """
        return self.agent.library_artist(plan.filters.artist)

    async def _discography(self, plan: Plan, *, found: int) -> list:
        """Last resort: go and read the artist's discography article.

        The failure this exists for is specific and common. "Хиты Канье после
        2020" has no page — nobody writes a listicle per artist per era — so the
        searches come back with charts, interviews and news, and the deterministic
        stage matches four tracks. Meanwhile Wikipedia has the whole singles
        discography in a table, and the era filter can do the rest itself.

        The query is built by CODE, not by the model: the artist is already
        resolved against the library, and "<artist> discography" is not a
        judgement call. Wikipedia's engine searches article titles, so it is also
        very likely to land exactly right.
        """
        artist = plan.filters.artist
        if not artist:
            return []
        artist = self._library_artist(plan) or artist

        self.sink.put("discography", artist=artist, found=found)
        logger.info("[playlist] only %d tracks — reading %s's discography", found,
                    artist)

        refs: list = []
        tried = 0
        for template in DISCOGRAPHY_TITLES:
            if tried >= self.cfg.discography_max_queries:
                break
            tried += 1
            query = template.format(artist=artist)
            hits = await asyncio.to_thread(self.sources.wikipedia, query, 2, True)
            pages = await self.fetcher.fetch_many(dedupe_by_url(hits), limit=1)
            for page in pages:
                got = await asyncio.to_thread(structured_tracks, page,
                                              default_artist=artist)
                logger.info("[playlist] discography %r -> %s -> %d rows", query,
                            page.url, len(got))
                refs += got
            if refs:
                break
            # Nothing on that page. Almost always the disambiguation stub —
            # "Kanye West discography" is three links and no table — so the next
            # spelling is tried rather than giving up.
            logger.info("[playlist] %r yielded no rows, trying the next title",
                        query)

        self.sink.put("discography_done", queries=tried, claims=len(refs))
        return refs

    # ── selection, all of it code ─────────────────────────────────────────

    def _resolve(self, claims: list, plan: Plan) -> tuple:
        """Claims → library tracks, weighted, era-filtered, best first."""
        with self.timings.span("resolve.library"):
            tracks, missing = select_tracks(
                self.agent.catalog, claims, era=plan.filters.era,
                source_weights=self.cfg.source_weights)
        if plan.filters.era:
            self.sink.put("era_filter", range=plan.filters.era, kept=len(tracks))
        return tracks, missing

    def _relax(self, plan: Plan, queries: list, done: int) -> Optional[list]:
        """Drop one constraint from the query TEXT (never from the filters).

        Order matters: the style word is the one that most distorts a web search
        ("спокойные хиты 80х" finds listicles about calm music, not the decade's
        hits), the era is second. Both keep filtering results afterwards — this
        only changes what we ask the internet for.
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


def _strip(text: str, token: str) -> str:
    """Remove ``token`` from ``text``, case-insensitively, as a whole phrase."""
    if not token:
        return text
    return " ".join(re.sub(re.escape(token), " ", text, flags=re.I).split())
