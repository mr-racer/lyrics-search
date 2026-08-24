"""Finding tracks by how they SOUND: CLAP text→audio, four angles, RRF.

The LLM is called twice and neither call is a judgement. The planner turned the
sentence into an intent and a set of filters; this branch asks it once more to
rewrite the user's description of sound into CLAP's own dialect, and then does
arithmetic. There is nothing here for a model to validate — the vector store
already answered — so the caption is written by code and no third call is made.

Why four rephrasings rather than one: a single prompt is one lucky or unlucky
point in CLAP's text space, and its top-10 swings hard on wording that means the
same thing to a human. Four prompts that differ only in which acoustic parameter
they emphasise, merged by reciprocal rank, turn "whatever this phrasing happened
to retrieve" into "what every phrasing agreed on".

The artist never enters the prompt text. It is doing its work as a filter, where
it is exact and free; inside a CLAP prompt it is noise that drags the vector
towards whatever that artist's most typical track sounds like.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.domain.models import SearchFilters
from app.services.assistant.contracts import AudioResult, Plan
from app.services.assistant.llm import as_str_list
from app.services.assistant.prompts import CLAP_REPHRASE_SYSTEM
from app.services.library_catalog import filter_by_era

logger = logging.getLogger(__name__)


class AudioBranch:
    def __init__(self, agent):
        self.agent = agent
        self.cfg = agent.cfg
        self.sink = agent.sink
        self.timings = agent.timings

    async def run(self, message: str, plan: Plan) -> AudioResult:
        service = self.agent.search_service
        if service is None:
            return AudioResult(title=message[:80], comment="",
                               notes=["search service unavailable"])

        artist, artist_slug = self.agent.library_artist_ref(plan.filters.artist)
        style = plan.filters.style or message

        with self.timings.span("llm.clap_rephrase"):
            queries = await self._rephrase(style, artist)
        self.sink.put("clap_rephrase", queries=queries)
        logger.info("[audio] %r -> %s", style, queries)

        if not queries:
            return AudioResult(title=_title(message),
                               comment=_no_prompt(self.cfg.lang),
                               notes=["clap rephrasing failed"])

        # A resolved artist is a filter Qdrant can apply itself: every point
        # carries artist_slugs for all of its participants, keyword-indexed, so
        # "Sade feat. …" matches on the same footing as "Sade" and CLAP ranks
        # WITHIN the artist. What that replaces was a funnel — top-K over the
        # whole library, then keep the few tracks that happened to belong to the
        # artist — and on a real library it ended with two tracks out of fifteen.
        #
        # Only what still runs after the search needs a deeper pool: the era, and
        # an artist whose name did not resolve to a slug (ambiguous, or spelled
        # in a way the catalog cannot pin), which falls back to matching names.
        post_filters = bool(plan.filters.era or (artist and not artist_slug))
        limit = self.cfg.clap_limit_per_query * (2 if post_filters else 1)
        filters = SearchFilters(artist_slug=artist_slug) if artist_slug else None
        rankings: list = []
        by_id: dict = {}
        with self.timings.span("search.clap"):
            for query in queries:
                try:
                    hits = await service.search(
                        query, mode="audio", limit=limit, filters=filters,
                        collection_name=self.agent.collection_name)
                except Exception as exc:  # noqa: BLE001 — one prompt of four
                    logger.warning("[audio] search failed for %r: %s", query, exc)
                    continue
                # Qdrant already answered the artist question when it had a
                # slug; re-checking the display name here could only drop a
                # track it was right about (a compilation credit, a group whose
                # tag names its members).
                hits = self._narrow(hits, era=plan.filters.era,
                                    artist=None if artist_slug else artist)
                order = []
                for hit in hits:
                    track_id = hit.track.track_id
                    if not track_id:
                        continue
                    by_id.setdefault(track_id, hit)
                    order.append(track_id)
                rankings.append(order)
                self.sink.put("search", source="clap", query=query,
                              found=len(order))

        if not by_id:
            return AudioResult(title=_title(message), comment=_nothing(self.cfg.lang),
                               queries=queries, notes=["nothing matched"])

        merged = _rrf(rankings, k=self.cfg.clap_rrf_k)
        tracks = [by_id[tid] for tid, _ in merged[:self.cfg.clap_result_count]]
        for rank, (tid, score) in enumerate(merged[:len(tracks)]):
            by_id[tid].score = float(score)

        self.sink.put("result", tracks=len(tracks), missing=0)
        return AudioResult(title=_title(message),
                           comment=_caption(len(tracks), artist, self.cfg.lang),
                           tracks=tracks, queries=queries)

    async def _rephrase(self, style: str, artist: Optional[str]) -> list:
        """The user's description of sound as ``clap_queries`` CLAP prompts.

        The artist name is given as CONTEXT for the model to reason from ("Sade
        sounds smooth and slow"), never to be echoed into the prompt — rule 8 of
        the system prompt covers that, and the prompt itself forbids the name.
        """
        hint = f"{style} (the listener also named the artist {artist})" if artist \
            else style
        messages = [
            {"role": "system",
             "content": CLAP_REPHRASE_SYSTEM.format(user_query=hint)},
            {"role": "user", "content": hint},
        ]

        def _usable(value) -> list:
            return [q for q in as_str_list(value, limit=self.cfg.clap_queries,
                                           item_limit=200) if q]

        queries = _usable(await self.agent.llm.ask_list(messages))
        if not queries:
            # One repair round before giving up. ``ask_list`` skips the repair
            # on the theory that every caller has a cheap deterministic
            # fallback — which is true here only for a listener who wrote in
            # English, and these failures are formatting rather than
            # capability: the model writes four good prompts and names the
            # field something the coercion did not read.
            last_raw = getattr(self.agent.llm, "last_raw", "") or ""
            logger.info("[audio] rephrasing unusable (%r) — one repair round",
                        last_raw[:160])
            queries = _usable(await self.agent.llm.ask_list(messages + [
                {"role": "assistant", "content": last_raw[:2000]},
                {"role": "user",
                 "content": "Reply with the JSON array of 4 plain strings and "
                            "nothing else. No objects, no keys, no prose, no "
                            "markdown fence."},
            ]))

        if not queries:
            # The old fallback pasted the style straight into the template and
            # searched that. It only works when the style is already English:
            # CLAP's text tower is trained on English audio captions, so
            # "This song is a спокойное" is not a weak query, it is noise, and
            # the ten tracks it returns are indistinguishable from random. An
            # empty list makes the branch say so instead of pretending.
            if _is_latin(style):
                logger.info("[audio] rephrasing failed — falling back to the "
                            "listener's own words")
                return [f"This song is a {style}"]
            logger.warning("[audio] rephrasing failed and %r is not usable as a "
                           "CLAP prompt — no audio search", style)
            return []
        return queries[:self.cfg.clap_queries]

    def _narrow(self, hits: list, *, artist: Optional[str],
                era: Optional[tuple]) -> list:
        if artist:
            from app.services.assistant.branches.lyrics import _artist_matches

            hits = [h for h in hits if _artist_matches(artist, h.track.artist)]
        if era:
            hits = filter_by_era(hits, era, year=lambda h: h.track.year)
        return hits


def _rrf(rankings: list, *, k: int = 60) -> list:
    """``[(track_id, score), ...]`` best first, fused across the ranked lists.

    Reciprocal rank rather than raw CLAP similarity: the four prompts are four
    different points in the text space and their score scales are not comparable,
    while their ORDERS are. A track that every prompt put near the top beats one
    that a single prompt loved.
    """
    scores: dict = {}
    for order in rankings:
        for rank, track_id in enumerate(order, start=1):
            scores[track_id] = scores.get(track_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _title(message: str) -> str:
    return " ".join((message or "").split())[:80] or "Подборка"


def _caption(count: int, artist: Optional[str], lang: str) -> str:
    ru = (lang or "").lower().startswith("ru")
    if ru:
        who = f" у {artist}" if artist else ""
        return f"Подобрал {count} трек(ов) по звучанию{who}."
    who = f" by {artist}" if artist else ""
    return f"{count} tracks picked by how they sound{who}."


def _is_latin(text: str) -> bool:
    """True when ``text`` carries at least one Latin letter and no other script.

    The gate for "may this go into a CLAP prompt": the model is English-only,
    so a Cyrillic (or CJK, or Greek) style word is out-of-distribution noise
    rather than a weak query. Digits and punctuation are neutral.
    """
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if not letters:
        return False
    return all("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in letters)


def _no_prompt(lang: str) -> str:
    return ("Не получилось перевести описание звучания в запрос для поиска по "
            "звуку — попробуй переформулировать."
            if (lang or "").lower().startswith("ru") else
            "Could not turn that description into an audio-search prompt — "
            "try rephrasing.")


def _nothing(lang: str) -> str:
    return ("По звучанию ничего не нашлось — попробуй описать иначе."
            if (lang or "").lower().startswith("ru") else
            "Nothing matched by sound — try describing it differently.")
