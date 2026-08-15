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

        artist = self.agent.library_artist(plan.filters.artist)
        style = plan.filters.style or message

        with self.timings.span("llm.clap_rephrase"):
            queries = await self._rephrase(style, artist)
        self.sink.put("clap_rephrase", queries=queries)
        logger.info("[audio] %r -> %s", style, queries)

        # Deeper per query when a filter is on, for the same reason as in the
        # lyrics branch: the artist filter runs here rather than in Qdrant, whose
        # payload filter is an exact match and would drop "Sade feat. …".
        limit = self.cfg.clap_limit_per_query * (2 if (artist or plan.filters.era)
                                                 else 1)
        rankings: list = []
        by_id: dict = {}
        with self.timings.span("search.clap"):
            for query in queries:
                try:
                    hits = await service.search(
                        query, mode="audio", limit=limit,
                        collection_name=self.agent.collection_name)
                except Exception as exc:  # noqa: BLE001 — one prompt of four
                    logger.warning("[audio] search failed for %r: %s", query, exc)
                    continue
                hits = self._narrow(hits, artist=artist, era=plan.filters.era)
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
        raw = await self.agent.llm.ask_list([
            {"role": "system",
             "content": CLAP_REPHRASE_SYSTEM.format(user_query=hint)},
            {"role": "user", "content": hint},
        ])
        queries = [q for q in as_str_list(raw, limit=self.cfg.clap_queries,
                                          item_limit=200) if q]
        if not queries:
            # A deterministic fallback rather than a second call: one prompt is
            # worse than four, and "no answer" is worse than one.
            logger.info("[audio] rephrasing produced nothing — searching the "
                        "user's own words")
            return [f"This song is a {style}"]
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


def _nothing(lang: str) -> str:
    return ("По звучанию ничего не нашлось — попробуй описать иначе."
            if (lang or "").lower().startswith("ru") else
            "Nothing matched by sound — try describing it differently.")
