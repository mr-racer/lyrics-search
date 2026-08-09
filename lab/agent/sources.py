"""Where results come from, and how a result is judged before it is downloaded.

Four sources, two very different treatments:

* **Wikipedia, Apple Music, Fandom** are searched with the host pinned. What
  comes back is trusted enough to fetch without a relevance pass — the host
  filter has already done the work a reranker would.
* **The open web** is pulled 20 deep and passed through the cross-encoder as
  ``title + snippet``. Only candidates above the threshold get downloaded.
  This is where the money is: fetching five pages costs seconds each, and the
  cross-encoder can tell in one batched forward pass which five are worth it.

What is NOT here any more: the hand-tuned authority table (billboard 2.8,
pitchfork 2.4, …). It was a static prior standing in for a relevance judgement
that we can now actually make. The junk-URL blacklist stays — a Spotify or
Instagram link is never the answer, and dropping it costs nothing and saves a
cross-encoder slot.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional
from urllib.parse import urlparse

from lab.agent.models import SearchHit

logger = logging.getLogger(__name__)

# Kept from the production ranker: these hosts never carry the text we want.
JUNK_URL = re.compile(
    r"""(?ix)
      genius\.com/artists/
    | genius\.com/[^/]+-annotated
    | genius\.com/[^/?#]+-lyrics\b
    | //(?:www\.)?instagram\.com
    | //(?:www\.)?facebook\.com
    | //(?:www\.)?x\.com
    | //(?:www\.)?twitter\.com
    | ticketmaster\.
    | //(?:www\.|music\.)?youtube\.com/(?:channel|@|watch|playlist)
    | /tickets?\b
    | //(?:www\.)?open\.spotify\.com
    | //(?:www\.)?deezer\.com
    """
)


def _host(url: str) -> str:
    return (urlparse(url).netloc or "").lower()


def is_junk(url: str) -> bool:
    return not url or bool(JUNK_URL.search(url))


class SearchSources:
    """SearXNG, sliced by host. One instance per run."""

    def __init__(self, config=None, sink=None):
        from lab.agent.config import AgentConfig

        self.cfg = config or AgentConfig()
        self.sink = sink
        self.searches = 0
        self._seen_queries: set[str] = set()
        self._publish_searxng_url()

    def _publish_searxng_url(self) -> None:
        """Point ``websearch_lab`` at the configured SearXNG.

        It reads its address from ``os.environ`` at call time (that is what its
        own ``configure()`` writes to, and what makes it survive %autoreload).
        Without this the AgentConfig field is decoration: every search would
        quietly go to whatever the notebook happened to configure last, or to
        localhost.
        """
        if self.cfg.searxng_url:
            os.environ["SEARXNG_URL"] = self.cfg.searxng_url

    def _emit(self, stage: str, **fields) -> None:
        if self.sink is not None:
            self.sink.put(stage, **fields)

    # ── the raw call ──────────────────────────────────────────────────────

    def _searx(self, query: str, *, engines: Optional[str],
               limit: int) -> list[dict]:
        """One SearXNG call. Budgeted, deduplicated, never raises.

        A repeat of a query already run returns nothing rather than the same
        pages again: small models rephrase cosmetically when they are stuck,
        and paying for that twice is how a run burns its budget without
        learning anything.
        """
        norm = " ".join((query or "").lower().split())
        if not norm:
            return []
        key = f"{engines or '*'}::{norm}"
        if key in self._seen_queries:
            logger.info("[sources] repeat refused: %r", query)
            return []
        if self.searches >= self.cfg.max_web_searches:
            logger.info("[sources] search budget spent (%d)", self.searches)
            return []
        self._seen_queries.add(key)
        self.searches += 1

        from lab import websearch_lab as L

        try:
            return L.search_searxng(query, max_results=limit, engines=engines)
        except Exception:
            logger.warning("[sources] searxng failed for %r", query, exc_info=True)
            return []

    @staticmethod
    def _to_hits(rows: list[dict], source: str,
                 *, host_filter: Optional[tuple[str, ...]] = None) -> list[SearchHit]:
        out: list[SearchHit] = []
        for i, row in enumerate(rows):
            url = row.get("url") or ""
            if is_junk(url):
                continue
            if host_filter:
                host = _host(url)
                if not any(host == h or host.endswith("." + h) or h in host
                           for h in host_filter):
                    continue
            out.append(SearchHit(
                url=url, title=(row.get("title") or "").strip(),
                snippet=(row.get("content") or "").strip(),
                source=source, rank=i))
        return out

    # ── the four sources ──────────────────────────────────────────────────

    def web(self, query: str) -> list[SearchHit]:
        """Open web, deep pool — the cross-encoder decides what survives."""
        rows = self._searx(query, engines=None, limit=self.cfg.searx_pool)
        hits = self._to_hits(rows, "web")
        self._emit("search", source="web", query=query, found=len(hits))
        return hits

    def wikipedia(self, query: str, limit: int = 2) -> list[SearchHit]:
        """Wikipedia only. The engine searches article TITLES, so any query
        naming an artist lands on their page — which is what we want here and
        exactly why this engine is excluded from the open-web call."""
        rows = self._searx(query, engines="wikipedia", limit=max(limit, 5))
        hits = self._to_hits(rows, "wikipedia",
                             host_filter=("wikipedia.org",))[:limit]
        self._emit("search", source="wikipedia", query=query, found=len(hits))
        return hits

    def apple_music(self, query: str, limit: int = 3) -> list[SearchHit]:
        rows = self._searx(f"site:music.apple.com {query}", engines=None,
                           limit=self.cfg.searx_pool)
        hits = self._to_hits(rows, "apple",
                             host_filter=("music.apple.com",))[:limit]
        self._emit("search", source="apple", query=query, found=len(hits))
        return hits

    def fandom(self, query: str, limit: int = 2) -> list[SearchHit]:
        rows = self._searx(f"site:fandom.com {query}", engines=None,
                           limit=self.cfg.searx_pool)
        hits = self._to_hits(rows, "fandom", host_filter=("fandom.com",))[:limit]
        self._emit("search", source="fandom", query=query, found=len(hits))
        return hits

    def wikipedia_title(self, term: str) -> Optional[str]:
        """The title of the best Wikipedia article for ``term``.

        Used to expand an abbreviation the model was not sure about: the
        article title for "GTA 5" is "Grand Theft Auto V", which is a better
        expansion than anything a 12b model invents.
        """
        rows = self._searx(term, engines="wikipedia", limit=3)
        for row in rows:
            if "wikipedia.org" not in _host(row.get("url") or ""):
                continue
            title = (row.get("title") or "").strip()
            # SearXNG appends " - Wikipedia" on some engines.
            title = re.sub(r"\s+[-–—]\s+Wikipedia\s*$", "", title).strip()
            if title:
                return title
        return None


def rerank_hits(hits: list[SearchHit], ce_query: str, *, hub,
                threshold: float) -> list[SearchHit]:
    """Keep the hits whose title+snippet the cross-encoder likes.

    With no cross-encoder available the list comes back untouched: an
    unfiltered pool is a worse pool, an empty one is no pool at all.
    """
    if not hits:
        return []
    docs = [f"{h.title}\n{h.snippet}".strip() for h in hits]
    probs = hub.ce_probabilities(ce_query, docs)
    if probs is None:
        logger.info("[sources] no cross-encoder — %d hits pass unfiltered",
                    len(hits))
        return hits
    for hit, p in zip(hits, probs):
        hit.ce_prob = p
    kept = [h for h in hits if (h.ce_prob or 0.0) >= threshold]
    kept.sort(key=lambda h: -(h.ce_prob or 0.0))
    logger.info("[sources] cross-encoder kept %d/%d at p>=%.2f",
                len(kept), len(hits), threshold)
    return kept
