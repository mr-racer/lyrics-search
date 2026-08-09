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
from lab.agent.spam import is_spam_host, spam_report
from lab.agent.urls import (dedupe_by_url, normalise_apple_url,
                            source_for_url)

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


def _engine_names(row: dict) -> list[str]:
    return row.get("engines") or ([row["engine"]] if row.get("engine") else [])


def engine_host_spread(results: list[dict]) -> dict[str, dict]:
    """Per engine: how many results, and from how many distinct hosts.

    This is the measurement that identifies a broken engine, and no blocklist
    can replace it. A scraper that has landed on the wrong page — a redirect, an
    error page, an ad portal — returns THAT page's navigation as results: ten
    links, one host, none of them about the query. Observed as a Polish TV
    guide's channel list, a Czech portal's sections and a run of Indonesian
    journal PDFs, all for music queries.

    A healthy engine returns many hosts. One host and several results is the
    signature.
    """
    out: dict[str, dict] = {}
    for row in results:
        host = _host(row.get("url") or "")
        for name in _engine_names(row):
            entry = out.setdefault(name, {"results": 0, "hosts": set()})
            entry["results"] += 1
            if host:
                entry["hosts"].add(host)
    return {name: {"results": e["results"], "hosts": len(e["hosts"]),
                   "top_host": max(e["hosts"], key=len, default="") if len(e["hosts"]) == 1
                   else ""}
            for name, e in sorted(out.items(), key=lambda kv: -kv[1]["results"])}


def dominating_host(results: list[dict], *, min_share: float,
                    min_count: int) -> Optional[str]:
    """The host that took over the result set, if there is one."""
    if len(results) < min_count:
        return None
    counts: dict[str, int] = {}
    for row in results:
        host = _host(row.get("url") or "")
        if host:
            counts[host] = counts.get(host, 0) + 1
    if not counts:
        return None
    host, count = max(counts.items(), key=lambda kv: kv[1])
    return host if count >= min_count and count / len(results) >= min_share else None


def _count_by_engine(results: list[dict]) -> dict[str, int]:
    """How many results each engine contributed, best-effort.

    SearXNG reports both a single ``engine`` and, when several found the same
    page, an ``engines`` list. Counting the list is what shows an engine that
    is technically responding but only ever corroborating others.
    """
    counts: dict[str, int] = {}
    for row in results:
        names = row.get("engines") or ([row["engine"]] if row.get("engine") else [])
        for name in names:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def is_junk(url: str) -> bool:
    """A URL not worth a slot: a known dead end, or outright spam.

    Two different reasons kept behind one gate at the edge. A dead end
    (Spotify, Instagram) is a real page we cannot read; spam is a broken
    engine's output. Both are dropped here, at ingestion, before dedup and
    before the cross-encoder — the counters that tell them apart live in
    ``SearchSources.diagnostics``.
    """
    return not url or bool(JUNK_URL.search(url)) or is_spam_host(url)


class SearchSources:
    """SearXNG, sliced by host. One instance per run."""

    def __init__(self, config=None, sink=None):
        from lab.agent.config import AgentConfig

        self.cfg = config or AgentConfig()
        self.sink = sink
        self.searches = 0
        self._seen_queries: set[str] = set()
        # One entry per SearXNG call: who answered, who did not, how many
        # results each engine contributed. Read it after a run to find out why
        # the same query by hand looked better.
        self.diagnostics: list[dict] = []
        self.last_response: dict = {}
        self._next_call_at = 0.0
        self._publish_searxng_url()

    def suspended_engines(self) -> dict[str, str]:
        """Engines that failed at least once this run, and why.

        The single most useful thing to look at after a run that returned
        nonsense: if google, duckduckgo and brave are all in here, the answer
        was assembled by whoever was left.
        """
        out: dict[str, str] = {}
        for entry in self.diagnostics:
            for name, reason in entry.get("unresponsive") or []:
                out[name] = reason
        return out

    def spam_by_engine(self) -> dict[str, int]:
        """Adult/gambling results contributed per engine, across the run.

        This is the number that answers "which engine do I remove?" — a broken
        engine's spam is fused in by rank and looks like any other result from
        inside, so without attribution the only symptom is a filthy answer.
        """
        out: dict[str, int] = {}
        for entry in self.diagnostics:
            for engine, count in (entry.get("spam_by_engine") or {}).items():
                out[engine] = out.get(engine, 0) + count
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def suspect_engines(self) -> dict[str, dict]:
        """Engines that returned several results from a SINGLE host.

        The one measurement that finds a broken scraper. It has landed on the
        wrong page and is returning that page's navigation, so its results are
        many and its hosts are one — while a healthy engine's results are
        spread across the web. No blocklist finds this, because the host is
        usually a perfectly real site that has nothing to do with the query.
        """
        totals: dict[str, dict] = {}
        for entry in self.diagnostics:
            for name, spread in (entry.get("engine_spread") or {}).items():
                if spread["hosts"] == 1 and spread["results"] >= 3:
                    row = totals.setdefault(
                        name, {"dumps": 0, "results": 0, "hosts": set()})
                    row["dumps"] += 1
                    row["results"] += spread["results"]
                    if spread["top_host"]:
                        row["hosts"].add(spread["top_host"])
        return {name: {"dumps": r["dumps"], "results": r["results"],
                       "hosts": sorted(r["hosts"])}
                for name, r in sorted(totals.items(),
                                      key=lambda kv: -kv[1]["results"])}

    def report(self) -> str:
        """One line per search: what was asked, who answered, who did not."""
        lines = []
        for entry in self.diagnostics:
            engines = ", ".join(f"{k}:{v}" for k, v in entry["per_engine"].items())
            lines.append(f"{entry['results']:>3} results  {entry['query'][:60]!r}")
            lines.append(f"     from: {engines or '(nobody)'}")
            for name, spread in (entry.get("engine_spread") or {}).items():
                if spread["hosts"] == 1 and spread["results"] >= 3:
                    lines.append(f"     DUMP: {name} returned "
                                 f"{spread['results']} results, all from "
                                 f"{spread['top_host'] or '(one host)'}")
            if entry.get("takeover_host"):
                lines.append(f"     TAKEOVER dropped: {entry['takeover_host']}")
            if entry.get("spam"):
                by = ", ".join(f"{k}:{v}" for k, v
                               in (entry.get("spam_by_engine") or {}).items())
                hosts = ", ".join(list(entry.get("spam_hosts") or {})[:4])
                lines.append(f"     SPAM: {entry['spam']} dropped "
                             f"[{by or 'engine not reported'}] {hosts}")
            if entry["unresponsive"]:
                dead = "; ".join(f"{n} ({r})" for n, r in entry["unresponsive"])
                lines.append(f"     DOWN: {dead}")
        return "\n".join(lines)

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

    def _searx(self, query: str, *, engines: Optional[str], limit: int,
               force: bool = False) -> list[dict]:
        """One SearXNG call. Budgeted, deduplicated, never raises.

        A repeat of a query already run returns nothing rather than the same
        pages again: small models rephrase cosmetically when they are stuck,
        and paying for that twice is how a run burns its budget without
        learning anything.

        The raw JSON is read here rather than through
        ``websearch_lab.search_searxng`` for one reason: that helper returns
        ``results`` and drops the rest, and the rest is where the diagnosis
        lives. ``unresponsive_engines`` is how you learn that DuckDuckGo timed
        out and the answer you are looking at came from three fallbacks — which
        is invisible otherwise, and shows up only as "the results got worse".
        """
        norm = " ".join((query or "").lower().split())
        if not norm:
            return []
        key = f"{engines or '*'}::{norm}"
        if key in self._seen_queries:
            logger.info("[sources] repeat refused: %r", query)
            return []
        if self.searches >= self.cfg.max_web_searches:
            if not force:
                logger.info("[sources] search budget spent (%d)", self.searches)
                return []
            # ``force`` is for the last-resort searches only — the rescue that
            # runs when a playlist would otherwise come back nearly empty. It
            # still counts against the budget, so it cannot loop.
            logger.info("[sources] budget spent, running %r anyway (forced)",
                        query)
        self._seen_queries.add(key)
        self.searches += 1

        results = self._searx_raw(query, engines=engines, limit=limit)
        if results is None:
            from lab import websearch_lab as L

            logger.info("[sources] falling back to DDG direct for %r", query)
            try:
                return L.search_ddg(query, max_results=limit)
            except Exception:
                logger.warning("[sources] DDG fallback failed", exc_info=True)
                return []
        return results

    def _searx_raw(self, query: str, *, engines: Optional[str],
                   limit: int) -> Optional[list[dict]]:
        """The JSON call. ``None`` means the instance itself did not answer."""
        import time

        import httpx

        # Space the calls out. See AgentConfig.searx_min_interval — a burst is
        # what gets the good engines suspended, and a suspended engine is
        # invisible except as "the results got worse".
        wait = self._next_call_at - time.monotonic()
        if wait > 0:
            logger.debug("[sources] pacing: waiting %.1fs", wait)
            time.sleep(wait)
        self._next_call_at = time.monotonic() + self.cfg.searx_min_interval

        base = self.cfg.searxng_url
        params = {
            "q": query,
            "format": "json",
            "language": self.cfg.searx_language,
            "safesearch": 0,
            "pageno": 1,
        }
        pinned = engines if engines is not None else self.cfg.searx_engines
        if pinned:
            params["engines"] = pinned

        try:
            resp = httpx.get(f"{base}/search", params=params, timeout=20, headers={
                "Accept": "application/json, text/javascript, */*",
                "Referer": f"{base}/",
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"),
            })
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[sources] searxng unreachable for %r: %s: %s",
                           query, type(exc).__name__, exc)
            return None

        all_results = data.get("results") or []
        # Spam is attributed BEFORE it is dropped. Knowing that an engine
        # returned nine cam sites is the only way to decide whether to keep
        # asking it; dropping the results silently leaves you with "the answers
        # got worse" and nothing to act on.
        spam_rows = [r for r in all_results if is_spam_host(r.get("url") or "")]
        clean = [r for r in all_results if not is_spam_host(r.get("url") or "")]

        # A query pinned to one HOST is supposed to come back from one host, so
        # the takeover rules are off for it. Two ways that happens: a `site:`
        # operator, or a single-engine request (engines=wikipedia).
        #
        # Note what does NOT count: `pinned` being set. The engine WHITELIST is
        # non-empty on every open-web search, and reading that as "host-pinned"
        # switched the whole check off silently — which is how it shipped
        # broken the first time.
        pinned_to_host = ("site:" in (query or "").lower()
                          or (engines is not None and "," not in engines))
        dominant = None
        if not pinned_to_host:
            dominant = dominating_host(
                clean, min_share=self.cfg.host_takeover_share,
                min_count=self.cfg.host_takeover_min)
            if dominant:
                clean = [r for r in clean if _host(r.get("url") or "") != dominant]

        capped, per_host = [], {}
        for row in clean:
            host = _host(row.get("url") or "")
            if not pinned_to_host and host:
                per_host[host] = per_host.get(host, 0) + 1
                if per_host[host] > self.cfg.max_results_per_host:
                    continue
            capped.append(row)
        clean = capped
        results = clean[:limit]

        self.last_response = {
            "query": query,
            "engines_asked": pinned or "(server default)",
            "results": len(clean),
            "per_engine": _count_by_engine(clean),
            "spam": len(spam_rows),
            "spam_by_engine": _count_by_engine(spam_rows),
            "spam_hosts": spam_report(r.get("url") or "" for r in spam_rows),
            "takeover_host": dominant,
            "engine_spread": engine_host_spread(all_results),
            "unresponsive": [list(e) for e in (data.get("unresponsive_engines") or [])],
        }
        self.diagnostics.append(self.last_response)

        if dominant:
            culprits = {name: e for name, e in
                        self.last_response["engine_spread"].items()
                        if e["top_host"] == dominant}
            logger.warning(
                "[sources] %r took over the results for %r — dropped. "
                "Engine(s) dumping it: %s", dominant, query,
                culprits or "(not reported; see engine_spread)")
            if self.sink is not None:
                self.sink.put("host_takeover", query=query, host=dominant,
                              engines=sorted(culprits))

        if spam_rows:
            logger.warning(
                "[sources] dropped %d adult/gambling results for %r — from %s",
                len(spam_rows), query,
                self.last_response["spam_by_engine"] or "(engine not reported)")
            if self.sink is not None:
                self.sink.put("spam_dropped", query=query, count=len(spam_rows),
                              engines=self.last_response["spam_by_engine"])

        dead = self.last_response["unresponsive"]
        if dead:
            # Loud, because this is the difference between "the web has nothing"
            # and "half the web was not asked".
            logger.warning("[sources] %d engine(s) did not answer %r: %s",
                           len(dead), query,
                           "; ".join(f"{name}: {reason}" for name, reason in dead))
        logger.info("[sources] %r -> %d results from %s", query, len(results),
                    self.last_response["per_engine"] or "nobody")
        if self.sink is not None and dead:
            self.sink.put("engines_down", query=query,
                          engines=[name for name, _ in dead])
        return results

    @staticmethod
    def _to_hits(rows: list[dict], source: str,
                 *, host_filter: Optional[tuple[str, ...]] = None) -> list[SearchHit]:
        out: list[SearchHit] = []
        for i, row in enumerate(rows):
            url = row.get("url") or ""
            if is_junk(url):
                continue
            # Rewritten here rather than at fetch time so that dedup, the
            # reranker and the fetcher all agree on which page this is.
            better = normalise_apple_url(url)
            if better != url:
                logger.info("[sources] %s -> %s (one storefront; the artist "
                            "landing page rotates and mixes in other "
                            "artists' songs)", url, better)
                url = better
            if host_filter:
                host = _host(url)
                if not any(host == h or host.endswith("." + h) or h in host
                           for h in host_filter):
                    continue
            out.append(SearchHit(
                url=url, title=(row.get("title") or "").strip(),
                snippet=(row.get("content") or "").strip(),
                # The host decides, not the stream. Google returning a
                # Wikipedia article makes it a Wikipedia article.
                source=source_for_url(url, fallback=source), rank=i))
        return out

    # ── the four sources ──────────────────────────────────────────────────

    def web(self, query: str) -> list[SearchHit]:
        """Open web, deep pool — the cross-encoder decides what survives."""
        rows = self._searx(query, engines=None, limit=self.cfg.searx_pool)
        hits = self._to_hits(rows, "web")
        self._emit("search", source="web", query=query, found=len(hits))
        return hits

    def wikipedia(self, query: str, limit: int = 2,
                  force: bool = False) -> list[SearchHit]:
        """Wikipedia only. The engine searches article TITLES, so any query
        naming an artist lands on their page — which is what we want here and
        exactly why this engine is excluded from the open-web call."""
        rows = self._searx(query, engines="wikipedia", limit=max(limit, 5),
                           force=force)
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

    Deduplicated FIRST, and that ordering matters. An iteration runs two
    queries and concatenates their results, so a page both queries found
    arrives twice — with different titles and snippets, because each engine
    words them its own way. Scored twice it takes two slots in the most
    expensive stage of the pipeline and comes back with two different
    probabilities for the same document, which is nonsense on its face.

    With no cross-encoder available the list comes back deduplicated but
    unfiltered: an unfiltered pool is a worse pool, an empty one is no pool.
    """
    if not hits:
        return []
    before = len(hits)
    hits = dedupe_by_url(hits)
    if before != len(hits):
        logger.info("[sources] %d duplicate URLs collapsed before reranking",
                    before - len(hits))
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
