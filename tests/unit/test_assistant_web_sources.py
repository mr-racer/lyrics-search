"""Search sources: the budget, the deduplication, the host pinning.

The config-plumbing test is here because the failure it guards against is
invisible: with the address unpublished every search still "works", it just
goes somewhere else, and the only symptom is worse results.
"""

import os

import pytest

from app.services.assistant.config import AgentConfig
from app.services.assistant.contracts import SearchHit
from app.services.assistant.web_sources import SearchSources, is_junk, rerank_hits


@pytest.fixture
def sources(monkeypatch):
    """A SearchSources whose transport is stubbed.

    Stubbed at ``_searx_raw``, which is the seam between "decide what to ask"
    and "ask it". An earlier version of this fixture patched a function the
    code no longer called, so the tests quietly started hitting a real SearXNG
    on the LAN — they passed, slowly, and depended on someone's server.
    """
    cfg = AgentConfig(max_web_searches=3)
    src = SearchSources(cfg)
    calls: list[dict] = []

    def fake_raw(query, *, engines=None, limit=10, host_pinned=False):
        calls.append({"query": query, "engines": engines, "limit": limit,
                      "host_pinned": host_pinned})
        return [{"url": "https://en.wikipedia.org/wiki/Kanye_West",
                 "title": "Kanye West", "content": "rapper"},
                {"url": "https://music.apple.com/us/playlist/x/pl.1",
                 "title": "Playlist", "content": "songs"},
                {"url": "https://open.spotify.com/track/1",
                 "title": "Spotify", "content": "listen"},
                {"url": "https://example.com/best-songs",
                 "title": "Best songs", "content": "a list"}]

    monkeypatch.setattr(src, "_searx_raw", fake_raw)
    src._calls = calls
    return src


class TestConfigPlumbing:
    """The instance address is NOT an agent config field.

    It lives in the environment and is read at call time, exactly like the one
    ``llm_web_search`` uses — a second copy on the config would be a second
    source of truth, and the failure it causes is invisible: every search still
    "works", it just goes somewhere else, and the only symptom is worse results.
    """

    def test_the_address_is_read_at_call_time(self, monkeypatch):
        from app.resources import searxng_client

        monkeypatch.setenv("SEARXNG_URL", "http://10.0.0.5:8088")
        assert searxng_client.base_url() == "http://10.0.0.5:8088"

    def test_a_trailing_slash_is_normalised_away(self, monkeypatch):
        from app.resources import searxng_client

        monkeypatch.setenv("SEARXNG_URL", "http://10.0.0.5:8088/")
        assert searxng_client.base_url() == "http://10.0.0.5:8088"


class TestJunk:
    @pytest.mark.parametrize("url", [
        "https://open.spotify.com/track/1",
        "https://www.instagram.com/kanyewest",
        "https://genius.com/artists/Kanye-west",
        "https://www.youtube.com/watch?v=1",
    ])
    def test_known_dead_ends_are_dropped(self, url):
        assert is_junk(url)

    @pytest.mark.parametrize("url", [
        "https://en.wikipedia.org/wiki/Kanye_West",
        "https://music.apple.com/us/playlist/x/pl.1",
        "https://gta.fandom.com/wiki/Soundtrack",
    ])
    def test_useful_hosts_survive(self, url):
        assert not is_junk(url)


class TestSources:
    def test_the_open_web_drops_junk_and_keeps_the_rest(self, sources):
        hits = sources.web("kanye west")
        urls = [h.url for h in hits]
        assert "https://open.spotify.com/track/1" not in urls
        assert "https://example.com/best-songs" in urls

    def test_wikipedia_pins_the_host_and_the_engine(self, sources):
        hits = sources.wikipedia("kanye west")
        assert all("wikipedia.org" in h.url for h in hits)
        assert sources._calls[-1]["engines"] == "wikipedia"

    def test_the_open_web_asks_for_the_configured_whitelist(self, sources):
        """None would mean the server's stock ~70-engine general set, which is
        where the Indonesian journal PDFs came from."""
        sources.web("kanye west")
        assert sources._calls[-1]["engines"] is None      # resolved downstream
        assert sources.cfg.searx_engines
        assert "duckduckgo" in sources.cfg.searx_engines

    def test_apple_pins_the_host_and_says_so_in_the_query(self, sources):
        hits = sources.apple_music("2000s club hits")
        assert all("music.apple.com" in h.url for h in hits)
        assert "site:music.apple.com" in sources._calls[-1]["query"]

    def test_reddit_is_not_an_engine_any_more(self, sources):
        """SearXNG's reddit engine queries Reddit unauthenticated and Reddit
        blocks the IP, so it answered "access denied" to every call and got
        itself suspended for it — noise that hid the engines that broke for a
        real reason."""
        assert "reddit" not in (sources.cfg.searx_engines or "")

    def test_reddit_is_reached_by_pinning_the_site_instead(self, sources):
        sources.reddit("los santos rock radio song list")
        assert "site:reddit.com" in sources._calls[0]["query"]
        assert sources._calls[0]["engines"] is None       # the open web answers

    def test_google_is_gone_from_the_whitelist(self, sources):
        """It answered every query with nothing and never reported itself
        unresponsive — a round trip per search for no results."""
        assert "google" not in (sources.cfg.searx_engines or "")


class TestHostPinning:
    """`site:` pins the host, and nothing softens it when it comes back empty.

    Retrying with the bare domain as an ordinary word was tried and measured:
    it makes the domain the strongest term in the query, so engines answer with
    the domain's OWN landing pages — music.apple.com/us/new,
    fandom.com/topics/home-page, the DC Comics Database — all of which pass the
    host check and cost a fetch slot each for nothing.
    """

    def test_an_empty_site_search_is_not_retried(self, sources):
        sources.reddit("anything")           # the stub has no reddit URLs
        assert len(sources._calls) == 1
        assert sources._calls[0]["query"].startswith("site:reddit.com ")

    def test_a_successful_site_search_costs_one_call(self, sources):
        sources.apple_music("2000s club hits")     # the stub HAS an apple URL
        assert len(sources._calls) == 1

    def test_the_host_filter_rules_regardless_of_the_query(self, sources):
        """The query is loose; the filter is not. A hit that is not on the
        domain cannot survive, so the engine is never trusted."""
        assert sources.reddit("anything") == []

    def test_the_domains_own_pages_do_not_take_a_slot(self, monkeypatch,
                                                      sources):
        """Asking for "fandom.com <question>" makes the domain name the
        strongest term in the query, so engines answer with the site's own
        marketing pages. All three of these arrived that way in a real run,
        passed the host check, and cost a fetch slot each."""
        monkeypatch.setattr(sources, "_searx_raw", lambda q, **kw: [
            {"url": "https://www.fandom.com/", "title": "Fandom", "content": ""},
            {"url": "https://www.fandom.com/topics/home-page", "title": "Topics",
             "content": ""},
            {"url": "https://about.fandom.com/what-is-fandom-home-legacy",
             "title": "About", "content": ""},
            {"url": "https://gta.fandom.com/wiki/Soundtrack",
             "title": "Soundtrack", "content": "songs"},
        ])
        hits = sources.fandom("gta v soundtrack", limit=2)
        assert [h.url for h in hits] == ["https://gta.fandom.com/wiki/Soundtrack"]

    @pytest.mark.parametrize("call,good,bad", [
        ("apple_music", "https://music.apple.com/us/album/x/1",
         "https://music.apple.com/us/new"),
        ("reddit", "https://www.reddit.com/r/x/comments/1/t/",
         "https://www.reddit.com/r/x/"),
    ])
    def test_only_content_paths_survive(self, monkeypatch, sources, call, good,
                                        bad):
        monkeypatch.setattr(sources, "_searx_raw", lambda q, **kw: [
            {"url": bad, "title": "landing", "content": ""},
            {"url": good, "title": "real", "content": "songs"},
        ])
        assert [h.url for h in getattr(sources, call)("q")] == [good]

    def test_a_pinned_search_carries_the_host_pinned_flag(self, sources):
        """Without it the takeover rule deletes the one host the query was for:
        a domain that IS the answer looks exactly like a scraper dumping one
        site's navigation."""
        sources.reddit("anything")
        assert sources._calls[0]["host_pinned"] is True

    def test_a_repeated_query_is_refused_without_spending_budget(self, sources):
        """Small models rephrase cosmetically when stuck; paying twice for the
        same pages is how a run burns its budget without learning anything."""
        sources.web("kanye west")
        before = sources.searches
        assert sources.web("Kanye  West") == []
        assert sources.searches == before

    def test_the_same_text_on_a_different_engine_is_not_a_repeat(self, sources):
        sources.web("kanye west")
        assert sources.wikipedia("kanye west")

    def test_the_search_budget_stops_the_run(self, sources):
        for i in range(5):
            sources.web(f"query {i}")
        assert sources.searches == sources.cfg.max_web_searches

    def test_an_empty_query_costs_nothing(self, sources):
        assert sources.web("   ") == []
        assert sources.searches == 0


class TestDiagnostics:
    """Surfacing what the JSON already said and the old code threw away."""

    def test_a_suspended_engine_is_recorded_not_swallowed(self, monkeypatch):
        cfg = AgentConfig(searx_min_interval=0)
        src = SearchSources(cfg)

        class _Resp:
            @staticmethod
            def raise_for_status():
                pass

            @staticmethod
            def json():
                return {"results": [{"url": "https://ex.com/a", "title": "A",
                                     "content": "c", "engine": "google",
                                     "engines": ["google", "bing"]}],
                        "unresponsive_engines": [["duckduckgo", "timeout"],
                                                 ["brave", "timeout"]]}

        monkeypatch.setattr("httpx.get", lambda *a, **kw: _Resp())
        src.web("kanye west")

        assert src.suspended_engines() == {"duckduckgo": "timeout",
                                           "brave": "timeout"}
        assert src.last_response["per_engine"] == {"google": 1, "bing": 1}
        assert "DOWN" in src.report()

    def test_an_unreachable_instance_falls_back_to_ddg_direct(self, monkeypatch):
        """A dead SearXNG is not a dead search — websearch_lab talks to DDG
        without it."""
        from app.resources import searxng_client

        cfg = AgentConfig(searx_min_interval=0)
        src = SearchSources(cfg)

        def boom(*a, **kw):
            raise OSError("connection refused")

        monkeypatch.setattr("httpx.get", boom)
        monkeypatch.setattr(searxng_client, "search_ddg", lambda q, limit=10: [
            {"url": "https://ex.com/x", "title": "X", "content": "c"}])
        assert [h.url for h in src.web("kanye west")] == ["https://ex.com/x"]


class TestRerank:
    class _Hub:
        def __init__(self, probs=None):
            self.probs = probs

        def ce_probabilities(self, query, docs):
            return self.probs

    def test_only_hits_above_the_threshold_are_fetched(self):
        hits = [SearchHit(url=f"https://ex/{i}", title=f"t{i}", snippet="s",
                          source="web", rank=i) for i in range(4)]
        kept = rerank_hits(hits, "q", hub=self._Hub([0.9, 0.1, 0.5, 0.05]),
                           threshold=0.2)
        assert [h.url for h in kept] == ["https://ex/0", "https://ex/2"]

    def test_without_a_cross_encoder_nothing_is_dropped(self):
        """An unfiltered pool is a worse pool; an empty one is no pool."""
        hits = [SearchHit(url="https://ex/0", title="t", snippet="s",
                          source="web", rank=0)]
        assert rerank_hits(hits, "q", hub=self._Hub(None), threshold=0.9) == hits

    def test_an_empty_list_stays_empty(self):
        assert rerank_hits([], "q", hub=self._Hub([]), threshold=0.2) == []


class TestRerankDeduplication:
    """An iteration runs two queries and concatenates their results, so a page
    both queries found arrives twice — with different titles and snippets,
    because each engine words them its own way."""

    class _Hub:
        def __init__(self):
            self.batches: list[list[str]] = []

        def ce_probabilities(self, query, docs):
            self.batches.append(list(docs))
            return [0.9] * len(docs)

    FEUD = "https://en.wikipedia.org/wiki/Taylor_Swift–Kanye_West_feud"
    FEUD_ENCODED = ("https://en.wikipedia.org/wiki/"
                    "Taylor_Swift%E2%80%93Kanye_West_feud")

    def _hits(self):
        return [
            SearchHit(url=self.FEUD, title="Taylor Swift-Kanye West feud",
                      snippet="from query one", source="web", rank=0),
            SearchHit(url="https://people.com/x", title="A Complete Timeline",
                      snippet="a", source="web", rank=1),
            SearchHit(url=self.FEUD_ENCODED, title="Taylor Swift–Kanye West feud",
                      snippet="from query two", source="web", rank=2),
            SearchHit(url="https://people.com/x", title="A Complete Timeline",
                      snippet="a", source="web", rank=3),
        ]

    def test_the_same_page_is_scored_once(self):
        """Two slots in the most expensive stage, and two different
        probabilities for one document."""
        hub = self._Hub()
        kept = rerank_hits(self._hits(), "q", hub=hub, threshold=0.2)
        assert len(hub.batches[0]) == 2
        assert len(kept) == 2

    def test_the_better_ranked_spelling_is_the_one_kept(self):
        kept = rerank_hits(self._hits(), "q", hub=self._Hub(), threshold=0.2)
        assert kept[0].url == self.FEUD or kept[1].url == self.FEUD
        assert all(h.url != self.FEUD_ENCODED for h in kept)

    def test_dedup_happens_even_without_a_cross_encoder(self):
        class _NoCE:
            @staticmethod
            def ce_probabilities(query, docs):
                return None

        kept = rerank_hits(self._hits(), "q", hub=_NoCE(), threshold=0.2)
        assert len(kept) == 2


class TestBrokenEngineDetection:
    """A scraper that landed on the wrong page returns THAT page's navigation.

    The results are real URLs on a real site with nothing to do with the query
    — a Polish TV guide's channel list, a Czech portal's sections. No blocklist
    finds this; the shape does. One host, many results.
    """

    TAKEOVER = [
        {"url": "https://telemagazyn.pl/", "title": "t", "content": "",
         "engine": "brokenengine", "engines": ["brokenengine"]},
        {"url": "https://playback.fm/artist/kanye-west-top-songs", "title": "t",
         "content": "", "engine": "google", "engines": ["google"]},
        {"url": "https://telemagazyn.pl/program-tv", "title": "t", "content": "",
         "engine": "brokenengine", "engines": ["brokenengine"]},
        {"url": "https://www.billboard.com/artist/kanye-west/", "title": "t",
         "content": "", "engine": "google", "engines": ["google"]},
        {"url": "https://telemagazyn.pl/stacje/tvp-1", "title": "t", "content": "",
         "engine": "brokenengine", "engines": ["brokenengine"]},
        {"url": "https://telemagazyn.pl/stacje/tv6", "title": "t", "content": "",
         "engine": "brokenengine", "engines": ["brokenengine"]},
        {"url": "https://kworb.net/spotify/x.html", "title": "t", "content": "",
         "engine": "google", "engines": ["google"]},
        {"url": "https://telemagazyn.pl/stacje/tv-4", "title": "t", "content": "",
         "engine": "brokenengine", "engines": ["brokenengine"]},
    ]

    def _sources(self, monkeypatch, rows, **cfg):
        cfg.setdefault("searx_min_interval", 0)
        src = SearchSources(AgentConfig(**cfg))

        class _Resp:
            status_code = 200

            @staticmethod
            def raise_for_status():
                pass

            @staticmethod
            def json():
                return {"results": rows}

        monkeypatch.setattr("httpx.get", lambda *a, **kw: _Resp())
        return src

    def test_the_takeover_host_is_dropped_entirely(self, monkeypatch):
        src = self._sources(monkeypatch, self.TAKEOVER)
        urls = [h.url for h in src.web("kanye west hit songs")]
        assert not any("telemagazyn" in u for u in urls)
        assert any("billboard" in u for u in urls)

    def test_the_engine_behind_it_is_named(self, monkeypatch):
        """The answer to "which engine do I remove?" — the whole point."""
        src = self._sources(monkeypatch, self.TAKEOVER)
        src.web("kanye west hit songs")
        assert src.last_response["takeover_host"] == "telemagazyn.pl"
        assert "brokenengine" in src.suspect_engines()
        assert src.suspect_engines()["brokenengine"]["hosts"] == ["telemagazyn.pl"]
        assert "DUMP" in src.report()

    def test_a_healthy_result_set_is_untouched(self, monkeypatch):
        rows = [{"url": f"https://site{i}.com/a", "title": "t", "content": "",
                 "engine": "google", "engines": ["google"]} for i in range(8)]
        src = self._sources(monkeypatch, rows)
        assert len(src.web("kanye west")) == 8
        assert src.last_response["takeover_host"] is None
        assert src.suspect_engines() == {}

    def test_a_host_pinned_query_is_exempt(self, monkeypatch):
        """site:music.apple.com is SUPPOSED to come back from one host."""
        rows = [{"url": f"https://music.apple.com/us/playlist/x/pl.{i}",
                 "title": "t", "content": "", "engine": "google",
                 "engines": ["google"]} for i in range(6)]
        src = self._sources(monkeypatch, rows)
        assert len(src.apple_music("2000s club hits")) == 3   # limited, not dropped

    def test_a_pinned_engine_is_exempt(self, monkeypatch):
        rows = [{"url": f"https://en.wikipedia.org/wiki/P{i}", "title": "t",
                 "content": "", "engine": "wikipedia", "engines": ["wikipedia"]}
                for i in range(5)]
        src = self._sources(monkeypatch, rows)
        assert len(src.wikipedia("kanye west", limit=4)) == 4

    def test_a_merely_popular_host_is_capped_not_dropped(self, monkeypatch):
        """Three Wikipedia articles for one query is normal; ten is a dump."""
        rows = ([{"url": f"https://en.wikipedia.org/wiki/A{i}", "title": "t",
                  "content": "", "engine": "google", "engines": ["google"]}
                 for i in range(4)]
                + [{"url": f"https://site{i}.com/x", "title": "t", "content": "",
                    "engine": "google", "engines": ["google"]} for i in range(8)])
        src = self._sources(monkeypatch, rows)
        urls = [h.url for h in src.web("kanye west")]
        assert sum("wikipedia" in u for u in urls) == 3
        assert len(urls) == 11
