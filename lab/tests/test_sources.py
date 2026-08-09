"""Search sources: the budget, the deduplication, the host pinning.

The config-plumbing test is here because the failure it guards against is
invisible: with the address unpublished every search still "works", it just
goes somewhere else, and the only symptom is worse results.
"""

import os

import pytest

from lab.agent.config import AgentConfig
from lab.agent.models import SearchHit
from lab.agent.sources import SearchSources, is_junk, rerank_hits


@pytest.fixture
def sources(monkeypatch):
    """A SearchSources whose transport is stubbed.

    Stubbed at ``_searx_raw``, which is the seam between "decide what to ask"
    and "ask it". An earlier version of this fixture patched a function the
    code no longer called, so the tests quietly started hitting a real SearXNG
    on the LAN — they passed, slowly, and depended on someone's server.
    """
    cfg = AgentConfig(searxng_url="http://searxng.invalid:8088", max_web_searches=3)
    src = SearchSources(cfg)
    calls: list[dict] = []

    def fake_raw(query, *, engines=None, limit=10):
        calls.append({"query": query, "engines": engines, "limit": limit})
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
    def test_the_configured_address_reaches_websearch_lab(self, monkeypatch):
        """websearch_lab reads its address from the environment at call time."""
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        SearchSources(AgentConfig(searxng_url="http://10.0.0.5:8088"))
        assert os.environ["SEARXNG_URL"] == "http://10.0.0.5:8088"

    def test_a_trailing_slash_is_normalised_away(self):
        cfg = AgentConfig(searxng_url="http://10.0.0.5:8088/")
        assert cfg.searxng_url == "http://10.0.0.5:8088"


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
        cfg = AgentConfig(searxng_url="http://searxng.invalid:8088",
                          searx_min_interval=0)
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
        import lab.websearch_lab as L

        cfg = AgentConfig(searxng_url="http://searxng.invalid:8088",
                          searx_min_interval=0)
        src = SearchSources(cfg)

        def boom(*a, **kw):
            raise OSError("connection refused")

        monkeypatch.setattr("httpx.get", boom)
        monkeypatch.setattr(L, "search_ddg", lambda q, max_results=10: [
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
