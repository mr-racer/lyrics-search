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
    cfg = AgentConfig(searxng_url="http://192.168.0.168:8088", max_web_searches=3)
    src = SearchSources(cfg)
    calls: list[dict] = []

    def fake_search(query, max_results=10, engines=None):
        calls.append({"query": query, "engines": engines, "limit": max_results})
        return [{"url": "https://en.wikipedia.org/wiki/Kanye_West",
                 "title": "Kanye West", "content": "rapper"},
                {"url": "https://music.apple.com/us/playlist/x/pl.1",
                 "title": "Playlist", "content": "songs"},
                {"url": "https://open.spotify.com/track/1",
                 "title": "Spotify", "content": "listen"},
                {"url": "https://example.com/best-songs",
                 "title": "Best songs", "content": "a list"}]

    import lab.websearch_lab as L
    monkeypatch.setattr(L, "search_searxng", fake_search)
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
