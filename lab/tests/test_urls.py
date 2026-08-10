"""One URL, one page.

The spellings here are the ones SearXNG actually returned for a single
Wikipedia article in one run: percent-encoded and decoded, from two different
search streams. Three of five fetch slots went to that one page, and its
chunks landed in the retriever three times.
"""

import pytest

from lab.agent.fetch import PageFetcher
from lab.agent.models import Page, SearchHit
from lab.agent.urls import (canonical_url, dedupe_by_url,
                            normalise_apple_url, source_for_url)

FEUD = "https://en.wikipedia.org/wiki/Taylor_Swift–Kanye_West_feud"
FEUD_ENCODED = "https://en.wikipedia.org/wiki/Taylor_Swift%E2%80%93Kanye_West_feud"


class TestCanonical:
    def test_percent_encoding_collapses(self):
        """The exact pair that caused the duplicate fetches."""
        assert canonical_url(FEUD) == canonical_url(FEUD_ENCODED)

    @pytest.mark.parametrize("other", [
        "http://en.wikipedia.org/wiki/Kanye_West",
        "https://www.en.wikipedia.org/wiki/Kanye_West",
        "https://en.wikipedia.org/wiki/Kanye_West/",
        "https://en.wikipedia.org/wiki/Kanye_West#Career",
        "https://en.m.wikipedia.org/wiki/Kanye_West",
        "https://en.wikipedia.org/wiki/Kanye_West?utm_source=searx&fbclid=123",
    ])
    def test_the_same_article_in_many_spellings(self, other):
        assert canonical_url("https://en.wikipedia.org/wiki/Kanye_West") == \
               canonical_url(other)

    def test_different_articles_stay_different(self):
        assert canonical_url("https://en.wikipedia.org/wiki/Kanye_West") != \
               canonical_url("https://en.wikipedia.org/wiki/Taylor_Swift")

    def test_a_meaningful_query_parameter_is_kept(self):
        """Dropping every parameter would merge distinct pages on sites that
        route by query string."""
        assert canonical_url("https://ex.com/p?id=1") != \
               canonical_url("https://ex.com/p?id=2")

    def test_parameter_order_does_not_matter(self):
        assert canonical_url("https://ex.com/p?b=2&a=1") == \
               canonical_url("https://ex.com/p?a=1&b=2")

    def test_junk_input_does_not_raise(self):
        assert canonical_url("") == ""
        assert canonical_url("not a url")


class TestDedupe:
    def test_the_first_spelling_wins_and_order_holds(self):
        hits = [SearchHit(url=FEUD_ENCODED, title="b", snippet="", source="web", rank=1),
                SearchHit(url=FEUD, title="a", snippet="", source="wikipedia", rank=0),
                SearchHit(url="https://ex.com/x", title="c", snippet="",
                          source="web", rank=2)]
        out = dedupe_by_url(hits)
        assert [h.title for h in out] == ["b", "c"]


class TestFetcherDeduplication:
    """The fetcher is the last line: whatever reaches it, one page is fetched
    once."""

    @pytest.fixture
    def fetcher(self, monkeypatch):
        f = PageFetcher()
        calls: list[str] = []

        def fake(url, *, source="web", title=""):
            calls.append(url)
            page = Page(url=url, title=title, markdown="body", source=source)
            f._cache[canonical_url(url)] = page
            return page

        monkeypatch.setattr(f, "fetch_sync", fake)
        f.calls = calls
        return f

    async def test_the_same_page_twice_in_one_batch_is_fetched_once(self, fetcher):
        hits = [SearchHit(url=FEUD, title="", snippet="", source="wikipedia", rank=0),
                SearchHit(url=FEUD, title="", snippet="", source="web", rank=1),
                SearchHit(url=FEUD_ENCODED, title="", snippet="", source="web", rank=2)]
        pages = await fetcher.fetch_many(hits)
        assert len(fetcher.calls) == 1
        assert len(pages) == 1

    async def test_a_page_read_last_iteration_is_not_read_again(self, fetcher):
        hit = SearchHit(url=FEUD, title="", snippet="", source="wikipedia", rank=0)
        await fetcher.fetch_many([hit])
        again = await fetcher.fetch_many(
            [SearchHit(url=FEUD_ENCODED, title="", snippet="", source="web", rank=0)])
        assert len(fetcher.calls) == 1
        assert again == []

    async def test_the_limit_counts_distinct_pages_not_links(self, fetcher):
        """A limit of 2 spent on one duplicated page was the actual waste."""
        hits = [SearchHit(url=FEUD, title="", snippet="", source="wikipedia", rank=0),
                SearchHit(url=FEUD_ENCODED, title="", snippet="", source="web", rank=1),
                SearchHit(url="https://ex.com/a", title="", snippet="",
                          source="web", rank=2)]
        pages = await fetcher.fetch_many(hits, limit=2)
        assert len(pages) == 2
        assert {p.url for p in pages} == {FEUD, "https://ex.com/a"}


class TestSourceKind:
    """What a page IS, not who found it."""

    def test_wikipedia_is_wikipedia_whoever_returned_it(self):
        """The bug this fixes: a discography article came back from the
        open-web stream, was labelled "web", and the table parser skipped it."""
        assert source_for_url(
            "https://en.wikipedia.org/wiki/Kanye_West_singles_discography") == "wikipedia"

    def test_fandom_and_apple_are_recognised(self):
        assert source_for_url("https://gta.fandom.com/wiki/Soundtrack") == "fandom"
        assert source_for_url("https://music.apple.com/us/playlist/x/pl.1") == "apple"

    def test_anything_else_falls_back(self):
        assert source_for_url("https://billboard.com/charts") == "web"
        assert source_for_url("", fallback="wikipedia") == "wikipedia"

    def test_a_mobile_mirror_counts_too(self):
        assert source_for_url("https://en.m.wikipedia.org/wiki/Power") == "wikipedia"


class TestAppleArtistPage:
    """The artist landing page is the worse source in three ways and the
    better one in none — measured on Kanye West's:

    it rotates (two fetches a second apart returned different songs), it mixes
    in "Appears On" ("Run This Town" is a JAY-Z song), and it is 830 KB against
    150 KB for the same twenty tracks. So it is rewritten, not read.
    """

    def test_a_landing_page_becomes_top_songs(self):
        assert normalise_apple_url(
            "https://music.apple.com/us/artist/kanye-west/2715720"
        ) == "https://music.apple.com/us/artist/kanye-west/2715720/top-songs"

    def test_the_storefront_is_pinned(self):
        """Search returns the same artist under any country code — /vc/ as
        readily as /us/ — and the catalogue and ordering differ per country."""
        assert normalise_apple_url(
            "https://music.apple.com/vc/artist/kanye-west/2715720"
        ) == "https://music.apple.com/us/artist/kanye-west/2715720/top-songs"

    def test_two_storefronts_become_one_page(self):
        """Otherwise the same artist costs two fetch slots and appears twice."""
        a = normalise_apple_url("https://music.apple.com/vc/artist/sade/462006")
        b = normalise_apple_url("https://music.apple.com/de-de/artist/sade/462006")
        assert a == b
        assert canonical_url(a) == canonical_url(b)

    def test_a_non_artist_apple_page_keeps_its_shape_but_moves_storefront(self):
        assert normalise_apple_url(
            "https://music.apple.com/de/album/yeezus/1440851894"
        ) == "https://music.apple.com/us/album/yeezus/1440851894"

    def test_a_trailing_slash_does_not_confuse_it(self):
        assert normalise_apple_url(
            "https://music.apple.com/us/artist/kanye-west/2715720/"
        ).endswith("/2715720/top-songs")

    @pytest.mark.parametrize("url", [
        # already the page we want
        "https://music.apple.com/us/artist/kanye-west/2715720/top-songs",
        # other Apple pages, which are not artist landings
        "https://music.apple.com/us/album/yeezus/1440851894",
        "https://music.apple.com/us/playlist/muse-essentials/pl.5d8ac",
        "https://music.apple.com/us/artist/kanye-west/2715720/see-all",
        # not Apple at all
        "https://en.wikipedia.org/wiki/Kanye_West",
    ])
    def test_everything_else_is_left_alone(self, url):
        assert normalise_apple_url(url) == url

    def test_the_artist_itself_is_never_changed(self):
        """The guarantee that makes the rewrite safe: the slug and the numeric
        id survive it, so it cannot land on a different artist. Only the
        storefront and the trailing section move."""
        for url in ("https://music.apple.com/de-de/artist/sade/462006",
                    "https://music.apple.com/vc/artist/kanye-west/2715720",
                    "https://music.apple.com/jp/artist/muse/1990/top-songs"):
            got = normalise_apple_url(url)
            slug, ident = url.split("/artist/")[1].split("/")[:2]
            assert f"/artist/{slug}/{ident}" in got, url


class TestRefill:
    """A limit is a number of pages READ, not a number attempted.

    The run this fixes: eight candidates cleared the cross-encoder, five slots
    were spent on the top five, two of those hosts answered 403, and the
    iteration got three pages while a hit scored 0.88 sat unread.
    """

    @staticmethod
    def _fetcher(dead: set[str], config=None):
        from lab.agent.config import AgentConfig

        f = PageFetcher(config or AgentConfig())
        tried: list[str] = []

        def fake(url, *, source="web", title=""):
            tried.append(url)
            page = (Page(url=url, title=title, markdown="", source=source,
                         error="403") if url in dead
                    else Page(url=url, title=title, markdown="body",
                              source=source))
            f._cache[canonical_url(url)] = page
            return page

        f.fetch_sync = fake
        f.tried = tried
        return f

    @staticmethod
    def _hits(n: int):
        return [SearchHit(url=f"https://h{i}.example/p", title="", snippet="",
                          source="web", rank=i) for i in range(n)]

    async def test_a_failure_pulls_in_the_next_candidate(self):
        f = self._fetcher(dead={"https://h1.example/p"})
        pages = await f.fetch_many(self._hits(6), limit=3)
        assert len(pages) == 3
        assert [p.url for p in pages] == ["https://h0.example/p",
                                          "https://h2.example/p",
                                          "https://h3.example/p"]

    async def test_the_ranking_order_is_respected_when_refilling(self):
        """The replacement is the next-best candidate, not an arbitrary one."""
        f = self._fetcher(dead={"https://h0.example/p", "https://h1.example/p"})
        pages = await f.fetch_many(self._hits(8), limit=2)
        assert [p.url for p in pages] == ["https://h2.example/p",
                                          "https://h3.example/p"]

    async def test_an_exhausted_pool_returns_what_it_got(self):
        f = self._fetcher(dead={"https://h1.example/p"})
        pages = await f.fetch_many(self._hits(2), limit=2)
        assert [p.url for p in pages] == ["https://h0.example/p"]

    async def test_a_dead_pool_cannot_cost_more_than_the_budget(self):
        """Without a cap, twenty unreachable candidates are twenty deadlines."""
        from lab.agent.config import AgentConfig

        f = self._fetcher(dead={f"https://h{i}.example/p" for i in range(20)},
                          config=AgentConfig(fetch_refill_attempts=4))
        pages = await f.fetch_many(self._hits(20), limit=3)
        assert pages == []
        assert len(f.tried) == 3 + 4

    async def test_nothing_is_fetched_twice_while_refilling(self):
        f = self._fetcher(dead={"https://h0.example/p"})
        await f.fetch_many(self._hits(5), limit=3)
        assert len(f.tried) == len(set(f.tried))

    async def test_an_uncapped_call_does_not_refill(self):
        """No limit means "read this list" — there is no reserve to draw on."""
        f = self._fetcher(dead={"https://h0.example/p"})
        pages = await f.fetch_many(self._hits(3))
        assert len(pages) == 2
        assert len(f.tried) == 3

    async def test_all_good_pages_means_one_wave(self):
        f = self._fetcher(dead=set())
        pages = await f.fetch_many(self._hits(9), limit=4)
        assert len(pages) == 4
        assert len(f.tried) == 4

    async def test_a_page_that_failed_is_not_retried_next_iteration(self):
        f = self._fetcher(dead={"https://h0.example/p"})
        await f.fetch_many(self._hits(3), limit=2)
        f.tried.clear()
        again = await f.fetch_many(self._hits(3), limit=2)
        assert again == []
        assert f.tried == []


class TestFetchDeadline:
    """A stuck page must cost one page, not the iteration.

    Note what the deadline does and does not do: it stops the pipeline WAITING.
    The worker thread runs to completion regardless — Python cannot cancel it —
    so the sleeps here are short on purpose, or every run of this file would
    wait for them at exit.
    """

    async def test_a_hanging_fetch_is_abandoned(self, monkeypatch):
        import time

        from lab.agent.config import AgentConfig

        f = PageFetcher(AgentConfig(fetch_deadline=0.05))
        monkeypatch.setattr(f, "fetch_sync",
                            lambda url, **kw: time.sleep(0.3) or Page(
                                url=url, title="", markdown="x", source="web"))
        page = await f.fetch("https://slow.example/a")
        assert not page.ok
        assert "deadline" in page.error

    async def test_the_run_continues_after_one_page_gives_up(self, monkeypatch):
        import time

        from lab.agent.config import AgentConfig
        from lab.agent.models import SearchHit

        f = PageFetcher(AgentConfig(fetch_deadline=0.05, fetch_concurrency=2))

        def maybe_hang(url, *, source="web", title=""):
            if "slow" in url:
                time.sleep(0.3)
            return Page(url=url, title="", markdown="body", source=source)

        monkeypatch.setattr(f, "fetch_sync", maybe_hang)
        pages = await f.fetch_many([
            SearchHit(url="https://slow.example/a", title="", snippet="",
                      source="web", rank=0),
            SearchHit(url="https://fast.example/b", title="", snippet="",
                      source="web", rank=1)])
        assert [p.url for p in pages] == ["https://fast.example/b"]
