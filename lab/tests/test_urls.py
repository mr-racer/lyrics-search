"""One URL, one page.

The spellings here are the ones SearXNG actually returned for a single
Wikipedia article in one run: percent-encoded and decoded, from two different
search streams. Three of five fetch slots went to that one page, and its
chunks landed in the retriever three times.
"""

import pytest

from lab.agent.fetch import PageFetcher
from lab.agent.models import Page, SearchHit
from lab.agent.urls import canonical_url, dedupe_by_url

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
