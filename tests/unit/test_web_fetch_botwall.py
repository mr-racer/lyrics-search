"""A challenge page must not pass for a source.

The run this comes from: asked for a thread from a blocked IP, Reddit answers
200 OK with 167 KB of markup titled "Prove your humanity" — no post, no
comments. Every signal the fetcher trusts says success, so the challenge got
chunked, embedded and ranked like any other page, and the only symptom was an
answer built out of nothing. A 403 would have been kinder.
"""

import pytest

from app.services.assistant.chunking import MarkdownChunker
from app.services.assistant.config import AgentConfig
from app.resources.web_fetch import looks_like_a_bot_wall as _looks_like_a_bot_wall
from app.services.assistant.fetcher import PageFetcher


class TestRecognisingAWall:
    @pytest.mark.parametrize("title", [
        "Reddit - Prove your humanity",
        "Just a moment...",
        "Attention Required! | Cloudflare",
        "Are you a robot?",
        "Verify you are human",
        "Access denied",
        "Security check",
    ])
    def test_known_walls(self, title):
        assert _looks_like_a_bot_wall(title)

    @pytest.mark.parametrize("title", [
        "Los Santos Rock Radio - GTA V Soundtrack",
        "Kanye West - Wikipedia",
        "The 100 best songs of the 2000s",
        "",
        "Human After All - Daft Punk",       # "human" alone is not a wall
        "Access to the Archive - discogs",   # "access" alone is not a wall
    ])
    def test_real_pages_survive(self, title):
        assert not _looks_like_a_bot_wall(title)


@pytest.fixture
def fetcher(monkeypatch):
    """A real PageFetcher with only the network and the extractor stubbed.

    Stubbed at the ``web_fetch`` seam, so everything the guard actually runs
    inside — the route choice, the Page construction, ``ok`` — is real code.
    """
    from app.resources import mediawiki
    from app.resources import reddit_feed as reddit_rss
    from app.resources import web_fetch

    # These URLs are reddit's, and reddit goes through the Atom feed first.
    # Saying the feed found nothing is exactly the case under test: the feed
    # failed, scraping "succeeded", and what it returned was the wall.
    monkeypatch.setattr(reddit_rss, "fetch_thread", lambda url, **kw: None)
    monkeypatch.setattr(mediawiki, "fetch_html", lambda url, **kw: None)

    state = {"title": "", "text": "a heading\n\nplenty of extracted prose"}

    monkeypatch.setattr(web_fetch, "fetch_html",
                        lambda url, **kw: ("<html>…</html>", "curl_chrome124"))
    monkeypatch.setattr(web_fetch, "extract_page", lambda html, url=None: {
        "text": state["text"], "meta": {"title": state["title"]}})

    made = PageFetcher(AgentConfig())
    made.state = state
    return made


class TestThroughTheFetcher:
    """``page.ok`` is what the chunker and the refill both read."""

    def test_a_wall_is_not_ok_even_with_a_full_body(self, fetcher):
        """167 KB of challenge markup extracts to plenty of text, so length is
        not the signal — the document's own title is."""
        fetcher.state["title"] = "Reddit - Prove your humanity"
        page = fetcher.fetch_sync("https://www.reddit.com/r/x/comments/1/",
                                  source="reddit", title="Real thread title")
        assert not page.ok
        assert "bot wall" in (page.error or "")
        assert page.markdown == ""

    def test_the_search_title_does_not_mask_it(self, fetcher):
        """``page.title`` prefers the search hit's wording, which still reads
        like the thread we wanted — so the check must read meta, not title."""
        fetcher.state["title"] = "Reddit - Prove your humanity"
        page = fetcher.fetch_sync(
            "https://www.reddit.com/r/x/comments/1/", source="reddit",
            title="Los Santos Rock Radio full song list")
        assert page.title == "Los Santos Rock Radio full song list"
        assert not page.ok

    def test_a_walled_page_yields_no_chunks(self, fetcher):
        fetcher.state["title"] = "Just a moment..."
        page = fetcher.fetch_sync("https://www.reddit.com/r/x/comments/2/",
                                  source="reddit")
        assert MarkdownChunker(AgentConfig()).split_page(page) == []

    def test_a_real_page_still_passes(self, fetcher):
        fetcher.state["title"] = "Los Santos Rock Radio | GTA Wiki"
        page = fetcher.fetch_sync("https://example.com/los-santos-rock-radio")
        assert page.ok
        assert page.markdown

    async def test_a_walled_page_frees_its_slot_for_the_next_candidate(
            self, fetcher, monkeypatch):
        """A wall has to fail like any other dead fetch, or the refill spends
        the iteration's page budget on challenge pages."""
        from app.resources import web_fetch
        from app.services.assistant.contracts import SearchHit

        def titles(url, **kw):
            fetcher.state["title"] = ("Reddit - Prove your humanity"
                                      if "reddit.com" in url else "A real page")
            return "<html>…</html>", "curl_plain"

        monkeypatch.setattr(web_fetch, "fetch_html", titles)

        hits = [SearchHit(url="https://www.reddit.com/r/x/comments/3/",
                          title="t", snippet="s", source="reddit", rank=0),
                SearchHit(url="https://example.com/real", title="t",
                          snippet="s", source="web", rank=1)]
        pages = await fetcher.fetch_many(hits, limit=1)
        assert [p.url for p in pages] == ["https://example.com/real"]
