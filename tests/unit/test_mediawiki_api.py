"""Reading a wiki through api.php.

Fandom answers a Cloudflare interstitial to every HTTP fetcher — measured:
bare httpx, a browser User-Agent, a full browser header set, ?action=raw and
Special:Export all returned 403 on
testdrive.fandom.com/wiki/Test_Drive_Unlimited_2/Soundtrack, while api.php
returned 200 and the article. No test here touches the network.
"""

import pytest

from app.services.assistant.config import AgentConfig
from app.resources.mediawiki import fetch_html, parse_article, prefers_api

FANDOM = "https://testdrive.fandom.com/wiki/Test_Drive_Unlimited_2/Soundtrack"
WIKI = "https://en.wikipedia.org/wiki/Kanye_West_singles_discography"


class TestArticleUrls:
    def test_a_fandom_article_is_recognised(self):
        assert parse_article(FANDOM) == ("https://testdrive.fandom.com",
                                         "Test Drive Unlimited 2/Soundtrack")

    def test_underscores_become_spaces(self):
        assert parse_article(WIKI)[1] == "Kanye West singles discography"

    def test_percent_encoding_is_decoded(self):
        url = "https://en.wikipedia.org/wiki/Taylor_Swift%E2%80%93Kanye_West_feud"
        assert parse_article(url)[1] == "Taylor Swift–Kanye West feud"

    @pytest.mark.parametrize("url", [
        "https://billboard.com/charts/hot-100",
        "https://en.wikipedia.org/w/index.php?title=Kanye",
        "https://en.wikipedia.org/wiki/Special:Export/Kanye",
        "", "not a url",
    ])
    def test_anything_else_is_not_an_article(self, url):
        assert parse_article(url) is None


class TestOrdering:
    """Which route goes first, and why it differs by host."""

    def test_fandom_uses_the_api_first(self):
        """Scraping it does not work at all."""
        assert prefers_api(FANDOM)

    def test_wikipedia_is_scraped_first(self):
        """The API hands back the whole raw article — 120k chars on a
        discography page against 57k extracted — so the ordinary path wins."""
        assert not prefers_api(WIKI)

    def test_a_subdomain_of_a_listed_host_counts(self):
        assert prefers_api("https://gta.fandom.com/wiki/Soundtrack",
                           ("fandom.com",))


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class TestFetch:
    def test_the_article_html_comes_back_wrapped(self, monkeypatch):
        monkeypatch.setattr("httpx.get", lambda *a, **kw: _Resp(
            {"parse": {"title": "Soundtrack", "text": "<table>rows</table>"}}))
        html, meta = fetch_html(FANDOM)
        assert html.startswith("<html><body>") and "<table>" in html
        assert meta["title"] == "Soundtrack"
        assert meta["hostname"] == "testdrive.fandom.com"

    def test_markup_in_the_display_title_is_stripped(self, monkeypatch):
        monkeypatch.setattr("httpx.get", lambda *a, **kw: _Resp(
            {"parse": {"displaytitle": "<i>Soundtrack</i>", "text": "<p>x</p>"}}))
        assert fetch_html(FANDOM)[1]["title"] == "Soundtrack"

    def test_it_tries_the_other_api_path(self, monkeypatch):
        """Wikimedia keeps api.php under /w/, Fandom at the root."""
        seen = []

        def fake(url, **kw):
            seen.append(url)
            if url.endswith("/w/api.php"):
                return _Resp({}, status=404)
            return _Resp({"parse": {"title": "T", "text": "<p>ok</p>"}})

        monkeypatch.setattr("httpx.get", fake)
        assert fetch_html(FANDOM) is not None
        assert len(seen) == 2

    def test_an_api_error_gives_up_rather_than_retrying(self, monkeypatch):
        """A missing page is an answer; trying the other path repeats it."""
        calls = []

        def fake(url, **kw):
            calls.append(url)
            return _Resp({"error": {"code": "missingtitle"}})

        monkeypatch.setattr("httpx.get", fake)
        assert fetch_html(FANDOM) is None
        assert len(calls) == 1

    def test_a_dead_endpoint_returns_none_without_raising(self, monkeypatch):
        def boom(*a, **kw):
            raise OSError("connection refused")

        monkeypatch.setattr("httpx.get", boom)
        assert fetch_html(FANDOM) is None

    def test_an_html_error_page_under_a_json_url_is_survived(self, monkeypatch):
        monkeypatch.setattr("httpx.get",
                            lambda *a, **kw: _Resp(ValueError("not json")))
        assert fetch_html(FANDOM) is None

    def test_a_non_article_url_is_not_fetched_at_all(self, monkeypatch):
        def boom(*a, **kw):
            raise AssertionError("should not have been called")

        monkeypatch.setattr("httpx.get", boom)
        assert fetch_html("https://billboard.com/charts") is None
