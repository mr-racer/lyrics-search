"""Which pages reach the model, and which are parsed for free.

The routing under test is the difference between a run that spends twenty
seconds retyping a table and one that does not, so the tests are written against
the property that matters — a page with a parser is never material for the model
unless its parser came back empty.
"""

import pytest

from app.services.assistant.branches.playlist import PlaylistBranch
from app.services.assistant.config import AgentConfig
from app.services.assistant.contracts import Page
from app.services.assistant.events import AgentSink
from app.services.assistant.timing import Timings

SOUNDTRACK = """
## Soundtrack

| # | Title | Artist | Year |
| --- | --- | --- | --- |
| 1 | Kids | MGMT | 2007 |
| 2 | Electric Feel | MGMT | 2007 |
"""


class _Agent:
    """The three attributes ``WebBranch.__init__`` reads off an agent."""

    def __init__(self):
        self.cfg = AgentConfig()
        self.sink = AgentSink()
        self.timings = Timings()


def _branch() -> PlaylistBranch:
    return PlaylistBranch(_Agent(), sources=None, fetcher=None)


def _page(url: str, *, markdown: str = "", source: str = "web") -> Page:
    return Page(url=url, title="", markdown=markdown, source=source)


class TestHarvest:
    async def test_a_wikipedia_page_found_by_the_open_web_search_is_parsed(self):
        """The regression this routing exists for.

        ``gather`` splits its result by SEARCH STREAM, so a Wikipedia article
        that only the open-web search surfaced arrived labelled "web" and was
        handed to the model — the most expensive way to read a table.
        """
        page = _page("https://en.wikipedia.org/wiki/MGMT_discography",
                     markdown=SOUNDTRACK, source="web")

        claims, prose = await _branch()._harvest([page], artist=None)

        assert [c.title for c in claims] == ["Kids", "Electric Feel"]
        assert prose == []

    async def test_a_page_with_no_parser_is_prose(self):
        page = _page("https://pitchfork.com/best-songs", markdown="Some prose.")

        claims, prose = await _branch()._harvest([page], artist=None)

        assert claims == []
        assert prose == [page]

    async def test_reddit_is_not_a_structured_host(self):
        """It has a dedicated FETCHER, which is a different thing from a parser."""
        page = _page("https://www.reddit.com/r/hiphopheads/comments/abc/",
                     markdown="- Kids by MGMT", source="reddit")

        claims, prose = await _branch()._harvest([page], artist=None)

        assert claims == []
        assert prose == [page]

    async def test_a_structured_page_that_parses_to_nothing_falls_back_to_prose(self):
        """``tracks_from_markdown`` reads TABLES. A Fandom soundtrack written as
        a bulleted list is a page full of real tracks and no rows, and dropping
        it because of its host would lose the whole page."""
        page = _page("https://gta.fandom.com/wiki/Radio_X",
                     markdown="* Kids - MGMT\n* Electric Feel - MGMT",
                     source="fandom")

        claims, prose = await _branch()._harvest([page], artist=None)

        assert claims == []
        assert prose == [page]

    async def test_the_two_lanes_are_filled_from_one_mixed_batch(self):
        wiki = _page("https://en.wikipedia.org/wiki/X", markdown=SOUNDTRACK)
        blog = _page("https://ex.com/list", markdown="Some prose.")
        empty_wiki = _page("https://en.wikipedia.org/wiki/Y", markdown="Prose.")

        claims, prose = await _branch()._harvest([wiki, blog, empty_wiki],
                                                 artist=None)

        assert len(claims) == 2
        assert prose == [blog, empty_wiki]

    async def test_a_parsed_page_is_announced_and_an_empty_one_is_not(self):
        """The sink drives the UI's progress line — an empty parse is not a find."""
        branch = _branch()
        pages = [_page("https://en.wikipedia.org/wiki/X", markdown=SOUNDTRACK),
                 _page("https://en.wikipedia.org/wiki/Y", markdown="Prose.")]

        await branch._harvest(pages, artist=None)

        found = [f for f in branch.sink.history if f.get("stage") == "structured"]
        assert len(found) == 1
        assert found[0]["url"].endswith("/X")
        assert found[0]["tracks"] == 2

    async def test_rows_that_name_no_artist_are_tagged_with_the_library_spelling(self):
        page = _page("https://en.wikipedia.org/wiki/X", markdown="""
| Title |
| --- |
| Kids |
""")

        claims, _ = await _branch()._harvest([page], artist="MGMT")

        assert [c.artist for c in claims] == ["MGMT"]
