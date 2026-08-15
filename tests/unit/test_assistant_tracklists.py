"""Markdown tables → track references.

This is the code that replaced the dash-splitting heuristics, so the tests are
mostly about what it must REFUSE: chart tables, infoboxes, numbered lists that
render as tables. A parser that returns rows from those is worse than the
regexes it replaced, because it looks authoritative.
"""

import pytest

from app.services.assistant.contracts import Page
from app.services.assistant.tracklists import (has_structured_parser,
                                               parse_markdown_tables,
                                               structured_tracks,
                                               tracks_from_markdown,
                                               tracks_from_table)

SOUNDTRACK = """
## Soundtrack

| # | Title | Artist | Year |
| --- | --- | --- | --- |
| 1 | Kids | MGMT | 2007 |
| 2 | Electric Feel | MGMT | 2007 |
| 3 | Time to Pretend[1] | MGMT | 2008 |
"""

CHART = """
| Chart (2009) | Peak position |
| --- | --- |
| US Billboard Hot 100 | 12 |
| UK Singles Chart | 4 |
"""


class TestParsing:
    def test_finds_a_table_with_its_header(self):
        tables = parse_markdown_tables(SOUNDTRACK)
        assert len(tables) == 1
        assert tables[0].header == ["#", "Title", "Artist", "Year"]
        assert len(tables[0].rows) == 3

    def test_a_pipe_line_without_a_separator_is_not_a_table(self):
        """Prose containing pipes is prose. The `| --- |` row is the only
        unambiguous signal that a table starts here."""
        assert parse_markdown_tables("| this | is | just text |\nand more") == []

    def test_several_tables_are_kept_apart(self):
        tables = parse_markdown_tables(SOUNDTRACK + "\n" + CHART)
        assert len(tables) == 2
        assert tables[1].header[0] == "Chart (2009)"


class TestExtraction:
    def test_reads_title_artist_and_year_by_header(self):
        refs = tracks_from_markdown(SOUNDTRACK, source_url="https://ex/1")
        assert [r.title for r in refs] == ["Kids", "Electric Feel", "Time to Pretend"]
        assert {r.artist for r in refs} == {"MGMT"}
        assert refs[0].year == 2007
        assert refs[0].source_url == "https://ex/1"

    def test_reference_markers_are_stripped(self):
        refs = tracks_from_markdown(SOUNDTRACK)
        assert refs[2].title == "Time to Pretend"

    def test_a_table_without_a_title_column_yields_nothing(self):
        """A chart table has a year and numbers and no songs. Guessing that
        "US Billboard Hot 100" is a track is exactly the old failure."""
        assert tracks_from_markdown(CHART) == []

    def test_column_order_does_not_matter(self):
        md = ("| Performer | Song |\n| --- | --- |\n"
              "| Queen | Bohemian Rhapsody |\n")
        refs = tracks_from_markdown(md)
        assert refs[0].title == "Bohemian Rhapsody"
        assert refs[0].artist == "Queen"

    def test_plain_artist_beats_original_artist(self):
        """Both headers contain "artist"; a cover table has both columns and
        the performing act is the one we want."""
        md = ("| Title | Original artist | Artist |\n| --- | --- | --- |\n"
              "| Hurt | Nine Inch Nails | Johnny Cash |\n")
        assert tracks_from_markdown(md)[0].artist == "Johnny Cash"

    def test_a_numeric_title_cell_alone_in_its_row_is_skipped(self):
        """Columns misread as a table put the track number under "Title", and
        such a row names nobody — that emptiness is the signal."""
        md = "| Title | Artist |\n| --- | --- |\n| 7 | |\n| Kids | MGMT |\n"
        assert [r.title for r in tracks_from_markdown(md)] == ["Kids"]

    def test_a_numeric_title_with_an_artist_is_a_real_song(self):
        """Measured on the TDU2 soundtrack: `| Phoenix | 1901 | 3:12 |` was
        dropped from an 86-row table because the title is four digits, and the
        one song of it the user owned never reached the playlist. 1901, 1234,
        1979, 212, 2112 are songs. The guard is for a numbering COLUMN."""
        md = ("| Artist | Song | Length |\n| --- | --- | --- |\n"
              "| Phoenix | 1901 | 3:12 |\n| Metric | Gold Guns Girls | 3:32 |\n")
        assert [r.title for r in tracks_from_markdown(md)] == \
            ["1901", "Gold Guns Girls"]

    def test_a_numbering_column_is_still_refused_whole(self):
        """The case the guard exists for, and the one a per-row rule cannot
        see: "Track" is a title header, it holds 1, 2, 3, and every row names a
        real artist. Only the column as a whole gives it away."""
        md = ("| Track | Artist |\n| --- | --- |\n"
              "| 1 | Phoenix |\n| 2 | Metric |\n| 3 | Feist |\n")
        assert tracks_from_markdown(md) == []

    def test_missing_artist_falls_back_to_the_page_subject(self):
        md = "| Title |\n| --- |\n| Runaway |\n"
        refs = tracks_from_markdown(md, default_artist="Kanye West")
        assert refs[0].artist == "Kanye West"

    def test_duplicate_rows_collapse(self):
        md = ("| Title | Artist |\n| --- | --- |\n"
              "| Kids | MGMT |\n| kids | mgmt |\n")
        assert len(tracks_from_markdown(md)) == 1

    def test_year_is_read_out_of_a_full_date(self):
        md = ("| Song | Release date |\n| --- | --- |\n"
              "| Creep | 21 September 1992 |\n")
        assert tracks_from_markdown(md)[0].year == 1992

    def test_a_ragged_row_does_not_crash_the_parse(self):
        md = "| Title | Artist |\n| --- | --- |\n| Kids |\n| Creep | Radiohead |\n"
        titles = [r.title for r in tracks_from_markdown(md)]
        assert "Creep" in titles

    def test_source_kind_is_carried_through(self):
        table = parse_markdown_tables(SOUNDTRACK)[0]
        refs = tracks_from_table(table, source="fandom")
        assert all(r.source == "fandom" for r in refs)


class TestProvenance:
    """Where a row was found is what the final triage pass judges by."""

    def test_the_heading_path_is_attached_to_every_row(self):
        md = ("# Kanye West\n\n## Discography\n\n### Singles\n\n"
              "| Title | Year |\n| --- | --- |\n| Power | 2010 |\n")
        ref = tracks_from_markdown(md)[0]
        assert ref.section == "Kanye West > Discography > Singles"

    def test_tables_under_different_headings_keep_their_own(self):
        md = ("## Soundtrack\n\n| Title |\n| --- |\n| Kids |\n\n"
              "## Other appearances\n\n| Title |\n| --- |\n| Electric Feel |\n")
        by_title = {r.title: r.section for r in tracks_from_markdown(md)}
        assert by_title["Kids"] == "Soundtrack"
        assert by_title["Electric Feel"] == "Other appearances"

    def test_the_row_is_kept_as_context(self):
        md = ("## Charts\n\n| Title | Peak | Year |\n| --- | --- | --- |\n"
              "| Power | 24 | 2010 |\n")
        ref = tracks_from_markdown(md)[0]
        assert "24" in ref.context and "2010" in ref.context

    def test_the_page_title_travels_with_the_row(self):
        md = "| Title |\n| --- |\n| Kids |\n"
        ref = tracks_from_markdown(md, page_title="TDU2 soundtrack")[0]
        assert ref.page_title == "TDU2 soundtrack"


DISCOGRAPHY = (
    "## As lead artist\n\n"
    "| Title | Year | Peak chart positions |  |  |  | Certifications | Album |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    "| US | US R&B | US Rap | AUS | CAN | WW |  |  |\n"
    '| "Through the Wire" | 2003 | 15 | 8 | 4 | 81 | RIAA: 3× Platinum | '
    "The College Dropout |\n"
    '| "All Falls Down" (featuring Syleena Johnson) | 2004 | 7 | 4 | 2 | — | '
    "RIAA: 6× Platinum |  |\n"
    '| "Can\'t Tell Me Nothing" | 41 | 20 | 8 | — | 16 | RIAA: 7× Platinum |  |\n'
    '| "—" denotes a recording that did not chart or was not released in that '
    "territory. |  |  |  |  |  |  |  |\n"
)


class TestRealWikipediaDiscography:
    """The shape that actually turned up: a two-tier header from a colspan, a
    legend row at the bottom, and every title in quotes."""

    def test_the_songs_come_out(self):
        titles = [r.title for r in tracks_from_markdown(DISCOGRAPHY)]
        assert "Through the Wire" in titles
        assert "Can't Tell Me Nothing" in titles

    def test_the_chart_subheader_is_not_a_song(self):
        """"Peak chart positions" spans several columns; markdown renders the
        span as empty header cells and puts "US | US R&B | ..." on the row
        below. Without the colspan signal every such table yields a track
        called "US"."""
        assert "US" not in [r.title for r in tracks_from_markdown(DISCOGRAPHY)]

    def test_the_legend_row_is_not_a_song(self):
        assert not any("denotes" in r.title
                       for r in tracks_from_markdown(DISCOGRAPHY))

    def test_the_closing_quote_inside_a_title_is_removed(self):
        """The cell reads `"All Falls Down" (featuring ...)`, so stripping the
        ends leaves a quote in the middle."""
        titles = [r.title for r in tracks_from_markdown(DISCOGRAPHY)]
        assert "All Falls Down (featuring Syleena Johnson)" in titles
        assert all('"' not in t for t in titles)

    def test_an_apostrophe_survives(self):
        """Double quotes never mean anything in a title cell; apostrophes do."""
        assert "Can't Tell Me Nothing" in [r.title
                                           for r in tracks_from_markdown(DISCOGRAPHY)]

    def test_a_row_whose_year_column_shifted_still_yields_the_song(self):
        """Wikipedia omits the year on rows continuing the previous one, which
        shifts every column left. The song is still real; the year is unknown,
        and unknown survives the era filter."""
        row = next(r for r in tracks_from_markdown(DISCOGRAPHY)
                   if r.title == "Can't Tell Me Nothing")
        assert row.year is None


class TestParserAvailability:
    """The tuple that routes a page away from the model.

    Kept honest against ``structured_tracks`` itself: a kind this says yes to and
    the parser then ignores is a page nobody reads at all.
    """

    @staticmethod
    def _page(url: str, source: str = "web") -> Page:
        return Page(url=url, title="", markdown="", source=source)

    @pytest.mark.parametrize("url", [
        "https://en.wikipedia.org/wiki/MGMT_discography",
        "https://gta.fandom.com/wiki/Radio_X",
        "https://sonic.wikia.org/wiki/Soundtrack",
        "https://music.apple.com/us/playlist/pl.1",
    ])
    def test_hosts_with_a_parser(self, url):
        assert has_structured_parser(self._page(url))

    @pytest.mark.parametrize("url", [
        "https://pitchfork.com/features/lists/best-songs",
        "https://www.reddit.com/r/hiphopheads/comments/abc/",
        "https://genius.com/albums/Mgmt/Oracular-spectacular",
    ])
    def test_hosts_without_one(self, url):
        assert not has_structured_parser(self._page(url))

    def test_an_unknown_host_falls_back_to_the_stream_label(self):
        """``source_for_url`` cannot read a host it does not know, and the label
        the search stream attached is the only other thing on offer."""
        assert has_structured_parser(self._page("https://mirror.example/x",
                                                source="wikipedia"))

    def test_a_kind_outside_the_tuple_is_never_parsed(self):
        """The other half of the contract. A page the branch sends to the model
        must be one the parser would have refused anyway — otherwise the two
        disagree about who reads it and it gets read twice, or not at all."""
        page = Page(url="https://pitchfork.com/x", title="",
                    markdown=SOUNDTRACK, source="web")
        assert not has_structured_parser(page)
        assert structured_tracks(page) == []
