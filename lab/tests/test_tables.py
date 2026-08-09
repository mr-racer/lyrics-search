"""Markdown tables → track references.

This is the code that replaced the dash-splitting heuristics, so the tests are
mostly about what it must REFUSE: chart tables, infoboxes, numbered lists that
render as tables. A parser that returns rows from those is worse than the
regexes it replaced, because it looks authoritative.
"""

from lab.agent.tables import (parse_markdown_tables, tracks_from_markdown,
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

    def test_a_numeric_title_cell_is_skipped(self):
        """Columns misread as a table put the track number under "Title"."""
        md = "| Title | Artist |\n| --- | --- |\n| 7 | |\n| Kids | MGMT |\n"
        assert [r.title for r in tracks_from_markdown(md)] == ["Kids"]

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
