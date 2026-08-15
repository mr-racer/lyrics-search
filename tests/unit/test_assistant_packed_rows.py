"""A tracklist that packs the whole entry into one cell.

GTA Wiki's radio pages are laid out `| Song: | Preview: | Notes: |` with the
last two empty and the first reading `"Title" by Artist (Year)`. Before this
was handled the resolver was asked to find a song called
`west end girls by pet shop boys 1985`, and Non-Stop-Pop FM resolved 1 of 41
rows against a library that had 15 of them.
"""

import pytest

from app.services.assistant.tracklists import (split_title_by_artist, tracks_from_markdown,
                              tracks_from_table, Table)


class TestSplitting:
    @pytest.mark.parametrize("cell,title,artist,year", [
        ('"Pure Shores" by All Saints (2000)', "Pure Shores", "All Saints", 2000),
        ('"West End Girls" by Pet Shop Boys (1985)', "West End Girls",
         "Pet Shop Boys", 1985),
        ('"Midnight City" by M83 (2011)', "Midnight City", "M83", 2011),
        ('"Glamorous" by Fergie feat. Ludacris (2007)', "Glamorous",
         "Fergie feat. Ludacris", 2007),
        ('"Anthem" by N-Joi', "Anthem", "N-Joi", None),
    ])
    def test_the_three_parts_come_apart(self, cell, title, artist, year):
        assert split_title_by_artist(cell) == (title, artist, year)

    def test_a_quote_inside_the_title_does_not_end_it(self):
        """The real row: the title carries a 7" and a lazy match would call the
        artist `Single)" by Corona`."""
        title, artist, year = split_title_by_artist(
            '"The Rhythm of the Night (Rapino Bros. 7" Single)" by Corona (1993)')
        assert artist == "Corona"
        assert year == 1993
        assert title.startswith("The Rhythm of the Night")

    @pytest.mark.parametrize("cell", [
        "Stand by Me",                       # the trap: no quotes, real title
        "Sitting by the Dock of the Bay",
        "Poisoned by Love (1987)",
        '"Just a Title"',                    # quoted, but nothing after it
        "",
        None,
    ])
    def test_an_unquoted_by_is_left_alone(self, cell):
        """Without quotes there is no way to tell the separator from the word,
        and cutting "Stand by Me" in half is worse than not splitting."""
        assert split_title_by_artist(cell) is None


class TestThroughTheTableParser:
    def _table(self, rows):
        return Table(header=["Song:", "Preview:", "Notes:"],
                     rows=[[r, "", ""] for r in rows])

    def test_a_packed_row_yields_a_usable_claim(self):
        refs = tracks_from_table(
            self._table(['"West End Girls" by Pet Shop Boys (1985)']),
            source="fandom")
        assert len(refs) == 1
        assert (refs[0].title, refs[0].artist, refs[0].year) == (
            "West End Girls", "Pet Shop Boys", 1985)

    def test_a_real_artist_column_still_wins(self):
        """Only a cell with nowhere else to get the artist gets unpacked."""
        table = Table(header=["Title", "Artist"],
                      rows=[['"Stand by Me"', "Ben E. King"]])
        refs = tracks_from_table(table, source="wikipedia")
        assert (refs[0].title, refs[0].artist) == ("Stand by Me", "Ben E. King")

    def test_the_default_artist_is_not_overwritten_by_the_split(self):
        refs = tracks_from_table(
            self._table(['"Power" by Kanye West (2010)']),
            source="fandom", default_artist="Someone Else")
        assert refs[0].artist == "Kanye West"

    def test_an_ordinary_row_is_untouched(self):
        refs = tracks_from_table(self._table(["Bohemian Rhapsody"]),
                                 source="fandom", default_artist="Queen")
        assert (refs[0].title, refs[0].artist) == ("Bohemian Rhapsody", "Queen")

    def test_the_station_page_shape_end_to_end(self):
        markdown = (
            "| Song: | Preview: | Notes: |\n"
            "| --- | --- | --- |\n"
            '| "Pure Shores" by All Saints (2000) |  |  |\n'
            '| "Midnight City" by M83 (2011) |  |  |\n'
            '| "Feel Good Inc." by Gorillaz feat. De La Soul (2005) |  |  |\n')
        refs = tracks_from_markdown(markdown, source="fandom")
        assert [(r.title, r.artist) for r in refs] == [
            ("Pure Shores", "All Saints"),
            ("Midnight City", "M83"),
            ("Feel Good Inc.", "Gorillaz feat. De La Soul"),
        ]
