"""Track lists out of markdown tables, by column header.

This is the piece that replaces the old ``tracklines()`` heuristics. The
difference is not cosmetic. The old code looked at a line, found a dash, and
guessed which side was the artist; it fired on prose, on navigation menus and
on discography footers, and it could not tell a 2005 release year from a track
number. This reads the header row, decides which column holds what, and takes
only rows from tables that actually have such a column.

Why not hand the page to the model instead: the soundtrack of a big game is two
hundred rows. A 12b model asked to transcribe that will sample a couple of
dozen and quietly drop the rest, and the intersection with the library comes
out empty. Tables are structure — parse them as structure. Prose goes to the
model, where judgement is actually needed (see ``extraction.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from lab.agent.models import SourceKind, TrackRef

# Header wordings seen on Wikipedia and Fandom soundtrack/discography tables.
# Matching is on the folded header cell, so case and punctuation do not matter.
TITLE_HEADERS = ("title", "song", "song title", "track", "track title", "name",
                 "titles", "single", "singles")
ARTIST_HEADERS = ("artist", "artists", "performer", "performed by", "performer(s)",
                  "band", "act", "musician", "recording artist", "original artist")
YEAR_HEADERS = ("year", "released", "release", "release date", "date", "original release")

# A cell that is only a number is a track/position number, not a title.
_NUMERIC_RE = re.compile(r"^\d{1,4}$")
_YEAR_RE = re.compile(r"(1[89]\d{2}|20\d{2})")
# Wikipedia leaves reference markers in cell text often enough to matter.
_REF_RE = re.compile(r"\[\s*\d+\s*\]|\[\s*(?:citation needed|note \d+)\s*\]", re.I)
_QUOTES = "\"'«»“”‘’„"


@dataclass(slots=True)
class Table:
    header: list[str]
    rows: list[list[str]]
    # Index in the document, so a caller can tell "the first table" from "the
    # eleventh" when deciding what to trust.
    position: int = 0
    caption: str = ""

    @property
    def folded_header(self) -> list[str]:
        return [_fold_cell(h) for h in self.header]


def _fold_cell(text: str) -> str:
    text = _REF_RE.sub(" ", text or "")
    return " ".join(re.sub(r"[^\w\s]", " ", text).lower().split())


def _clean_value(text: str) -> str:
    text = _REF_RE.sub(" ", text or "")
    return " ".join(text.split()).strip().strip(_QUOTES).strip()


def parse_markdown_tables(markdown: str) -> list[Table]:
    """Every pipe table in the document, header row separated from the body.

    A markdown table here is what ``websearch_lab``'s extractor emits: a header
    row, a ``| --- |`` separator, then body rows. The separator is what makes
    this unambiguous — without it a run of pipe-containing lines is just text.
    """
    tables: list[Table] = []
    lines = (markdown or "").split("\n")
    i = 0
    while i < len(lines) - 1:
        header_line = lines[i].strip()
        sep_line = lines[i + 1].strip()
        if not (header_line.startswith("|") and _is_separator(sep_line)):
            i += 1
            continue
        header = _split_row(header_line)
        rows: list[list[str]] = []
        j = i + 2
        while j < len(lines) and lines[j].strip().startswith("|"):
            row = _split_row(lines[j].strip())
            if row:
                rows.append(row)
            j += 1
        if header and rows:
            tables.append(Table(header=header, rows=rows, position=len(tables)))
        i = j
    return tables


def _is_separator(line: str) -> bool:
    if not line.startswith("|"):
        return False
    cells = _split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c.strip())
                               for c in cells if c.strip())


def _split_row(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [c.strip() for c in body.split("|")]


def _column_index(folded: list[str], candidates: tuple[str, ...]) -> Optional[int]:
    """The column for this role, resolved by CANDIDATE priority, not by position.

    Iterating the candidates first is what makes a cover table come out right:
    "Original artist" and "Artist" both satisfy the substring test, and a scan
    over columns would take whichever comes first in the table. Scanning over
    candidates takes the plain "artist" wherever it sits, because that is the
    performing act — the one the library will have.
    """
    for candidate in candidates:
        for i, head in enumerate(folded):
            if head == candidate:
                return i
    for candidate in candidates:
        for i, head in enumerate(folded):
            if candidate in head:
                return i
    return None


def tracks_from_table(table: Table, *, source: SourceKind = "wikipedia",
                      source_url: str = "",
                      default_artist: Optional[str] = None) -> list[TrackRef]:
    """Rows of ``table`` as track references, or ``[]`` if it is not a tracklist.

    A table without a recognisable title column is not a tracklist — a chart
    position table, an album infobox, a cast list — and returning nothing is
    the correct answer for it.
    """
    folded = table.folded_header
    title_col = _column_index(folded, TITLE_HEADERS)
    if title_col is None:
        return []
    artist_col = _column_index(folded, ARTIST_HEADERS)
    year_col = _column_index(folded, YEAR_HEADERS)

    out: list[TrackRef] = []
    for row in table.rows:
        if title_col >= len(row):
            continue
        title = _clean_value(row[title_col])
        # A bare number in the title column means the columns were misread
        # (a numbered list rendered as a table) — skip rather than guess.
        if not title or _NUMERIC_RE.match(title) or len(title) > 200:
            continue

        artist = None
        if artist_col is not None and artist_col < len(row):
            artist = _clean_value(row[artist_col]) or None
        if not artist:
            artist = default_artist

        year = None
        if year_col is not None and year_col < len(row):
            m = _YEAR_RE.search(row[year_col] or "")
            if m:
                year = int(m.group(1))

        out.append(TrackRef(title=title, artist=artist, year=year,
                            source=source, source_url=source_url))
    return out


def tracks_from_markdown(markdown: str, *, source: SourceKind = "wikipedia",
                         source_url: str = "",
                         default_artist: Optional[str] = None) -> list[TrackRef]:
    """Every tracklist row on the page, deduplicated on (title, artist)."""
    seen: set[tuple[str, str]] = set()
    out: list[TrackRef] = []
    for table in parse_markdown_tables(markdown):
        for ref in tracks_from_table(table, source=source, source_url=source_url,
                                     default_artist=default_artist):
            key = (_fold_cell(ref.title), _fold_cell(ref.artist or ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
    return out
