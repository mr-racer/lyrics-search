"""Getting track references off a page: structure first, model last.

Three paths, chosen by what the page actually is:

1. **Apple Music** — ``apple_music.parse_apple_playlist`` reads the page's own
   embedded JSON. Positions, artists, albums, durations, ids. No model, no
   ambiguity. Only Apple's OWN material is accepted: an Apple-curated list is a
   professionally assembled answer to "2000s club hits", a stranger's public
   playlist is noise with the same URL shape.
2. **Wikipedia / Fandom** — markdown tables, read by column header. This is what
   replaces line-by-line heuristics: the old approach looked at a line, found a
   dash and guessed which side was the artist, so it fired on prose, on
   navigation menus and on discography footers, and could not tell a 2005 release
   year from a track number. Reading the header row and taking only rows from
   tables that have such a column is not a better guess — it is not a guess.
3. **Everything else** — prose and listicles, where there is no structure and
   judgement is genuinely needed. That is the model's job.

Why not hand a soundtrack page to the model instead: a big game's soundtrack is
two hundred rows, and a 12b model asked to transcribe that will sample a couple
of dozen and quietly drop the rest, so the intersection with the library comes
out empty. Tables are structure — parse them as structure.

The rule that makes path 3 safe: a title the model produced is a CLAIM. It
becomes a track only after ``LibraryCatalog`` matches it against the library. A
hallucinated title matches nothing and disappears.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

from app.services.assistant.config import AgentConfig
from app.services.assistant.contracts import Page, TrackRef
from app.services.assistant.llm import LLMClient, as_int, as_str
from app.services.assistant.prompts import EXTRACT_TRACKS_SYSTEM
from app.services.assistant.web_urls import source_for_url

logger = logging.getLogger(__name__)

# ── markdown tables ─────────────────────────────────────────────────────────

# Header wordings seen on Wikipedia and Fandom soundtrack/discography tables.
# Matching is on the folded header cell, so case and punctuation do not matter.
TITLE_HEADERS = ("title", "song", "song title", "track", "track title", "name",
                 "titles", "single", "singles")
ARTIST_HEADERS = ("artist", "artists", "performer", "performed by",
                  "performer(s)", "band", "act", "musician", "recording artist",
                  "original artist")
YEAR_HEADERS = ("year", "released", "release", "release date", "date",
                "original release")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMERIC_RE = re.compile(r"^\d{1,4}$")
_YEAR_RE = re.compile(r"(1[89]\d{2}|20\d{2})")
# Wikipedia leaves reference markers in cell text often enough to matter.
_REF_RE = re.compile(r"\[\s*\d+\s*\]|\[\s*(?:citation needed|note \d+)\s*\]", re.I)
_QUOTES = "\"'«»“”‘’„"


@dataclass(slots=True)
class Table:
    header: list
    rows: list
    # Index in the document, so a caller can tell "the first table" from "the
    # eleventh" when deciding what to trust.
    position: int = 0
    caption: str = ""
    # Heading path the table sits under: "Discography > Singles". The single most
    # useful thing to know about a tracklist, because "Soundtrack" and "Other
    # appearances" are different claims and the rows look identical.
    section: str = ""

    @property
    def folded_header(self) -> list:
        return [_fold_cell(h) for h in self.header]


def _fold_cell(text: str) -> str:
    text = _REF_RE.sub(" ", text or "")
    return " ".join(re.sub(r"[^\w\s]", " ", text).lower().split())


def _clean_value(text: str) -> str:
    text = _REF_RE.sub(" ", text or "")
    return " ".join(text.split()).strip().strip(_QUOTES).strip()


def _clean_title(text: str) -> str:
    """A title cell, with the quoting Wikipedia wraps every song in removed.

    Stripping only the ends is not enough: the cell reads
    ``"All Falls Down" (featuring Syleena Johnson)``, so the closing quote sits in
    the middle and survives. Double quotes never carry meaning inside a song
    title, apostrophes routinely do ("Can't Tell Me Nothing") — so only the
    former go.
    """
    return " ".join(_clean_value(text).replace('"', " ")
                    .replace("«", " ").replace("»", " ")
                    .replace("“", " ").replace("”", " ").split())


# A one-column tracklist: the whole entry packed into the song cell as
# `"Title" by Artist (Year)`. GTA Wiki's radio station pages are laid out this
# way — the table's other columns are "Preview" and "Notes", both empty — and
# without this the entire row became the title. Measured on Non-Stop-Pop FM: 41
# rows, 40 of them unmatchable, because the resolver was handed
# `west end girls by pet shop boys 1985` as a song name.
#
# The quotes are REQUIRED, and that is the whole safety argument. An unquoted
# `X by Y` cannot be told from a title that simply contains the word: "Stand by
# Me", "Sitting by the Dock", "Poisoned by Love" would all be cut in half. The
# quotes are the page saying which part is the name.
_QUOTED_BY_ARTIST = re.compile(
    r'^\s*["“«](?P<title>.+)["”»]\s+by\s+(?P<artist>.+?)\s*$', re.I)
_TRAILING_YEAR = re.compile(r"\s*[\(\[](1[89]\d{2}|20\d{2})[\)\]]\s*$")


def split_title_by_artist(cell: str) -> Optional[tuple]:
    """``"Title" by Artist (Year)`` → ``(title, artist, year)``, or None.

    The title group is greedy on purpose. One row reads
    ``"The Rhythm of the Night (Rapino Bros. 7" Single)" by Corona (1993)`` — the
    title contains its own double quote — and a lazy match would end the title at
    ``7`` and call the artist ``Single)" by Corona``. Greedy takes the LAST
    ``" by `` in the cell, which is the real separator.
    """
    found = _QUOTED_BY_ARTIST.match(cell or "")
    if not found:
        return None

    artist = found.group("artist").strip()
    year = None
    tail = _TRAILING_YEAR.search(artist)
    if tail:
        year = int(tail.group(1))
        artist = artist[:tail.start()].strip()

    title = _clean_title(found.group("title"))
    artist = _clean_value(artist)
    if not title or not artist:
        return None
    return title, artist, year


def _is_subheader_row(header: list, row: list) -> bool:
    """True when this row is the second tier of a two-row header.

    Wikipedia's chart tables span one header cell across many columns ("Peak
    chart positions") and put the individual chart names underneath. The markdown
    conversion turns the span into empty header cells, so the empty cells ARE the
    signal — and the row below them is column labels, not a song. Without this,
    every such table contributes a track called "US".
    """
    if sum(1 for cell in header if not cell.strip()) < 2:
        return False
    values = [c.strip() for c in row if c.strip()]
    if not values:
        return False
    # Column labels are short and none of them is a year.
    return (all(len(v) <= 12 for v in values)
            and not any(_YEAR_RE.fullmatch(v) for v in values))


def _is_footnote_row(row: list, title: str) -> bool:
    """A table's trailing legend: one long sentence, every other cell empty."""
    filled = [c for c in row[1:] if c.strip()]
    return not filled and (len(title) > 60 or " denotes " in title.lower())


def parse_markdown_tables(markdown: str) -> list:
    """Every pipe table in the document, header row separated from the body.

    The ``| --- |`` separator is what makes this unambiguous — without it a run of
    pipe-containing lines is just text.
    """
    tables: list = []
    lines = (markdown or "").split("\n")
    stack: list = []
    i = 0
    while i < len(lines) - 1:
        heading = _HEADING_RE.match(lines[i])
        if heading:
            level, text = len(heading.group(1)), heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            i += 1
            continue

        header_line = lines[i].strip()
        sep_line = lines[i + 1].strip()
        if not (header_line.startswith("|") and _is_separator(sep_line)):
            i += 1
            continue
        section = " > ".join(text for _, text in stack)
        header = _split_row(header_line)
        rows: list = []
        j = i + 2
        while j < len(lines) and lines[j].strip().startswith("|"):
            row = _split_row(lines[j].strip())
            if row:
                rows.append(row)
            j += 1
        if header and rows:
            tables.append(Table(header=header, rows=rows, position=len(tables),
                                section=section))
        i = j
    return tables


def _is_separator(line: str) -> bool:
    if not line.startswith("|"):
        return False
    cells = _split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c.strip())
                               for c in cells if c.strip())


def _split_row(line: str) -> list:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [c.strip() for c in body.split("|")]


def _column_index(folded: list, candidates: tuple) -> Optional[int]:
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


def _is_a_numbering_column(table: Table, title_col: int) -> bool:
    """True when the column read as titles is really the track numbers.

    ``TITLE_HEADERS`` contains "track", which is both "the song" and "the position
    on the disc" depending on the page, so this cannot be settled from the header.
    It can be settled from the values: a numbering column is numeric in every row,
    and a tracklist that happens to contain "1901" is numeric in one row out of
    eighty. Two rows minimum, because a one-row table of a single numeric title is
    not evidence of a pattern.
    """
    filled = [t for t in (_clean_title(row[title_col]) for row in table.rows
                          if title_col < len(row)) if t]
    numeric = [t for t in filled if _NUMERIC_RE.match(t)]
    return len(numeric) >= 2 and len(numeric) * 2 >= len(filled)


def tracks_from_table(table: Table, *, source: str = "wikipedia",
                      source_url: str = "", page_title: str = "",
                      default_artist: Optional[str] = None) -> list:
    """Rows of ``table`` as track references, or ``[]`` if it is not a tracklist.

    A table without a recognisable title column is not a tracklist — a chart
    position table, an album infobox, a cast list — and returning nothing is the
    correct answer for it.
    """
    folded = table.folded_header
    title_col = _column_index(folded, TITLE_HEADERS)
    if title_col is None:
        return []
    artist_col = _column_index(folded, ARTIST_HEADERS)
    year_col = _column_index(folded, YEAR_HEADERS)
    numbering = _is_a_numbering_column(table, title_col)

    out: list = []
    for row in table.rows:
        if title_col >= len(row):
            continue
        if _is_subheader_row(table.header, row):
            continue
        title = _clean_title(row[title_col])
        year = None

        artist = None
        if artist_col is not None and artist_col < len(row):
            artist = _clean_value(row[artist_col]) or None
        if not artist:
            # No artist column, so the cell may be carrying all three parts.
            # Tried BEFORE the validity checks below, which would otherwise judge
            # a packed row by a title that is really a whole sentence.
            packed = split_title_by_artist(row[title_col])
            if packed:
                title, artist, year = packed
        # Whether THIS ROW named somebody. ``default_artist`` is the page's
        # subject applied to every row alike, so it says nothing about this one,
        # and the numeric check below needs the difference.
        row_names_an_artist = artist is not None
        if not artist:
            artist = default_artist

        if not title or len(title) > 200:
            continue
        # A bare number under "Title" is usually the track number — the columns
        # were misread, or "Track" was a numbering header. But sometimes it is the
        # song: 1901, 1234, 1979, 212, 2112. Two signals separate them, and
        # neither works alone. The column, because a numbering column is numeric
        # all the way down while a numeric TITLE is one row in eighty; and the
        # row, because a misread number sits in a line with nothing else in it,
        # where a song has its artist beside it.
        if _NUMERIC_RE.match(title) and (numbering or not row_names_an_artist):
            continue
        if _is_footnote_row(row, title):
            continue

        if year_col is not None and year_col < len(row):
            # A real column beats the year packed into the song cell.
            m = _YEAR_RE.search(row[year_col] or "")
            if m:
                year = int(m.group(1))

        # The whole row is the context here: the other columns are what tell a
        # reader that this line is a chart position, a B-side or a remix.
        context = " | ".join(_clean_value(c) for c in row if _clean_value(c))
        out.append(TrackRef(title=title, artist=artist, year=year, source=source,
                            source_url=source_url, section=table.section,
                            page_title=page_title, context=context[:200]))
    return out


def tracks_from_markdown(markdown: str, *, source: str = "wikipedia",
                         source_url: str = "", page_title: str = "",
                         default_artist: Optional[str] = None) -> list:
    """Every tracklist row on the page, deduplicated on (title, artist)."""
    seen: set = set()
    out: list = []
    for table in parse_markdown_tables(markdown):
        for ref in tracks_from_table(table, source=source, source_url=source_url,
                                     page_title=page_title,
                                     default_artist=default_artist):
            key = (_fold_cell(ref.title), _fold_cell(ref.artist or ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
    return out


# ── Apple Music ─────────────────────────────────────────────────────────────

# Editorial playlists are credited to "Apple Music" plus a genre desk — "Apple
# Music Alternative", "Apple Music Hip-Hop", or plain "Apple Music". Anything else
# under /playlist/ is somebody's personal collection wearing the same URL shape.
_EDITORIAL_AUTHOR = "apple music"
# User-created playlist ids carry a `u-` marker. A hard reject on top of the
# author check: it needs no parsing to have succeeded, so it still holds when the
# page shape changes and the author comes back empty.
_USER_PLAYLIST_PREFIX = "pl.u-"
# Paths that are Apple's OWN catalogue. There is no user-generated variant of
# these — an artist's top songs is Apple's chart of that artist, an album is an
# album — so there is no curator to check and none to find.
_CATALOGUE_PATHS = ("/artist/", "/album/", "/song/")


def apple_page_kind(url: str) -> str:
    """``"playlist"``, ``"catalogue"`` or ``"other"`` from the URL alone."""
    path = (urlsplit(url or "").path or "").lower()
    if "/playlist/" in path:
        return "playlist"
    if any(marker in path for marker in _CATALOGUE_PATHS):
        return "catalogue"
    return "other"


def is_usable_apple_page(playlist, url: str) -> bool:
    """Whether this Apple page is Apple's own material rather than a listener's.

    The rule depends on the KIND of page, and conflating them is what made
    ``/artist/kanye-west/…/top-songs`` come back empty: it has no author and no
    ``pl.`` id because it is not a playlist at all, and the editorial check
    rejected it for failing to be one.
    """
    kind = apple_page_kind(url)
    if kind == "catalogue":
        return True
    if kind != "playlist":
        return False
    playlist_id = (getattr(playlist, "playlist_id", "") or "").lower()
    if playlist_id.startswith(_USER_PLAYLIST_PREFIX):
        return False
    author = (getattr(playlist, "author", "") or "").strip().lower()
    return author.startswith(_EDITORIAL_AUTHOR)


def apple_tracks(page: Page) -> list:
    """Tracks from an Apple Music page, or [] if it is a listener's playlist.

    Uses the HTML the fetcher already kept. Apple's track list lives in embedded
    JSON that markdown extraction throws away, so the fetcher keeps the raw body
    for these pages. With no body there is nothing to do here — re-downloading
    would be a second request for a page the run already decided about.
    """
    from app.services.assistant.apple_music import parse_apple_playlist

    if not page.html:
        logger.info("[tracklists] %s: no raw html kept, nothing to parse", page.url)
        return []
    try:
        playlist = parse_apple_playlist(page.html, url=page.url)
    except Exception:  # noqa: BLE001
        logger.info("[tracklists] apple parse failed for %s", page.url,
                    exc_info=True)
        return []

    # Logged before the verdict, and always: "Apple gave us nothing" and "the
    # parser found forty tracks and the gate refused them" look identical from
    # outside, and they need opposite fixes.
    logger.info("[tracklists] apple %s: kind=%s author=%r id=%r parsed=%d tracks",
                page.url, apple_page_kind(page.url), playlist.author,
                playlist.playlist_id, len(playlist.tracks))

    if not is_usable_apple_page(playlist, page.url):
        logger.info("[tracklists] %s: skipped — a listener's playlist, not "
                    "Apple's", page.url)
        return []

    where = playlist.title or apple_page_kind(page.url)
    out: list = []
    for track in playlist.tracks:
        if not track.title:
            continue
        out.append(TrackRef(
            title=track.title,
            artist=(track.artists[0] if track.artists else None),
            year=None, source="apple", source_url=page.url,
            section=where, page_title=playlist.title or "",
            context=" · ".join(p for p in (", ".join(track.artists),
                                           track.album) if p)[:200]))
    return out


# The hosts a parser exists for, as ONE tuple: the playlist branch decides what
# to put in front of the model by asking whether a page is in here, so a kind
# listed in one place and not the other is a page routed away from the model and
# then dropped unparsed — silently, and only for that host.
STRUCTURED_KINDS = ("apple", "wikipedia", "fandom")


def has_structured_parser(page: Page) -> bool:
    """Whether ``structured_tracks`` has anything to try on this page.

    Says nothing about whether it will FIND anything: a Wikipedia article with no
    tables is in here and yields nothing. That difference is the caller's to
    handle — see ``PlaylistBranch._harvest``.
    """
    return source_for_url(page.url, fallback=page.source) in STRUCTURED_KINDS


def structured_tracks(page: Page,
                      *, default_artist: Optional[str] = None) -> list:
    """Whatever the page yields without asking a model. May be empty.

    Dispatched on the HOST, not on ``page.source``. The source label records which
    search stream found the page, and a Wikipedia article found by the open-web
    search used to be labelled "web" — so the table parser skipped a 900-row
    discography without a word.
    """
    kind = source_for_url(page.url, fallback=page.source)
    if kind not in STRUCTURED_KINDS:
        return []
    if kind == "apple":
        return apple_tracks(page)
    return tracks_from_markdown(page.markdown, source=kind,
                                source_url=page.url, page_title=page.title,
                                default_artist=default_artist)


# ── prose ───────────────────────────────────────────────────────────────────


class TrackExtractor:
    """The model's leg: titles out of prose, one call for a batch of passages."""

    def __init__(self, llm: LLMClient, config: Optional[AgentConfig] = None,
                 sink=None):
        self.llm = llm
        self.cfg = config or AgentConfig()
        self.sink = sink

    async def from_passages(self, passages: list, *, request: str) -> list:
        """``passages`` is ``[(url, text), ...]``. Returns claims, not tracks.

        One call for all of them rather than one per passage: the model needs to
        see the request beside the material, and repeating that framing five times
        costs five times as much for no more accuracy.
        """
        passages = [(u, t) for u, t in passages if (t or "").strip()]
        if not passages:
            return []

        blocks = "\n\n".join(f"[{i + 1}] ({url})\n{text}"
                             for i, (url, text) in enumerate(passages))
        raw = await self.llm.ask_json([
            {"role": "system", "content": EXTRACT_TRACKS_SYSTEM},
            {"role": "user",
             "content": f"Request: {request}\n\nPassages:\n{blocks}"},
        ], required=("tracks",))
        if raw is None:
            return []

        items = raw.get("tracks")
        if not isinstance(items, list):
            return []

        primary_url = passages[0][0]
        out: list = []
        seen: set = set()
        for item in items[:120]:
            if not isinstance(item, dict):
                continue
            title = as_str(item.get("title"), 200)
            if not title:
                continue
            artist = as_str(item.get("artist"), 160) or None
            key = (title.lower(), (artist or "").lower())
            if key in seen:
                continue
            seen.add(key)
            year = as_int(item.get("year"))
            if year is not None and not (1900 <= year <= 2100):
                year = None
            out.append(TrackRef(title=title, artist=artist, year=year,
                                source="web", source_url=primary_url))

        if self.sink is not None:
            self.sink.put("extract", passages=len(passages), claims=len(out))
        logger.info("[tracklists] model produced %d claims from %d passages",
                    len(out), len(passages))
        return out
