"""Lyrics preprocessing for the gems pipeline. Pure stdlib — no torch here.

Genius-style pages carry a lot of structure that is not lyrics, and every bit
of it that survives into the model input becomes a false gem. Measured on the
2026-07-26 production snapshot: 40 of 585 stored gems (6.8%) — and 34 of 273
namedrops (12.5%) — were built on markup rather than on a sung line:

  17  nested brackets      [Chorus: Jay-Z & (Kanye West)]
  10  wiki bold            '''Daft Punk:'''
   5  speaker labels       Verse Two: Foxy Brown / Lead Vocals : Freddie Mercury
   2  stage directions     *Instrumental for ... plays in the background*

The old ``_SECTION_LINE`` only matched a line made entirely of brackets with
NO nested pair, so ``(Kanye West)`` inside ``[Chorus: ...]`` defeated it; wiki
bold and ``Name:`` labels it never knew at all.

Separately, some lyric pulls use bare ``\\r`` as the line separator. Splitting
on ``\\n`` alone then collapses the whole song into one line — the section
filter can't fire and quotes come out as 200-char blobs (4 of 585 stored
quotes, 3 of them in songref, including The Black Keys' "Sister" -> "Wake Up").
Everything here therefore runs on newline-normalized text.
"""
from __future__ import annotations

import re
from typing import List, Optional

# Tracklist artifacts: "14. Apologize (Timbaland Remix) [0:03:04.82]"
_TRACKLIST_LINE = re.compile(r"^\s*\d{1,2}\.\s.+\[\d+:\d")
_TIMESTAMP = re.compile(r"\[\d+:\d+(?::\d+)?(?:\.\d+)?\]")
# Wiki-style bold used by some lyric mirrors around speaker labels.
_WIKI_BOLD = re.compile(r"'{3,}")
# A whole line wrapped in asterisks: "*Instrumental for X plays in the back*"
_STAGE_LINE = re.compile(r"^\s*\*.*\*\s*$")
# Credit headers from Chinese lyric providers, seen verbatim in production:
# "作词 : Kevin Parker", "作曲 : Curtis Jackson & Khari Cain". The Latin-script
# label rules cannot see these, and they name real artists.
_CJK_CREDIT_LINE = re.compile("^\\s*[一-鿿]{2,8}\\s*[:：]")
_OPEN = {"[": "]", "(": ")", "{": "}"}

# "Name: rest of line" / "Verse 2: rest" / "Charlie Wilson and Brenda Lee:".
# Every word of the label is Capitalized (connectors excepted), so the real
# Jay-Z line "My portfolio reads: leads to Don Corleone" is NOT matched — its
# second word is lowercase. The colon may also end the line, which is how the
# wiki-bold mirrors write it: "'''Daft Punk:'''".
_NAME_TOKEN = "[A-Z0-9][\\w.'&$\u2010-\u2015-]*"
_CONNECTOR = r"(?:and|&|x|with|feat\.?|ft\.?)"
_SPEAKER_PREFIX = re.compile(
    "^\\s*(" + _NAME_TOKEN
    + "(?:(?:\\s*,\\s*|\\s+(?:" + _CONNECTOR + "\\s+)?)" + _NAME_TOKEN + "){0,4})"
    + "\\s*:(?=\\s|$)",
)
# Labels that make the WHOLE line structural rather than just a prefix to peel:
# a role or a lyrics-provider metadata header.
_ROLE_LABELS = frozenset({
    "verse", "chorus", "hook", "bridge", "intro", "outro", "refrain",
    "interlude", "breakdown", "pre", "post", "part",
    "lead vocals", "backing vocals", "vocals", "featuring", "feat",
    "produced by", "written by", "producer", "writer", "composer",
    "artist", "title", "album", "song", "track", "lyrics", "genre", "year",
})
# Leftovers that never make a line meaningful on their own.
_EMPTY_AFTER = re.compile("[\\s:\\-\u2010-\u2015]+")

CHUNK_CHARS = 1200
MAX_CHUNKS = 12  # ~14k chars — covers even long Genius pulls
# A sung line longer than this is not a line — it's a merged blob. Rejecting
# instead of truncating: a gem whose "quote" is a wall of text reads as a bug.
MAX_QUOTE_LEN = 140


def normalize_newlines(text: str) -> str:
    r"""``\r\n`` and bare ``\r`` -> ``\n``. Run before ANY line splitting."""
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def cyr_ratio(text: str) -> float:
    """Share of Cyrillic letters among all letters (0.0 for no letters)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    n_cyr = sum(1 for c in letters if "а" <= c.lower() <= "я" or c.lower() == "ё")
    return n_cyr / len(letters)


def _strip_bracket_groups(line: str) -> str:
    """Drop every balanced bracket group, honoring nesting.

    ``"[Chorus: Jay-Z & (Kanye West)]"`` -> ``""``; a plain lyric keeps its
    text. Unbalanced brackets are left alone rather than eating the line.
    """
    out: List[str] = []
    stack: List[str] = []
    for ch in line:
        if ch in _OPEN:
            stack.append(_OPEN[ch])
            continue
        if stack and ch == stack[-1]:
            stack.pop()
            continue
        if not stack:
            out.append(ch)
    if stack:
        return line  # unbalanced — not our business
    return "".join(out)


def _is_role_label(label: str) -> bool:
    """True for section/role/provider-metadata labels (drop the whole line)."""
    key = re.sub(r"[\s\d]+", " ", (label or "").lower()).strip()
    if key in _ROLE_LABELS:
        return True
    return key.split(" ")[0] in _ROLE_LABELS if key else False


def _peel(line: str) -> tuple[str, bool]:
    """``(line without markup, whole line is structural)``.

    Shared by :func:`is_structural_line` and :func:`strip_line_markup` so the
    two can never disagree about what counts as markup.
    """
    if _TRACKLIST_LINE.match(line) or _STAGE_LINE.match(line) or _CJK_CREDIT_LINE.match(line):
        return "", True
    bare = _WIKI_BOLD.sub("", _TIMESTAMP.sub("", line))
    m = _SPEAKER_PREFIX.match(bare)
    if m is not None:
        if _is_role_label(m.group(1)):
            return "", True
        bare = bare[m.end():]
    return bare, not _EMPTY_AFTER.sub("", _strip_bracket_groups(bare))


def is_structural_line(line: str) -> bool:
    """True when the line is markup, not a sung line."""
    return _peel(line)[1]


def strip_line_markup(line: str) -> str:
    """A line reduced to what was actually sung.

    Timestamps and wiki bold go away; a personal speaker label ("Raekwon: Yo,
    Meth") is peeled off so the name cannot be mined as a mention while the
    words that follow stay quotable.
    """
    return _peel(line)[0]


def clean_lyrics(text: str) -> str:
    """Drop structural lines; peel markup off the ones that remain."""
    kept: List[str] = []
    for line in normalize_newlines(text).split("\n"):
        bare, structural = _peel(line)
        if structural:
            continue
        kept.append(bare)
    return "\n".join(kept)


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, max_chunks: int = MAX_CHUNKS) -> List[str]:
    """Split on line boundaries into ~chunk_chars pieces (GLiNER input size)."""
    chunks: List[str] = []
    cur = ""
    for line in text.split("\n"):
        if len(cur) + len(line) > chunk_chars and cur.strip():
            chunks.append(cur)
            if len(chunks) >= max_chunks:
                return chunks
            cur = ""
        cur += line + "\n"
    if cur.strip():
        chunks.append(cur)
    return chunks[:max_chunks]


def whole_word_re(surface: str) -> re.Pattern:
    """``surface`` as a whole word — "Wee" must not match inside "Between"."""
    pat = re.escape(surface)
    lead = r"(?<!\w)" if (surface[:1].isalnum() or surface[:1] == "_") else ""
    tail = r"(?!\w)" if (surface[-1:].isalnum() or surface[-1:] == "_") else ""
    return re.compile(lead + pat + tail, re.IGNORECASE)


def find_quote(surface: str, original_lyrics: str, max_len: int = MAX_QUOTE_LEN) -> Optional[str]:
    """The sung line containing ``surface`` as a whole word, or None.

    Structural lines are never valid quotes, and a "line" longer than
    ``max_len`` is a merged blob rather than a lyric — both return None so the
    caller drops the gem entirely. Silence over noise.
    """
    if not surface or not original_lyrics:
        return None
    pat = whole_word_re(surface)
    for line in normalize_newlines(original_lyrics).split("\n"):
        bare, structural = _peel(line)
        if structural:
            continue
        quote = bare.strip()
        if not quote or len(quote) > max_len:
            continue
        if pat.search(quote):
            return quote
    return None


def appears_capitalized(surface: str, original_lyrics: str) -> bool:
    """True when ``surface`` occurs at least once starting with an uppercase
    letter in the original text (brands/names in lyrics keep their caps;
    common words used as words don't)."""
    if not surface:
        return False
    for m in whole_word_re(surface).finditer(normalize_newlines(original_lyrics)):
        if m.group(0)[:1].isupper():
            return True
    return False
