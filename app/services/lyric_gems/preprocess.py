"""Lyrics preprocessing for the gems pipeline. Pure stdlib — no torch here.

The two structural noise sources (measured on the 2026-07-19 dry run over
5422 tracks) are Genius-style section headers (``[Verse 1: Eminem]`` — the
top false-namedrop source, 809→~200 candidates once stripped) and embedded
tracklist lines (``14. Apologize (Timbaland Remix) [0:03:04]``).
"""
from __future__ import annotations

import re
from typing import List, Optional

# A line that is nothing but bracketed/parenthesised markers: "[Chorus: X]",
# "(Hook){50 Cent}", "[Intro]" — possibly several groups on one line.
_SECTION_LINE = re.compile(r"^\s*(?:[\[({][^\])}]*[\])}]\s*[:\-]?\s*)+$")
# Tracklist artifacts: "14. Apologize (Timbaland Remix) [0:03:04.82]"
_TRACKLIST_LINE = re.compile(r"^\s*\d{1,2}\.\s.+\[\d+:\d")
_TIMESTAMP = re.compile(r"\[\d+:\d+(?::\d+)?(?:\.\d+)?\]")

CHUNK_CHARS = 1200
MAX_CHUNKS = 12  # ≈14k chars — covers even long Genius pulls


def cyr_ratio(text: str) -> float:
    """Share of Cyrillic letters among all letters (0.0 for no letters)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    n_cyr = sum(1 for c in letters if "а" <= c.lower() <= "я" or c.lower() == "ё")
    return n_cyr / len(letters)


def clean_lyrics(text: str) -> str:
    """Drop section-header and tracklist lines; keep everything else verbatim."""
    kept: List[str] = []
    for line in (text or "").split("\n"):
        if _SECTION_LINE.match(line) or _TRACKLIST_LINE.match(line):
            continue
        kept.append(_TIMESTAMP.sub("", line))
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


def find_quote(surface: str, original_lyrics: str, max_len: int = 200) -> Optional[str]:
    """The original lyrics line containing ``surface`` (case-insensitive).

    Section-header lines are never valid quotes. Returns None when the
    surface only ever appears inside structural markup or not at all.
    """
    if not surface or not original_lyrics:
        return None
    pat = re.compile(re.escape(surface), re.IGNORECASE)
    for line in original_lyrics.split("\n"):
        if _SECTION_LINE.match(line) or _TRACKLIST_LINE.match(line):
            continue
        if pat.search(line):
            quote = line.strip()
            if len(quote) > max_len:
                quote = quote[: max_len].rstrip() + "…"
            return quote
    return None


def appears_capitalized(surface: str, original_lyrics: str) -> bool:
    """True when ``surface`` occurs at least once starting with an uppercase
    letter in the original text (brands/names in lyrics keep their caps;
    common words used as words don't)."""
    if not surface:
        return False
    pat = re.compile(re.escape(surface), re.IGNORECASE)
    for m in pat.finditer(original_lyrics):
        if m.group(0)[:1].isupper():
            return True
    return False
