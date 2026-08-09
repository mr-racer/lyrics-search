"""Code gate over the per-track "why this fits" line.

Ported from ``app/services/assistant/reason_gate.py`` with its reasoning
intact, because the reasoning is the valuable part: the prompt already forbids
filler and a 12b model obeys that unevenly, so nothing reaches a card that this
has not cleared. Dropping a reason is cheap — the row renders without a second
line and nothing looks broken. Showing "отличный трек" is what makes the whole
feature read as noise.
"""

from __future__ import annotations

import re

from lab.websearch_lab import fold

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_STOPWORDS = frozenset({
    "трек", "песня", "песню", "композиция", "это", "тут", "здесь", "и", "а",
    "the", "a", "an", "track", "song", "this", "is", "it", "of", "by",
})

MIN_LEN = 12
MAX_LEN = 90

_FILLER = tuple(fold(p) for p in (
    "отличный трек", "отличная песня", "хороший трек", "хорошая песня",
    "подходит под настроение", "подходит к настроению", "создаёт настроение",
    "хороший выбор", "просто хит",
    "great track", "great song", "good track", "fits the mood",
    "fits the vibe", "matches the mood", "perfect for this",
))


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _restates_track(text: str, title: str, artist: str) -> bool:
    """True when the reason adds nothing to the row printed above it.

    Measured by subtraction, not containment: the model both quotes the title
    verbatim and pads around it, so what matters is what is left once the words
    already on the row are removed.
    """
    known = set(_tokens(title)) | set(_tokens(artist))
    rest = [t for t in _tokens(text) if t not in known and t not in _STOPWORDS]
    return len(rest) <= 1


def clean_reason(reason, *, title: str = "", artist: str = ""):
    """The reason to show, or None for no second line."""
    if not isinstance(reason, str):
        return None
    text = reason.strip()
    if not (MIN_LEN <= len(text) <= MAX_LEN):
        return None
    folded = fold(text)
    if not folded or any(f in folded for f in _FILLER):
        return None
    if _restates_track(text, title, artist):
        return None
    return text
