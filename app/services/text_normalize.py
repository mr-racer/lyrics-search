"""Text analyzer for catalog search — diacritic folding, tokenization, and
Cyrillic↔Latin transliteration. Pure functions, no I/O; shared at index time and
query time so the same normalization applies to both.

The lower half (``similar``, ``strip_qualifiers``, ``title_key``) serves title
and name MATCHING rather than indexing: the assistant's library catalog resolves
what a web page claims against what the user owns, and both sides have to be
normalised the same way or the match is decided by punctuation.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


def fold(text: str) -> str:
    """Normalize for diacritic/case/punctuation-insensitive matching.

    NFKD-decompose → drop combining marks (folds é→e, ё→е, й→и) → lowercase →
    map every non-alphanumeric char to a space → collapse runs of whitespace.
    Cyrillic and Latin letters and digits survive; everything else becomes a gap.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    out = []
    for ch in decomposed:
        if unicodedata.category(ch) == "Mn":  # combining mark
            continue
        out.append(ch if ch.isalnum() else " ")
    return " ".join("".join(out).lower().split())


# Collaboration / production noise tokens dropped before matching.
_NOISE_TOKENS = {"feat", "ft", "featuring", "prod", "vs"}


def tokenize(text: str) -> list[str]:
    """``fold`` then split on whitespace, dropping collaboration noise tokens."""
    return [t for t in fold(text).split() if t and t not in _NOISE_TOKENS]


# Cyrillic→Latin: deterministic char map (folded text has no ё/й — already е/и).
_CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh",
    "з": "z", "и": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h",
    "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y",
    "ь": "", "э": "e", "ю": "yu", "я": "ya",
}

# Latin→Cyrillic: longest-match digraphs first, then singles. Best-effort
# (Latin→Cyrillic is inherently ambiguous); matching survives because both the
# index and the query also carry the well-defined Cyrillic→Latin direction.
_LAT_DIGRAPHS = {
    "shch": "щ", "zh": "ж", "kh": "х", "ts": "ц", "ch": "ч", "sh": "ш",
    "yu": "ю", "ya": "я", "yo": "е",
}
_LAT_SINGLES = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "и", "z": "з",
}


def _cyr_to_lat(s: str) -> str:
    return "".join(_CYR_TO_LAT.get(c, c) for c in s)


def _lat_to_cyr(s: str) -> str:
    out, i, n = [], 0, len(s)
    while i < n:
        matched = False
        for size in (4, 3, 2):  # shch / digraphs
            chunk = s[i:i + size]
            if chunk in _LAT_DIGRAPHS:
                out.append(_LAT_DIGRAPHS[chunk])
                i += size
                matched = True
                break
        if not matched:
            out.append(_LAT_SINGLES.get(s[i], s[i]))
            i += 1
    return "".join(out)


def translit_variants(token: str) -> set[str]:
    """Return the folded token plus its cross-script transliteration.

    Cyrillic token → adds its Latin form; Latin token → adds its Cyrillic form.
    A token in neither script (digits, etc.) returns just itself.
    """
    t = fold(token)
    if not t:
        return set()
    variants = {t}
    if any("Ѐ" <= c <= "ӿ" for c in t):
        variants.add(_cyr_to_lat(t))
    elif any("a" <= c <= "z" for c in t):
        variants.add(_lat_to_cyr(t))
    variants.discard("")
    return variants


def analyze(text: str) -> list[str]:
    """Token bag for indexing: every token's folded form + translit variants."""
    bag: list[str] = []
    for tok in tokenize(text):
        bag.extend(sorted(translit_variants(tok)))
    return bag


def to_latin(text: str) -> str:
    """Cyrillic → Latin, one direction, deterministic.

    Public because the assistant's catalog needs the *single* Latin form of a
    string, not the variant set: it precomputes one per library track and
    compares them pairwise, where building a set per comparison is what made the
    fuzzy leg quadratic in practice.
    """
    return _cyr_to_lat(text)


def similar(a: str, b: str) -> float:
    """Cross-script similarity of two names, 0..1.

    A simplified ``name_match.score_names``: fold both, try both alphabets, take
    the best ratio. Used to SHORTLIST candidates, never to decide identity —
    measured on a real library, "Muse"/"Fuse" scores 0.750 (wrong) while
    «канье»/"Kanye West" scores 0.571 (right), so no threshold separates the two
    and whoever consumes the shortlist has to judge.
    """
    fa, fb = fold(a), fold(b)
    if not fa or not fb:
        return 0.0
    return max(SequenceMatcher(None, x, y).ratio()
               for x in {fa, _cyr_to_lat(fa)} for y in {fb, _cyr_to_lat(fb)})


_FEAT = {"feat", "ft", "featuring"}

# Words that make a bracketed tail a QUALIFIER of the recording rather than part
# of its name. Measured on a 5630-track library: 968 titles end in a bracket,
# 781 of them like this — 14% of the library that a plain claim for "Work" could
# never reach while the file is called "Work (Freemasons Remix)".
#
# The list is deliberately a whitelist and not "strip any bracket". The other
# 187 carry brackets that ARE the title — "See You on Monday (You're Lost)",
# "I Just Wanna Love U (Give It 2 Me)", "Eh, Eh (Nothing Else I Can Say)" — and
# cutting those merges songs that merely share a first half.
_QUALIFIER = re.compile(
    r"(?i)\b(remix|mix|version|edit|live|acoustic|instrumental|demo"
    r"|remaster(?:ed)?|radio|single|extended|club|dub|reprise|bonus|deluxe"
    r"|mono|stereo|cover|feat|ft|featuring|with|explicit|clean|original"
    r"|album)\b")

# TRAILING only. 22 titles in the same library OPEN with a bracket —
# "(I Can't Get No) Satisfaction", "(Sittin' On) the Dock of the Bay" — and
# there the bracket is the beginning of the sentence, not a note about the mix.
_TRAILING_BRACKET = re.compile(r"[\(\[]([^)\]]*)[\)\]]\s*$")


def strip_qualifiers(s: str) -> str:
    """Drop trailing "(remix)" / "[live]" / "(feat. X)" tails, repeatedly.

    Repeatedly because they stack: "Faint [Live] [bonus track]" needs two passes
    to become "Faint". A title that is nothing BUT a bracket is left alone —
    "[Premade Sandwiches]" is a real track name.
    """
    out = (s or "").strip()
    while True:
        found = _TRAILING_BRACKET.search(out)
        if not found or not _QUALIFIER.search(found.group(1)):
            return out
        shorter = out[:found.start()].strip()
        if not shorter:
            return out
        out = shorter


def title_key(s: str) -> str:
    """The key two titles must share to be the same song.

    Qualifiers go before folding, because ``fold`` turns brackets into spaces and
    by then "Work (Freemasons Remix)" is indistinguishable from a song genuinely
    called "Work Freemasons Remix".
    """
    toks = fold(strip_qualifiers(s)).split()
    for i, t in enumerate(toks):
        if t in _FEAT:
            return " ".join(toks[:i]) or " ".join(toks)
    return " ".join(toks)
