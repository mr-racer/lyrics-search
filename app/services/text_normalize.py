"""Text analyzer for catalog search — diacritic folding, tokenization, and
Cyrillic↔Latin transliteration. Pure functions, no I/O; shared at index time and
query time so the same normalization applies to both.
"""
from __future__ import annotations

import unicodedata


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
