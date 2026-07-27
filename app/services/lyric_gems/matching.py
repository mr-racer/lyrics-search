"""Canonicalization of GLiNER spans against the library and gazetteers.

Pure stdlib. All name comparison goes through :func:`norm_name`, which folds
case, punctuation and unicode dash variants (the library has both "Jay-Z" and
"JAY-Z" with a non-ASCII hyphen — measured on the dry run).

Two hard-won rules live here, both from the 2026-07-26 production audit:

- **Namedrop identity is a slug, not a string.** ``own_name_keys`` only ever
  held display names, so an alias slipped straight through: "Life," by Marshall
  Mathers on an *Eminem* track produced a namedrop of Eminem, and "Pablo bought
  a Roley" did the same for Kanye. Comparing the resolved ``artist_slug``
  against the track's own slugs closes that regardless of alias coverage.
- **A song title in lyrics means nothing without a citation marker.** Matching
  plain text against library titles turned "Wake up" into a reference to
  Fergie's "Wake Up" and "'cos I love you" into Billie Eilish's "i love you".
  A title now counts only when it is quoted or introduced by a citation word.
"""
from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Optional, Tuple

from app.services.lyric_gems.gazetteers import (
    GENERIC_ARTIST_WORDS,
    STOPWORDS,
    load_popculture,
)
from app.services.lyric_gems.preprocess import whole_word_re

MIN_NAME_LEN = 4
MIN_SCORE = 0.6
LLM_CANDIDATE_MIN_SCORE = 0.75
# songref is the noisiest kind by far — it gets its own, higher bar.
SONGREF_MIN_SCORE = 0.75
# An artist credited on fewer tracks than this is a weak identity: a one-word
# name ("Kobe") is then more likely a basketball player in someone else's
# lyrics than a namedrop, so it goes to the LLM verifier instead of straight in.
RARE_ARTIST_TRACKS = 3
CITATION_WINDOW = 40

# Song/album titles that are too generic to mean anything as a plain match.
COMMON_TITLE_WORDS = frozenset(
    """
    home love baby tonight forever yesterday today tomorrow money
    fire water heaven hell angel dreams dream life night day time
    happy sad alone together closer higher lower stay go run
    intro outro interlude freestyle remix
    """.split()
)

_QUOTE_CHARS = "\"'‘’“”«»‹›‚„"
# Words that introduce a musical work by name in a lyric.
_CITATION_WORDS = re.compile(
    r"\b(?:songs?|tracks?|albums?|records?|singles?|joint|cut|"
    r"called|titled|named|"
    r"listen(?:ing)? to|throw on|bump(?:in['g]?)?|put on|play(?:ed|in['g]?)?|"
    r"off (?:my|the|his|her|their|that)|on (?:my|the|his|her|their|that) album)\b",
    re.IGNORECASE,
)
_PAREN_TAIL = re.compile(r"\s*[\[(].*?[\])]\s*")


def norm_name(s: str) -> str:
    """'JAY-Z' / 'Jay-Z ' / 'jay z' -> 'jay z'."""
    return re.sub(r"[^\w]+", " ", (s or "").lower()).strip()


def bare_title(title: str) -> str:
    """Title without its bracketed tail: "Fallin' (Feat. Bilal)" -> "Fallin'".

    The catalog is keyed on this form, so every own-title comparison must use
    it too. They used not to, which is why Queen's "Las Palabras de Amor (The
    Words of Love)" ended up referencing itself — along with six more.
    """
    return _PAREN_TAIL.sub(" ", title or "").strip()


def build_artist_index(
    artists: Dict[str, str], aliases: Dict[str, str],
) -> Dict[str, Tuple[str, str]]:
    """{norm_name: (slug, display)} over library artists + their aliases.

    ``artists`` is {slug: display_name}; ``aliases`` is {alias_lower: slug}.
    Names shorter than MIN_NAME_LEN never enter (too ambiguous to match).
    """
    index: Dict[str, Tuple[str, str]] = {}
    for slug, display in artists.items():
        key = norm_name(display)
        if len(key) >= MIN_NAME_LEN:
            index[key] = (slug, display)
    for alias, slug in aliases.items():
        key = norm_name(alias)
        if len(key) >= MIN_NAME_LEN and slug in artists:
            index.setdefault(key, (slug, artists[slug]))
    return index


def own_name_keys(track_artist: str, track_artists: List[str], title: str) -> frozenset:
    """Normalized names that can never be a namedrop for this track: the
    performer(s), everyone featured, and anything already in the title
    ("Renegade (feat. Eminem)" mentioning Eminem is not a surprise)."""
    keys = set()
    for name in [track_artist, *(track_artists or [])]:
        k = norm_name(name)
        if k:
            keys.add(k)
    title_norm = norm_name(title)
    if title_norm:
        keys.add(title_norm)
    return frozenset(keys)


def _in_title(candidate_norm: str, title: str) -> bool:
    title_norm = norm_name(title)
    return bool(candidate_norm) and candidate_norm in title_norm


def match_namedrop(
    span: str,
    score: Optional[float],
    artist_index: Dict[str, Tuple[str, str]],
    own_keys: frozenset,
    title: str,
    own_slugs: FrozenSet[str] = frozenset(),
    rare_slugs: FrozenSet[str] = frozenset(),
) -> Tuple[Optional[dict], bool]:
    """Match one GLiNER celebrity span against the library artist index.

    Returns ``(gem_or_None, needs_llm)``. ``own_slugs`` are the track's own
    performer slugs — a span resolving to one of them is the performer under
    an alias, never a namedrop. ``rare_slugs`` are artists with very few
    library tracks; a one-word name among them is ambiguous enough that the
    LLM decides. A GENERIC-word artist name is likewise never auto-accepted.
    """
    key = norm_name(span)
    if not key or len(key) < MIN_NAME_LEN or key in STOPWORDS:
        return None, False
    if score is not None and score < MIN_SCORE:
        return None, False
    hit = artist_index.get(key)
    if hit is None:
        return None, False
    slug, display = hit
    if slug in own_slugs:
        return None, False
    if key in own_keys or any(key in own or own in key for own in own_keys):
        return None, False
    if _in_title(key, title):
        return None, False
    gem = {
        "kind": "namedrop",
        "canonical": display,
        "display": display,
        "quote": None,  # caller fills from original lyrics
        "detail": {"artist_slug": slug},
        "score": score,
    }
    needs_llm = key in GENERIC_ARTIST_WORDS or (
        slug in rare_slugs and len(key.split()) == 1
    )
    return gem, needs_llm


def build_song_catalog(tracks: List[dict]) -> Dict[str, dict]:
    """{norm_title: {title, kind, artist, ambiguous}} over the collection.

    Generic one-word titles are excluded — matching them by text is
    meaningless. A title held by more than one artist is kept but flagged
    ``ambiguous``: there is no way to tell which one a lyric meant, and a
    confidently wrong attribution ("Wake Up" credited to Fergie) is worse than
    no gem at all.
    """
    catalog: Dict[str, dict] = {}

    def _eligible(t: str) -> bool:
        key = norm_name(t)
        if len(key) < 5:
            return False
        words = key.split()
        if len(words) == 1 and (key in COMMON_TITLE_WORDS or len(key) < 6):
            return False
        return True

    def _put(key: str, title: str, kind: str, artist: str) -> None:
        entry = catalog.get(key)
        if entry is None:
            catalog[key] = {
                "title": title, "kind": kind, "artist": artist, "ambiguous": False,
            }
            return
        if artist and norm_name(artist) != norm_name(entry["artist"]):
            entry["ambiguous"] = True

    for tr in tracks:
        artist = tr.get("artist") or ""
        album = (tr.get("album") or "").strip()
        if album and _eligible(album):
            _put(norm_name(album), album, "album", artist)
        title = (tr.get("title") or "").strip()
        bare = bare_title(title)
        if bare and _eligible(bare):
            _put(norm_name(bare), bare, "track", artist)
    return catalog


def has_citation_marker(span: str, quote: str, window: int = CITATION_WINDOW) -> bool:
    """True when ``span`` is presented in ``quote`` as the name of a work.

    Either the title is wrapped in quotation marks — 'I made "Jesus Walks"' —
    or a citation word sits nearby — 'The song Guilty Conscience has gotten…'.
    Bare text that merely happens to equal a title never qualifies.
    """
    if not span or not quote:
        return False
    m = whole_word_re(span).search(quote)
    if m is None:
        return False
    start, end = m.start(), m.end()
    pre, post = quote[max(0, start - 3):start], quote[end:end + 3]
    if any(c in _QUOTE_CHARS for c in pre) and any(c in _QUOTE_CHARS for c in post):
        return True
    near = quote[max(0, start - window):end + window]
    return bool(_CITATION_WORDS.search(near))


def match_songref(
    span: str,
    score: Optional[float],
    catalog: Dict[str, dict],
    own_title: str,
    own_album: str,
    quote: Optional[str] = None,
) -> Optional[dict]:
    """Match one GLiNER song_or_album span against the library catalog.

    ``quote`` is the lyric line the span was found on; without a citation
    marker on it the match is rejected. Passing ``None`` skips that gate and
    exists only for callers that have already applied it.
    """
    key = norm_name(span)
    if not key or (score is not None and score < SONGREF_MIN_SCORE):
        return None
    if len(key.split()) < 2:
        return None
    entry = catalog.get(key)
    if entry is None or entry.get("ambiguous"):
        return None
    own_key = norm_name(bare_title(own_title))
    own_album_key = norm_name(bare_title(own_album))
    if key and key in (own_key, own_album_key):
        return None
    # Contained in the track's OWN title: 'This next song\'s called "Novacaine"'
    # on "Give Me Novacaine (Live at …)" is the track announcing itself. The
    # album is matched by equality only — "Off That" off "The Blueprint 3" may
    # legitimately name "The Blueprint".
    if own_key and re.search(rf"(?:^|\s){re.escape(key)}(?:\s|$)", own_key):
        return None
    if quote is not None and not has_citation_marker(span, quote):
        return None
    return {
        "kind": "songref",
        "canonical": entry["title"],
        "display": entry["title"],  # "album"/"track" label is added at render, by lang
        "quote": quote,
        "detail": {"ref_kind": entry["kind"], "artist": entry["artist"]},
        "score": score,
    }


def match_popculture(span: str, score: Optional[float]) -> Tuple[Optional[dict], bool]:
    """Match one fictional_character/movie span against the pop gazetteer.

    Returns ``(gem_or_None, needs_llm)``: unknown spans with a high score are
    LLM candidates, known ones are accepted as-is."""
    key = (span or "").lower().strip()
    if not key or key in STOPWORDS or (score is not None and score < MIN_SCORE):
        return None, False
    entry = load_popculture().get(key)
    if entry is not None:
        return {
            "kind": "popculture",
            "canonical": entry["canonical"],
            "display": entry["canonical"],
            "quote": None,
            "detail": {k: v for k, v in entry.items() if k != "canonical"},
            "score": score,
        }, False
    if score is not None and score >= LLM_CANDIDATE_MIN_SCORE and len(key) >= MIN_NAME_LEN:
        return None, True  # unknown but confident — worth one cached LLM look
    return None, False
