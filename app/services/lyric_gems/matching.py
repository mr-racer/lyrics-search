"""Canonicalization of GLiNER spans against the library and gazetteers.

Pure stdlib. All name comparison goes through :func:`norm_name`, which folds
case, punctuation and unicode dash variants (the library has both "Jay-Z" and
"JAY‐Z" with a non-ASCII hyphen — measured on the dry run).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from app.services.lyric_gems.gazetteers import (
    GENERIC_ARTIST_WORDS,
    STOPWORDS,
    load_popculture,
)

MIN_NAME_LEN = 4
MIN_SCORE = 0.6
LLM_CANDIDATE_MIN_SCORE = 0.75

# Song/album titles that are too generic to mean anything as a plain match.
COMMON_TITLE_WORDS = frozenset(
    """
    home love baby tonight forever yesterday today tomorrow money
    fire water heaven hell angel dreams dream life night day time
    happy sad alone together closer higher lower stay go run
    intro outro interlude freestyle remix
    """.split()
)


def norm_name(s: str) -> str:
    """'JAY‐Z' / 'Jay-Z ' / 'jay z' -> 'jay z'."""
    return re.sub(r"[^\w]+", " ", (s or "").lower()).strip()


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
) -> Tuple[Optional[dict], bool]:
    """Match one GLiNER celebrity span against the library artist index.

    Returns ``(gem_or_None, needs_llm)``. A gem for a GENERIC-word artist
    name is never auto-accepted — it comes back with ``needs_llm=True`` and
    the caller decides (LLM verification or drop).
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
    needs_llm = key in GENERIC_ARTIST_WORDS
    return gem, needs_llm


def build_song_catalog(tracks: List[dict]) -> Dict[str, dict]:
    """{norm_title: {title, kind, artist}} over the collection's albums and
    track titles. Generic one-word titles are excluded — matching them by
    text is meaningless."""
    catalog: Dict[str, dict] = {}

    def _eligible(t: str) -> bool:
        key = norm_name(t)
        if len(key) < 5:
            return False
        words = key.split()
        if len(words) == 1 and (key in COMMON_TITLE_WORDS or len(key) < 6):
            return False
        return True

    for tr in tracks:
        album = (tr.get("album") or "").strip()
        if album and _eligible(album):
            catalog.setdefault(
                norm_name(album),
                {"title": album, "kind": "album", "artist": tr.get("artist") or ""},
            )
        title = (tr.get("title") or "").strip()
        # strip "(feat. X)" / "(Remix)" tails for the catalog key
        bare = re.sub(r"\s*[\[(].*?[\])]\s*", " ", title).strip()
        if bare and _eligible(bare):
            catalog.setdefault(
                norm_name(bare),
                {"title": bare, "kind": "track", "artist": tr.get("artist") or ""},
            )
    return catalog


def match_songref(
    span: str,
    score: Optional[float],
    catalog: Dict[str, dict],
    own_title: str,
    own_album: str,
) -> Optional[dict]:
    """Match one GLiNER song_or_album span against the library catalog."""
    key = norm_name(span)
    if not key or (score is not None and score < MIN_SCORE):
        return None
    entry = catalog.get(key)
    if entry is None:
        return None
    if key in (norm_name(own_title), norm_name(own_album)):
        return None
    return {
        "kind": "songref",
        "canonical": entry["title"],
        "display": entry["title"],  # "album"/"track" label is added at render, by lang
        "quote": None,
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
