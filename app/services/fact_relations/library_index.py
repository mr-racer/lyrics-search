"""What the sample gates need to know about the user's own tracks.

A claimed link is only checkable when the other side is something we actually
hold — that is 275 of 2977 links on the 2026-07-26 production snapshot (9%).
This index is what turns "Kanye West — Jesus Walks" in a fact into a year, an
album and a ``songs.slug``.

One deliberate choice: :meth:`resolve` reports the **earliest** year among all
library copies of a title. Library ``year`` is the year of the *copy*, not of
the recording — the same snapshot has Bee Gees' "Stayin' Alive" at 2017, ABBA's
"Dancing Queen" at 2012 and Michael Jackson's "Billie Jean" at 2008, all from
compilations. Taking the minimum is what keeps a remaster from making a real
sample look like a time-travelling one.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

_PAREN_TAIL = re.compile(r"\s*[\[(][^)\]]*[\])]")
_FEAT_TAIL = re.compile(r"\s*\b(?:feat|ft|featuring|with)\b\.?.*$", re.IGNORECASE)
# Dropped-g elision, endemic in song titles: "Sweet Nothin's" and "Sweet
# Nothings" are the same record, and Bound 2 stored both as separate samples.
# Restoring the g before folding punctuation makes them one key.
_ELISION = re.compile(r"in['‘’ʼ]", re.IGNORECASE)


def norm(s: str) -> str:
    """Fold case, elision, punctuation and unicode dashes to a comparison key."""
    return re.sub(r"[^\w]+", " ", _ELISION.sub("ing", (s or "").lower())).strip()


def title_key(title: str) -> str:
    """Title without bracketed tails or a trailing "feat. X"."""
    return norm(_FEAT_TAIL.sub("", _PAREN_TAIL.sub(" ", title or "")))


def artist_key(artist: str) -> str:
    """Artist slug or display name folded to one comparison key."""
    return norm((artist or "").replace("-", " "))


def artists_match(a: str, b: str) -> bool:
    """Substring-tolerant artist equality ("Ye" vs "Kanye West" does NOT pass).

    Whole-word containment only: "Jay-Z" matches "JAY-Z", and "Kanye West"
    matches "Kanye West & Jay-Z", but a short name is never swallowed by an
    unrelated longer one.
    """
    ka, kb = artist_key(a), artist_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    short, long = (ka, kb) if len(ka) < len(kb) else (kb, ka)
    return re.search(rf"(?:^|\s){re.escape(short)}(?:\s|$)", long) is not None


class LibraryIndex:
    """Title-keyed view of the collection, plus the song-slug lookup."""

    def __init__(self, tracks: Iterable[dict] = (), songs: Iterable[dict] = ()):
        self._by_title: Dict[str, List[dict]] = {}
        for tr in tracks:
            key = title_key(tr.get("title") or "")
            if not key:
                continue
            self._by_title.setdefault(key, []).append({
                "title": tr.get("title") or "",
                "artist": tr.get("artist") or "",
                "album": tr.get("album") or "",
                "year": tr.get("year"),
            })
        self._slugs: Dict[str, str] = {}
        for s in songs:
            key = (title_key(s.get("title") or ""), artist_key(s.get("artist_slug") or ""))
            if key[0]:
                self._slugs.setdefault(key, s.get("slug") or "")

    def resolve(self, title: Optional[str], artist: Optional[str]) -> Optional[dict]:
        """``{title, artist, album, year}`` for a claimed work, or None.

        Without an artist a title is only resolved when the library holds
        exactly one candidate — guessing which "Hey Mama" was meant is how a
        1987 Michael Jackson track ended up "sampling" Dua Lipa.
        """
        cands = self._by_title.get(title_key(title or ""), [])
        if not cands:
            return None
        if artist:
            cands = [c for c in cands if artists_match(artist, c["artist"])]
        elif len(cands) > 1:
            return None
        if not cands:
            return None
        years = [c["year"] for c in cands if isinstance(c.get("year"), int)]
        best = cands[0]
        return {
            "title": best["title"],
            "artist": best["artist"],
            "album": best["album"],
            "year": min(years) if years else None,
            "albums": {c["album"] for c in cands if c.get("album")},
        }

    def song_slug(self, title: Optional[str], artist: Optional[str]) -> Optional[str]:
        """``songs.slug`` for a claimed work, or None when we don't hold it."""
        tkey = title_key(title or "")
        if not tkey:
            return None
        akey = artist_key(artist or "")
        if akey:
            for (t, a), slug in self._slugs.items():
                if t == tkey and (a == akey or artists_match(a, akey)):
                    return slug or None
            return None
        hits = [slug for (t, _a), slug in self._slugs.items() if t == tkey]
        return hits[0] if len(hits) == 1 else None
