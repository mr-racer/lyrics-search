"""The user's library, read from SQLite.

Deliberately NOT Qdrant. The production paths this replaces both go to the
vector store for something that is plain relational data:
``agent_deps.resolve_filter_values`` scrolls the collection 128 points at a
time to collect artist names, and ``catalog_search_service`` reads a memoised
full-collection scroll. ``track_metadata`` has all of it in one indexed table —
and, unlike the scroll, it hands over the release year in the same row, which
is what the era filter needs.

Matching is the part worth preserving from the current assistant, so it is kept
exactly: exact on a folded, feat-stripped title key with an artist containment
check, then fuzzy with a strict similarity gate, and every result tagged
``exact`` / ``fuzzy`` / ``none``. The normalisation helpers come from
``websearch_lab`` rather than being written again — they are the same
transliteration-aware functions the production resolver uses.

If ``track_metadata`` is empty (a dump taken before the backfill, or a fresh
instance) the catalog falls back to ``songs`` + ``artists`` for name resolution
and says so. That is degraded, not broken: the general branch needs names, only
the playlist branch needs track ids.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Iterable, Optional

from lab.agent.models import MatchMode, ResolvedTrack, TrackRef
from lab.websearch_lab import fold, similar, title_key

logger = logging.getLogger(__name__)

# Below this, a fuzzy title match is not a match. Lifted from the production
# resolver, where it was tuned against real "best of" listicles.
FUZZY_TITLE_MIN = 0.75
FUZZY_ARTIST_MIN = 0.75
# An artist candidate below this never reaches the model.
ARTIST_SCORE_MIN = 0.5


def _artist_contains(query: str, value: str) -> bool:
    """Loose containment both ways: "Kanye" matches "Kanye West", and a track
    tagged "Kanye West, Jay-Z" matches a query for "Kanye West"."""
    a, b = fold(query or ""), fold(value or "")
    return bool(a) and bool(b) and (a in b or b in a)


class LibraryCatalog:
    """Everything the agent needs to know about what the user actually owns."""

    def __init__(self, db_path: str, collection_name: Optional[str] = None):
        self.db_path = str(db_path)
        self.collection_name = collection_name or self._largest_collection()
        self.songs: list[dict] = []
        self.by_title: dict[str, list[dict]] = {}
        self.artists: list[str] = []
        self.artist_slugs: dict[str, str] = {}     # folded name -> slug
        self.degraded = False
        self._load()

    # ── loading ───────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _largest_collection(self) -> Optional[str]:
        """The account with the most tracks — the one a dev dump is about."""
        try:
            with self._connect() as conn:
                for table, column in (("track_metadata", "collection_name"),
                                      ("songs", "collection_name")):
                    row = conn.execute(
                        f"SELECT {column} AS c, COUNT(*) AS n FROM {table} "
                        f"GROUP BY 1 ORDER BY n DESC LIMIT 1").fetchone()
                    if row and row["c"]:
                        return row["c"]
        except Exception:
            logger.warning("[catalog] could not pick a collection", exc_info=True)
        return None

    def _load(self) -> None:
        rows: list[sqlite3.Row] = []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT track_id, title, artist, artists, artist_slugs, "
                    "       primary_artist_slug, album, year "
                    "FROM track_metadata WHERE collection_name = ?",
                    (self.collection_name,)).fetchall()
        except Exception:
            logger.warning("[catalog] track_metadata unreadable", exc_info=True)

        if rows:
            for r in rows:
                song = {
                    "track_id": r["track_id"],
                    "title": r["title"] or "",
                    "artist": r["artist"] or "",
                    "album": r["album"] or "",
                    "year": r["year"],
                    "artist_slug": r["primary_artist_slug"] or "",
                    "artist_slugs": _json_list(r["artist_slugs"]),
                    "artists": _json_list(r["artists"]),
                }
                self.songs.append(song)
                self.by_title.setdefault(title_key(song["title"]), []).append(song)
        else:
            self.degraded = True
            logger.warning(
                "[catalog] no track_metadata for %r — falling back to songs/artists. "
                "Name resolution works; track ids, years and playlist assembly do not.",
                self.collection_name)
            self._load_degraded()

        self._load_artists()
        logger.info("[catalog] %s: %d tracks, %d artists%s", self.collection_name,
                    len(self.songs), len(self.artists),
                    " (degraded)" if self.degraded else "")

    def _load_degraded(self) -> None:
        """Titles from ``songs`` when the track mirror has not been backfilled."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT s.slug, s.title, s.artist_slug, a.name AS artist_name "
                    "FROM songs s LEFT JOIN artists a "
                    "  ON a.slug = s.artist_slug AND a.collection_name = s.collection_name "
                    "WHERE s.collection_name = ?", (self.collection_name,)).fetchall()
        except Exception:
            logger.warning("[catalog] songs table unreadable", exc_info=True)
            return
        for r in rows:
            song = {"track_id": "", "title": r["title"] or "",
                    "artist": r["artist_name"] or r["artist_slug"] or "",
                    "album": "", "year": None,
                    "artist_slug": r["artist_slug"] or "",
                    "artist_slugs": [r["artist_slug"]] if r["artist_slug"] else [],
                    "artists": [], "song_slug": r["slug"]}
            self.songs.append(song)
            self.by_title.setdefault(title_key(song["title"]), []).append(song)

    def _load_artists(self) -> None:
        names: dict[str, str] = {}
        try:
            with self._connect() as conn:
                for r in conn.execute(
                        "SELECT slug, name FROM artists WHERE collection_name = ?",
                        (self.collection_name,)).fetchall():
                    if r["name"]:
                        names[fold(r["name"])] = r["slug"]
        except Exception:
            logger.warning("[catalog] artists table unreadable", exc_info=True)

        # Tags carry spellings the artists table does not (features, casing),
        # so both sources feed the candidate list the model is shown.
        for song in self.songs:
            if song["artist"]:
                names.setdefault(fold(song["artist"]), song.get("artist_slug") or "")
        self.artist_slugs = names
        by_fold = {fold(s["artist"]): s["artist"] for s in self.songs if s["artist"]}
        try:
            with self._connect() as conn:
                for r in conn.execute(
                        "SELECT name FROM artists WHERE collection_name = ?",
                        (self.collection_name,)).fetchall():
                    if r["name"]:
                        by_fold.setdefault(fold(r["name"]), r["name"])
        except Exception:
            pass
        self.artists = sorted(by_fold.values(), key=str.lower)

    # ── resolution ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.songs)

    def resolve_artist(self, query: str, limit: int = 5) -> list[dict]:
        """How the user spelled an artist → what the library actually calls them.

        Transliteration-aware, so «канье» finds "Kanye West". Returns at most
        ``limit`` candidates with a score, best first.
        """
        if not (query or "").strip() or not self.artists:
            return []
        scored = [(similar(query, name), name) for name in self.artists]
        scored.sort(key=lambda pair: -pair[0])
        return [{"artist": name, "score": round(score, 2),
                 "slug": self.artist_slugs.get(fold(name), "")}
                for score, name in scored[:limit] if score >= ARTIST_SCORE_MIN]

    def artist_slug_for(self, name: str) -> Optional[str]:
        slug = self.artist_slugs.get(fold(name or ""))
        if slug:
            return slug
        best = self.resolve_artist(name, limit=1)
        return (best[0]["slug"] or None) if best else None

    def song_slug_for(self, title: str, artist: Optional[str] = None) -> Optional[str]:
        """The ``song_facts.song_slug`` of a track, looked up rather than derived.

        Deriving it would mean re-implementing ``song_facts_service._slugify``
        AND ``artist_split.primary_artist``, and CLAUDE.md is explicit that the
        three ``_slugify`` variants in the app are not interchangeable. The
        ``songs`` table already stores the answer.
        """
        key = title_key(title)
        if not key:
            return None
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT slug, title, artist_slug FROM songs "
                    "WHERE collection_name = ?", (self.collection_name,)).fetchall()
        except Exception:
            logger.warning("[catalog] songs lookup failed", exc_info=True)
            return None

        artist_slug = self.artist_slug_for(artist) if artist else None
        best: Optional[tuple[float, str]] = None
        for r in rows:
            if title_key(r["title"] or "") != key:
                continue
            score = 1.0
            if artist_slug and r["artist_slug"]:
                score += 1.0 if r["artist_slug"] == artist_slug else 0.0
            elif artist:
                score += similar(artist, r["artist_slug"] or "")
            if best is None or score > best[0]:
                best = (score, r["slug"])
        return best[1] if best else None

    def resolve_tracks(self, refs: Iterable[TrackRef],
                       *, max_fuzzy: int = 150) -> tuple[list[ResolvedTrack], list[TrackRef]]:
        """Match page claims against the library.

        Returns ``(resolved, missing)``. ``max_fuzzy`` bounds the expensive leg:
        a 200-row soundtrack against a 5000-track library is 10^6 comparisons
        if every miss is retried fuzzily.
        """
        resolved: list[ResolvedTrack] = []
        missing: list[TrackRef] = []
        seen_ids: set[str] = set()
        fuzzy_left = max_fuzzy

        for ref in refs:
            picked, mode = self._match_one(ref, allow_fuzzy=fuzzy_left > 0)
            if mode == "fuzzy":
                fuzzy_left -= 1
            if not picked:
                missing.append(ref)
                continue
            track_id = picked.get("track_id") or ""
            if track_id and track_id in seen_ids:
                continue
            if track_id:
                seen_ids.add(track_id)
            resolved.append(ResolvedTrack(
                track_id=track_id, title=picked.get("title") or ref.title,
                artist=picked.get("artist") or (ref.artist or ""),
                # The library's own year wins: a listicle prints the year of
                # the compilation it is plugging as often as the release year.
                year=picked.get("year") or ref.year,
                match=mode, sources=[ref.source]))
        return resolved, missing

    def _match_one(self, ref: TrackRef,
                   *, allow_fuzzy: bool) -> tuple[Optional[dict], MatchMode]:
        # Both orientations: pages write "Artist — Title" and "Title — Artist"
        # about equally often, and a structured table can have them swapped too.
        for title, artist in ((ref.title, ref.artist), (ref.artist or "", ref.title)):
            if not title:
                continue
            for candidate in self.by_title.get(title_key(title), []):
                if not artist or _artist_contains(artist, candidate.get("artist", "")):
                    return candidate, "exact"

        if not allow_fuzzy:
            return None, "none"

        key = title_key(ref.title)
        best: Optional[tuple[float, dict]] = None
        for candidate in self.songs:
            ck = title_key(candidate.get("title") or "")
            if not ck:
                continue
            title_score = similar(ref.title, candidate.get("title") or "")
            title_ok = (key and ck and key in ck) or title_score >= FUZZY_TITLE_MIN
            if not title_ok:
                continue
            artist_ok = (
                not ref.artist
                or _artist_contains(ref.artist, candidate.get("artist", ""))
                or similar(ref.artist, candidate.get("artist") or "") >= FUZZY_ARTIST_MIN
            )
            if not artist_ok:
                continue
            if best is None or title_score > best[0]:
                best = (title_score, candidate)
        return (best[1], "fuzzy") if best else (None, "none")

    # ── introspection ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        years = [s["year"] for s in self.songs if s.get("year")]
        return {"collection": self.collection_name, "tracks": len(self.songs),
                "artists": len(self.artists), "degraded": self.degraded,
                "years": (min(years), max(years)) if years else None}


def _json_list(raw) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []
