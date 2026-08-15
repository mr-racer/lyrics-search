"""What the user actually owns, in memory, indexed for matching.

Deliberately SQLite and not Qdrant. The paths this replaces both went to the
vector store for what is plain relational data: ``agent_deps.resolve_filter_values``
scrolls the collection 128 points at a time to collect artist names, and
``catalog_search_service`` reads a memoised full-collection scroll.
``track_metadata`` has all of it in one indexed table — and, unlike the scroll,
it hands over the release year in the same row, which is what the era filter
needs.

**Per-account gating is not uniform across these tables, and getting it wrong is
silent.** ``track_metadata`` has a composite primary key ``(collection_name,
track_id)``, so filtering it by ``collection_name`` is correct. ``songs`` and
``artists`` do NOT: their primary key is a global slug and ``collection_name`` is
one mutable column on it, which is exactly the bug ``fact_visibility`` was
created to fix — whichever account indexed a slug last used to steal every other
account's visibility. So those two are gated by joining ``fact_visibility``, and
never by their own ``collection_name`` column.

Matching is the part worth preserving from the previous assistant, so it is kept
exactly: exact on a folded, feat-stripped title key with an artist containment
check, then fuzzy with a strict similarity gate, and every result tagged
``exact`` / ``fuzzy`` / ``none``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from difflib import SequenceMatcher
from typing import Iterable, Optional

from app.services.assistant.contracts import (MatchMode, ResolvedTrack, Subject,
                                              TrackRef)
from app.services.text_normalize import fold, similar, title_key, to_latin

logger = logging.getLogger(__name__)

# Below this, a fuzzy title match is not a match. Tuned against real "best of"
# listicles in the previous resolver.
FUZZY_TITLE_MIN = 0.75
FUZZY_ARTIST_MIN = 0.75
# An artist candidate below this never reaches the SHORTLIST. It is not, and must
# not become, a threshold for deciding identity on its own — measured on a real
# library, "Muse" scores 0.750 against "Fuse" (wrong) while «канье» scores 0.571
# against "Kanye West" (right), so no single number separates the two. Whoever
# consumes the shortlist decides; this only bounds its length.
ARTIST_SCORE_MIN = 0.5

_FEAT_MARKERS = ("feat", "ft", "featuring", "with", "vs", "x", "и")

# How long a built catalog is served before it is rebuilt. Indexing calls
# ``invalidate()`` explicitly, so this is only the safety net for anything that
# writes to the library without saying so.
CACHE_TTL_SEC = 300.0

_cache: dict = {}
_cache_lock = threading.Lock()


def get_catalog(collection_name: str) -> "LibraryCatalog":
    """The catalog for one account, built at most once per ``CACHE_TTL_SEC``.

    Building it scans the whole library and precomputes a token index, which is
    real work at 5–6k tracks — and the fuzzy leg degrades to quadratic without
    that index, so per-request rebuilds are not an option.
    """
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(collection_name)
        if entry is not None and now - entry[0] < CACHE_TTL_SEC:
            return entry[1]

    catalog = LibraryCatalog(collection_name)
    with _cache_lock:
        _cache[collection_name] = (time.monotonic(), catalog)
    return catalog


def invalidate(collection_name: Optional[str] = None) -> None:
    """Drop the cached catalog. Called when indexing changes the library.

    With no argument, drops every account's — which is what tests want.
    """
    with _cache_lock:
        if collection_name is None:
            _cache.clear()
        else:
            _cache.pop(collection_name, None)


class LibraryCatalog:
    """Everything the assistant needs to know about what the user owns."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.songs: list = []
        self.by_title: dict = {}
        # token -> songs containing it. Blocking: the fuzzy leg used to score
        # every claim against every track in the library, recomputing the folded
        # title key each time. On 5000 tracks and 900 claims that is ~20 seconds
        # of SequenceMatcher; with the index it is a few milliseconds, because
        # two titles with no token in common cannot be a fuzzy match anyway.
        self.by_token: dict = {}
        self.artists: list = []
        self.artist_slugs: dict = {}     # folded name -> slug
        self.artist_names: dict = {}     # slug -> display name
        # Rows of the `songs` table: the only place that pairs a song slug with
        # its artist slug, which is what lets a named song answer "whose facts?"
        # without any name matching at all.
        self.song_rows: list = []
        self.degraded = False
        self._load()

    # ── loading ───────────────────────────────────────────────────────────

    @staticmethod
    def _rows(sql: str, params: tuple) -> list:
        from app.resources.metadata_db import MetadataDB

        MetadataDB.init()
        return MetadataDB.get().execute(sql, params).fetchall()

    def _load(self) -> None:
        rows: list = []
        try:
            rows = self._rows(
                "SELECT track_id, title, artist, artists, artist_slugs, "
                "       primary_artist_slug, album, year, cover_art_path, "
                "       duration, file_path "
                "FROM track_metadata WHERE collection_name = ?",
                (self.collection_name,))
        except Exception:  # noqa: BLE001
            logger.warning("[catalog] track_metadata unreadable", exc_info=True)

        for r in rows:
            song = {
                "track_id": r[0], "title": r[1] or "", "artist": r[2] or "",
                "artists": _json_list(r[3]), "artist_slugs": _json_list(r[4]),
                "artist_slug": r[5] or "", "album": r[6] or "", "year": r[7],
                "cover_art_path": r[8],
                # Carried so a matched claim can be serialised as a playable
                # track without a second trip to the database per row.
                "duration_sec": r[9] or 0.0, "file_path": r[10] or "",
            }
            self.songs.append(song)
            self.by_title.setdefault(title_key(song["title"]), []).append(song)

        if not rows:
            self.degraded = True
            logger.warning(
                "[catalog] no track_metadata for %r — name resolution will work "
                "off the songs table, but track ids, years and playlist assembly "
                "will not.", self.collection_name)

        self._build_indexes()
        self._load_artists()
        self._load_song_rows()
        logger.info("[catalog] %s: %d tracks, %d artists, %d song rows%s",
                    self.collection_name, len(self.songs), len(self.artists),
                    len(self.song_rows), " (degraded)" if self.degraded else "")

    def _build_indexes(self) -> None:
        """Precompute what the matcher would otherwise recompute per comparison.

        ``_key`` and ``_artist_variants`` are derived from the same functions the
        matcher uses; doing it once at load turns the inner loop from "normalise
        two strings, then score them" into "score them".
        """
        self.by_token = {}
        for song in self.songs:
            key = title_key(song.get("title") or "")
            song["_key"] = key
            song["_key_variants"] = _variants(key)
            song["_artist_variants"] = _variants(fold(song.get("artist") or ""))
            for token in _index_tokens(key):
                self.by_token.setdefault(token, []).append(song)

    def _load_song_rows(self) -> None:
        try:
            rows = self._rows(
                "SELECT s.slug, s.title, s.artist_slug FROM songs s "
                "JOIN fact_visibility v ON v.kind = 'song' AND v.slug = s.slug "
                " AND v.collection_name = ?", (self.collection_name,))
        except Exception:  # noqa: BLE001
            logger.warning("[catalog] songs table unreadable", exc_info=True)
            return
        self.song_rows = [{"slug": r[0], "title": r[1] or "",
                           "artist_slug": r[2] or "",
                           "key": title_key(r[1] or "")} for r in rows]

    def _load_artists(self) -> None:
        slugs: dict = {}      # folded name -> slug
        display: dict = {}    # folded name -> name as written
        names_by_slug: dict = {}
        try:
            rows = self._rows(
                "SELECT a.slug, a.name FROM artists a "
                "JOIN fact_visibility v ON v.kind = 'artist' AND v.slug = a.slug "
                " AND v.collection_name = ?", (self.collection_name,))
        except Exception:  # noqa: BLE001
            logger.warning("[catalog] artists table unreadable", exc_info=True)
            rows = []
        for slug, name in rows:
            if not name:
                continue
            slugs[fold(name)] = slug
            display[fold(name)] = name
            names_by_slug.setdefault(slug, name)

        # Tags carry spellings the artists table does not ("Amerie feat. Nas"
        # where the table only has the collab as one row, or nothing at all), so
        # both sources feed the shortlist.
        for song in self.songs:
            artist = song.get("artist") or ""
            if not artist:
                continue
            key = fold(artist)
            slugs.setdefault(key, song.get("artist_slug") or "")
            display.setdefault(key, artist)

        self.artist_slugs = slugs
        self.artist_names = names_by_slug
        self.artists = sorted(display.values(), key=str.lower)

    # ── resolution ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.songs)

    def track(self, track_id: str) -> Optional[dict]:
        """One library track by id, or None."""
        if not track_id:
            return None
        return next((s for s in self.songs if s.get("track_id") == track_id), None)

    def resolve_artist(self, query: str, limit: int = 5) -> list:
        """A SHORTLIST of who the user might mean, best first.

        Transliteration-aware, so «канье» reaches "Kanye West". This is a
        suggestion generator, not an identity decision — see
        :meth:`resolve_subject` for the difference and why it matters.
        """
        if not (query or "").strip() or not self.artists:
            return []
        scored = [(similar(query, name), name) for name in self.artists]
        scored.sort(key=lambda pair: -pair[0])
        return [{"artist": name, "score": round(score, 2),
                 "slug": self.artist_slugs.get(fold(name), "")}
                for score, name in scored[:limit] if score >= ARTIST_SCORE_MIN]

    def _artist_identity(self, name: str) -> tuple:
        """``(slug, how)`` for an artist we are sure about, else ``(None, ...)``.

        Three tiers, all of them structural:

        1. **exact-name** — the folded name matches a library name outright.
        2. **transliteration** — the two are EQUAL once Cyrillic is mapped across
           ("Эминем" and "Eminem"). Not a fuzzy tier: it is the same string under
           the same normalisation the exact tier uses. Uniqueness is required,
           because two library artists collapsing onto one query is an ambiguity,
           not a match.
        3. **participant** — the query is the leading participant of a collab
           tag: "Amerie" of "Amerie feat. Nas", "RAYE" of "RAYE, Regard". That is
           parsing the tag, not measuring similarity.

        There is deliberately NO similarity tier. One is what made "Amerie"
        resolve to "Fergie" (they score 0.667) and load a stranger's facts — an
        identity decided silently by a spelling distance, somewhere being wrong
        is invisible. And no threshold could have saved it: "Muse"/"Fuse" scores
        0.750 while «канье»/"Kanye West", which is right, scores 0.571. Ambiguity
        goes to :meth:`resolve_subject`, which hands it to something able to
        actually judge.
        """
        folded = fold(name or "")
        if not folded:
            return None, "none"

        slug = self.artist_slugs.get(folded)
        if slug:
            return slug, "exact-name"

        equal = [n for n in self.artists if similar(name, n) >= 1.0]
        if len(equal) == 1:
            slug = self.artist_slugs.get(fold(equal[0]))
            if slug:
                logger.info("[catalog] %r is %r across alphabets", name, equal[0])
                return slug, "transliteration"

        # Shortest first: "Amerie feat. Nas" beats "Amerie feat. Nas & Eve" as
        # the reading of a bare "Amerie".
        participants = sorted(
            (n for n in self.artist_slugs if self._is_leading_participant(folded, n)),
            key=len)
        for candidate in participants:
            found = self.artist_slugs.get(candidate)
            if found:
                logger.info("[catalog] %r resolved to %r as the leading participant",
                            name, candidate)
                return found, "participant"
        return None, "none"

    @staticmethod
    def _is_leading_participant(folded_query: str, folded_candidate: str) -> bool:
        """True when the candidate tag STARTS with the query as whole tokens.

        The separator has to be a real one — a feature marker or nothing — so that
        "Kanye" does not claim "Kanye Wester" while "Kanye West" does claim
        "Kanye West, Jay-Z".
        """
        if not folded_candidate.startswith(folded_query + " "):
            return False
        rest = folded_candidate[len(folded_query):].split()
        return bool(rest) and rest[0] in _FEAT_MARKERS

    def resolve_subject(self, *, song: Optional[str] = None,
                        artist: Optional[str] = None) -> Subject:
        """Who a question is about — structure first, a shortlist as last resort.

        The order is what matters. A named song that the library has answers the
        artist question outright: its row in ``songs`` carries the artist slug, so
        no name is compared to any other name and no wrong artist is reachable.
        Only when there is no such row does the artist's name get looked at, and
        only when THAT is ambiguous does anyone have to judge.
        """
        artist_slug, artist_how = (self._artist_identity(artist) if artist
                                   else (None, "none"))

        if song and self.song_rows:
            key = title_key(song)
            matches = [r for r in self.song_rows if r["key"] == key] if key else []
            if artist_slug:
                narrowed = [r for r in matches if r["artist_slug"] == artist_slug]
                matches = narrowed or matches
            distinct = {r["artist_slug"] for r in matches}
            if len(matches) == 1 or (matches and len(distinct) == 1):
                row = matches[0]
                return Subject(song_slug=row["slug"],
                               artist_slug=row["artist_slug"] or artist_slug,
                               artist_name=self.artist_names.get(row["artist_slug"]),
                               song_title=row["title"], how="song-row")
            if len(distinct) > 1:
                # The same title by several artists in this library. The name the
                # user gave did not separate them, so someone must choose.
                return Subject(
                    how="shortlist",
                    candidates=[{"artist": self.artist_names.get(s, s), "slug": s,
                                 "score": 1.0,
                                 "song_slug": next(r["slug"] for r in matches
                                                   if r["artist_slug"] == s)}
                                for s in sorted(distinct)])

        if artist_slug:
            return Subject(artist_slug=artist_slug,
                           artist_name=self.artist_names.get(artist_slug) or artist,
                           how=artist_how)

        if artist:
            candidates = self.resolve_artist(artist)
            if candidates:
                return Subject(how="shortlist", candidates=candidates)

        return Subject(how="none")

    # ── pinned subjects (the UI already knows who it means) ───────────────

    def subject_for_track(self, track_id: str) -> Optional[Subject]:
        """The subject of a track the CALLER identified by id.

        No name matching happens and none may: the id came from a card the user
        tapped, and re-deriving the artist from a string would put the "Amerie
        became Fergie" failure back into a path that had already been settled.

        The song slug is built with ``get_song_facts_key`` — the one of the three
        ``_slugify`` variants in this codebase that actually wrote ``songs`` and
        ``song_facts``. The other two produce different strings for the same
        track and would silently find nothing.
        """
        from app.services.song_facts_service import get_song_facts_key

        row = self.track(track_id)
        if row is None:
            logger.info("[catalog] pinned track %r is not in %s", track_id,
                        self.collection_name)
            return None
        artist_slug = row.get("artist_slug") or ""
        return Subject(
            song_slug=get_song_facts_key(row.get("artist") or "",
                                         row.get("title") or ""),
            artist_slug=artist_slug or None,
            artist_name=self.artist_names.get(artist_slug) or row.get("artist"),
            song_title=row.get("title"), track_id=track_id, how="pinned")

    def subject_for_artist(self, artist_slug: str) -> Optional[Subject]:
        """The subject of an artist the CALLER identified by slug."""
        if not artist_slug:
            return None
        name = self.artist_names.get(artist_slug)
        if name is None:
            logger.info("[catalog] pinned artist %r is not visible to %s",
                        artist_slug, self.collection_name)
            return None
        return Subject(artist_slug=artist_slug, artist_name=name, how="pinned")

    # ── claim matching ────────────────────────────────────────────────────

    def resolve_tracks(self, refs: Iterable,
                       *, max_fuzzy: int = 150) -> tuple:
        """Match page claims against the library.

        Returns ``(resolved, missing)``. ``max_fuzzy`` bounds the expensive leg: a
        200-row soundtrack against a 5000-track library is 10^6 comparisons if
        every miss is retried fuzzily.
        """
        from app.services.assistant.web_urls import canonical_url

        resolved: list = []
        missing: list = []
        # (track, page) — one page cannot vote for the same track twice, but a
        # second page can. Deduplicating on the track alone threw away every
        # corroboration before it could be counted: the weights never summed, and
        # a track found by Apple, Wikipedia and a listicle scored the same as one
        # found once.
        seen: set = set()
        fuzzy_left = max_fuzzy

        for ref in refs:
            picked, mode = self._match_one(ref, allow_fuzzy=fuzzy_left > 0)
            if mode == "fuzzy":
                fuzzy_left -= 1
            if not picked:
                missing.append(ref)
                continue
            track_id = picked.get("track_id") or ""
            key = (track_id, canonical_url(ref.source_url))
            if track_id and key in seen:
                continue
            if track_id:
                seen.add(key)
            resolved.append(ResolvedTrack(
                track_id=track_id, title=picked.get("title") or ref.title,
                artist=picked.get("artist") or (ref.artist or ""),
                # The library's own year wins: a listicle prints the year of the
                # compilation it is plugging as often as the release year.
                year=picked.get("year") or ref.year,
                match=mode, sources=[ref.source],
                section=ref.section, page_title=ref.page_title,
                context=ref.context,
                cover_art_path=picked.get("cover_art_path"),
                album=picked.get("album") or None,
                duration_sec=float(picked.get("duration_sec") or 0.0),
                file_path=picked.get("file_path") or ""))
        return resolved, missing

    def _match_one(self, ref: TrackRef, *, allow_fuzzy: bool) -> tuple:
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
        if not key:
            return None, "none"

        # Only titles sharing a token can be a fuzzy match, and the index knows
        # which those are. Scanning the whole library instead was the three
        # minutes this used to take on a 900-row discography page.
        seen: set = set()
        pool: list = []
        for token in _index_tokens(key):
            for candidate in self.by_token.get(token, ()):
                marker = id(candidate)
                if marker not in seen:
                    seen.add(marker)
                    pool.append(candidate)
        if not pool:
            return None, "none"

        # Folded once per claim, not once per comparison.
        title_variants = _variants(key)
        artist_variants = _variants(fold(ref.artist or "")) if ref.artist else ()

        best = None
        for candidate in pool:
            ck = candidate.get("_key") or ""
            if not ck:
                continue
            title_score = _ratio(title_variants, candidate["_key_variants"],
                                 FUZZY_TITLE_MIN)
            if not ((key in ck) or title_score >= FUZZY_TITLE_MIN):
                continue
            artist_ok = (
                not ref.artist
                or _artist_contains(ref.artist, candidate.get("artist", ""))
                or _ratio(artist_variants, candidate["_artist_variants"],
                          FUZZY_ARTIST_MIN) >= FUZZY_ARTIST_MIN
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


def _artist_contains(query: str, value: str) -> bool:
    """Loose containment both ways: "Kanye" matches "Kanye West", and a track
    tagged "Kanye West, Jay-Z" matches a query for "Kanye West"."""
    a, b = fold(query or ""), fold(value or "")
    return bool(a) and bool(b) and (a in b or b in a)


def _variants(folded: str) -> tuple:
    """The folded string and its transliteration, deduplicated.

    Precomputed per library track so the matcher never re-folds the same text
    once per comparison — which is what ``similar()`` does internally, and what
    made the fuzzy leg quadratic in practice.
    """
    return tuple({folded, to_latin(folded)} - {""})


def _ratio(a_variants: tuple, b_variants: tuple, floor: float) -> float:
    """Best similarity across the variant pairs, with a free early exit.

    ``SequenceMatcher`` can never exceed ``2*min(len)/sum(len)``, so a pair whose
    lengths differ enough is skipped without building a matcher at all.
    """
    best = 0.0
    for a in a_variants:
        for b in b_variants:
            total = len(a) + len(b)
            if not total:
                continue
            if 2.0 * min(len(a), len(b)) / total < floor:
                continue
            best = max(best, SequenceMatcher(None, a, b).ratio())
            if best >= 1.0:
                return best
    return best


def filter_by_era(items, era: Optional[tuple], *, year=lambda item: item.year):
    """Drop what the era rules out. Anything with no year survives.

    Both halves matter. Filtering is the whole point of extracting an era; but a
    missing year is not evidence of the wrong decade, and dropping year-less
    tracks would quietly delete everything the library has no metadata for.
    """
    if not era:
        return list(items)
    low, high = era
    return [item for item in items
            if not year(item) or low <= int(year(item)) <= high]


def _index_tokens(folded_key: str) -> set:
    """Tokens a title is indexed and looked up under.

    Both the folded form and its transliteration, so a Cyrillic title still lands
    in the same bucket as its Latin spelling — blocking must not hide the
    cross-script matches the scorer exists to find.
    """
    tokens = set(folded_key.split())
    tokens |= {to_latin(t) for t in tokens}
    return {t for t in tokens if t}


def _json_list(raw) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []
