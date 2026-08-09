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
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Optional

from lab.agent.models import MatchMode, ResolvedTrack, TrackRef
from lab.agent.urls import canonical_url
# _translit is private to websearch_lab but it is the same Cyrillic->Latin map
# `similar` uses; the token index has to agree with the scorer or blocking
# would hide exactly the cross-script matches the scorer exists to find.
from lab.websearch_lab import _translit, fold, similar, title_key

logger = logging.getLogger(__name__)

# Below this, a fuzzy title match is not a match. Lifted from the production
# resolver, where it was tuned against real "best of" listicles.
FUZZY_TITLE_MIN = 0.75
FUZZY_ARTIST_MIN = 0.75
# An artist candidate below this never reaches the SHORTLIST. It is not, and
# must not become, a threshold for deciding identity on its own — measured on
# this library, "Muse" scores 0.750 against "Fuse" (wrong) while "канье" scores
# 0.571 against "Kanye West" (right), so no single number separates the two.
# Whoever consumes the shortlist decides; this only bounds its length.
ARTIST_SCORE_MIN = 0.5

_FEAT_MARKERS = ("feat", "ft", "featuring", "with", "vs", "x", "и")


@dataclass(slots=True)
class Subject:
    """Who and what a question is about, once the library has had its say.

    ``how`` records which tier answered, because the tiers differ in kind and
    not just in confidence:

    ``song-row``     the named song is in the library and its row carries the
                     artist slug — no name matching happened at all
    ``exact-name``   the artist name matched a library name exactly (folded)
    ``participant``  the query is the leading participant of a collab tag
                     ("Amerie" of "Amerie feat. Nas") — structure, not similarity
    ``shortlist``    genuinely ambiguous; ``candidates`` is for a judgement call
    ``none``         nothing plausible. No facts, and that is the right answer.
    """

    song_slug: Optional[str] = None
    artist_slug: Optional[str] = None
    artist_name: Optional[str] = None
    how: str = "none"
    candidates: list[dict] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return bool(self.song_slug or self.artist_slug)


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
        # token -> songs containing it. Blocking: the fuzzy leg used to score
        # every claim against every track in the library, recomputing the
        # folded title key each time. On 5000 tracks and 900 claims that is
        # ~20 seconds of SequenceMatcher; with the index it is a few
        # milliseconds, because two titles with no token in common cannot be
        # a fuzzy match anyway.
        self.by_token: dict[str, list[dict]] = {}
        self.artists: list[str] = []
        self.artist_slugs: dict[str, str] = {}     # folded name -> slug
        self.artist_names: dict[str, str] = {}     # slug -> display name
        # Rows of the `songs` table: the only place that pairs a song slug with
        # its artist slug, which is what lets a named song answer "whose facts?"
        # without any name matching at all.
        self.song_rows: list[dict] = []
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

        self._build_indexes()
        self._load_artists()
        self._load_song_rows()
        logger.info("[catalog] %s: %d tracks, %d artists, %d song rows%s",
                    self.collection_name, len(self.songs), len(self.artists),
                    len(self.song_rows), " (degraded)" if self.degraded else "")

    def _build_indexes(self) -> None:
        """Precompute what the matcher would otherwise recompute per comparison.

        ``_key`` and ``_artist_fold`` are derived from the same functions the
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
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT slug, title, artist_slug FROM songs "
                    "WHERE collection_name = ?", (self.collection_name,)).fetchall()
        except Exception:
            logger.warning("[catalog] songs table unreadable", exc_info=True)
            return
        self.song_rows = [{"slug": r["slug"], "title": r["title"] or "",
                           "artist_slug": r["artist_slug"] or "",
                           "key": title_key(r["title"] or "")} for r in rows]

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
        slugs: dict[str, str] = {}      # folded name -> slug
        display: dict[str, str] = {}    # folded name -> name as written
        names_by_slug: dict[str, str] = {}
        try:
            with self._connect() as conn:
                for r in conn.execute(
                        "SELECT slug, name FROM artists WHERE collection_name = ?",
                        (self.collection_name,)).fetchall():
                    if not r["name"]:
                        continue
                    slugs[fold(r["name"])] = r["slug"]
                    display[fold(r["name"])] = r["name"]
                    names_by_slug.setdefault(r["slug"], r["name"])
        except Exception:
            logger.warning("[catalog] artists table unreadable", exc_info=True)

        # Tags carry spellings the artists table does not ("Amerie feat. Nas"
        # where the table only has the collab as one row, or nothing at all),
        # so both sources feed the shortlist.
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

    def resolve_artist(self, query: str, limit: int = 5) -> list[dict]:
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

    def artist_slug_for(self, name: str) -> Optional[str]:
        """The slug of an artist we are SURE about, or None."""
        return self._artist_identity(name)[0]

    def _artist_identity(self, name: str) -> tuple[Optional[str], str]:
        """``(slug, how)`` for an artist we are sure about, else ``(None, ...)``.

        Three tiers, all of them structural:

        1. **exact-name** — the folded name matches a library name outright.
        2. **transliteration** — the two are EQUAL once Cyrillic is mapped
           across ("Эминем" and "Eminem"). Not a fuzzy tier: it is the same
           string under the same normalisation the exact tier uses. Uniqueness
           is required, because two library artists collapsing onto one query
           is an ambiguity, not a match.
        3. **participant** — the query is the leading participant of a collab
           tag: "Amerie" of "Amerie feat. Nas", "RAYE" of "RAYE, Regard". That
           is parsing the tag, not measuring similarity.

        There is deliberately NO similarity tier. One is what made "Amerie"
        resolve to "Fergie" (they score 0.667) and load a stranger's facts —
        an identity decided silently by a spelling distance, somewhere being
        wrong is invisible. And no threshold could have saved it: on this
        library "Muse"/"Fuse" scores 0.750 while «канье»/"Kanye West", which is
        right, scores 0.571. Ambiguity goes to :meth:`resolve_subject`, which
        hands it to something able to actually judge.
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

        The separator has to be a real one — a feature marker or nothing — so
        that "Kanye" does not claim "Kanye Wester" while "Kanye West" does
        claim "Kanye West, Jay-Z".
        """
        if not folded_candidate.startswith(folded_query + " "):
            return False
        rest = folded_candidate[len(folded_query):].split()
        return bool(rest) and rest[0] in _FEAT_MARKERS

    def song_slug_for(self, title: str, artist: Optional[str] = None) -> Optional[str]:
        """The ``song_facts.song_slug`` of a track, looked up rather than derived.

        Deriving it would mean re-implementing ``song_facts_service._slugify``
        AND ``artist_split.primary_artist``, and CLAUDE.md is explicit that the
        three ``_slugify`` variants in the app are not interchangeable. The
        ``songs`` table already stores the answer.
        """
        return self.resolve_subject(song=title, artist=artist).song_slug

    def resolve_subject(self, *, song: Optional[str] = None,
                        artist: Optional[str] = None) -> Subject:
        """Who a question is about — structure first, a shortlist as last resort.

        The order is what matters. A named song that the library has answers
        the artist question outright: its row in ``songs`` carries the artist
        slug, so no name is compared to any other name and no wrong artist is
        reachable. Only when there is no such row does the artist's name get
        looked at, and only when THAT is ambiguous does anyone have to judge.
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
                               how="song-row")
            if len(distinct) > 1:
                # The same title by several artists in this library. The name
                # the user gave did not separate them, so someone must choose.
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

    def resolve_tracks(self, refs: Iterable[TrackRef],
                       *, max_fuzzy: int = 150) -> tuple[list[ResolvedTrack], list[TrackRef]]:
        """Match page claims against the library.

        Returns ``(resolved, missing)``. ``max_fuzzy`` bounds the expensive leg:
        a 200-row soundtrack against a 5000-track library is 10^6 comparisons
        if every miss is retried fuzzily.
        """
        resolved: list[ResolvedTrack] = []
        missing: list[TrackRef] = []
        # (track, page) — one page cannot vote for the same track twice, but a
        # second page can. Deduplicating on the track alone (which this did)
        # threw away every corroboration before it could be counted: the
        # weights in `merge_claims` never summed, and a track found by Apple,
        # Wikipedia and a listicle scored the same as one found once.
        seen: set[tuple[str, str]] = set()
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
                # The library's own year wins: a listicle prints the year of
                # the compilation it is plugging as often as the release year.
                year=picked.get("year") or ref.year,
                match=mode, sources=[ref.source],
                section=ref.section, page_title=ref.page_title,
                context=ref.context))
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
        if not key:
            return None, "none"

        # Only titles sharing a token can be a fuzzy match, and the index knows
        # which those are. Scanning the whole library instead was the three
        # minutes this used to take on a 900-row discography page.
        seen: set[int] = set()
        pool: list[dict] = []
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

        best: Optional[tuple[float, dict]] = None
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


def _variants(folded: str) -> tuple[str, ...]:
    """The folded string and its transliteration, deduplicated.

    Precomputed per library track so the matcher never re-folds the same text
    once per comparison — which is what ``similar()`` does internally, and what
    made the fuzzy leg quadratic in practice.
    """
    return tuple({folded, _translit(folded)} - {""})


def _ratio(a_variants: tuple[str, ...], b_variants: tuple[str, ...],
           floor: float) -> float:
    """Best similarity across the variant pairs, with a free early exit.

    ``SequenceMatcher`` can never exceed ``2*min(len)/sum(len)``, so a pair
    whose lengths differ enough is skipped without building a matcher at all.
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


def filter_by_era(items, era: Optional[tuple[int, int]], *,
                  year=lambda item: item.year):
    """Drop what the era rules out. Anything with no year survives.

    Both halves matter. Filtering is the whole point of extracting an era; but
    a missing year is not evidence of the wrong decade, and dropping year-less
    tracks would quietly delete everything the library has no metadata for.
    """
    if not era:
        return list(items)
    low, high = era
    return [item for item in items
            if not year(item) or low <= int(year(item)) <= high]


def _index_tokens(folded_key: str) -> set[str]:
    """Tokens a title is indexed and looked up under.

    Both the folded form and its transliteration, so a Cyrillic title still
    lands in the same bucket as its Latin spelling — blocking must not hide the
    cross-script matches the scorer exists to find.
    """
    tokens = set(folded_key.split())
    tokens |= {_translit(t) for t in tokens}
    return {t for t in tokens if t}


def _json_list(raw) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []
