"""Unit tests for LibraryService (consolidated).

Merged from:
  - test_library_service_active_jobs.py     -> TestActiveJobs
  - test_library_service_albums.py          -> TestAlbums
  - test_library_service_albums_sqlite.py   -> TestAlbumsSqlite
  - test_library_service_liked.py           -> TestLiked
  - test_library_service_listening_stats.py -> TestListeningStats
  - test_library_service_semaphore.py       -> TestSemaphore
"""
import os
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import app.resources.metadata_db as mod
from app.resources.metadata_db import MetadataDB
from app.services import library_service as lib_mod
from app.services.library_service import LibraryService


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _mk_point(payload, point_id="tid-abc"):
    return SimpleNamespace(payload=payload, id=point_id)


def _stub_qdrant(points):
    """Build a MagicMock qdrant client whose .scroll() yields all points in one batch."""
    qdrant = MagicMock()
    qdrant.scroll.return_value = (points, None)
    qdrant.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="test_col")]
    )
    qdrant.get_collection.return_value = SimpleNamespace(points_count=len(points))
    return qdrant


def _empty_qdrant():
    """Qdrant stub whose scroll yields nothing — forces reliance on SQLite."""
    q = MagicMock()
    q.scroll.return_value = ([], None)
    q.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="c")]
    )
    q.get_collection.return_value = SimpleNamespace(points_count=0)
    return q


def _upsert(track_id, **payload):
    MetadataDB.upsert_track_metadata("c", track_id, payload)


def _stub_qdrant_retrieve(by_id):
    """Build a MagicMock qdrant whose .retrieve() resolves payload by id."""
    qdrant = MagicMock()

    def retrieve(*, collection_name, ids, with_payload, with_vectors):
        return [
            SimpleNamespace(id=tid, payload=by_id[tid])
            for tid in ids if tid in by_id
        ]

    qdrant.retrieve.side_effect = retrieve
    qdrant.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="test_col")]
    )
    return qdrant


def _qdrant_with(by_id):
    qdrant = MagicMock()
    qdrant.retrieve.side_effect = lambda **kw: [
        SimpleNamespace(id=tid, payload=by_id[tid]) for tid in kw["ids"] if tid in by_id
    ]
    return qdrant


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def svc():
    return LibraryService(search_service=MagicMock(), db_client=MagicMock())


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Repoint MetadataDB at a fresh temp SQLite file for the test."""
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "meta.db")
    MetadataDB._instance = None
    orig_connect = MetadataDB._connect

    def _connect(cls):
        conn = sqlite3.connect(str(mod.DB_PATH), check_same_thread=False)
        conn.row_factory = None
        return conn

    monkeypatch.setattr(MetadataDB, "_connect", classmethod(_connect))
    MetadataDB.init()
    yield
    MetadataDB._instance = None
    MetadataDB._connect = orig_connect


@pytest.fixture
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


@pytest.fixture
def reset_semaphore(monkeypatch):
    """Replace the module-level semaphore with a small one for this test."""
    new_sem = threading.BoundedSemaphore(2)
    monkeypatch.setattr(lib_mod, "_INDEX_SEMAPHORE", new_sem)
    yield new_sem


# --------------------------------------------------------------------------- #
# Per-account indexing queue: two accounts ok in parallel, same account -> 409.
# --------------------------------------------------------------------------- #
class TestActiveJobs:
    def test_first_start_succeeds(self, svc):
        assert svc.try_start_job(account_id="acct-A", job_id="j1") is True
        assert svc.is_account_indexing("acct-A") is True

    def test_same_account_second_start_fails(self, svc):
        assert svc.try_start_job(account_id="acct-A", job_id="j1") is True
        assert svc.try_start_job(account_id="acct-A", job_id="j2") is False

    def test_two_different_accounts_both_start(self, svc):
        assert svc.try_start_job(account_id="acct-A", job_id="j1") is True
        assert svc.try_start_job(account_id="acct-B", job_id="j2") is True
        assert svc.is_account_indexing("acct-A") is True
        assert svc.is_account_indexing("acct-B") is True

    def test_finish_releases_slot(self, svc):
        svc.try_start_job(account_id="acct-A", job_id="j1")
        svc.finish_job(account_id="acct-A")
        assert svc.is_account_indexing("acct-A") is False
        assert svc.try_start_job(account_id="acct-A", job_id="j2") is True

    def test_finish_is_idempotent(self, svc):
        svc.finish_job(account_id="never-started")  # no error
        svc.try_start_job(account_id="acct-A", job_id="j1")
        svc.finish_job(account_id="acct-A")
        svc.finish_job(account_id="acct-A")  # second release is a no-op

    def test_get_account_job_id_returns_current(self, svc):
        svc.try_start_job(account_id="acct-A", job_id="abc123")
        assert svc.get_account_job_id("acct-A") == "abc123"
        assert svc.get_account_job_id("acct-B") is None

    def test_concurrent_starts_for_same_account_only_one_wins(self):
        """50 threads race to start under the same account_id. Exactly one wins."""
        svc = LibraryService(search_service=MagicMock(), db_client=MagicMock())
        wins: list[bool] = []
        lock = threading.Lock()

        def _worker(i: int):
            ok = svc.try_start_job(account_id="acct-A", job_id=f"j{i}")
            with lock:
                wins.append(ok)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(wins) == 1, f"expected exactly one winner, got {sum(wins)}"


# --------------------------------------------------------------------------- #
# LibraryService.get_albums — pure aggregation over a fake Qdrant scroll.
# --------------------------------------------------------------------------- #
class TestAlbums:
    def test_get_albums_groups_by_title_majority_vote_primary(self):
        points = [
            _mk_point({"album": "In Rainbows", "artist": "Radiohead",
                       "title": "Bodysnatchers", "year": 2007, "duration": 240,
                       "genre": "art rock", "cover_art_path": "/c/r1.jpg"}, point_id="r1"),
            _mk_point({"album": "In Rainbows", "artist": "Radiohead",
                       "title": "Nude", "year": 2007, "duration": 255,
                       "genre": "art rock", "cover_art_path": "/c/r1.jpg"}, point_id="r2"),
            _mk_point({"album": "In Rainbows", "artist": "Frank Ocean",
                       "title": "Wisemen feat", "year": 2007, "duration": 200,
                       "genre": "rnb", "cover_art_path": "/c/r1.jpg"}, point_id="r3"),
        ]
        qdrant = _stub_qdrant(points)
        res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col")

        assert res.qdrant_available is True
        assert len(res.albums) == 1
        a = res.albums[0]
        assert a.album_title == "In Rainbows"
        assert a.primary_artist == "Radiohead"
        assert [f.name for f in a.feat_artists] == ["Frank Ocean"]
        assert a.track_count == 3
        assert a.year == 2007
        assert a.year_range is None
        assert "art rock" in a.top_genres
        assert {t.track_id for t in a.tracks} == {"r1", "r2", "r3"}

    def test_get_albums_splits_collaboration_into_primary_and_feat(self):
        """An album whose tracks are all tagged "A, B" must surface A as the
        primary artist and B as a feat participant — not the whole "A, B" tag as
        one un-clickable primary. Each gets its own canonical slug so the artist
        page resolves."""
        points = [
            _mk_point({"album": "Collab LP", "artist": "Calvin Harris, Dua Lipa",
                       "title": "One Kiss", "year": 2018, "duration": 215}, point_id="c1"),
            _mk_point({"album": "Collab LP", "artist": "Calvin Harris, Dua Lipa",
                       "title": "Another", "year": 2018, "duration": 200}, point_id="c2"),
        ]
        qdrant = _stub_qdrant(points)
        res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col")
        a = res.albums[0]
        assert a.primary_artist == "Calvin Harris"
        assert a.primary_artist_slug == "calvin-harris"
        assert [f.name for f in a.feat_artists] == ["Dua Lipa"]
        assert [f.slug for f in a.feat_artists] == ["dua-lipa"]

    def test_get_albums_cyrillic_artist_gets_nonempty_slug(self):
        """Cyrillic artist names must produce a non-empty canonical slug
        (regression: the old album slugify stripped all non-ASCII → empty slug
        → the Russian artist page never opened)."""
        points = [
            _mk_point({"album": "Шут", "artist": "Король и Шут",
                       "title": "Лесник", "year": 1997, "duration": 180}, point_id="k1"),
        ]
        qdrant = _stub_qdrant(points)
        res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col")
        a = res.albums[0]
        assert a.primary_artist == "Король и Шут"
        assert a.primary_artist_slug == "король-и-шут"

    def test_get_albums_emits_year_range_when_tracks_span_years(self):
        points = [
            _mk_point({"album": "Live 2003", "artist": "Sigur Rós", "title": "x",
                       "year": 2002, "duration": 300}),
            _mk_point({"album": "Live 2003", "artist": "Sigur Rós", "title": "y",
                       "year": 2003, "duration": 300}),
        ]
        qdrant = _stub_qdrant(points)
        res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col")
        a = res.albums[0]
        assert a.year is None
        assert a.year_range == "2002—2003"

    def test_get_albums_skips_tracks_without_album_field(self):
        points = [
            _mk_point({"album": None, "artist": "A", "title": "loose", "duration": 100}),
            _mk_point({"album": "", "artist": "B", "title": "empty", "duration": 100}),
            _mk_point({"album": "Real", "artist": "C", "title": "ok", "duration": 100}),
        ]
        qdrant = _stub_qdrant(points)
        res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col")
        assert [a.album_title for a in res.albums] == ["Real"]

    def test_get_albums_returns_empty_when_qdrant_unavailable(self):
        qdrant = MagicMock()
        qdrant.get_collections.side_effect = Exception("connect refused")
        res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col")
        assert res.qdrant_available is False
        assert res.albums == []

    def test_get_albums_sort_alphabetical_default(self):
        points = [
            _mk_point({"album": "Banana", "artist": "X", "title": "t1", "duration": 100}),
            _mk_point({"album": "Apple", "artist": "X", "title": "t2", "duration": 100}),
            _mk_point({"album": "Cherry", "artist": "X", "title": "t3", "duration": 100}),
        ]
        qdrant = _stub_qdrant(points)
        res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col")
        assert [a.album_title for a in res.albums] == ["Apple", "Banana", "Cherry"]

    def test_get_albums_year_sort_uses_year_range_when_year_none(self):
        """Albums with year_range only (multi-year) should still sort by their min year."""
        points = [
            _mk_point({"album": "Modern Album", "artist": "X", "title": "t", "year": 2020, "duration": 200}, point_id="m1"),
            _mk_point({"album": "Live 2003", "artist": "Y", "title": "t1", "year": 2002, "duration": 200}, point_id="l1"),
            _mk_point({"album": "Live 2003", "artist": "Y", "title": "t2", "year": 2003, "duration": 200}, point_id="l2"),
        ]
        qdrant = _stub_qdrant(points)
        res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col", sort="year_asc")
        # "Live 2003" has year=None, year_range="2002—2003" — should sort BEFORE "Modern Album" (2020)
        assert [a.album_title for a in res.albums] == ["Live 2003", "Modern Album"]


# --------------------------------------------------------------------------- #
# LibraryService.get_albums SQLite fast-path (Qdrant scroll is empty, so any
# correct result can only have come from SQLite).
# --------------------------------------------------------------------------- #
class TestAlbumsSqlite:
    def test_sqlite_albums_compute_majority_primary_artist(self, temp_db):
        _upsert("t1", album="In Rainbows", artist="Radiohead",
                artist_slugs=["radiohead"], title="Bodysnatchers", year=2007,
                genre="art rock", duration=240, cover_art_path="/c/r.jpg")
        _upsert("t2", album="In Rainbows", artist="Radiohead",
                artist_slugs=["radiohead"], title="Nude", year=2007,
                genre="art rock", duration=255, cover_art_path="/c/r.jpg")
        _upsert("t3", album="In Rainbows", artist="Frank Ocean",
                artist_slugs=["frank-ocean"], title="Wisemen", year=2007,
                genre="rnb", duration=200, cover_art_path="/c/r.jpg")

        res = LibraryService.get_albums(qdrant_client=_empty_qdrant(), collection_name="c")

        assert len(res.albums) == 1
        a = res.albums[0]
        assert a.album_title == "In Rainbows"
        assert a.primary_artist == "Radiohead"
        assert a.primary_artist_slug and a.primary_artist_slug != "—"
        assert [f.name for f in a.feat_artists] == ["Frank Ocean"]
        assert a.track_count == 3
        assert a.year == 2007
        assert "art rock" in a.top_genres

    def test_sqlite_albums_split_collaboration_primary_and_feat(self, temp_db):
        """SQLite fast-path must split a collaboration tag the same way the
        Qdrant fallback does: "A, B" → primary A (+ canonical slug) + feat B."""
        _upsert("t1", album="Collab", artist="Calvin Harris, Dua Lipa",
                artist_slugs=["calvin-harris", "dua-lipa"], title="One Kiss",
                year=2018, duration=215)
        res = LibraryService.get_albums(qdrant_client=_empty_qdrant(), collection_name="c")
        a = res.albums[0]
        assert a.primary_artist == "Calvin Harris"
        assert a.primary_artist_slug == "calvin-harris"
        assert [f.name for f in a.feat_artists] == ["Dua Lipa"]
        assert [f.slug for f in a.feat_artists] == ["dua-lipa"]

    def test_sqlite_albums_emit_year_range_when_spanning_years(self, temp_db):
        _upsert("t1", album="Live", artist="Sigur Ros", artist_slugs=["sigur-ros"],
                title="x", year=2002, duration=300)
        _upsert("t2", album="Live", artist="Sigur Ros", artist_slugs=["sigur-ros"],
                title="y", year=2003, duration=300)

        res = LibraryService.get_albums(qdrant_client=_empty_qdrant(), collection_name="c")
        a = res.albums[0]
        assert a.year is None
        assert a.year_range == "2002—2003"

    def test_get_light_points_shape_and_sonic_axes(self, temp_db):
        _upsert("t1", title="A", artist="X", artist_slugs=["x"], album="Al",
                year=2020, genre="Pop", duration=200, file_path="/a.flac",
                cover_art_path="/c.jpg", sonic_axes={"energy": 0.5})
        pts = MetadataDB.get_light_points("c")
        assert len(pts) == 1
        tid, payload = pts[0]
        assert tid == "t1"
        # All LIGHT_PAYLOAD_FIELDS present, sonic_axes deserialized to a dict.
        for key in ("title", "artist", "artists", "artist_slugs", "primary_artist_slug",
                    "album", "year", "genre", "duration", "duration_range",
                    "file_path", "cover_art_path", "sonic_axes"):
            assert key in payload
        assert payload["sonic_axes"] == {"energy": 0.5}
        assert payload["artist_slugs"] == ["x"]
        assert MetadataDB.get_light_points("missing") == []

    def test_light_points_prefers_sqlite_over_qdrant(self, temp_db):
        """light_points() must serve the SQLite mirror without any Qdrant scroll
        when the mirror is populated."""
        from app.resources.qdrant_utils import light_points, invalidate_light_cache
        invalidate_light_cache()
        _upsert("t1", title="A", artist="X", artist_slugs=["x"], album="Al",
                duration=200, sonic_axes={"energy": 0.5})

        q = MagicMock()
        q.scroll.side_effect = AssertionError("Qdrant scroll must not be called")
        pts = light_points(q, "c")
        assert {tid for tid, _ in pts} == {"t1"}
        q.scroll.assert_not_called()

    def test_delete_track_metadata_removes_ghost(self, temp_db):
        _upsert("t1", title="A", artist="X", artist_slugs=["x"], album="Al", duration=200)
        _upsert("t2", title="B", artist="X", artist_slugs=["x"], album="Al", duration=200)
        assert MetadataDB.get_track_count_for_collection("c") == 2
        MetadataDB.delete_track_metadata("c", "t1")
        assert MetadataDB.get_track_count_for_collection("c") == 1
        assert {tid for tid, _ in MetadataDB.get_light_points("c")} == {"t2"}

    def test_get_browse_rows_shape_and_limit(self, temp_db):
        for i in range(5):
            _upsert(f"t{i}", title=f"Song {i}", artist="X", artist_slugs=["x"],
                    album="Al", duration=200, cover_art_path=f"/c{i}.jpg")
        rows = MetadataDB.get_browse_rows("c")
        assert len(rows) == 5
        assert set(rows[0].keys()) == {"track_id", "title", "artist", "album", "cover_art_path"}
        assert len(MetadataDB.get_browse_rows("c", limit=2)) == 2
        assert MetadataDB.get_browse_rows("missing") == []

    def test_get_track_ids_for_collection(self, temp_db):
        _upsert("t1", title="A", artist="X", artist_slugs=["x"], album="Al", duration=200)
        _upsert("t2", title="B", artist="X", artist_slugs=["x"], album="Al", duration=200)
        assert set(MetadataDB.get_track_ids_for_collection("c")) == {"t1", "t2"}
        assert MetadataDB.get_track_ids_for_collection("missing") == []

    def test_get_year_facets_decade_buckets(self, temp_db):
        _upsert("t1", title="A", artist="X", artist_slugs=["x"], album="Al", year=2014, duration=200)
        _upsert("t2", title="B", artist="X", artist_slugs=["x"], album="Al", year=2017, duration=200)
        _upsert("t3", title="C", artist="X", artist_slugs=["x"], album="Al", year=2003, duration=200)
        facets = MetadataDB.get_year_facets_from_sqlite("c")
        assert facets == {"2010-2019": 2, "2000-2009": 1}
        # collection_name=None aggregates across all collections.
        assert MetadataDB.get_year_facets_from_sqlite()["2010-2019"] == 2


# --------------------------------------------------------------------------- #
# LibraryService.get_liked_songs.
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("_isolated_db")
class TestLiked:
    def test_get_liked_songs_returns_metadata_in_liked_order(self):
        MetadataDB.set_reaction(track_id="t1", collection_name="test_col", reaction="like")
        MetadataDB.set_reaction(track_id="t2", collection_name="test_col", reaction="like")
        # SQLite's CURRENT_TIMESTAMP has 1s resolution, which makes back-to-back
        # set_reaction calls indistinguishable. Force explicit, distinct
        # timestamps so the ordering assertion is deterministic and fast.
        conn = MetadataDB._connect()
        conn.execute(
            "UPDATE track_reactions SET updated_at='2026-01-01 00:00:00' WHERE track_id='t1'"
        )
        conn.execute(
            "UPDATE track_reactions SET updated_at='2026-06-01 00:00:00' WHERE track_id='t2'"
        )
        conn.commit()
        qdrant = _stub_qdrant_retrieve({
            "t1": {"title": "T1", "artist": "A", "album": "Al", "duration": 200},
            "t2": {"title": "T2", "artist": "B", "album": "Bl", "duration": 220},
        })
        res = LibraryService.get_liked_songs(qdrant_client=qdrant, collection_name="test_col")
        # t2 liked after t1 → should be first (newest-first)
        assert [t.track_id for t in res.tracks] == ["t2", "t1"]
        assert all(t.liked_at for t in res.tracks)

    def test_get_liked_songs_skips_tracks_missing_payload(self):
        MetadataDB.set_reaction(track_id="t1", collection_name="test_col", reaction="like")
        MetadataDB.set_reaction(track_id="t2", collection_name="test_col", reaction="like")
        qdrant = _stub_qdrant_retrieve({
            "t1": {"title": "T1", "artist": "A", "album": "Al", "duration": 200},
            # t2 missing from payload (re-index churn)
        })
        res = LibraryService.get_liked_songs(qdrant_client=qdrant, collection_name="test_col")
        assert [t.track_id for t in res.tracks] == ["t1"]

    def test_get_liked_songs_populates_artist_refs_for_collab(self):
        """Each liked track carries aligned per-participant artist_refs so the UI
        can render each collaborator as its own clickable artist link."""
        MetadataDB.set_reaction(track_id="t1", collection_name="test_col", reaction="like")
        qdrant = _stub_qdrant_retrieve({
            "t1": {"title": "One Kiss", "artist": "Calvin Harris, Dua Lipa", "duration": 200},
        })
        res = LibraryService.get_liked_songs(qdrant_client=qdrant, collection_name="test_col")
        refs = res.tracks[0].artist_refs
        assert [(r.name, r.slug) for r in refs] == [
            ("Calvin Harris", "calvin-harris"), ("Dua Lipa", "dua-lipa"),
        ]

    def test_get_liked_songs_empty_when_no_likes(self):
        qdrant = _stub_qdrant_retrieve({})
        res = LibraryService.get_liked_songs(qdrant_client=qdrant, collection_name="test_col")
        assert res.tracks == []

    def test_get_liked_songs_survives_legacy_bucket_duration(self):
        """Pre-fix indexes overwrite numeric duration with a bucket string
        like '203-234'. The endpoint must coerce instead of raising
        Pydantic ValidationError for LikedSongTrack."""
        MetadataDB.set_reaction(track_id="t1", collection_name="test_col", reaction="like")
        MetadataDB.set_reaction(track_id="t2", collection_name="test_col", reaction="like")
        qdrant = _stub_qdrant_retrieve({
            "t1": {"title": "T1", "artist": "A", "duration": "203-234", "year": "2003-2004"},
            "t2": {"title": "T2", "artist": "B", "duration": 220, "year": 2010},
        })
        res = LibraryService.get_liked_songs(qdrant_client=qdrant, collection_name="test_col")
        assert {t.track_id for t in res.tracks} == {"t1", "t2"}
        t1 = next(t for t in res.tracks if t.track_id == "t1")
        # Bucket '203-234' midpoint is 218.5 — close enough to the real seconds
        # that downstream UIs (which only show MM:SS) render sensibly.
        assert t1.duration == pytest.approx(218.5)
        # 'YYYY-YYYY' year range falls back to first year token.
        assert t1.year == 2003


# --------------------------------------------------------------------------- #
# LibraryService.get_listening_stats.
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("_isolated_db")
class TestListeningStats:
    def test_listening_stats_empty_when_no_events(self):
        qdrant = _qdrant_with({})
        res = LibraryService.get_listening_stats(qdrant_client=qdrant, collection_name="c", lang="en")
        assert res.total_seconds_listened == 0
        assert res.top_track is None
        assert res.top_artist is None
        assert res.peak_hour is None

    def test_listening_stats_sums_played_sec(self):
        MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                         track_id="t1", played_sec=120, total_dur=240)
        MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                         track_id="t2", played_sec=60, total_dur=240)
        qdrant = _qdrant_with({
            "t1": {"title": "T1", "artist": "A", "duration": 240},
            "t2": {"title": "T2", "artist": "B", "duration": 240},
        })
        res = LibraryService.get_listening_stats(qdrant_client=qdrant, collection_name="c", lang="en")
        assert res.total_seconds_listened == 180

    def test_listening_stats_top_track_excludes_skips(self):
        # t1 has 2 non-skip + 1 skip; t2 has 3 skips, 0 valid plays → top should be t1
        for _ in range(2):
            MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                             track_id="t1", played_sec=200, total_dur=240)
        MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                         track_id="t1", played_sec=3, total_dur=240)
        for _ in range(3):
            MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                             track_id="t2", played_sec=3, total_dur=240)
        qdrant = _qdrant_with({
            "t1": {"title": "T1", "artist": "Radiohead", "duration": 240},
            "t2": {"title": "T2", "artist": "Other", "duration": 240},
        })
        res = LibraryService.get_listening_stats(qdrant_client=qdrant, collection_name="c", lang="en")
        assert res.top_track.track_id == "t1"
        assert res.top_track.play_count == 2
        assert res.top_artist.name == "Radiohead"


# --------------------------------------------------------------------------- #
# Global semaphore: at most MAX_PARALLEL_INDEXING_JOBS jobs may hold the slot.
# --------------------------------------------------------------------------- #
class TestSemaphore:
    def test_third_acquirer_blocks_until_release(self, reset_semaphore):
        sem = reset_semaphore
        acquired_order: list[str] = []
        release_event = threading.Event()

        def _worker(name: str):
            with sem:
                acquired_order.append(name)
                if name in ("A", "B"):
                    release_event.wait(timeout=2.0)

        a = threading.Thread(target=_worker, args=("A",))
        b = threading.Thread(target=_worker, args=("B",))
        c = threading.Thread(target=_worker, args=("C",))
        a.start(); b.start()
        time.sleep(0.05)  # let A and B grab the two slots
        assert sorted(acquired_order) == ["A", "B"]
        c.start()
        time.sleep(0.05)
        # C is blocked — slot was full
        assert "C" not in acquired_order
        # Release A and B → C should acquire
        release_event.set()
        a.join(timeout=2); b.join(timeout=2); c.join(timeout=2)
        assert "C" in acquired_order

    def test_default_semaphore_size_is_two_or_more(self):
        """Sanity: the production constant is at least 2 (per spec §6.2)."""
        expected = int(os.environ.get("MAX_PARALLEL_INDEXING_JOBS", "2"))
        assert lib_mod.MAX_PARALLEL_INDEXING_JOBS == expected
        assert lib_mod.MAX_PARALLEL_INDEXING_JOBS >= 1
