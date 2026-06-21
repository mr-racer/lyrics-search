"""Unit tests for LibraryService.get_albums SQLite fast-path.

These verify the SQLite-backed album grid derives a real ``primary_artist``
(majority vote), feat artists, genres and year/year_range — NOT the "—"
placeholder. The Qdrant client is stubbed to return an EMPTY scroll, so any
correct result can only have come from SQLite (proving the fast-path is used).
"""
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import app.resources.metadata_db as mod
from app.resources.metadata_db import MetadataDB
from app.services.library_service import LibraryService


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


def test_sqlite_albums_compute_majority_primary_artist(temp_db):
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


def test_sqlite_albums_emit_year_range_when_spanning_years(temp_db):
    _upsert("t1", album="Live", artist="Sigur Ros", artist_slugs=["sigur-ros"],
            title="x", year=2002, duration=300)
    _upsert("t2", album="Live", artist="Sigur Ros", artist_slugs=["sigur-ros"],
            title="y", year=2003, duration=300)

    res = LibraryService.get_albums(qdrant_client=_empty_qdrant(), collection_name="c")
    a = res.albums[0]
    assert a.year is None
    assert a.year_range == "2002—2003"


def test_get_light_points_shape_and_sonic_axes(temp_db):
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


def test_light_points_prefers_sqlite_over_qdrant(temp_db):
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


def test_delete_track_metadata_removes_ghost(temp_db):
    _upsert("t1", title="A", artist="X", artist_slugs=["x"], album="Al", duration=200)
    _upsert("t2", title="B", artist="X", artist_slugs=["x"], album="Al", duration=200)
    assert MetadataDB.get_track_count_for_collection("c") == 2
    MetadataDB.delete_track_metadata("c", "t1")
    assert MetadataDB.get_track_count_for_collection("c") == 1
    assert {tid for tid, _ in MetadataDB.get_light_points("c")} == {"t2"}


def test_get_browse_rows_shape_and_limit(temp_db):
    for i in range(5):
        _upsert(f"t{i}", title=f"Song {i}", artist="X", artist_slugs=["x"],
                album="Al", duration=200, cover_art_path=f"/c{i}.jpg")
    rows = MetadataDB.get_browse_rows("c")
    assert len(rows) == 5
    assert set(rows[0].keys()) == {"track_id", "title", "artist", "album", "cover_art_path"}
    assert len(MetadataDB.get_browse_rows("c", limit=2)) == 2
    assert MetadataDB.get_browse_rows("missing") == []


def test_get_track_ids_for_collection(temp_db):
    _upsert("t1", title="A", artist="X", artist_slugs=["x"], album="Al", duration=200)
    _upsert("t2", title="B", artist="X", artist_slugs=["x"], album="Al", duration=200)
    assert set(MetadataDB.get_track_ids_for_collection("c")) == {"t1", "t2"}
    assert MetadataDB.get_track_ids_for_collection("missing") == []


def test_get_year_facets_decade_buckets(temp_db):
    _upsert("t1", title="A", artist="X", artist_slugs=["x"], album="Al", year=2014, duration=200)
    _upsert("t2", title="B", artist="X", artist_slugs=["x"], album="Al", year=2017, duration=200)
    _upsert("t3", title="C", artist="X", artist_slugs=["x"], album="Al", year=2003, duration=200)
    facets = MetadataDB.get_year_facets_from_sqlite("c")
    assert facets == {"2010-2019": 2, "2000-2009": 1}
    # collection_name=None aggregates across all collections.
    assert MetadataDB.get_year_facets_from_sqlite()["2010-2019"] == 2
