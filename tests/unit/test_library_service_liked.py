"""Unit tests for LibraryService.get_liked_songs."""
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.resources.metadata_db import MetadataDB
from app.services.library_service import LibraryService


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


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


def test_get_liked_songs_returns_metadata_in_liked_order():
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


def test_get_liked_songs_skips_tracks_missing_payload():
    MetadataDB.set_reaction(track_id="t1", collection_name="test_col", reaction="like")
    MetadataDB.set_reaction(track_id="t2", collection_name="test_col", reaction="like")
    qdrant = _stub_qdrant_retrieve({
        "t1": {"title": "T1", "artist": "A", "album": "Al", "duration": 200},
        # t2 missing from payload (re-index churn)
    })
    res = LibraryService.get_liked_songs(qdrant_client=qdrant, collection_name="test_col")
    assert [t.track_id for t in res.tracks] == ["t1"]


def test_get_liked_songs_empty_when_no_likes():
    qdrant = _stub_qdrant_retrieve({})
    res = LibraryService.get_liked_songs(qdrant_client=qdrant, collection_name="test_col")
    assert res.tracks == []


def test_get_liked_songs_survives_legacy_bucket_duration():
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
