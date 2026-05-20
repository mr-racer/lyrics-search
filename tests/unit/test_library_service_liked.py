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
    qdrant = _stub_qdrant_retrieve({
        "t1": {"title": "T1", "artist": "A", "album": "Al", "duration": 200},
        "t2": {"title": "T2", "artist": "B", "album": "Bl", "duration": 220},
    })
    res = LibraryService.get_liked_songs(qdrant_client=qdrant, collection_name="test_col")
    assert {t.track_id for t in res.tracks} == {"t1", "t2"}
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
