from unittest.mock import MagicMock

import pytest

from app.resources.metadata_db import MetadataDB
from app.services import personalization_service as ps


@pytest.fixture(autouse=True)
def _db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _pt(tid):
    pt = MagicMock(); pt.id = tid
    pt.payload = {"title": f"T{tid}", "artist": "A", "cover_art_path": f"/{tid}.jpg"}
    return pt


def test_seed_prefers_liked(monkeypatch):
    monkeypatch.setattr(MetadataDB, "get_liked_track_ids_with_updated_at",
                        classmethod(lambda cls, c: [("liked1", "2026-01-01T00:00:00")]))
    qd = MagicMock()
    qd.retrieve.side_effect = lambda **kw: [_pt(kw["ids"][0])]
    res = ps.pick_for_you_seed(qdrant_client=qd, collection_name="c")
    assert res.seed_track_id == "liked1"
    assert res.source == "liked"
    assert res.track.track_id == "liked1"


def test_seed_falls_back_to_random_scroll(monkeypatch):
    monkeypatch.setattr(MetadataDB, "get_liked_track_ids_with_updated_at",
                        classmethod(lambda cls, c: []))
    qd = MagicMock()
    qd.scroll.return_value = ([_pt("r1"), _pt("r2")], None)
    qd.retrieve.side_effect = lambda **kw: [_pt(kw["ids"][0])]
    res = ps.pick_for_you_seed(qdrant_client=qd, collection_name="c")
    assert res.source == "random"
    assert res.seed_track_id in {"r1", "r2"}
