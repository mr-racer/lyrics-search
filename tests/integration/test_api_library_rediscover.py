"""Integration tests for GET /library/rediscover.

Phase D (D-soft): the collection is derived from the JWT (``acct_<user.id>``),
not from the supplied ``collection_name`` param. Tests that need recency data to
line up with the queried collection pin a fixed user via dependency_overrides so
the derived collection name (``acct_user-A``) is known and can be seeded.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import library as lib_route
from app.resources.metadata_db import MetadataDB

_DERIVED = "acct_user-A"


def _pt(tid, title="T", artist="A"):
    pt = MagicMock()
    pt.id = tid
    pt.payload = {"title": title, "artist": artist, "album": "Al",
                  "year": 1999, "duration": 200.0, "cover_art_path": f"/c/{tid}.jpg",
                  "genre": "rock"}
    return pt


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    qdrant = MagicMock()
    qdrant.scroll.return_value = ([_pt("t1"), _pt("t2"), _pt("t3")], None)
    qdrant.retrieve.side_effect = lambda **kw: [_pt(kw["ids"][0])]
    db = MagicMock(); db.qdrant = qdrant
    app.state.db_client = db
    fixed = SimpleNamespace(id="user-A", email="a@x")
    app.dependency_overrides[lib_route.get_current_user] = lambda: fixed
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()
    MetadataDB._reset_for_tests()
    app.state.db_client = None


def test_rediscover_prefers_never_played(client):
    # Record the play under the DERIVED collection so recency lines up with the
    # collection the endpoint actually queries (D-soft ignores the param).
    MetadataDB.record_playback_event(session_id="s", collection_name=_DERIVED,
                                     track_id="t1", played_sec=150.0, total_dur=200.0)
    # Supplied collection_name is bogus on purpose — it must be ignored.
    resp = client.get("/api/v1/library/rediscover", params={"collection_name": "c"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["never_played"] is True
    assert body["track"]["track_id"] in {"t2", "t3"}


def test_rediscover_qdrant_down_returns_empty(client):
    app.state.db_client = None
    resp = client.get("/api/v1/library/rediscover", params={"collection_name": "c"})
    assert resp.status_code == 200
    assert resp.json()["track"] is None


def test_rediscover_missing_collection_now_optional(client):
    """D-soft relaxed the previously-required collection_name param: omitting it
    no longer 422s — the collection is derived from the JWT."""
    resp = client.get("/api/v1/library/rediscover")
    assert resp.status_code == 200
