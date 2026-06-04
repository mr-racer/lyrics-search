"""Integration tests for /recommend/autoplay-queue and /recommend/sonic-sibling (501 stub)."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client


def _mk_point(track_id, score, artist="A"):
    pt = MagicMock()
    pt.id = track_id
    pt.score = score
    pt.payload = {
        "track_id": track_id,
        "title": f"Title-{track_id}",
        "artist": artist,
        "duration_sec": 200.0,
        "file_path": f"/tmp/{track_id}.mp3",
    }
    pt.vector = {"clap": [0.1] * 512}
    return pt


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()

    # Inject a stub Qdrant client into app state.
    qdrant = MagicMock()
    qdrant.retrieve.return_value = [_mk_point("seed", 1.0)]
    qdrant.search.return_value = [
        _mk_point("t1", 0.9, artist="A"),
        _mk_point("t2", 0.85, artist="B"),
        _mk_point("t3", 0.8, artist="C"),
    ]
    db_stub = MagicMock()
    db_stub.qdrant = qdrant
    app.state.db_client = db_stub

    c = TestClient(app)
    authenticate_test_client(c, app)
    yield c
    MetadataDB._reset_for_tests()


def test_autoplay_queue_returns_filtered_tracks(client):
    resp = client.get(
        "/api/v1/recommend/autoplay-queue",
        params={"collection": "music", "seed_track_id": "seed", "limit": 10},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["seed_track_id"] == "seed"
    ids = [t["track_id"] for t in body["tracks"]]
    assert "seed" not in ids
    assert ids == ["t1", "t2", "t3"]
    assert body["diagnostics"]["returned"] == 3


def test_autoplay_queue_excludes_via_param(client):
    resp = client.get(
        "/api/v1/recommend/autoplay-queue",
        params={"collection": "music", "seed_track_id": "seed",
                "exclude_ids": "t1,t2", "limit": 10},
    )
    assert resp.status_code == 200
    body = resp.json()
    ids = [t["track_id"] for t in body["tracks"]]
    assert ids == ["t3"]
    # seed + t1 + t2 all dropped
    assert body["diagnostics"]["dropped_excluded"] >= 2


def test_autoplay_queue_db_unavailable_returns_503(client):
    app.state.db_client = None
    resp = client.get(
        "/api/v1/recommend/autoplay-queue",
        params={"collection": "music", "seed_track_id": "seed"},
    )
    assert resp.status_code == 503


def test_sonic_sibling_returns_501(client):
    resp = client.get(
        "/api/v1/recommend/sonic-sibling",
        params={"track_id": "t1", "collection": "music"},
    )
    assert resp.status_code == 501
    detail = resp.json()["detail"].lower()
    assert "deferred" in detail or "not implemented" in detail
