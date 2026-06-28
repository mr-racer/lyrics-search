"""Replay test (design §10): a whole synthetic session driven through the API.

Invariants checked on every chunk and across the session:
  1. no track is served twice within the session window;
  2. liked share per chunk = slider quota ± 1 slot;
  3. a disliked track is never served.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.api.main import app
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client
from .test_api_stream import FakeQdrant, N_PER_CLUSTER

N_CHUNKS = 6
CHUNK_N = 3
LIKED_SHARE = 0.3


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    db = MagicMock(); db.qdrant = FakeQdrant()
    app.state.db_client = db
    c = TestClient(app)
    authenticate_test_client(c, app)
    yield c
    MetadataDB._reset_for_tests()
    app.state.db_client = None


def _owner_collection(client) -> str:
    me = client.get("/api/v1/auth/me").json()
    return f"acct_{me['user']['id'] if 'user' in me else me['id']}"


def test_synthetic_session_invariants(client):
    coll = _owner_collection(client)

    # Long-term history + taste markers.
    for i in range(4):
        client.post("/api/v1/playback/events", json={
            "session_id": "hist", "track_id": f"a{i}",
            "played_sec": 190.0, "total_dur": 200.0, "interacted": True,
        })
    for tid in ("a0", "a1", "a2", "b0", "b1"):
        MetadataDB.set_reaction(tid, coll, "like")
    MetadataDB.set_reaction("b11", coll, "dislike")

    session = "replay"
    served: list[str] = []
    quota = round(CHUNK_N * LIKED_SHARE)

    for chunk_idx in range(N_CHUNKS):
        resp = client.get("/api/v1/recommend/stream/next", params={
            "session_id": session, "n": CHUNK_N, "liked_share": LIKED_SHARE,
        })
        assert resp.status_code == 200
        tracks = resp.json()["tracks"]
        ids = [t["track_id"] for t in tracks]

        # invariant 3: дизлайкнутый трек не появляется никогда
        assert "b11" not in ids, f"disliked track served in chunk {chunk_idx}"

        # invariant 2: доля любимых = слайдер ± 1 слот (пока пул C не иссяк)
        liked_count = sum(1 for t in tracks if t["pool"] == "liked")
        assert abs(liked_count - quota) <= 1, (
            f"chunk {chunk_idx}: liked={liked_count}, quota={quota}"
        )

        # «проигрываем» каждый выданный трек полностью
        for tid in ids:
            r = client.post("/api/v1/playback/events", json={
                "session_id": session, "track_id": tid,
                "played_sec": 195.0, "total_dur": 200.0, "interacted": True,
            })
            assert r.status_code == 200
        served.extend(ids)

    # invariant 1: нет повторов в окне сессии
    assert len(served) == len(set(served)), (
        f"repeats inside session: { {t for t in served if served.count(t) > 1} }"
    )
    # sanity: the session actually streamed a meaningful number of tracks
    assert len(served) >= N_CHUNKS * (CHUNK_N - 1)


def test_session_with_dislikes_mid_stream(client):
    """A dislike set mid-session vanishes from the very next chunk."""
    coll = _owner_collection(client)
    for i in range(3):
        client.post("/api/v1/playback/events", json={
            "session_id": "s", "track_id": f"a{i}",
            "played_sec": 190.0, "total_dur": 200.0, "interacted": True,
        })

    first = client.get("/api/v1/recommend/stream/next",
                       params={"session_id": "s", "n": 3}).json()
    victim = first["tracks"][0]["track_id"]
    MetadataDB.set_reaction(victim, coll, "dislike")

    for _ in range(4):
        body = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "s", "n": 3}).json()
        assert victim not in {t["track_id"] for t in body["tracks"]}
