"""Integration tests for POST /recommend/ai-playlist/stream (NDJSON progress)."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()

    db_stub = MagicMock()
    db_stub.qdrant = MagicMock()
    app.state.db_client = db_stub
    app.state.search_service = MagicMock()

    c = TestClient(app)
    authenticate_test_client(c, app)
    yield c
    MetadataDB._reset_for_tests()


def _parse_ndjson(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_stream_emits_status_then_result(client, monkeypatch):
    async def fake_ai_playlist(*, on_status=None, **kw):
        on_status({"stage": "plan"})
        on_status({"stage": "web_search", "query": "Kanye West greatest hits"})
        return {"title": "Хиты", "steps": [{"tool": "web_search", "query": "q", "found": 1}],
                "tracks": [{"track_id": "k1", "title": "Stronger", "artist": "Kanye West",
                            "duration": 200.0, "file_path": "/k1.mp3"}]}

    monkeypatch.setattr(
        "app.api.routes.recommend.recsys_ai_service.ai_playlist", fake_ai_playlist)

    resp = client.post("/api/v1/recommend/ai-playlist/stream",
                       json={"prompt": "хиты канье", "lang": "ru", "limit": 15})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")

    events = _parse_ndjson(resp.text)
    assert [e["type"] for e in events] == ["status", "status", "result"]
    assert events[0]["stage"] == "plan"
    assert events[1]["query"] == "Kanye West greatest hits"
    payload = events[2]["payload"]
    assert payload["title"] == "Хиты"
    assert payload["tracks"][0]["track_id"] == "k1"


def test_stream_emits_error_event_on_failure(client, monkeypatch):
    async def broken_ai_playlist(*, on_status=None, **kw):
        on_status({"stage": "plan"})
        raise RuntimeError("llm down")

    monkeypatch.setattr(
        "app.api.routes.recommend.recsys_ai_service.ai_playlist", broken_ai_playlist)

    resp = client.post("/api/v1/recommend/ai-playlist/stream",
                       json={"prompt": "что-то", "lang": "ru", "limit": 15})
    assert resp.status_code == 200
    events = _parse_ndjson(resp.text)
    assert [e["type"] for e in events] == ["status", "error"]
    assert "llm down" in events[1]["message"]
