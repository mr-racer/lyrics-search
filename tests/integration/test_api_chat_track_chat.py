"""Integration test for POST /chat/track-chat endpoint."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    app = create_app()
    yield TestClient(app)
    MetadataDB._reset_for_tests()


@pytest.fixture
def mock_agent(monkeypatch):
    """Patch answer_track_chat to return a predictable response without calling a real LLM."""
    from app.domain.models import TrackChatResponse
    from app.services import track_chat_service

    async def fake_answer(req):
        return TrackChatResponse(
            message=f"Mocked reply for mode={req.mode}",
            web_search_used=(req.mode == "lyric_explain"),
        )

    monkeypatch.setattr(track_chat_service, "answer_track_chat", fake_answer)


def test_track_chat_song_mode_returns_message(client, mock_agent):
    body = {
        "track_context": {
            "title": "T", "artist": "A", "album": None, "year": None,
            "genre": None, "full_lyrics": "lyrics here",
        },
        "mode": "song",
        "message": "What's this song about?",
    }
    r = client.post("/api/v1/chat/track-chat", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "Mocked reply for mode=song" in data["message"]
    assert data["web_search_used"] is False


def test_track_chat_lyric_explain_returns_message(client, mock_agent):
    body = {
        "track_context": {
            "title": "T", "artist": "A", "album": None, "year": None,
            "genre": None, "full_lyrics": "line 1\nline 2",
        },
        "mode": "lyric_explain",
        "selected_line": "line 1",
        "message": "Explain this line",
    }
    r = client.post("/api/v1/chat/track-chat", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "Mocked reply for mode=lyric_explain" in data["message"]
    assert data["web_search_used"] is True


def test_track_chat_lyric_explain_without_selected_line_returns_400(client, mock_agent):
    body = {
        "track_context": {
            "title": "T", "artist": "A", "album": None, "year": None,
            "genre": None, "full_lyrics": "",
        },
        "mode": "lyric_explain",
        # selected_line omitted
        "message": "Explain",
    }
    r = client.post("/api/v1/chat/track-chat", json=body)
    assert r.status_code == 400
    assert "selected_line" in r.text


def test_track_chat_unknown_mode_returns_422(client):
    """Pydantic should reject an unknown mode at request validation."""
    body = {
        "track_context": {
            "title": "T", "artist": "A", "album": None, "year": None,
            "genre": None, "full_lyrics": "",
        },
        "mode": "INVALID",
        "message": "hi",
    }
    r = client.post("/api/v1/chat/track-chat", json=body)
    assert r.status_code == 422
