"""Integration tests for the playlist-builder route.

The agent itself (web_search/LLM) is mocked — these verify the route wiring:
response shape, hallucinated-track_id filtering, and the unavailable-service
guard."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app


def _app_with_user():
    from app.api.routes import chat as chat_route

    fixed = SimpleNamespace(id="user-A", email="a@x")
    app = create_app()
    app.dependency_overrides[chat_route.get_current_user] = lambda: fixed
    return app


@pytest.mark.integration
def test_playlist_builder_returns_draft():
    from app.domain.models import PlaylistDraft

    draft = PlaylistDraft(title="Хиты", track_ids=["1"], comment="нашлось мало", missing=["X — Y"])
    svc = MagicMock()  # only needs .lyrics_db.qdrant_client for CatalogAdapter (agent is mocked)
    app = _app_with_user()
    with TestClient(app) as c:
        c.app.state.search_service = svc
        with patch("app.api.routes.chat.run_playlist_agent", new=AsyncMock(return_value=draft)):
            r = c.post("/api/v1/chat/playlist-builder", json={"prompt": "хиты Канье", "lang": "ru"})
    assert r.status_code == 200
    body = r.json()
    assert body["draft"]["title"] == "Хиты"
    assert body["draft"]["missing"] == ["X — Y"]
    # state was never populated (agent mocked) -> hallucinated id "1" filtered out
    assert body["tracks"] == []


@pytest.mark.integration
def test_playlist_builder_503_when_no_service():
    app = _app_with_user()
    with TestClient(app) as c:
        c.app.state.search_service = None
        r = c.post("/api/v1/chat/playlist-builder", json={"prompt": "хиты Канье", "lang": "ru"})
    assert r.status_code == 503
