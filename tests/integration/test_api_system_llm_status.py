"""Integration tests for POST /system/llm-status."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_status_returns_available_true_on_2xx_response(client):
    fake_response = MagicMock(
        status_code=200,
        json=lambda: {"data": [{"id": "x"}]},  # endpoint checks model in /models response
    )
    fake_response.raise_for_status = MagicMock()
    mock_get = AsyncMock(return_value=fake_response)
    with patch("app.api.routes.system.httpx.AsyncClient") as ctx:
        ctx.return_value.__aenter__.return_value.get = mock_get
        r = client.post(
            "/api/v1/system/llm-status",
            json={"base_url": "http://localhost:1234/v1", "model": "x"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["base_url"] == "http://localhost:1234/v1"
    assert body["error"] is None


def test_status_returns_available_false_on_connection_error(client):
    import httpx
    mock_get = AsyncMock(side_effect=httpx.ConnectError("could not connect"))
    with patch("app.api.routes.system.httpx.AsyncClient") as ctx:
        ctx.return_value.__aenter__.return_value.get = mock_get
        r = client.post(
            "/api/v1/system/llm-status",
            json={"base_url": "http://nope:9999/v1"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert "could not connect" in (body["error"] or "")


def test_status_uses_env_fallback_when_body_empty(client, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://envhost:5555/v1")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    fake_response = MagicMock(
        status_code=200,
        json=lambda: {"data": [{"id": "env-model"}]},
    )
    fake_response.raise_for_status = MagicMock()
    mock_get = AsyncMock(return_value=fake_response)
    with patch("app.api.routes.system.httpx.AsyncClient") as ctx:
        ctx.return_value.__aenter__.return_value.get = mock_get
        r = client.post("/api/v1/system/llm-status", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["base_url"] == "http://envhost:5555/v1"
    assert body["model"] == "env-model"


def test_status_reports_error_when_no_base_url_anywhere(client, monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    r = client.post("/api/v1/system/llm-status", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert "no base_url" in (body["error"] or "").lower()
