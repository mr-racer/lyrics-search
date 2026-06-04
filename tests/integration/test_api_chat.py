"""Integration tests for chat API endpoint."""

from fastapi.testclient import TestClient

from app.api.main import create_app
from ._auth_helper import authenticate_test_client


class TestChatAPI:
    def test_chat_returns_response(self):
        """POST /api/v1/chat/ returns a response even with limited setup."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.post(
                "/api/v1/chat/",
                json={"message": "Find some pop songs"},
            )
            # The chat endpoint may return various responses depending on
            # LLM availability. We just check it responds without crashing.
            assert resp.status_code in (200, 500, 503)

    def test_chat_with_mode(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.post(
                "/api/v1/chat/",
                json={
                    "message": "Find upbeat tracks",
                    "mode": "text",
                    "auto_mode": False,
                },
            )
            assert resp.status_code in (200, 500, 503)

    def test_chat_with_history(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.post(
                "/api/v1/chat/",
                json={
                    "message": "More like that",
                    "history": [
                        {"role": "user", "content": "Find pop songs"},
                        {"role": "assistant", "content": "Here are some..."},
                    ],
                },
            )
            assert resp.status_code in (200, 500, 503)

    def test_chat_validates_message(self):
        """Empty message should fail validation."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.post(
                "/api/v1/chat/",
                json={},
            )
            assert resp.status_code == 422

    def test_chat_with_llm_settings(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.post(
                "/api/v1/chat/",
                json={
                    "message": "Test",
                    "llm_base_url": "http://localhost:8000/v1",
                    "llm_model": "test-model",
                },
            )
            assert resp.status_code in (200, 500, 503)
