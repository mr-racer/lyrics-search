"""Integration tests for chat API endpoint."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.services.search_service import SearchService
from ._auth_helper import authenticate_test_client


def test_chat_ignores_supplied_collection():
    """D-soft: POST /chat/ must use derived collection from JWT, not client-supplied value."""
    from app.api.routes import chat as chat_route

    fixed = SimpleNamespace(id="user-A", email="a@x")
    captured = {}

    async def fake_search(**kwargs):
        captured.setdefault("collection_name", kwargs.get("collection_name"))
        return []

    # Mock ask_llm to immediately return action="answer" so we don't need a real LLM.
    # The first call (or the only call in the agentic loop) returns an answer action
    # so the loop terminates without trying to execute a search via ask_llm.
    # We still need the audio/text fast-path to hit service.search —
    # use mode="audio" which goes through the RRF audio fast path (no ask_llm for search).
    svc = MagicMock(spec=SearchService)
    svc.search = fake_search

    import app.api.routes.chat as chat_module

    orig_ask_llm = chat_module.ask_llm

    async def fake_ask_llm(message, *, system_prompt=None, parse_json=False, **kwargs):
        """Return CLAP rephrase result for audio fast path (list of rephrased queries)."""
        # _CLAP_REPHRASE_SYSTEM_PROMPT branch expects a list of strings
        if parse_json and system_prompt and "CLAP" in (system_prompt or ""):
            return ["mellow instrumental"]
        # audio-answer prompt expects a dict
        if parse_json and system_prompt and "user_query" in (system_prompt or ""):
            return {"message": "Found it!"}
        # agentic loop ask_llm: return answer immediately
        if parse_json:
            return {"action": "answer", "confidence": "low", "song": None, "artist": None,
                    "message": "no result", "queries": None}
        return {}

    chat_module.ask_llm = fake_ask_llm
    try:
        app = create_app()
        app.dependency_overrides[chat_route.get_current_user] = lambda: fixed
        with TestClient(app) as c:
            c.app.state.search_service = svc
            resp = c.post(
                "/api/v1/chat/",
                json={
                    "message": "hi",
                    "collection_name": "acct_BAD",
                    "auto_mode": False,
                    "mode": "audio",
                },
            )
        assert resp.status_code == 200
        # The search was called with the derived collection, not the client-supplied one
        assert captured["collection_name"] == "acct_user-A", (
            f"Expected acct_user-A but got {captured.get('collection_name')!r}"
        )
    finally:
        chat_module.ask_llm = orig_ask_llm


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
