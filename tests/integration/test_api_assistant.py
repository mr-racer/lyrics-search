"""Integration tests for the unified assistant endpoints.

Cover the plumbing the unit tests can't: router registration order, the
per-account collection derivation, the NDJSON envelope, and — the one that
would otherwise only blow up in production — that a sync ``on_status`` callback
fired from a WORKER THREAD still lands in the stream, in order, before the
terminal ``result`` frame.

The intent router and the three executors are stubbed; what runs for real is
``assistant_service.run_assistant`` plus the route.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.domain.models import AssistantRoute
from app.services.search_service import SearchService


@contextmanager
def _client(monkeypatch, **state):
    """TestClient with auth overridden and app.state wired to fakes.

    The fakes are installed AFTER entering the context: lifespan runs on
    ``__enter__`` and would overwrite ``db_client``/``search_service`` (Qdrant
    is unreachable in CI, so the app boots in limited mode and the route would
    503).
    """
    from app.api.routes import assistant as assistant_route

    fixed = SimpleNamespace(id="user-A", email="a@x")
    app = create_app()
    app.dependency_overrides[assistant_route.get_current_user] = lambda: fixed
    with TestClient(app) as client:
        client.app.state.search_service = state.get(
            "search_service", MagicMock(spec=SearchService),
        )
        client.app.state.db_client = SimpleNamespace(
            qdrant=state.get("qdrant", MagicMock()),
        )
        yield client


def _stub_route(monkeypatch, intent, **kw):
    from app.services.assistant import router as R

    # ``**_kw`` keeps this stub honest about the router's real signature
    # (last_message and the llm_* passthrough) without pinning it here —
    # what this file tests is the HTTP layer, not the routing decision.
    async def fake_route(message, slots=None, *, explicit_intent=None, **_kw):
        if explicit_intent:
            return AssistantRoute(intent=explicit_intent, source="explicit", confidence=1.0)
        return AssistantRoute(intent=intent, confidence=0.9, margin=0.5,
                              source="gliner", **kw)

    monkeypatch.setattr(R, "route", fake_route)


def _ndjson(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


# ── routing outcomes ─────────────────────────────────────────────────────────


def test_unclear_intent_returns_clarify_options(monkeypatch):
    from app.services.assistant import router as R

    # ``**_kw`` keeps this stub honest about the router's real signature
    # (last_message and the llm_* passthrough) without pinning it here —
    # what this file tests is the HTTP layer, not the routing decision.
    async def fake_route(message, slots=None, *, explicit_intent=None, **_kw):
        return AssistantRoute(intent=None, source="unclear")

    monkeypatch.setattr(R, "route", fake_route)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/", json={"message": "эээ ну такое", "lang": "ru"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] is None
    assert [o["intent"] for o in body["clarify"]] == ["search", "playlist", "facts"]
    # The UI must never have to invent the button captions.
    assert all(o["label"] for o in body["clarify"])


def test_search_intent_delegates_to_the_search_engine(monkeypatch):
    _stub_route(monkeypatch, "search")
    captured = {}

    async def fake_run_chat_search(req, search_service, current_user, emit=None):
        captured["planner_enabled"] = req.planner_enabled
        captured["message"] = req.message
        return {"message": "found it", "song": "Runaway", "artist": "Kanye West",
                "confidence": "high", "best_hit": {"track": {"track_id": "t-9"}},
                "hits": [], "attempts": 1, "classification": {}}

    monkeypatch.setattr("app.services.chat_search_service.run_chat_search",
                        fake_run_chat_search)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/", json={"message": "найди трек про дождь"})
    body = resp.json()
    assert body["intent"] == "search"
    assert body["search"]["song"] == "Runaway"
    # Intent is already known, so the planner runs and the old LLM classifier
    # round-trip is skipped.
    assert captured["planner_enabled"] is True
    # The resolved track flows into the slots for the next turn.
    assert body["slots"]["last_track_id"] == "t-9"
    assert body["slots"]["last_intent"] == "search"


def test_playlist_intent_honours_an_explicit_count(monkeypatch):
    _stub_route(monkeypatch, "playlist", count=7)
    captured = {}

    async def fake_ai_playlist(**kwargs):
        captured.update(kwargs)
        return {"title": "Run", "steps": [],
                "tracks": [{"track_id": "t-1", "title": "A", "artist": "B",
                            "file_path": "/a.mp3"}]}

    monkeypatch.setattr("app.services.recsys_ai_service.ai_playlist", fake_ai_playlist)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/",
                      json={"message": "собери 7 треков под бег", "limit": 15})
    body = resp.json()
    assert body["intent"] == "playlist"
    assert captured["limit"] == 7  # the user's number beats the request default
    assert body["playlist"]["title"] == "Run"
    assert body["slots"]["last_playlist_ids"] == ["t-1"]


def test_facts_intent_can_ask_which_subject(monkeypatch):
    _stub_route(monkeypatch, "facts")

    async def fake_facts_run(**kwargs):
        return None, [
            {"kind": "artist", "title": "Queen", "artist_slug": "queen",
             "subtitle": None, "image_path": None, "track_id": None},
            {"kind": "song", "title": "Queen Bee", "artist_slug": None,
             "subtitle": "Taj Mahal", "image_path": None, "track_id": "t-2"},
        ]

    monkeypatch.setattr("app.services.assistant.facts_executor.run", fake_facts_run)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/", json={"message": "расскажи про Queen"})
    body = resp.json()
    assert body["intent"] == "facts"
    assert [o["title"] for o in body["disambiguate"]] == ["Queen", "Queen Bee"]
    assert body["facts"] is None


def test_focus_fact_reaches_the_executor_and_comes_back_on_the_payload(monkeypatch):
    """Tapping a fact asks about THAT fact: the statement has to survive the
    round trip, and an unexplained one must not be dressed up as an answer."""
    _stub_route(monkeypatch, "facts")
    seen = {}

    async def fake_facts_run(**kwargs):
        seen["focus_fact"] = kwargs.get("focus_fact")
        return {
            "subject_kind": "song", "subject_title": "Runaway", "answer": "…",
            "grounded": False, "explained": False, "focus_fact": kwargs["focus_fact"],
            "items": [], "related_tracks": [],
        }, []

    monkeypatch.setattr("app.services.assistant.facts_executor.run", fake_facts_run)

    fact = "«Runaway» сэмплирует «Expo 83»"
    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/",
                      json={"message": f"объясни: {fact}", "focus_fact": fact})
    body = resp.json()
    assert seen["focus_fact"] == fact
    assert body["facts"]["focus_fact"] == fact
    assert body["facts"]["explained"] is False
    assert body["facts"]["items"] == []      # no consolation list of other facts


def test_explicit_intent_from_a_clarify_reply_is_obeyed(monkeypatch):
    """The second leg of a clarify: the client echoes the chosen intent and the
    router must not be consulted again."""
    from app.services.assistant import router as R

    calls = []

    # ``**_kw`` keeps this stub honest about the router's real signature
    # (last_message and the llm_* passthrough) without pinning it here —
    # what this file tests is the HTTP layer, not the routing decision.
    async def fake_route(message, slots=None, *, explicit_intent=None, **_kw):
        calls.append(explicit_intent)
        return AssistantRoute(intent=explicit_intent or None,
                              source="explicit" if explicit_intent else "unclear")

    monkeypatch.setattr(R, "route", fake_route)

    async def fake_facts_run(**kwargs):
        return {"subject_kind": "artist", "subject_title": "Queen", "answer": "A band.",
                "grounded": True, "items": [], "related_tracks": []}, []

    monkeypatch.setattr("app.services.assistant.facts_executor.run", fake_facts_run)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/",
                      json={"message": "эээ ну такое", "intent": "facts"})
    assert resp.json()["intent"] == "facts"
    assert calls == ["facts"]


# ── the NDJSON envelope ──────────────────────────────────────────────────────


def test_stream_emits_route_then_status_then_result(monkeypatch):
    _stub_route(monkeypatch, "search")

    async def fake_run_chat_search(req, search_service, current_user, emit=None):
        if emit:
            await emit({"type": "classify", "mode": "text"})
            await emit({"type": "search", "found": 3, "queries": ["rain"]})
        return {"message": "ok", "song": None, "artist": None, "confidence": "low",
                "best_hit": None, "hits": [], "attempts": 1, "classification": {}}

    monkeypatch.setattr("app.services.chat_search_service.run_chat_search",
                        fake_run_chat_search)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/stream",
                      json={"message": "найди трек про дождь", "lang": "ru"})
    frames = _ndjson(resp)
    assert frames[0]["type"] == "route" and frames[0]["intent"] == "search"
    stages = [f["stage"] for f in frames if f["type"] == "status"]
    assert stages == ["classify", "search"]
    assert frames[-1]["type"] == "result"
    assert frames[-1]["payload"]["intent"] == "search"
    # Every frame ships a ready-to-render caption — the SPA phrases nothing.
    assert all(f.get("human") for f in frames if f["type"] in ("route", "status"))


def test_stream_preserves_events_emitted_from_a_worker_thread(monkeypatch):
    """The regression this design note exists for: ``playlist_agent`` tools run
    inside ``asyncio.to_thread`` and call the SYNC ``on_status``. Without the
    ``call_soon_threadsafe`` hop those events are lost or overtake ``result``."""
    _stub_route(monkeypatch, "playlist")

    async def fake_ai_playlist(**kwargs):
        on_status = kwargs["on_status"]

        def in_thread():
            on_status({"stage": "web_search", "query": "Kanye greatest hits"})
            on_status({"stage": "auto_matched", "query": "Kanye greatest hits",
                       "found": 14})

        await asyncio.to_thread(in_thread)
        on_status({"stage": "select_done", "picked": 1})  # from the loop thread
        return {"title": "Hits", "steps": [],
                "tracks": [{"track_id": "t-1", "title": "A", "artist": "Kanye West",
                            "file_path": "/a.mp3"}]}

    monkeypatch.setattr("app.services.recsys_ai_service.ai_playlist", fake_ai_playlist)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/stream",
                      json={"message": "собери хиты Канье", "lang": "ru"})
    frames = _ndjson(resp)
    stages = [f["stage"] for f in frames if f["type"] == "status"]
    assert stages == ["web_search", "auto_matched", "select_done"]
    assert frames[-1]["type"] == "result"
    # The web query reaches the user verbatim inside the rendered caption.
    web = next(f for f in frames if f.get("stage") == "web_search")
    assert "Kanye greatest hits" in web["human"]


def test_stream_reports_a_failure_as_a_terminal_error_frame(monkeypatch):
    _stub_route(monkeypatch, "search")

    async def boom(req, search_service, current_user, emit=None):
        raise RuntimeError("LLM is down")

    monkeypatch.setattr("app.services.chat_search_service.run_chat_search", boom)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/stream", json={"message": "x"})
    frames = _ndjson(resp)
    assert frames[-1]["type"] == "error"
    assert "LLM is down" in frames[-1]["message"]


# ── wiring guards ────────────────────────────────────────────────────────────


def test_collection_is_derived_from_the_jwt_not_the_client(monkeypatch):
    """Per-account isolation: the assistant must pass ``acct_{user.id}`` down,
    whatever the body says."""
    _stub_route(monkeypatch, "playlist")
    captured = {}

    async def fake_ai_playlist(**kwargs):
        captured.update(kwargs)
        return {"title": "x", "steps": [], "tracks": []}

    monkeypatch.setattr("app.services.recsys_ai_service.ai_playlist", fake_ai_playlist)

    with _client(monkeypatch) as c:
        c.post("/api/v1/assistant/",
               json={"message": "собери подборку", "collection_name": "acct-someone-else"})
    assert captured["collection_name"] == "acct_user-A"


def test_unavailable_database_is_a_503(monkeypatch):
    from app.api.routes import assistant as assistant_route

    fixed = SimpleNamespace(id="user-A", email="a@x")
    app = create_app()
    app.dependency_overrides[assistant_route.get_current_user] = lambda: fixed
    with TestClient(app) as c:
        c.app.state.db_client = None
        resp = c.post("/api/v1/assistant/", json={"message": "x"})
    assert resp.status_code == 503
