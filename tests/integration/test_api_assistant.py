"""Integration tests for the unified assistant endpoints.

Cover the plumbing the unit tests can't: router registration, the per-account
collection derivation, the NDJSON envelope, and — the one that would otherwise
only blow up in production — that a sync status callback fired from a WORKER
THREAD still lands in the stream, in order, before the terminal ``result``
frame.

``agent.Assistant`` is stubbed; what runs for real is
``assistant.service.run_assistant`` (slot merging, the four payload builders,
the EventSink's frame normalisation) plus the route. That boundary is the point:
these tests pin the contract between the agent and the HTTP layer, and the
agent's own decisions belong to tests/unit/test_assistant_*.py.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.services.assistant.contracts import (ClarifyRequest, Evidence,
                                              GeneralResult, LyricsResult,
                                              PlaylistResult, Subject)
from app.services.search_service import SearchService


@contextmanager
def _client(monkeypatch, **state):
    """TestClient with auth overridden and app.state wired to fakes.

    The fakes are installed AFTER entering the context: lifespan runs on
    ``__enter__`` and would overwrite ``db_client``/``search_service`` (Qdrant
    is unreachable under test, so the app boots in limited mode and the route
    would 503).
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


def _stub_agent(monkeypatch, run):
    """Replace agent.Assistant with one whose ``run`` is ``run``.

    ``run`` is called as ``run(agent, message, **kwargs)`` so a test can assert
    on what the service forwarded, and reach ``agent.sink`` to emit events.
    """
    from app.services.assistant import agent as agent_module

    class _StubAssistant:
        def __init__(self, collection_name, *, config=None, sink=None,
                     search_service=None):
            self.collection_name = collection_name
            self.config = config
            self.sink = sink
            self.search_service = search_service
            self.catalog = None

        async def run(self, message, **kwargs):
            return await run(self, message, **kwargs)

    monkeypatch.setattr(agent_module, "Assistant", _StubAssistant)


def _returns(result):
    async def _run(agent, message, **kwargs):
        return result
    return _run


def _ndjson(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def _track(track_id="t-1", title="A", artist="Kanye West"):
    """A resolved playlist track, as ``_playlist_track`` expects one."""
    return SimpleNamespace(
        track_id=track_id, title=title, artist=artist, album=None, year=None,
        duration_sec=100.0, file_path="/a.mp3", cover_art_path=None,
        reason="because", sources=["web"], weight=1.0,
    )


# ── the four payload shapes ──────────────────────────────────────────────────


def test_lyrics_result_becomes_a_search_payload(monkeypatch):
    _stub_agent(monkeypatch, _returns(LyricsResult(
        message="found it", song="Runaway", artist="Kanye West",
        confidence="high", best_hit=None, hits=[],
    )))

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/", json={"message": "найди трек про дождь"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "lyrics_search"
    assert body["search"]["message"] == "found it"
    assert body["search"]["song"] == "Runaway"
    # The turn's findings are carried into slots so the next message can say
    # "ещё у этого артиста" without repeating the name.
    assert body["slots"]["last_artist"] == "Kanye West"
    assert body["slots"]["last_intent"] == "lyrics_search"


def test_playlist_result_becomes_a_playlist_payload(monkeypatch):
    _stub_agent(monkeypatch, _returns(PlaylistResult(
        title="Hits", comment="", tracks=[_track("t-1"), _track("t-2", "B")],
    )))

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/", json={"message": "лучшее у канье"})

    body = resp.json()
    assert body["intent"] == "playlist"
    assert [t["track_id"] for t in body["playlist"]["tracks"]] == ["t-1", "t-2"]
    assert body["slots"]["last_playlist_ids"] == ["t-1", "t-2"]


def test_general_result_becomes_an_answer_payload(monkeypatch):
    _stub_agent(monkeypatch, _returns(GeneralResult(
        answer="Because of the sample.", grounded=True, iterations=2, used=[1],
        evidence=[Evidence(n=1, kind="fact", text="a fact",
                           source="example.com", url="http://example.com")],
        subject=Subject(song_slug="kanye-west-runaway", song_title="Runaway",
                        artist_name="Kanye West", how="song-row"),
    )))

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/", json={"message": "почему runaway такой"})

    body = resp.json()
    assert body["intent"] == "general"
    assert body["answer"]["answer"] == "Because of the sample."
    assert body["answer"]["grounded"] is True
    # `used` marks which evidence the answer actually leaned on — the UI greys
    # out the rest, so the flag has to survive the payload builder.
    assert [e["used"] for e in body["answer"]["evidence"]] == [True]


def test_clarify_is_surfaced_as_the_question_it_is(monkeypatch):
    """An un-expandable abbreviation asks one question, not a branch menu."""
    _stub_agent(monkeypatch, _returns(GeneralResult(
        answer="", evidence=[], used=[], grounded=False, iterations=0,
        clarify=ClarifyRequest(kind="abbreviation",
                               question="Кого именно ты имеешь в виду под «БГ»?"),
    )))

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/", json={"message": "расскажи про БГ",
                                                  "lang": "ru"})

    body = resp.json()
    assert body["intent"] == "general"
    assert body["human"] == "Кого именно ты имеешь в виду под «БГ»?"
    assert body["answer"]["grounded"] is False


# ── what the service forwards to the agent ───────────────────────────────────


def test_focus_fact_reaches_the_agent_and_comes_back_on_the_payload(monkeypatch):
    seen = {}

    async def _run(agent, message, **kwargs):
        seen.update(kwargs)
        return GeneralResult(answer="explained", evidence=[], used=[],
                             grounded=True, iterations=1,
                             focus_fact=kwargs.get("focus_fact"), explained=True)

    _stub_agent(monkeypatch, _run)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/", json={
            "message": "что это значит", "focus_fact": "«Runaway» сэмплирует «Expo 83»",
        })

    assert seen["focus_fact"] == "«Runaway» сэмплирует «Expo 83»"
    body = resp.json()
    assert body["answer"]["focus_fact"] == "«Runaway» сэмплирует «Expo 83»"
    assert body["answer"]["explained"] is True


def test_explicit_intent_from_a_clarify_reply_is_forwarded(monkeypatch):
    """The user tapped a branch — that choice outranks the planner's reading."""
    seen = {}

    async def _run(agent, message, **kwargs):
        seen.update(kwargs)
        return PlaylistResult(title="Hits", comment="", tracks=[])

    _stub_agent(monkeypatch, _run)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/", json={"message": "канье",
                                                   "intent": "playlist"})

    assert seen["forced_intent"] == "playlist"
    assert resp.json()["intent"] == "playlist"


def test_collection_is_derived_from_the_jwt_not_the_client(monkeypatch):
    """A client-supplied collection must never be able to reach the agent."""
    seen = {}

    async def _run(agent, message, **kwargs):
        seen["collection"] = agent.collection_name
        return GeneralResult(answer="ok", evidence=[], used=[], grounded=False,
                             iterations=0)

    _stub_agent(monkeypatch, _run)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/", json={
            "message": "привет", "collection_name": "acct_someone_else",
        })

    assert resp.status_code == 200
    assert seen["collection"] == "acct_user-A"


def test_unavailable_database_is_a_503(monkeypatch):
    from app.api.routes import assistant as assistant_route

    fixed = SimpleNamespace(id="user-A", email="a@x")
    app = create_app()
    app.dependency_overrides[assistant_route.get_current_user] = lambda: fixed
    with TestClient(app) as c:
        c.app.state.db_client = None
        resp = c.post("/api/v1/assistant/", json={"message": "привет"})
    assert resp.status_code == 503


# ── the NDJSON envelope ──────────────────────────────────────────────────────


def test_stream_emits_route_then_status_then_result(monkeypatch):
    async def _run(agent, message, **kwargs):
        # A plan event carrying an intent is what colours the client's orb, so
        # the sink promotes it to a `route` frame ahead of its status line.
        agent.sink.put("plan", intent="lyrics_search")
        agent.sink.put("search", found=3)
        return LyricsResult(message="ok", song=None, artist=None)

    _stub_agent(monkeypatch, _run)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/stream",
                      json={"message": "найди трек про дождь", "lang": "ru"})

    frames = _ndjson(resp)
    assert frames[0]["type"] == "route" and frames[0]["intent"] == "lyrics_search"
    assert [f["stage"] for f in frames if f["type"] == "status"] == ["plan", "search"]
    assert frames[-1]["type"] == "result"
    assert frames[-1]["payload"]["intent"] == "lyrics_search"
    # Every frame ships a ready-to-render caption — the SPA phrases nothing.
    assert all(f.get("human") for f in frames if f["type"] in ("route", "status"))


def test_stream_preserves_events_emitted_from_a_worker_thread(monkeypatch):
    """The regression this design note exists for: branch tools run inside
    ``asyncio.to_thread`` and call the SYNC sink. Without the
    ``call_soon_threadsafe`` hop those events are lost or overtake ``result``."""
    async def _run(agent, message, **kwargs):
        def in_thread():
            agent.sink.put("web_search", query="Kanye greatest hits")
            agent.sink.put("auto_matched", query="Kanye greatest hits", found=14)

        await asyncio.to_thread(in_thread)
        agent.sink.put("select_done", picked=1)  # from the loop thread
        return PlaylistResult(title="Hits", comment="", tracks=[_track("t-1")])

    _stub_agent(monkeypatch, _run)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/stream",
                      json={"message": "лучшее у канье", "lang": "ru"})

    frames = _ndjson(resp)
    stages = [f["stage"] for f in frames if f["type"] == "status"]
    assert stages == ["web_search", "auto_matched", "select_done"]
    # Ordering against the terminal frame is the whole point.
    assert frames[-1]["type"] == "result"


def test_stream_reports_a_failure_as_a_terminal_error_frame(monkeypatch):
    async def _run(agent, message, **kwargs):
        raise RuntimeError("the planner exploded")

    _stub_agent(monkeypatch, _run)

    with _client(monkeypatch) as c:
        resp = c.post("/api/v1/assistant/stream", json={"message": "привет"})

    frames = _ndjson(resp)
    assert frames[-1]["type"] == "error"
    assert "the planner exploded" in frames[-1]["message"]
    # A stream that dies silently looks identical to one still working, so the
    # error frame must carry a caption the UI can show as-is.
    assert frames[-1]["human"]
