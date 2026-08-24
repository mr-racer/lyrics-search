"""The HTTP side of the local-first work: turn contexts and the samples payload.

What these cover that the unit tests cannot: the new route is registered and
sits behind the auth gate, the context handle survives the round trip through
the NDJSON envelope, and the fields the card renders (``related_tracks``,
``focus_kind``) actually reach it.

The context store's own rules — TTL, eviction, cross-account isolation — are
pinned in ``tests/unit/test_assistant_page_store.py``. Here the question is only
whether the wiring carries them.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.services.assistant.contracts import Evidence, GeneralResult
from app.services.search_service import SearchService

pytestmark = pytest.mark.integration


@contextmanager
def _client(user_id="user-A", **state):
    from app.api.routes import assistant as assistant_route

    fixed = SimpleNamespace(id=user_id, email=f"{user_id}@x")
    app = create_app()
    app.dependency_overrides[assistant_route.get_current_user] = lambda: fixed
    with TestClient(app) as client:
        client.app.state.search_service = state.get(
            "search_service", MagicMock(spec=SearchService))
        client.app.state.db_client = SimpleNamespace(
            qdrant=state.get("qdrant", MagicMock()))
        yield client


def _stub_agent(monkeypatch, result, *, chunks=(), record=None):
    """Replace agent.Assistant with one returning ``result``.

    ``chunks`` become the branch's chunks, which is what decides whether a turn
    context is worth saving at all.
    """
    from app.services.assistant import agent as agent_module

    class _StubAssistant:
        def __init__(self, collection_name, *, config=None, sink=None,
                     search_service=None):
            self.collection_name = collection_name
            self.sink = sink
            self.catalog = None
            self.last_branch = SimpleNamespace(chunks=list(chunks),
                                               used_queries=["q1"])

        async def run(self, message, **kwargs):
            if record is not None:
                record.update(kwargs)
            return result

    monkeypatch.setattr(agent_module, "Assistant", _StubAssistant)


def _general(**kw):
    base = dict(answer="An answer.", evidence=[
        Evidence(n=1, text="a fact", kind="fact", source="songfacts")],
        used=[1], grounded=True, iterations=0)
    base.update(kw)
    return GeneralResult(**base)


def _chunk(body="passage"):
    from app.services.assistant.contracts import Chunk

    return Chunk(id=0, path=["T"], body=body, url="https://e.org/a", title="T")


def _post(client, **body):
    payload = {"message": "why six minutes?", "lang": "ru"}
    payload.update(body)
    return client.post("/api/v1/assistant/", json=payload)


# ── the context handle ───────────────────────────────────────────────────────


def test_a_turn_that_read_pages_hands_back_a_context_id(monkeypatch):
    _stub_agent(monkeypatch, _general(), chunks=[_chunk()])
    with _client() as client:
        body = _post(client).json()

    assert body["context_id"]


def test_a_turn_that_read_nothing_hands_back_none(monkeypatch):
    """Answered out of SQLite: there is nothing a later question could not read
    again just as cheaply, and an id for an empty context is a lie."""
    _stub_agent(monkeypatch, _general(), chunks=[])
    with _client() as client:
        body = _post(client).json()

    assert body["context_id"] is None


def test_the_context_id_comes_back_on_the_next_request(monkeypatch):
    seen = {}
    _stub_agent(monkeypatch, _general(), chunks=[_chunk()], record=seen)
    with _client() as client:
        first = _post(client).json()
        _post(client, context_id=first["context_id"])

    assert seen["context_id"] == first["context_id"]
    assert seen["context"] is not None, "the stored context did not come back"


def test_another_account_cannot_load_the_context(monkeypatch):
    seen = {}
    _stub_agent(monkeypatch, _general(), chunks=[_chunk()], record=seen)
    with _client(user_id="user-A") as client:
        first = _post(client).json()
    with _client(user_id="user-B") as other:
        _post(other, context_id=first["context_id"])

    assert seen["context"] is None, "one account read another's turn context"


def test_an_unknown_context_id_is_answered_normally(monkeypatch):
    seen = {}
    _stub_agent(monkeypatch, _general(), record=seen)
    with _client() as client:
        resp = _post(client, context_id="nope")

    assert resp.status_code == 200
    assert seen["context"] is None
    assert seen["context_id"] == "nope"    # the planner-skip rule still applies


# ── releasing it ─────────────────────────────────────────────────────────────


def test_release_drops_the_context(monkeypatch):
    seen = {}
    _stub_agent(monkeypatch, _general(), chunks=[_chunk()], record=seen)
    with _client() as client:
        cid = _post(client).json()["context_id"]
        resp = client.delete(f"/api/v1/assistant/context/{cid}")
        _post(client, context_id=cid)

    assert resp.status_code == 204
    assert seen["context"] is None


def test_release_of_an_unknown_context_is_still_204(monkeypatch):
    _stub_agent(monkeypatch, _general())
    with _client() as client:
        resp = client.delete("/api/v1/assistant/context/nothing-here")

    assert resp.status_code == 204


def test_another_account_cannot_release_it(monkeypatch):
    seen = {}
    _stub_agent(monkeypatch, _general(), chunks=[_chunk()], record=seen)
    with _client(user_id="user-A") as client:
        cid = _post(client).json()["context_id"]
    with _client(user_id="user-B") as other:
        assert other.delete(f"/api/v1/assistant/context/{cid}").status_code == 204
    with _client(user_id="user-A") as client:
        _post(client, context_id=cid)

    assert seen["context"] is not None, "a stranger's DELETE dropped the context"


# ── the samples payload ──────────────────────────────────────────────────────


def test_samples_fields_reach_the_card(monkeypatch):
    """The row shape here is what ``track_store`` actually returns: the mirror
    column is ``duration``, the model field is ``duration_sec``, and both it and
    ``file_path`` are required. Handing the row over unconverted is a 502 on
    every samples answer."""
    result = _general(
        focus_kind="samples",
        related_tracks=[{"track_id": "t-9", "title": "Expo 83",
                         "artist": "Backyard Heavies", "duration": 214.0,
                         "file_path": "/music/expo83.mp3"}])
    _stub_agent(monkeypatch, result)
    with _client() as client:
        answer = _post(client, focus_kind="samples",
                       subject_track_id="t-1").json()["answer"]

    assert answer["focus_kind"] == "samples"
    assert [t["track_id"] for t in answer["related_tracks"]] == ["t-9"]
    assert answer["related_tracks"][0]["duration_sec"] == 214.0


def test_a_sparse_related_row_still_validates(monkeypatch):
    """A mirror row missing the optional halves must not take the answer down
    with it — the track just renders without a duration."""
    result = _general(focus_kind="samples",
                      related_tracks=[{"track_id": "t-9", "title": "Expo 83",
                                       "artist": "Backyard Heavies"}])
    _stub_agent(monkeypatch, result)
    with _client() as client:
        resp = _post(client, focus_kind="samples", subject_track_id="t-1")

    assert resp.status_code == 200
    assert resp.json()["answer"]["related_tracks"][0]["duration_sec"] == 0.0


def test_the_request_fields_reach_the_agent(monkeypatch):
    seen = {}
    _stub_agent(monkeypatch, _general(), record=seen)
    with _client() as client:
        _post(client, focus_kind="samples", allow_web=True,
              subject_track_id="t-1")

    assert seen["focus_kind"] == "samples"
    assert seen["allow_web"] is True
    assert seen["subject_track_id"] == "t-1"


def test_allow_web_defaults_to_none_not_false(monkeypatch):
    """None means "the mode decides". A False default would mute the web for
    every ordinary turn, which is the kind of bug that looks like a slow day."""
    seen = {}
    _stub_agent(monkeypatch, _general(), record=seen)
    with _client() as client:
        _post(client)

    assert seen["allow_web"] is None


def test_the_stream_carries_the_context_id_too(monkeypatch):
    _stub_agent(monkeypatch, _general(), chunks=[_chunk()])
    with _client() as client:
        resp = client.post("/api/v1/assistant/stream",
                           json={"message": "why?", "lang": "ru"})
        frames = [json.loads(line) for line in resp.text.splitlines()
                  if line.strip()]

    terminal = [f for f in frames if f.get("type") == "result"]
    assert terminal and terminal[0]["payload"]["context_id"]
