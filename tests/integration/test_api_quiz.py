"""Quiz API surface, end to end through the live ASGI app.

Two of these tests exist to guard the feature's primary invariant rather than
any behaviour a user sees: I-1 (the quiz never feeds the recommender) and I-2
(a round's snippet is not listening history). Both are the kind of coupling
that gets introduced by accident months later, and neither is visible to a
unit test.

The library is seeded into the SQLite ``track_metadata`` mirror, which is what
``light_points`` reads first — so these tests exercise the real read path with
no Qdrant round-trip and no fake client.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §12, §14.
"""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import quiz as quiz_route
from app.resources.metadata_db import MetadataDB

pytestmark = pytest.mark.integration

AXES = ("energy", "vocal_lead", "spacious", "experimental",
        "brightness", "acousticness")

_USER = SimpleNamespace(id="quiz-test", email="quiz@example.com")
_COLL = "acct_quiz-test"


def _make_app(authenticated=True):
    app = create_app()
    if authenticated:
        app.dependency_overrides[quiz_route.get_current_user] = lambda: _USER
    return app


def _seed_library(n=30, plays=3, collection=_COLL):
    """Populate the SQLite mirror plus real playback events."""
    MetadataDB.init()
    for i in range(1, n + 1):
        MetadataDB.upsert_track_metadata(collection, f"t{i}", {
            "title": f"Song {i}",
            "artist": f"Artist {i}",
            "artists": [f"Artist {i}"],
            "artist_slugs": [f"artist-{i}"],
            "primary_artist_slug": f"artist-{i}",
            "album": f"Album {i}",
            "year": 2000 + (i % 20),
            "genre": "rock" if i % 2 else "pop",
            "duration": 200.0,
            "file_path": f"/music/t{i}.mp3",
            "cover_art_path": f"/covers/t{i}.jpg",
            "sonic_axes": dict(zip(AXES, (0.03 * i, 0.2, 0.3, 0.4, 0.5, 0.6))),
        })
        for _ in range(plays):
            MetadataDB.record_playback_event(
                session_id="seed", collection_name=collection,
                track_id=f"t{i}", played_sec=180.0, total_dur=200.0,
            )


def _count(table):
    row = MetadataDB._connect().execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def _correct_option_id(round_id):
    spec = json.loads(MetadataDB.get_quiz_round(round_id)["spec_json"])
    return spec["correct_option_id"]


# ── The auth gate ────────────────────────────────────────────────────────────

def test_modes_requires_a_token():
    """Every /api/v1 route sits behind the blanket gate."""
    with TestClient(_make_app(authenticated=False)) as client:
        assert client.get("/api/v1/quiz/modes").status_code == 401


# ── Registry reachability ────────────────────────────────────────────────────

def test_modes_lists_every_registered_mode():
    """The registry check a unit test cannot make: is the mode wired into the
    running app, not merely importable?"""
    _seed_library()
    with TestClient(_make_app()) as client:
        resp = client.get("/api/v1/quiz/modes")
        assert resp.status_code == 200
        keys = {m["key"] for m in resp.json()["modes"]}
        assert "track_snippet" in keys


def test_modes_reports_pool_and_availability():
    _seed_library()
    with TestClient(_make_app()) as client:
        entry = next(m for m in client.get("/api/v1/quiz/modes").json()["modes"]
                     if m["key"] == "track_snippet")
        assert entry["pool_size"] == 30
        assert entry["available"] is True


def test_thin_library_marks_the_mode_unavailable():
    _seed_library(n=6)
    with TestClient(_make_app()) as client:
        entry = next(m for m in client.get("/api/v1/quiz/modes").json()["modes"]
                     if m["key"] == "track_snippet")
        assert entry["available"] is False


# ── Playing a round ──────────────────────────────────────────────────────────

def test_a_round_is_created_without_the_answer():
    _seed_library()
    with TestClient(_make_app()) as client:
        resp = client.post("/api/v1/quiz/rounds", json={"mode": "track_snippet"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["options"]) == 4
        assert "correct_option_id" not in body
        assert "track_id" not in json.dumps(body)
        assert body["length_sec"] == 3.0
        assert body["start_sec"] > 0.0


def test_answering_reveals_the_truth():
    _seed_library()
    with TestClient(_make_app()) as client:
        created = client.post("/api/v1/quiz/rounds",
                              json={"mode": "track_snippet"}).json()
        correct_id = _correct_option_id(created["round_id"])
        resp = client.post(
            f"/api/v1/quiz/rounds/{created['round_id']}/answer",
            json={"option_id": correct_id},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["correct"] is True
        assert body["score"] == 100.0
        assert body["truth"]["title"]


def test_answering_twice_returns_409():
    _seed_library()
    with TestClient(_make_app()) as client:
        created = client.post("/api/v1/quiz/rounds",
                              json={"mode": "track_snippet"}).json()
        correct_id = _correct_option_id(created["round_id"])
        url = f"/api/v1/quiz/rounds/{created['round_id']}/answer"
        assert client.post(url, json={"option_id": correct_id}).status_code == 200
        assert client.post(url, json={"option_id": correct_id}).status_code == 409


def test_an_unknown_round_returns_404():
    _seed_library()
    with TestClient(_make_app()) as client:
        resp = client.post("/api/v1/quiz/rounds/nope/answer",
                           json={"option_id": "x"})
        assert resp.status_code == 404


def test_a_round_from_another_collection_returns_404():
    """Someone else's round must not be answerable, nor confirmable."""
    _seed_library()
    MetadataDB.create_quiz_round(
        round_id="foreign", collection_name="acct_someone-else",
        mode="track_snippet", track_id="t1",
        spec_json=json.dumps({"correct_option_id": "abc", "options": []}),
        expires_at=9_999_999_999.0,
    )
    with TestClient(_make_app()) as client:
        resp = client.post("/api/v1/quiz/rounds/foreign/answer",
                           json={"option_id": "abc"})
        assert resp.status_code == 404


def test_a_thin_library_refuses_a_round_with_409():
    _seed_library(n=6)
    with TestClient(_make_app()) as client:
        resp = client.post("/api/v1/quiz/rounds", json={"mode": "track_snippet"})
        assert resp.status_code == 409


# ── The invariants ───────────────────────────────────────────────────────────

def test_a_full_round_writes_no_playback_events():
    """I-2: a quiz snippet is not listening. Playing the game must not rewrite
    the listener's history."""
    _seed_library()
    before = _count("playback_events")
    with TestClient(_make_app()) as client:
        created = client.post("/api/v1/quiz/rounds",
                              json={"mode": "track_snippet"}).json()
        client.post(f"/api/v1/quiz/rounds/{created['round_id']}/answer",
                    json={"option_id": _correct_option_id(created["round_id"])})
    assert _count("playback_events") == before


def test_a_full_round_writes_no_taste_signals():
    """I-1: a score must never reach the recommender."""
    _seed_library()
    before = _count("taste_signals")
    with TestClient(_make_app()) as client:
        created = client.post("/api/v1/quiz/rounds",
                              json={"mode": "track_snippet"}).json()
        client.post(f"/api/v1/quiz/rounds/{created['round_id']}/answer",
                    json={"option_id": _correct_option_id(created["round_id"])})
    assert _count("taste_signals") == before


def test_a_wrong_answer_also_leaves_the_recommender_alone():
    """The tempting place to 'helpfully' log a miss as a signal."""
    _seed_library()
    before_events = _count("playback_events")
    before_signals = _count("taste_signals")
    with TestClient(_make_app()) as client:
        created = client.post("/api/v1/quiz/rounds",
                              json={"mode": "track_snippet"}).json()
        correct_id = _correct_option_id(created["round_id"])
        wrong_id = next(o["option_id"] for o in created["options"]
                        if o["option_id"] != correct_id)
        client.post(f"/api/v1/quiz/rounds/{created['round_id']}/answer",
                    json={"option_id": wrong_id})
    assert _count("playback_events") == before_events
    assert _count("taste_signals") == before_signals
