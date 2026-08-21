"""Round lifecycle: building, answering, expiry, audio resolution.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §6, §12.
"""
import pytest

import app.resources.metadata_db as mod
from app.resources.metadata_db import MetadataDB
from app.services.quiz import rounds
from app.services.quiz.errors import (
    AlreadyAnswered,
    NoRoundAvailable,
    RoundNotFound,
)

pytestmark = pytest.mark.unit

AXES = ("energy", "vocal_lead", "spacious", "experimental",
        "brightness", "acousticness")
COLL = "acct_1"
OTHER = "acct_2"


def _tracks(n=30):
    return [
        {
            "track_id": f"t{i}",
            "title": f"Song {i}",
            "artist": f"Artist {i}",
            "primary_artist_slug": f"artist-{i}",
            "artists": [f"Artist {i}"],
            "album": f"Album {i}",
            "year": 2000 + (i % 20),
            "genre": "rock" if i % 2 else "pop",
            "duration": 200.0,
            "cover_art_path": f"/covers/t{i}.jpg",
            "sonic_axes": dict(zip(AXES, (0.03 * i, 0.2, 0.3, 0.4, 0.5, 0.6))),
        }
        for i in range(1, n + 1)
    ]


@pytest.fixture(autouse=True)
def reset_db(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "quiz_rounds.db")
    MetadataDB.close()
    MetadataDB.init()
    yield
    MetadataDB.close()


@pytest.fixture(autouse=True)
def library(monkeypatch):
    tracks = _tracks()
    monkeypatch.setattr(rounds, "load_library",
                        lambda qdrant_client, collection_name: tracks)
    return tracks


def _seed_plays(track_ids, collection_name=COLL, times=3):
    """Real playback events, so play counts come from production code."""
    for track_id in track_ids:
        for _ in range(times):
            MetadataDB.record_playback_event(
                session_id="s1", collection_name=collection_name,
                track_id=track_id, played_sec=180.0, total_dur=200.0,
            )


def _build(mode="track_snippet", snippet_sec=3):
    return rounds.build_round(
        qdrant_client=None, collection_name=COLL, mode=mode,
        snippet_sec=snippet_sec,
    )


def _all_played():
    _seed_plays([f"t{i}" for i in range(1, 31)])


# ── Building ─────────────────────────────────────────────────────────────────

def test_built_round_never_reveals_the_answer():
    _all_played()
    out = _build()
    blob = repr(out)
    assert "correct_option_id" not in out
    assert "track_id" not in blob
    assert len(out["options"]) == 4


def test_built_round_is_persisted_with_its_answer():
    _all_played()
    out = _build()
    row = MetadataDB.get_quiz_round(out["round_id"])
    assert row is not None
    assert row["collection_name"] == COLL
    assert row["mode"] == "track_snippet"
    assert row["track_id"] is not None
    assert row["answered_at"] is None


def test_unknown_mode_is_refused():
    _all_played()
    with pytest.raises(NoRoundAvailable):
        _build(mode="no_such_mode")


def test_mode_below_the_pool_floor_is_refused():
    _seed_plays([f"t{i}" for i in range(1, 6)])   # 5 played, floor is 20
    with pytest.raises(NoRoundAvailable):
        _build()


# ── Answering ────────────────────────────────────────────────────────────────

def _correct_option_id(round_id):
    import json
    spec = json.loads(MetadataDB.get_quiz_round(round_id)["spec_json"])
    return spec["correct_option_id"]


def _answer(round_id, option_id):
    return rounds.submit_answer(
        qdrant_client=None, collection_name=COLL, round_id=round_id,
        answer={"option_id": option_id},
    )


def test_correct_answer_is_reported_as_correct():
    _all_played()
    out = _build()
    result = _answer(out["round_id"], _correct_option_id(out["round_id"]))
    assert result["correct"] is True
    assert result["score"] == 100.0


def test_wrong_answer_is_reported_with_the_truth():
    _all_played()
    out = _build()
    correct_id = _correct_option_id(out["round_id"])
    wrong_id = next(o["option_id"] for o in out["options"]
                    if o["option_id"] != correct_id)
    result = _answer(out["round_id"], wrong_id)
    assert result["correct"] is False
    assert result["score"] == 0.0
    assert result["correct_option_id"] == correct_id
    assert result["truth"]["title"]
    assert result["truth"]["track_id"]


def test_answering_twice_raises():
    _all_played()
    out = _build()
    _answer(out["round_id"], _correct_option_id(out["round_id"]))
    with pytest.raises(AlreadyAnswered):
        _answer(out["round_id"], _correct_option_id(out["round_id"]))


def test_unknown_round_raises_not_found():
    with pytest.raises(RoundNotFound):
        _answer("does-not-exist", "abc")


def test_another_accounts_round_raises_not_found():
    """Same error as 'unknown': saying it exists but is not yours is a leak."""
    _all_played()
    out = _build()
    with pytest.raises(RoundNotFound):
        rounds.submit_answer(
            qdrant_client=None, collection_name=OTHER,
            round_id=out["round_id"], answer={"option_id": "abc"},
        )


# ── Skill bookkeeping ────────────────────────────────────────────────────────

def test_a_correct_answer_raises_skill():
    _all_played()
    before = MetadataDB.get_quiz_skill(COLL, "track_snippet")["skill"]
    out = _build()
    _answer(out["round_id"], _correct_option_id(out["round_id"]))
    after = MetadataDB.get_quiz_skill(COLL, "track_snippet")
    assert after["skill"] > before
    assert after["n_answered"] == 1


def test_a_wrong_answer_lowers_skill():
    _all_played()
    out = _build()
    correct_id = _correct_option_id(out["round_id"])
    wrong_id = next(o["option_id"] for o in out["options"]
                    if o["option_id"] != correct_id)
    _answer(out["round_id"], wrong_id)
    assert MetadataDB.get_quiz_skill(COLL, "track_snippet")["skill"] < 0.5


def test_an_expired_round_scores_wrong_and_leaves_skill_alone():
    """Walking off to make tea must not quietly lower your difficulty."""
    _all_played()
    out = _build()
    conn = MetadataDB._connect()
    conn.execute("UPDATE quiz_rounds SET expires_at = 0 WHERE round_id = ?",
                 (out["round_id"],))
    conn.commit()

    result = _answer(out["round_id"], _correct_option_id(out["round_id"]))
    assert result["correct"] is False
    assert result["expired"] is True
    skill = MetadataDB.get_quiz_skill(COLL, "track_snippet")
    assert skill["skill"] == 0.5
    assert skill["n_answered"] == 0


# ── Modes listing ────────────────────────────────────────────────────────────

def test_modes_report_pool_and_availability():
    _all_played()
    infos = rounds.list_modes(qdrant_client=None, collection_name=COLL)
    entry = next(i for i in infos if i["key"] == "track_snippet")
    assert entry["pool_size"] == 30
    assert entry["available"] is True


def test_thin_library_marks_the_mode_unavailable():
    _seed_plays([f"t{i}" for i in range(1, 6)])
    infos = rounds.list_modes(qdrant_client=None, collection_name=COLL)
    entry = next(i for i in infos if i["key"] == "track_snippet")
    assert entry["pool_size"] == 5
    assert entry["available"] is False


# ── Audio resolution ─────────────────────────────────────────────────────────

def test_audio_resolves_to_the_answer_track_and_window():
    _all_played()
    out = _build(snippet_sec=3)
    track_id, start_sec, length_sec = rounds.resolve_round_audio(
        collection_name=COLL, round_id=out["round_id"],
    )
    row = MetadataDB.get_quiz_round(out["round_id"])
    assert track_id == row["track_id"]
    assert length_sec == 3.0
    assert start_sec == out["start_sec"]


def test_audio_for_another_accounts_round_is_refused():
    _all_played()
    out = _build()
    with pytest.raises(RoundNotFound):
        rounds.resolve_round_audio(
            collection_name=OTHER, round_id=out["round_id"],
        )


def test_audio_for_an_unknown_round_is_refused():
    with pytest.raises(RoundNotFound):
        rounds.resolve_round_audio(collection_name=COLL, round_id="nope")
