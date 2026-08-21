"""Storage layer for the library quiz — tables and accessors.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §5.
"""
import pytest

import app.resources.metadata_db as mod
from app.resources.metadata_db import MetadataDB

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_db(tmp_path, monkeypatch):
    """Point MetadataDB at a fresh file per test.

    close() clears both the connection map and the _schema_ready memo, so
    init() genuinely rebuilds the schema for the new path instead of assuming
    a previous test's file already has it.
    """
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "quiz_test.db")
    MetadataDB.close()
    MetadataDB.init()
    yield
    MetadataDB.close()


def _mk_round(round_id="r1", *, collection_name="acct_1", mode="track_snippet",
              track_id="t1", created_at=None, expires_at=9999.0):
    MetadataDB.create_quiz_round(
        round_id=round_id,
        collection_name=collection_name,
        mode=mode,
        track_id=track_id,
        spec_json='{"options": []}',
        expires_at=expires_at,
        created_at=created_at,
    )


# ── Schema ───────────────────────────────────────────────────────────────────

def test_init_creates_quiz_tables():
    conn = MetadataDB._connect()
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert {"quiz_rounds", "quiz_skill", "quiz_streak"} <= names


# ── Rounds ───────────────────────────────────────────────────────────────────

def test_created_round_reads_back_unanswered():
    _mk_round()
    row = MetadataDB.get_quiz_round("r1")
    assert row["collection_name"] == "acct_1"
    assert row["mode"] == "track_snippet"
    assert row["track_id"] == "t1"
    assert row["answered_at"] is None
    assert row["correct"] is None


def test_get_quiz_round_returns_none_for_unknown_id():
    assert MetadataDB.get_quiz_round("nope") is None


def test_answer_records_verdict():
    _mk_round()
    assert MetadataDB.answer_quiz_round(
        round_id="r1", answer_json='{"choice": "a"}', correct=True, score=100.0,
    ) is True
    row = MetadataDB.get_quiz_round("r1")
    assert row["correct"] == 1
    assert row["score"] == 100.0
    assert row["answered_at"] is not None


def test_second_answer_is_rejected_and_does_not_overwrite():
    _mk_round()
    MetadataDB.answer_quiz_round(
        round_id="r1", answer_json='{"choice": "a"}', correct=True, score=100.0,
    )
    assert MetadataDB.answer_quiz_round(
        round_id="r1", answer_json='{"choice": "b"}', correct=False, score=0.0,
    ) is False
    row = MetadataDB.get_quiz_round("r1")
    assert row["correct"] == 1
    assert row["score"] == 100.0


# ── Skill ────────────────────────────────────────────────────────────────────

def test_skill_defaults_to_starter_band_for_new_collection():
    s = MetadataDB.get_quiz_skill("acct_1", "track_snippet")
    assert s["skill"] == 0.5
    assert s["n_answered"] == 0
    assert (s["band_lo"], s["band_hi"]) == (60.0, 100.0)
    assert s["out_of_band"] == 0


def test_saved_skill_reads_back():
    MetadataDB.save_quiz_skill(
        collection_name="acct_1", mode="track_snippet", skill=0.9,
        n_answered=12, band_lo=10.0, band_hi=45.0, out_of_band=2,
    )
    s = MetadataDB.get_quiz_skill("acct_1", "track_snippet")
    assert s["skill"] == 0.9
    assert s["n_answered"] == 12
    assert (s["band_lo"], s["band_hi"]) == (10.0, 45.0)
    assert s["out_of_band"] == 2


def test_skill_is_scoped_per_collection():
    MetadataDB.save_quiz_skill(
        collection_name="acct_1", mode="track_snippet", skill=0.9,
        n_answered=12, band_lo=10.0, band_hi=45.0, out_of_band=0,
    )
    other = MetadataDB.get_quiz_skill("acct_2", "track_snippet")
    assert other["skill"] == 0.5
    assert other["n_answered"] == 0


# ── Anti-repeat ──────────────────────────────────────────────────────────────

def test_recent_track_ids_excludes_rounds_before_the_window():
    now = 1000.0
    _mk_round("old", track_id="t_old", created_at=now - 500)
    _mk_round("new", track_id="t_new", created_at=now - 10)
    got = MetadataDB.recent_quiz_track_ids(
        "acct_1", "track_snippet", since_ts=now - 100,
    )
    assert got == {"t_new"}


def test_recent_track_ids_is_scoped_per_collection():
    now = 1000.0
    _mk_round("mine", collection_name="acct_1", track_id="t_mine", created_at=now)
    _mk_round("theirs", collection_name="acct_2", track_id="t_theirs", created_at=now)
    got = MetadataDB.recent_quiz_track_ids(
        "acct_1", "track_snippet", since_ts=now - 100,
    )
    assert got == {"t_mine"}


def test_recent_track_ids_honours_limit_newest_first():
    now = 1000.0
    for i in range(5):
        _mk_round(f"r{i}", track_id=f"t{i}", created_at=now - (10 - i))
    got = MetadataDB.recent_quiz_track_ids(
        "acct_1", "track_snippet", limit=2, since_ts=now - 100,
    )
    assert got == {"t3", "t4"}
