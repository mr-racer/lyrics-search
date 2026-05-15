"""Unit tests for the Sonic Vibe AI task."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.resources.metadata_db import MetadataDB
from app.services.ai_indexing_service import JobState
from app.services.ai_tasks import sonic_vibe


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def test_build_user_prompt_includes_tags():
    user_msg = sonic_vibe._build_user_prompt(
        tags=["dreamy", "synth", "melancholy"],
        payload={"title": "Foo", "artist": "Bar", "year": 1985},
        facts=["From the cold-wave revival era"],
        lang="ru",
    )
    assert "dreamy" in user_msg
    assert "synth" in user_msg


def test_build_user_prompt_includes_decade_when_year_present():
    user_msg = sonic_vibe._build_user_prompt(
        tags=["a"], payload={"year": 1985}, facts=[], lang="en",
    )
    assert "1980s" in user_msg


def test_validate_phrase_strips_quotes_and_caps_length():
    short = sonic_vibe._validate('"a clean phrase."')
    assert short == "a clean phrase."
    long_input = "x" * 200
    capped = sonic_vibe._validate(long_input)
    assert len(capped) <= sonic_vibe.MAX_PHRASE_CHARS + 1  # +1 for ellipsis


def test_skip_track_without_tags_and_facts():
    """If track has no sonic_tags_json AND no facts, skip — don't call LLM."""
    import asyncio

    job = JobState(
        job_id="job-1", task_type="sonic_vibe", collection_name="music",
        lang="en", n_total=1,
    )
    qdrant = MagicMock()
    # Single point without tags / song_slug.
    pt = MagicMock()
    pt.id = "t1"
    pt.payload = {"track_id": "t1", "title": "A", "artist": "B"}
    qdrant.scroll.return_value = ([pt], None)

    db_client = MagicMock()
    db_client.qdrant = qdrant

    with patch("app.services.ai_tasks.sonic_vibe.ask_llm", new_callable=AsyncMock) as mock_llm:
        asyncio.run(sonic_vibe.run(job, db_client, llm=None))
        mock_llm.assert_not_called()

    assert MetadataDB.get_sonic_vibe("t1", "music", "en") is None


def test_generates_and_caches_when_tags_present():
    """Tags + LLM response → phrase cached."""
    import asyncio

    job = JobState(
        job_id="job-1", task_type="sonic_vibe", collection_name="music",
        lang="en", n_total=1,
    )
    qdrant = MagicMock()
    pt = MagicMock()
    pt.id = "t1"
    pt.payload = {
        "track_id": "t1", "title": "A", "artist": "B", "year": 1990,
        "sonic_tags_json": json.dumps(["dreamy", "synth"]),
    }
    qdrant.scroll.return_value = ([pt], None)
    db_client = MagicMock()
    db_client.qdrant = qdrant

    with patch(
        "app.services.ai_tasks.sonic_vibe.ask_llm",
        new_callable=AsyncMock,
        return_value="A neon-lit drift through the late eighties.",
    ):
        asyncio.run(sonic_vibe.run(job, db_client, llm=None))

    cached = MetadataDB.get_sonic_vibe("t1", "music", "en")
    assert cached is not None
    assert "neon" in cached["phrase"]


def test_skip_track_already_cached():
    """If a vibe is already cached for (track, collection, lang), don't re-call LLM."""
    import asyncio

    MetadataDB.set_sonic_vibe("t1", "music", "en", "already cached phrase")

    job = JobState(
        job_id="job-2", task_type="sonic_vibe", collection_name="music",
        lang="en", n_total=1,
    )
    qdrant = MagicMock()
    pt = MagicMock()
    pt.id = "t1"
    pt.payload = {
        "track_id": "t1", "title": "A", "artist": "B",
        "sonic_tags_json": json.dumps(["dreamy"]),
    }
    qdrant.scroll.return_value = ([pt], None)
    db_client = MagicMock()
    db_client.qdrant = qdrant

    with patch("app.services.ai_tasks.sonic_vibe.ask_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "should not be saved"
        asyncio.run(sonic_vibe.run(job, db_client, llm=None))
        mock_llm.assert_not_called()

    cached = MetadataDB.get_sonic_vibe("t1", "music", "en")
    assert cached["phrase"] == "already cached phrase"  # not overwritten
