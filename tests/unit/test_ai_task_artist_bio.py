"""Unit tests for the artist_bio AI-Indexing task."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service
from app.services.ai_tasks import artist_bio


class FakeQdrant:
    """Minimal qdrant stub that returns one scroll page of points."""
    def __init__(self, points):
        self._points = points
    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        if offset is None:
            return list(self._points), None
        return [], None


class FakePoint:
    def __init__(self, id, payload):
        self.id = id
        self.payload = payload


class FakeDb:
    def __init__(self, points):
        self.qdrant = FakeQdrant(points)


@pytest.fixture(autouse=True)
def _reset_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "t.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _seed_facts(slug: str, collection: str, facts: list[str]) -> None:
    """Skip the artists FK by direct insert (test-only)."""
    conn = MetadataDB.get()
    conn.execute(
        "INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?) "
        "ON CONFLICT(slug) DO NOTHING",
        (slug, slug.replace("-", " ").title(), collection),
    )
    MetadataDB.add_artist_facts_batch(slug, collection, facts, source="test")


def _make_job(collection: str = "c", lang: str = "en"):
    return ai_indexing_service.JobState(
        job_id="job-1", task_type="artist_bio",
        collection_name=collection, lang=lang, n_total=1,
    )


@pytest.mark.asyncio
async def test_run_calls_llm_with_artist_facts_and_persists_bio():
    _seed_facts("dua-lipa", "c", ["Born in London 1995", "Released Future Nostalgia 2020"])
    points = [FakePoint("p1", {"artist": "Dua Lipa", "title": "Physical"})]
    with patch.object(artist_bio, "_SYSTEM_PROMPT", "be brief"), \
         patch.object(artist_bio, "ask_llm", side_effect=lambda *a, **kw: "From London, indie-pop.") as mock_llm:
        await artist_bio.run(_make_job(), FakeDb(points), None)
    mock_llm.assert_called_once()
    assert MetadataDB.get_artist_bio("dua-lipa", "c", "en") == "From London, indie-pop."


@pytest.mark.asyncio
async def test_run_skips_artists_with_no_facts():
    points = [FakePoint("p1", {"artist": "Unknown", "title": "x"})]
    with patch.object(artist_bio, "_SYSTEM_PROMPT", "be brief"), \
         patch.object(artist_bio, "ask_llm") as mock_llm:
        await artist_bio.run(_make_job(), FakeDb(points), None)
    mock_llm.assert_not_called()
    assert MetadataDB.get_artist_bio("unknown", "c", "en") is None


@pytest.mark.asyncio
async def test_run_dedupes_artists_across_tracks():
    _seed_facts("dua-lipa", "c", ["fact"])
    points = [
        FakePoint("p1", {"artist": "Dua Lipa", "title": "A"}),
        FakePoint("p2", {"artist": "Dua Lipa", "title": "B"}),
        FakePoint("p3", {"artist": "Dua Lipa", "title": "C"}),
    ]
    with patch.object(artist_bio, "_SYSTEM_PROMPT", "be brief"), \
         patch.object(artist_bio, "ask_llm", side_effect=lambda *a, **kw: "bio") as mock_llm:
        await artist_bio.run(_make_job(), FakeDb(points), None)
    assert mock_llm.call_count == 1  # one LLM call per artist, not per track


@pytest.mark.asyncio
async def test_run_fails_fast_when_prompt_empty():
    with patch.object(artist_bio, "_SYSTEM_PROMPT", ""):
        with pytest.raises(RuntimeError, match="_SYSTEM_PROMPT is empty"):
            await artist_bio.run(_make_job(), FakeDb([]), None)
