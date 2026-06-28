"""Unit tests for AI-Indexing tasks (artist_bio, refined_facts, sonic_vibe).

Consolidated from test_ai_task_artist_bio.py, test_ai_task_refined_facts.py and
test_ai_task_sonic_vibe.py. Each former module's tests live in its own class.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service
from app.services.ai_indexing_service import JobState
from app.services.ai_tasks import artist_bio, refined_facts, sonic_vibe


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Isolate MetadataDB to a fresh per-test SQLite file."""
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


# --- artist_bio helpers (module-level) ---------------------------------------

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


class TestArtistBio:
    """Tests for the artist_bio AI-Indexing task."""

    @pytest.mark.asyncio
    async def test_run_calls_llm_with_artist_facts_and_persists_bio(self):
        _seed_facts("dua-lipa", "c", ["Born in London 1995", "Released Future Nostalgia 2020"])
        points = [FakePoint("p1", {"artist": "Dua Lipa", "title": "Physical"})]
        with patch("app.services.ai_tasks.artist_bio.web_research_bio",
                   return_value="From London, indie-pop.") as mock_bio:
            await artist_bio.run(_make_job(), FakeDb(points), None)
        mock_bio.assert_called_once()
        assert MetadataDB.get_artist_bio("dua-lipa", "c", "en") == "From London, indie-pop."

    @pytest.mark.asyncio
    async def test_run_processes_all_artists(self):
        """artist_bio processes every distinct artist regardless of facts."""
        points = [FakePoint("p1", {"artist": "Unknown", "title": "x"})]
        with patch("app.services.ai_tasks.artist_bio.web_research_bio",
                   return_value="Some bio.") as mock_bio:
            await artist_bio.run(_make_job(), FakeDb(points), None)
        mock_bio.assert_called_once()
        assert MetadataDB.get_artist_bio("unknown", "c", "en") == "Some bio."

    @pytest.mark.asyncio
    async def test_run_dedupes_artists_across_tracks(self):
        _seed_facts("dua-lipa", "c", ["fact"])
        points = [
            FakePoint("p1", {"artist": "Dua Lipa", "title": "A"}),
            FakePoint("p2", {"artist": "Dua Lipa", "title": "B"}),
            FakePoint("p3", {"artist": "Dua Lipa", "title": "C"}),
        ]
        with patch("app.services.ai_tasks.artist_bio.web_research_bio",
                   return_value="bio") as mock_bio:
            await artist_bio.run(_make_job(), FakeDb(points), None)
        assert mock_bio.call_count == 1  # one LLM call per artist, not per track

    @pytest.mark.asyncio
    async def test_run_skips_empty_bio_result(self):
        points = [FakePoint("p1", {"artist": "Empty Bio", "title": "x"})]
        with patch("app.services.ai_tasks.artist_bio.web_research_bio",
                   return_value=""):
            await artist_bio.run(_make_job(), FakeDb(points), None)
        assert MetadataDB.get_artist_bio("empty-bio", "c", "en") is None


class TestRefinedFacts:
    """Tests for the Refined Facts AI task — accessors, parser, batching."""

    def test_parse_llm_response_accepts_valid_dict(self):
        raw = json.dumps({
            "selected_facts": [
                {"reasoning": "interesting", "short_fact": "Hello"},
                {"reasoning": "weird", "short_fact": "World"},
            ]
        })
        parsed = refined_facts._parse_llm_response(raw)
        assert parsed == [
            {"refined_text": "Hello"},
            {"refined_text": "World"},
        ]

    def test_parse_llm_response_accepts_empty_selected_facts(self):
        raw = json.dumps({"selected_facts": []})
        parsed = refined_facts._parse_llm_response(raw)
        assert parsed == []

    def test_parse_llm_response_rejects_malformed_json(self):
        with pytest.raises(ValueError):
            refined_facts._parse_llm_response("not json")

    def test_parse_llm_response_rejects_non_dict(self):
        with pytest.raises(ValueError):
            refined_facts._parse_llm_response(json.dumps([{"id": 1}]))

    def test_parse_llm_response_rejects_missing_selected_facts(self):
        with pytest.raises(ValueError):
            refined_facts._parse_llm_response(json.dumps({"facts": []}))

    def test_get_refined_facts_returns_none_when_absent(self):
        assert MetadataDB.get_refined_facts(
            scope="song", scope_key="t1", collection_name="music", lang="en",
        ) is None

    def test_set_and_get_refined_facts_roundtrip(self):
        MetadataDB.set_refined_facts(
            scope="song", scope_key="t1", collection_name="music", lang="en",
            refined=["short fact a", "short fact b"],
        )
        out = MetadataDB.get_refined_facts(
            scope="song", scope_key="t1", collection_name="music", lang="en",
        )
        assert out == ["short fact a", "short fact b"]

    def test_set_empty_refined_still_creates_row(self):
        """Empty list is explicit: AI judged nothing interesting. Row exists,
        /facts should return [] instead of falling back to originals."""
        MetadataDB.set_refined_facts(
            scope="song", scope_key="t1", collection_name="music", lang="en",
            refined=[],
        )
        out = MetadataDB.get_refined_facts(
            scope="song", scope_key="t1", collection_name="music", lang="en",
        )
        assert out == []  # explicit empty, not None

    def test_delete_refined_facts_returns_count(self):
        MetadataDB.set_refined_facts(
            scope="song", scope_key="t1", collection_name="music", lang="en",
            refined=["a"],
        )
        MetadataDB.set_refined_facts(
            scope="artist", scope_key="bar", collection_name="music", lang="en",
            refined=["b"],
        )
        n = MetadataDB.delete_refined_facts("music")
        assert n == 2

    @pytest.mark.asyncio
    async def test_run_refines_each_participant_of_collaboration_separately(self):
        """A multi-artist track must store refined facts under EACH participant's
        canonical slug (calvin-harris, dua-lipa) — not under a combined slug.

        The artist page queries refined facts by the per-participant canonical
        slug, so a collaboration whose refined facts land under a combined slug
        is never matched and silently falls back to raw/un-refined facts. This
        mirrors artist_bio.run, which already splits via artist_slugs.
        """
        _seed_facts("calvin-harris", "c", ["A weird studio habit of Calvin's"])
        _seed_facts("dua-lipa", "c", ["An unusual incident at a Dua Lipa show"])
        points = [FakePoint("p1", {
            "artist": "Calvin Harris, Dua Lipa",
            "artist_slugs": ["calvin-harris", "dua-lipa"],
            "title": "One Kiss",
        })]
        llm_json = json.dumps({"selected_facts": [
            {"reasoning": "weird", "short_fact": "A refined, interesting fact."}
        ]})
        with patch("app.services.ai_tasks.refined_facts.ask_llm",
                   new_callable=AsyncMock, return_value=llm_json):
            await refined_facts.run(
                _make_job(collection="c", lang="en"), FakeDb(points), None,
            )

        assert MetadataDB.get_refined_facts(
            scope="artist", scope_key="calvin-harris",
            collection_name="c", lang="en",
        ) is not None
        assert MetadataDB.get_refined_facts(
            scope="artist", scope_key="dua-lipa",
            collection_name="c", lang="en",
        ) is not None


class TestSonicVibe:
    """Tests for the Sonic Vibe AI task."""

    def test_build_user_prompt_includes_tags(self):
        user_msg = sonic_vibe._build_user_prompt(
            tags=["dreamy", "synth", "melancholy"],
            payload={"title": "Foo", "artist": "Bar", "year": 1985},
            facts=["From the cold-wave revival era"],
            lang="ru",
        )
        assert "dreamy" in user_msg
        assert "synth" in user_msg

    def test_build_user_prompt_includes_decade_when_year_present(self):
        user_msg = sonic_vibe._build_user_prompt(
            tags=["a"], payload={"year": 1985}, facts=[], lang="en",
        )
        assert "1980s" in user_msg

    def test_validate_phrase_strips_quotes_and_caps_length(self):
        short = sonic_vibe._validate('"a clean phrase."')
        assert short == "a clean phrase."
        long_input = "x" * 200
        capped = sonic_vibe._validate(long_input)
        assert len(capped) <= sonic_vibe.MAX_PHRASE_CHARS + 1  # +1 for ellipsis

    def test_skip_track_without_tags_and_facts(self):
        """If track has no sonic_tags_json AND no facts, skip — don't call LLM."""
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

    def test_generates_and_caches_when_facts_present(self):
        """Facts + a non-SKIP LLM line → phrase cached."""
        from app.services.song_facts_service import get_song_facts_key

        MetadataDB.add_song_facts_batch(
            get_song_facts_key("B", "A"), "music",
            ["Recorded in one night in a hotel room before a flight"], source="test",
        )

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
            return_value="Recorded in one night, hours before a flight.",
        ):
            asyncio.run(sonic_vibe.run(job, db_client, llm=None))

        cached = MetadataDB.get_sonic_vibe("t1", "music", "en")
        assert cached is not None
        assert "one night" in cached["phrase"]

    def test_skip_track_already_cached(self):
        """If a vibe is already cached for (track, collection, lang), don't re-call LLM."""
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

    def test_skip_response_leaves_slot_empty(self):
        """Facts present but the LLM answers SKIP → LLM is called, nothing persisted."""
        from app.services.song_facts_service import get_song_facts_key

        MetadataDB.add_song_facts_batch(
            get_song_facts_key("B", "A"), "music",
            ["The song is about heartbreak"], source="test",
        )

        job = JobState(
            job_id="job-1", task_type="sonic_vibe", collection_name="music",
            lang="en", n_total=1,
        )
        qdrant = MagicMock()
        pt = MagicMock()
        pt.id = "t1"
        pt.payload = {"track_id": "t1", "title": "A", "artist": "B"}
        qdrant.scroll.return_value = ([pt], None)
        db_client = MagicMock()
        db_client.qdrant = qdrant

        with patch(
            "app.services.ai_tasks.sonic_vibe.ask_llm",
            new_callable=AsyncMock, return_value="SKIP",
        ) as mock_llm:
            asyncio.run(sonic_vibe.run(job, db_client, llm=None))
            mock_llm.assert_called_once()  # facts existed → the model was consulted

        assert MetadataDB.get_sonic_vibe("t1", "music", "en") is None  # SKIP → no vibe

    def test_tags_without_facts_are_skipped(self):
        """Tags alone are no longer enough — no facts means no LLM call, no vibe."""
        job = JobState(
            job_id="job-1", task_type="sonic_vibe", collection_name="music",
            lang="en", n_total=1,
        )
        qdrant = MagicMock()
        pt = MagicMock()
        pt.id = "t1"
        pt.payload = {
            "track_id": "t1", "title": "A", "artist": "B",
            "sonic_tags_json": json.dumps(["dreamy", "synth"]),
        }
        qdrant.scroll.return_value = ([pt], None)
        db_client = MagicMock()
        db_client.qdrant = qdrant

        with patch("app.services.ai_tasks.sonic_vibe.ask_llm", new_callable=AsyncMock) as mock_llm:
            asyncio.run(sonic_vibe.run(job, db_client, llm=None))
            mock_llm.assert_not_called()

        assert MetadataDB.get_sonic_vibe("t1", "music", "en") is None
