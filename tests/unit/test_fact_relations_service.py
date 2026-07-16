import json

import pytest

from app.resources.metadata_db import MetadataDB
from app.services.fact_relations.service import process_song_facts


class _IsolatedDB:
    """Fresh tmp-file MetadataDB per test (matches test_metadata_db.py's pattern)."""

    @pytest.fixture(autouse=True)
    def _isolated_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
        MetadataDB._reset_for_tests()
        MetadataDB.init()
        yield
        MetadataDB._reset_for_tests()


class FakeExtractor:
    def extract(self, fact):
        return {"entities": {"producer": [{"text": "Rick Rubin", "confidence": 0.99}]},
                "relation_extraction": {"produced_by": [
                    {"head": {"text": "album", "confidence": 0.95},
                     "tail": {"text": "Rick Rubin", "confidence": 0.99}}]},
                "sample_source": [], "sample_usage": []}


class WriterRiskExtractor:
    """Producer candidate that triage bumps to the LLM bucket (writer-risk
    words near the name -- same trap as test_producer_writer_words_push_to_llm
    in test_fact_relations_triage.py)."""

    def extract(self, fact):
        return {"entities": {"producer": [{"text": "Sia Furler", "confidence": 0.99}]},
                "relation_extraction": {"produced_by": [
                    {"head": {"text": "the song", "confidence": 0.95},
                     "tail": {"text": "Sia Furler", "confidence": 0.99}}]},
                "sample_source": [], "sample_usage": []}


class Boom(Exception):
    pass


class TestProcessSongFacts(_IsolatedDB):
    @pytest.mark.unit
    def test_process_writes_producers(self, tmp_path):
        MetadataDB.add_song_facts_batch(
            "a-b", "acct_test", ["The album was produced by Rick Rubin."], source="test",
        )
        res = process_song_facts(
            "a-b", ["The album was produced by Rick Rubin."],
            "a b", "a", MetadataDB, ask_llm_fn=lambda *a, **k: None,
            extractor=FakeExtractor(),
        )
        assert res["producers"] == ["Rick Rubin"]

        conn = MetadataDB._connect()
        row = conn.execute(
            "SELECT producers, samples_json FROM songs WHERE slug = ?", ("a-b",)
        ).fetchone()
        assert json.loads(row[0]) == ["Rick Rubin"]
        assert json.loads(row[1]) == {"samples": [], "sampled_by": []}

    @pytest.mark.unit
    def test_llm_bucket_uses_ask_llm_fn_and_merges(self):
        fact = "The song was written by Sia Furler."
        MetadataDB.add_song_facts_batch("x-y", "acct_test", [fact], source="test")
        calls = []

        def fake_llm(messages):
            calls.append(messages)
            # Confirm the LLM leg does NOT think Sia Furler is a producer here
            # (she's a writer per the fact) -- exercise the merge path with an
            # empty LLM producers list plus a distinct LLM-only sample.
            return {"producers": [], "samples": [{"song": "Other Song", "artist": "Someone"}],
                    "sampled_by": []}

        res = process_song_facts(
            "x-y", [fact], "x y", "some-artist", MetadataDB, ask_llm_fn=fake_llm,
            extractor=WriterRiskExtractor(),
        )
        assert calls, "expected the LLM leg to be invoked for the LLM-bucket producer candidate"
        assert "[Person: Sia Furler]" in calls[0][1]["content"]
        # Triage alone never confirms her as producer (writer-risk); the LLM
        # leg also says no producers -> final producers list stays empty.
        assert res["producers"] == []
        assert {"song": "Other Song", "artist": "Someone"} in res["samples"]

    @pytest.mark.unit
    def test_llm_unavailable_still_writes_as_is(self):
        MetadataDB.add_song_facts_batch(
            "x-y", "acct_test", ["The album was produced by Rick Rubin."], source="test",
        )

        def failing_llm(messages):
            raise Boom("LLM unreachable")

        res = process_song_facts(
            "x-y", ["The album was produced by Rick Rubin."],
            "x y", "some-artist", MetadataDB, ask_llm_fn=failing_llm,
            extractor=FakeExtractor(),
        )
        # AS_IS producer must still land even though ask_llm_fn always raises.
        assert res["producers"] == ["Rick Rubin"]

    @pytest.mark.unit
    def test_extractor_failure_on_one_fact_does_not_abort_others(self):
        MetadataDB.add_song_facts_batch(
            "a-b", "acct_test",
            ["broken fact", "The album was produced by Rick Rubin."],
            source="test",
        )

        class FlakyExtractor:
            def extract(self, fact):
                if fact == "broken fact":
                    raise Boom("nlp blew up")
                return FakeExtractor().extract(fact)

        res = process_song_facts(
            "a-b", ["broken fact", "The album was produced by Rick Rubin."],
            "a b", "a", MetadataDB, ask_llm_fn=lambda *a, **k: None,
            extractor=FlakyExtractor(),
        )
        assert res["producers"] == ["Rick Rubin"]
