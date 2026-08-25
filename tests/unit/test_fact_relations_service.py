import json

import pytest

from app.resources.metadata_db import MetadataDB
from app.services.fact_relations.gates import SongContext, apply_gates
from app.services.fact_relations.service import collect_claims, process_song_facts


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


class SampleExtractor:
    """A sample candidate in the LLM bucket (no SOURCE_CUE match on its own)."""

    def extract(self, fact):
        return {
            "entities": {"producer": [], "sampled_song": [{"text": "Bound", "confidence": 0.9}]},
            "relation_extraction": {"samples": [
                {"head": {"text": "the track", "confidence": 0.9},
                 "tail": {"text": "Bound", "confidence": 0.6}}]},
            "sample_source": [{"song": "Bound", "artist": "Ponderosa Twins Plus One"}],
            "sample_usage": [],
        }


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
        # samples_json belongs to facts_v2 now and must be left alone. Writing
        # it from here blanked the sample cache of every song this task saw.
        assert row[1] is None

    @pytest.mark.unit
    def test_llm_bucket_uses_ask_llm_fn_and_merges(self):
        fact = "The song was written by Sia Furler."
        MetadataDB.add_song_facts_batch("x-y", "acct_test", [fact], source="test")
        calls = []

        def fake_llm(messages):
            calls.append(messages)
            # Confirm the LLM leg does NOT think Sia Furler is a producer here
            # (she's a writer per the fact).
            return {"producers": [], "links": []}

        res = process_song_facts(
            "x-y", [fact], "x y", "some-artist", MetadataDB, ask_llm_fn=fake_llm,
            extractor=WriterRiskExtractor(),
        )
        assert calls, "expected the LLM leg to be invoked for the LLM-bucket producer candidate"
        assert "[Person: Sia Furler]" in calls[0][1]["content"]
        # Triage alone never confirms her as producer (writer-risk); the LLM
        # leg also says no producers -> final producers list stays empty.
        assert res["producers"] == []

    @pytest.mark.unit
    def test_production_path_never_writes_sample_links(self):
        """Samples have one owner now, and it is not this pipeline.

        This function used to persist with ``replace_sample_links`` — a
        delete-then-insert — and ran AFTER ``refined_facts`` in the auto
        pipeline, so it silently replaced every link facts_v2 had extracted.
        """
        fact = 'The track samples "Bound" by Ponderosa Twins Plus One.'
        MetadataDB.add_song_facts_batch(
            "s-1", "acct_test", [fact], title="Bound 2",
            artist_slug="kanye-west", source="test",
        )
        MetadataDB.replace_sample_links("acct_test", "s-1", [{
            "direction": "source", "dst_key": "ponderosa|bound",
            "dst_title": "Bound", "dst_artist": "Ponderosa Twins Plus One",
            "dst_slug": None, "relation": "sample",
        }])

        def eager_llm(messages):
            return {"producers": [], "links": [{
                "song": "Something Else", "artist": "Someone Else",
                "direction": "source", "relation": "sample",
            }]}

        res = process_song_facts(
            "s-1", [fact], "Bound 2", "kanye-west", MetadataDB,
            ask_llm_fn=eager_llm, extractor=SampleExtractor(),
            ctx=SongContext(slug="s-1", title="Bound 2", artist="kanye-west",
                            collection_name="acct_test"),
        )
        assert "samples" not in res
        stored = MetadataDB.get_sample_links("acct_test", "s-1")
        assert [e["song"] for e in stored["samples"]] == ["Bound"]

    @pytest.mark.unit
    def test_dry_run_leg_still_classifies_samples(self):
        """The same fact shape, two classifier verdicts, two outcomes.

        The sampling leg is no longer reached from production, so it is tested
        through the pair ``scripts/dry_run_relations.py`` calls — which is now
        its only caller.
        """
        fact = 'The track samples "Bound" by Ponderosa Twins Plus One.'
        ctx = SongContext(slug="s-1", title="Bound 2", artist="kanye-west")

        def llm(relation):
            def _call(messages):
                return {"producers": [], "links": [{
                    "song": "Bound", "artist": "Ponderosa Twins Plus One",
                    "direction": "source", "relation": relation,
                }]}
            return _call

        claims = collect_claims([fact], "Bound 2", "kanye-west",
                                llm("sample"), SampleExtractor())
        kept, _rejected = apply_gates(claims["links"], ctx)
        assert [ln["dst_title"] for ln in kept] == ["Bound"]

        claims = collect_claims([fact], "Bound 2", "kanye-west",
                                llm("lyrical_reference"), SampleExtractor())
        kept, _rejected = apply_gates(claims["links"], ctx)
        assert kept == []

    @pytest.mark.unit
    def test_production_leg_asks_nothing_about_samples(self):
        """`samples=False` also means no Song/Artist candidates in the prompt.

        Running the leg anyway would pay a second LLM call per sampling fact
        for an answer that is then thrown away.
        """
        fact = 'The track samples "Bound" by Ponderosa Twins Plus One.'
        calls = []

        def spy_llm(messages):
            calls.append(messages)
            return {"producers": [], "links": []}

        process_song_facts(
            "s-2", [fact], "Bound 2", "kanye-west", MetadataDB,
            ask_llm_fn=spy_llm, extractor=SampleExtractor(),
        )
        assert not calls, "no LLM call is warranted: the only candidates were samples"

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
    def test_reverse_link_is_derived_not_stored(self):
        """The half of the feature that was missing entirely.

        A link is written once, on the song whose fact stated it. Before this,
        only 32 of 274 in-library links had a reverse entry (12%) — tapping
        through to the sampled track showed an empty "Sampled by".
        """
        MetadataDB.add_song_facts_batch(
            "kanye-west-bound-2", "acct_test", ["f"], title="Bound 2",
            artist_slug="kanye-west", source="test",
        )
        MetadataDB.add_song_facts_batch(
            "ponderosa-bound", "acct_test", ["f"], title="Bound",
            artist_slug="ponderosa-twins-plus-one", source="test",
        )
        MetadataDB.replace_sample_links("acct_test", "kanye-west-bound-2", [{
            "direction": "source", "dst_key": "ponderosa|bound",
            "dst_title": "Bound", "dst_artist": "Ponderosa Twins Plus One",
            "dst_slug": "ponderosa-bound", "relation": "sample",
            "src_year": 2013, "dst_year": 1971, "evidence": "…", "confidence": 0.9,
        }])

        forward = MetadataDB.get_sample_links("acct_test", "kanye-west-bound-2")
        assert [e["song"] for e in forward["samples"]] == ["Bound"]
        assert forward["sampled_by"] == []

        backward = MetadataDB.get_sample_links("acct_test", "ponderosa-bound")
        assert backward["samples"] == []
        assert [e["song"] for e in backward["sampled_by"]] == ["Bound 2"]

        # …and the read cache the player uses carries the same two sides.
        MetadataDB.rebuild_samples_cache("acct_test")
        conn = MetadataDB._connect()
        cached = json.loads(conn.execute(
            "SELECT samples_json FROM songs WHERE slug = ?", ("ponderosa-bound",),
        ).fetchone()[0])
        assert cached["sampled_by"] == [{"song": "Bound 2", "artist": "kanye west"}]

    @pytest.mark.unit
    def test_rebuild_clears_links_that_no_longer_exist(self):
        """A re-extraction that rejects a link must make it disappear."""
        MetadataDB.add_song_facts_batch(
            "a-song", "acct_test", ["f"], title="A", artist_slug="a", source="test",
        )
        MetadataDB.replace_sample_links("acct_test", "a-song", [{
            "direction": "source", "dst_key": "x|y", "dst_title": "Y",
            "dst_artist": "X", "dst_slug": None, "relation": "sample",
        }])
        MetadataDB.rebuild_samples_cache("acct_test")

        MetadataDB.replace_sample_links("acct_test", "a-song", [])
        MetadataDB.rebuild_samples_cache("acct_test")
        conn = MetadataDB._connect()
        cached = json.loads(conn.execute(
            "SELECT samples_json FROM songs WHERE slug = ?", ("a-song",),
        ).fetchone()[0])
        assert cached == {"samples": [], "sampled_by": []}

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
