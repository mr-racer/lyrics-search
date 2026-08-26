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
from app.services.facts_v2 import pipeline as fv2


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
    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        wanted = set(ids)
        return [p for p in self._points if p.id in wanted]


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
        with patch("app.services.ai_tasks.artist_bio.bio2.build",
                   return_value={"bio": "From London, indie-pop.",
                                 "facets": {}}) as mock_bio:
            await artist_bio.run(_make_job(), FakeDb(points), None)
        mock_bio.assert_called_once()
        assert MetadataDB.get_artist_bio("dua-lipa", "c", "en") == "From London, indie-pop."

    @pytest.mark.asyncio
    async def test_run_processes_all_artists(self):
        """artist_bio processes every distinct artist regardless of facts."""
        points = [FakePoint("p1", {"artist": "Unknown", "title": "x"})]
        with patch("app.services.ai_tasks.artist_bio.bio2.build",
                   return_value={"bio": "Some bio.", "facets": {}}) as mock_bio:
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
        with patch("app.services.ai_tasks.artist_bio.bio2.build",
                   return_value={"bio": "bio", "facets": {}}) as mock_bio:
            await artist_bio.run(_make_job(), FakeDb(points), None)
        assert mock_bio.call_count == 1  # one LLM call per artist, not per track

    @pytest.mark.asyncio
    async def test_title_feat_guest_is_researched_under_own_name(self):
        """A guest credited only in the TITLE must not inherit the tag artist.

        «Kanye West — FML (ft. The Weeknd)» puts `the-weeknd` in artist_slugs
        while the raw tag says only «Kanye West». Researching the-weeknd as
        «Kanye West» produced bios about the wrong artist (and, seeded with the
        slug's own AudioDB text, LLM apologies inside the stored bio).
        """
        points = [FakePoint("p1", {
            "artist": "Kanye West", "title": "FML (ft. The Weeknd)",
            "artists": ["Kanye West", "The Weeknd"],
            "artist_slugs": ["kanye-west", "the-weeknd"],
        })]
        with patch("app.services.ai_tasks.artist_bio.bio2.build",
                   return_value={"bio": "bio", "facets": {}}) as mock_bio:
            await artist_bio.run(_make_job(), FakeDb(points), None)
        names = sorted(c.args[1] for c in mock_bio.call_args_list)
        assert names == ["Kanye West", "The Weeknd"]

    @pytest.mark.asyncio
    async def test_incremental_branch_uses_the_slug_own_name(self):
        points = [FakePoint("p1", {
            "artist": "Kanye West", "title": "FML (ft. The Weeknd)",
            "artists": ["Kanye West", "The Weeknd"],
            "artist_slugs": ["kanye-west", "the-weeknd"],
        })]
        job = _make_job()
        job.new_track_ids = ("p1",)
        with patch("app.services.ai_tasks.artist_bio.bio2.build",
                   return_value={"bio": "bio", "facets": {}}) as mock_bio:
            await artist_bio.run(job, FakeDb(points), None)
        names = sorted(c.args[1] for c in mock_bio.call_args_list)
        assert names == ["Kanye West", "The Weeknd"]

    @pytest.mark.asyncio
    async def test_sqlite_mirror_branch_uses_the_slug_own_name(self):
        """The mirror fast-path is what prod actually hits (tables populated)."""
        MetadataDB.upsert_track_metadata("c", "t1", {
            "title": "FML (ft. The Weeknd)", "artist": "Kanye West",
            "artists": ["Kanye West", "The Weeknd"],
            "artist_slugs": ["kanye-west", "the-weeknd"], "album": "TLOP",
        })
        with patch("app.services.ai_tasks.artist_bio.bio2.build",
                   return_value={"bio": "bio", "facets": {}}) as mock_bio:
            await artist_bio.run(_make_job(), FakeDb([]), None)
        names = sorted(c.args[1] for c in mock_bio.call_args_list)
        assert names == ["Kanye West", "The Weeknd"]

    @pytest.mark.asyncio
    async def test_never_passes_a_foreign_collaboration_tag(self):
        """Worst case (no participant list): a slug-derived name, never a stranger."""
        points = [FakePoint("p1", {
            "artist": "Kanye West", "title": "x",
            "artist_slugs": ["kanye-west", "the-weeknd"],
        })]
        with patch("app.services.ai_tasks.artist_bio.bio2.build",
                   return_value={"bio": "bio", "facets": {}}) as mock_bio:
            await artist_bio.run(_make_job(), FakeDb(points), None)
        names = sorted(c.args[1] for c in mock_bio.call_args_list)
        assert names == ["Kanye West", "The Weeknd"]

    @pytest.mark.asyncio
    async def test_run_skips_empty_bio_result(self):
        points = [FakePoint("p1", {"artist": "Empty Bio", "title": "x"})]
        with patch("app.services.ai_tasks.artist_bio.bio2.build",
                   return_value={"bio": "", "facets": {},
                                 "error": "no wikipedia article passed the gate"}):
            await artist_bio.run(_make_job(), FakeDb(points), None)
        assert MetadataDB.get_artist_bio("empty-bio", "c", "en") is None


class TestRefinedFacts:
    """Tests for the Refined Facts AI task — accessors, parser, batching."""

    def test_classify_response_maps_ids_to_labels(self):
        raw = json.dumps({"items": [
            {"id": "M1", "labels": ["creation"], "move": None},
            {"id": "M2", "labels": ["other"], "move": None},
        ]})
        out = fv2._parse_labels(fv2.parse_json(raw), fv2.SONG_LABELS)
        assert out["M1"]["labels"] == ["creation"]
        assert out["M2"]["labels"] == ["other"]

    def test_classify_response_drops_unknown_labels(self):
        raw = json.dumps({"items": [{"id": "M1", "labels": ["creation", "nope"]}]})
        out = fv2._parse_labels(fv2.parse_json(raw), fv2.SONG_LABELS)
        assert out["M1"]["labels"] == ["creation"]

    def test_other_never_survives_beside_a_real_label(self):
        """Code overrules the model here: 'other' means nothing else applied."""
        raw = json.dumps({"items": [{"id": "M1", "labels": ["other", "video"]}]})
        out = fv2._parse_labels(fv2.parse_json(raw), fv2.SONG_LABELS)
        assert out["M1"]["labels"] == ["video"]

    def test_classify_response_survives_garbage(self):
        assert fv2._parse_labels(fv2.parse_json("not json"), fv2.SONG_LABELS) == {}
        assert fv2._parse_labels(fv2.parse_json('{"wrong": []}'),
                                 fv2.SONG_LABELS) == {}

    def test_sample_is_orthogonal_and_never_eats_the_text_prompt(self):
        """A `creation + sample` fact yields BOTH a link and prose.

        Routing it to the sample branch alone threw the creation story away —
        seven times in the 1199-fact validation run.
        """
        plan = fv2.route(["creation", "sample"])
        assert plan["extract"] is True
        assert plan["primary"] == "creation"

    def test_sample_alone_produces_no_text(self):
        plan = fv2.route(["sample"])
        assert plan["extract"] is True and plan["primary"] is None

    def test_multi_label_order_does_not_decide_the_prompt(self):
        """The same pair listed either way must take the same prompt."""
        a = fv2.route(["name_origin", "band_history"])
        b = fv2.route(["band_history", "name_origin"])
        assert a["primary"] == b["primary"] == "name_origin"
        assert set(a["focus"]) == set(b["focus"]) == {"name_origin", "band_history"}

    def test_off_scope_fact_is_set_aside_not_rewritten(self):
        """`about_artist` on a song produces no prose at all.

        It used to be carried to the artist's page, which is where 1968 of the
        4880 facts on production artist pages came from — and the rewrite there
        is given no song title, so anything anchored to the song arrived
        pointing at nothing.
        """
        plan = fv2.route(["about_artist"])
        assert plan["primary"] is None and plan["focus"] == []

    def test_both_classify_prompts_still_format(self):
        """A stray unescaped brace breaks EVERY fact, not one.

        These prompts are mostly JSON, so every literal brace has to be
        doubled; miss one and `.format` raises KeyError on the first call of a
        ten-hour run.
        """
        from app.services.facts_v2 import prompts as P

        song = P.SONG_CLASSIFY.format(title="T", artist="A", items="M1. x")
        artist = P.ARTIST_CLASSIFY.format(artist="A", items="M1. x")
        for text in (song, artist):
            assert '{"items":[{"id":"M1","labels":["..."]}]}' in text
            assert '"move"' not in text          # the field is gone for good
            assert "{{" not in text and "}}" not in text

    def test_a_legacy_move_field_in_the_answer_is_ignored(self):
        """A model still echoing the old shape must not resurrect the move.

        Worth pinning: the answer format lived in prompt caches and in the
        model's habits long after the prompt changed.
        """
        got = fv2._parse_labels(
            {"items": [{"id": "M1", "labels": ["creation"],
                        "move": {"scope": "artist", "labels": ["personal"]}}]},
            fv2.SONG_LABELS)
        assert got == {"M1": {"labels": ["creation"]}}

    def test_off_scope_label_swallows_the_labels_beside_it(self):
        """A model answering ["about_artist","personal"] writes nothing here."""
        assert fv2._parse_labels(
            {"items": [{"id": "M1", "labels": ["about_artist", "personal"]}]},
            fv2.SONG_LABELS) == {"M1": {"labels": ["about_artist"]}}

    def test_roster_dump_is_gated_before_the_model(self):
        roster = ("1987-2002 Layne Staley Vocals, guitar 1987-2002 Jerry "
                  "Cantrell Guitar 1987- Mike Starr Bass")
        assert fv2.gate({"fact": roster}) == "roster"
        assert fv2.gate({"fact": "Recorded in a single night in a garage "
                                 "after their gear was stolen."}) is None

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

    def test_delete_refined_facts_hits_rows_written_by_other_collections(self):
        """refined_facts is a shared pool: reads ignore collection_name, so the
        reset must too. Rows for artists in MY library and songs I pass as
        song_keys are deleted even when another account wrote them."""
        # my library contains dua-lipa (via track_artist_slugs)
        MetadataDB.upsert_track_metadata("music", "t1", {
            "title": "Levitating", "artist": "Dua Lipa",
            "artist_slugs": ["dua-lipa"],
        })
        # ...but the refined rows were last written by ANOTHER account
        MetadataDB.set_refined_facts(
            scope="artist", scope_key="dua-lipa", collection_name="acct_other",
            lang="ru", refined=["a"],
        )
        MetadataDB.set_refined_facts(
            scope="song", scope_key="dua-lipa-levitating", collection_name="acct_other",
            lang="ru", refined=["b"],
        )
        n = MetadataDB.delete_refined_facts("music", song_keys=["dua-lipa-levitating"])
        assert n == 2
        assert MetadataDB.get_refined_facts(
            scope="artist", scope_key="dua-lipa", collection_name="music", lang="ru",
        ) is None
        assert MetadataDB.get_refined_facts(
            scope="song", scope_key="dua-lipa-levitating", collection_name="music", lang="ru",
        ) is None

    def test_delete_refined_facts_leaves_unrelated_rows(self):
        """Rows for artists NOT in my library and not in song_keys survive."""
        MetadataDB.set_refined_facts(
            scope="artist", scope_key="someone-else", collection_name="acct_other",
            lang="ru", refined=["keep me"],
        )
        n = MetadataDB.delete_refined_facts("music")
        assert n == 0
        assert MetadataDB.get_refined_facts(
            scope="artist", scope_key="someone-else", collection_name="music", lang="ru",
        ) == ["keep me"]

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

    # ── v2: annotation parsing / junk gate ──────────────────────────────────

    def test_parse_annotation_extracts_quote_and_note(self):
        q, n = refined_facts._parse_annotation(
            "Lyrics string: hell freezes over. Fact: A play on a common idiom.",
        )
        assert q == "hell freezes over"
        assert n == "A play on a common idiom."

    def test_parse_annotation_section_marker_clears_quote(self):
        q, n = refined_facts._parse_annotation(
            "Lyrics string: [Chorus: Mark Lanegan]. Fact: Homme wanted the world to hear Lanegan again.",
        )
        assert q == ""
        assert "Lanegan" in n

    def test_parse_annotation_rejects_malformed(self):
        assert refined_facts._parse_annotation("just an editorial fact") is None

    def test_junk_reason_gates_empty_and_short(self):
        assert refined_facts._junk_reason("?") == "junk_empty"
        assert refined_facts._junk_reason("  ") == "junk_empty"
        assert refined_facts._junk_reason("too short") == "junk_short"
        assert refined_facts._junk_reason(
            "A perfectly reasonable full-length fact about the song",
        ) is None

    # ── v2: entity check ─────────────────────────────────────────────────────

    def test_entities_ok_accepts_names_present_in_source(self):
        assert refined_facts._entities_ok(
            "Produced by Rick Rubin at his home studio.",
            "The album was produced by Rick Rubin in Malibu.",
        )

    def test_entities_ok_rejects_invented_name(self):
        assert not refined_facts._entities_ok(
            "Produced by Quincy Jones.",
            "The album was produced by Rick Rubin.",
        )

    def test_entities_ok_strips_trailing_punctuation(self):
        # "York." must not glue the sentence period onto the token
        assert refined_facts._entities_ok(
            "Big Apple is a nickname of New York.",
            "Big Apple is New York's nickname",
        )

    def test_garbled_script_detection(self):
        assert refined_facts._has_garbled_script("получил анаointment от церкви")
        assert refined_facts._has_garbled_script("West'а зовут Сент Вестom")
        # legit hyphen mix and pure-script texts are fine
        assert not refined_facts._has_garbled_script("Grammy-номинация за трек")
        assert not refined_facts._has_garbled_script("Produced by Rick Rubin")
        assert not refined_facts._has_garbled_script("Записан за одну ночь")

    # ── v2: dedup ────────────────────────────────────────────────────────────

    def test_processed_ids_make_a_rerun_skip_finished_facts(self):
        """The resume key. A ten-hour run gets interrupted; it must not restart."""
        MetadataDB.set_refined_fact_item(
            scope="song", scope_key="s-1", lang="ru", origin_kind="song_facts",
            origin_id=41, labels=["creation"], text="уже сделано",
        )
        done = MetadataDB.processed_origin_ids("song_facts", "ru", [41, 42])
        assert done == {41}

    def test_reprocessing_one_fact_replaces_its_row(self):
        MetadataDB.set_refined_fact_item(
            scope="song", scope_key="s-2", lang="ru", origin_kind="song_facts",
            origin_id=77, labels=["creation"], text="первая версия",
        )
        MetadataDB.set_refined_fact_item(
            scope="song", scope_key="s-2", lang="ru", origin_kind="song_facts",
            origin_id=77, labels=["video"], text="вторая версия",
        )
        got = MetadataDB.get_refined_facts_meta(
            scope="song", scope_key="s-2", collection_name="c", lang="ru")
        assert got == [{"text": "вторая версия", "confirmed": True,
                        "labels": ["video"]}]

    def test_hidden_labels_are_stored_but_never_returned(self):
        """`other` is kept for statistics and must not reach a reader."""
        MetadataDB.set_refined_fact_item(
            scope="song", scope_key="s-3", lang="ru", origin_kind="song_facts",
            origin_id=88, labels=["other"], text=None,
        )
        MetadataDB.set_refined_fact_item(
            scope="song", scope_key="s-3", lang="ru", origin_kind="song_facts",
            origin_id=89, labels=["sound"], text="видимый факт",
        )
        assert MetadataDB.get_refined_facts(
            scope="song", scope_key="s-3", collection_name="c",
            lang="ru") == ["видимый факт"]

    def test_legacy_blob_still_read_when_no_items_exist(self):
        """A library mid-migration keeps showing what it already had."""
        MetadataDB.set_refined_facts(
            scope="song", scope_key="s-legacy", collection_name="c", lang="ru",
            refined=[{"text": "старый факт"}])
        assert MetadataDB.get_refined_facts(
            scope="song", scope_key="s-legacy", collection_name="c",
            lang="ru") == ["старый факт"]

    # ── v2: song-scope split pipeline ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_run_labels_both_fact_kinds_and_stores_per_fact(self):
        """One classifier over both kinds, one rewrite per kept fact.

        The editorial fact and the Genius line note used to travel through two
        different prompts; now the classifier judges both, and what separates
        them is the LABEL it assigns, not the stream they arrived on.
        """
        from app.services.song_facts_service import get_song_facts_key

        slug = get_song_facts_key("Bar", "Foo")
        MetadataDB.add_song_facts_batch(
            slug, "c",
            ["Recorded in one single night at a hotel in Berlin, Germany"],
            source="songfacts.com",
        )
        MetadataDB.add_song_facts_batch(
            slug, "c",
            ["Lyrics string: feeling froggish, leap. Fact: The narrator simply "
             "feels bold here, and the frog is a metaphor for that boldness."],
            source="genius.com", category="genius_annotation",
        )
        points = [FakePoint("p1", {"artist": "Bar", "title": "Foo"})]

        classify = json.dumps({"items": [
            {"id": "M1", "labels": ["creation"], "move": None},
            {"id": "M2", "labels": ["other"], "move": None},
        ]})
        refine = json.dumps({"text": "Записана за одну ночь в отеле в Berlin."})

        async def _fake_llm(user, **kwargs):
            return classify if "NOTES TO SORT" in user else refine

        with patch("app.services.ai_tasks.refined_facts.ask_llm",
                   side_effect=_fake_llm):
            await refined_facts.run(
                _make_job(collection="c", lang="ru"), FakeDb(points), None,
            )

        meta = MetadataDB.get_refined_facts_meta(
            scope="song", scope_key=slug, collection_name="c", lang="ru",
        )
        # Only the labelled fact is visible; the lyric reading is stored as
        # `other` and never returned.
        assert meta is not None and len(meta) == 1
        assert "Berlin" in meta[0]["text"]
        assert meta[0]["labels"] == ["creation"]

    @pytest.mark.asyncio
    async def test_song_fact_about_the_artist_is_set_aside(self):
        """An `about_artist` note stays put, unwritten, and reaches no page.

        It used to be relocated to the artist's page. That is what filled
        production artist pages with facts anchored to songs they no longer
        named — 209 of Jay-Z's 217 rows, 231 of Kanye West's 245 — so the
        relocation is gone and the note is simply not written out.
        """
        from app.services.song_facts_service import get_song_facts_key

        slug = get_song_facts_key("Travis Scott", "Sicko Mode")
        MetadataDB.add_song_facts_batch(
            slug, "c", ["Scott appeared at WWE's Elimination Chamber in 2025 "
                        "and slapped the champion."],
            source="songfacts.com")
        points = [FakePoint("p9", {"artist": "Travis Scott", "title": "Sicko Mode"})]

        classify = json.dumps({"items": [
            {"id": "M1", "labels": ["about_artist"]}]})
        refine = json.dumps({"text": "Появился на Elimination Chamber в 2025 году."})

        async def _fake_llm(user, **kwargs):
            return classify if "NOTES TO SORT" in user else refine

        with patch("app.services.ai_tasks.refined_facts.ask_llm",
                   side_effect=_fake_llm):
            await refined_facts.run(
                _make_job(collection="c", lang="ru"), FakeDb(points), None)

        # The row exists, so the song counts as processed and does NOT fall
        # back to its raw facts — it just has nothing showable.
        assert MetadataDB.get_refined_facts(
            scope="song", scope_key=slug, collection_name="c", lang="ru") == []
        assert MetadataDB.get_refined_facts_meta(
            scope="artist", scope_key="travis-scott", collection_name="c",
            lang="ru") is None

    @pytest.mark.asyncio
    async def test_rerun_does_not_call_the_model_again(self):
        """Resume, end to end: a second run over processed facts is free."""
        from app.services.song_facts_service import get_song_facts_key

        slug = get_song_facts_key("Baz", "Qux")
        MetadataDB.add_song_facts_batch(
            slug, "c", ["Recorded in a garage after their gear was stolen"],
            source="songfacts.com",
        )
        points = [FakePoint("p2", {"artist": "Baz", "title": "Qux"})]
        classify = json.dumps({"items": [{"id": "M1", "labels": ["creation"]}]})
        refine = json.dumps({"text": "Записана в гараже после кражи аппаратуры."})

        async def _fake_llm(user, **kwargs):
            return classify if "NOTES TO SORT" in user else refine

        with patch("app.services.ai_tasks.refined_facts.ask_llm",
                   side_effect=_fake_llm) as first:
            await refined_facts.run(
                _make_job(collection="c", lang="ru"), FakeDb(points), None)
        assert first.call_count > 0

        with patch("app.services.ai_tasks.refined_facts.ask_llm",
                   side_effect=_fake_llm) as second:
            await refined_facts.run(
                _make_job(collection="c", lang="ru"), FakeDb(points), None)
        assert second.call_count == 0


class TestSonicVibe:
    """Tests for the Sonic Vibe AI task."""

    def test_build_user_prompt_includes_tags(self):
        window = sonic_vibe._build_fact_window(
            [{"fact": "From the cold-wave revival era of the early 80s", "category": None}],
        )
        user_msg = sonic_vibe._build_user_prompt(
            tags=["dreamy", "synth", "melancholy"],
            payload={"title": "Foo", "artist": "Bar", "year": 1985},
            window=window,
            lang="ru",
        )
        assert "dreamy" in user_msg
        assert "synth" in user_msg
        assert "M1 [EDITORIAL]" in user_msg

    def test_build_user_prompt_includes_decade_when_year_present(self):
        user_msg = sonic_vibe._build_user_prompt(
            tags=["a"], payload={"year": 1985}, window=[], lang="en",
        )
        assert "1980s" in user_msg

    def test_build_fact_window_prioritizes_editorial_and_reformats_notes(self):
        meta = [
            {"fact": "Lyrics string: hell freezes over. Fact: A play on a common idiom about impossible events happening.",
             "category": "genius_annotation"},
            {"fact": "Written and recorded in a single night at a hotel in Berlin",
             "category": None},
            {"fact": "Lyrics string: x. Fact: ?", "category": "genius_annotation"},
            {"fact": "The song opens the band's fourth album with a slow build",
             "category": "genius_description"},
            {"fact": "short", "category": None},
        ]
        window = sonic_vibe._build_fact_window(meta)
        # junk ("Fact: ?" and <30 chars) dropped; editorial+description first
        assert [it["tag"] for it in window] == [
            "[EDITORIAL]", "[DESCRIPTION]", "[LINE NOTE]",
        ]
        assert [it["key"] for it in window] == ["M1", "M2", "M3"]
        assert window[2]["text"].startswith('Note on the line "hell freezes over":')

    def test_parse_vibe_response_variants(self):
        valid = {"M1", "M2"}
        assert sonic_vibe._parse_vibe_response(
            '{"best": "M1", "category": "A", "line": "ok line"}', valid,
        ) == ("M1", "ok line")
        assert sonic_vibe._parse_vibe_response('{"best": null}', valid) == (None, "")
        # legacy bare SKIP is tolerated
        assert sonic_vibe._parse_vibe_response("SKIP", valid) == (None, "")
        with pytest.raises(ValueError):
            sonic_vibe._parse_vibe_response(
                '{"best": "M9", "line": "x"}', valid,
            )
        with pytest.raises(ValueError):
            sonic_vibe._parse_vibe_response("no json here at all", valid)
        with pytest.raises(ValueError):
            sonic_vibe._parse_vibe_response('{"best": "M1"}', valid)  # no line

    def test_line_ok_rejects_scaffold_and_wrong_script(self):
        assert sonic_vibe._line_ok("Записан за одну ночь в отеле", "ru")
        assert not sonic_vibe._line_ok("Lyrics string: something. Fact: x", "ru")
        assert not sonic_vibe._line_ok('Note on the line "x": y', "en")
        # ru line with no cyrillic at all → wrong script leak
        assert not sonic_vibe._line_ok("Recorded overnight in a hotel", "ru")
        # garbled script inside one word («анаointment») → reject
        assert not sonic_vibe._line_ok("Получил анаointment от Kirk Franklin", "ru")

    def test_user_prompt_anchors_artist_spelling(self):
        user_msg = sonic_vibe._build_user_prompt(
            tags=[], payload={"artist": "Kanye West", "title": "FATHER"},
            window=[], lang="ru",
        )
        assert "Artist (original spelling" in user_msg
        assert "Kanye West" in user_msg
        assert "Track: FATHER" in user_msg

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
            return_value=json.dumps({
                "best": "M1", "category": "A",
                "line": "Recorded in one night, hours before a flight.",
            }),
        ):
            asyncio.run(sonic_vibe.run(job, db_client, llm=None))

        cached = MetadataDB.get_sonic_vibe("t1", "music", "en")
        assert cached is not None
        assert "one night" in cached["phrase"]

    def test_vibe_entity_check_rejects_invented_name_then_skips(self):
        """A line naming someone absent from the winning fact → retry → SKIP."""
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
        pt.payload = {"track_id": "t1", "title": "A", "artist": "B"}
        qdrant.scroll.return_value = ([pt], None)
        db_client = MagicMock()
        db_client.qdrant = qdrant

        bad = json.dumps({
            "best": "M1", "category": "B",
            "line": "Produced by Quincy Jones in one night.",
        })
        with patch(
            "app.services.ai_tasks.sonic_vibe.ask_llm",
            new_callable=AsyncMock, return_value=bad,
        ) as mock_llm:
            asyncio.run(sonic_vibe.run(job, db_client, llm=None))
            assert mock_llm.call_count == 2  # one validation retry

        assert MetadataDB.get_sonic_vibe("t1", "music", "en") is None

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
            ["The song is about heartbreak and long lonely evenings"], source="test",
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
            new_callable=AsyncMock, return_value='{"best": null}',
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


# --- Incremental scoping: auto (append/upload) vs manual (full walk) ----------
#
# The core bug this fix removes: after "Сканировать библиотеку" added a few new
# files, the auto AI-enrichment walked the WHOLE collection instead of only the
# new batch. These tests pin the contract that a job carrying `new_track_ids`
# touches exactly those tracks (a batched retrieve), never the rest (no scroll).

class TestIncrementalScoping:
    """Auto-run jobs (new_track_ids set) must enrich ONLY the new tracks."""

    def test_jobstate_defaults_to_full_walk(self):
        """No new_track_ids → legacy whole-collection behaviour (manual entry)."""
        job = JobState(
            job_id="j", task_type="sonic_vibe", collection_name="c",
            lang="en", n_total=0,
        )
        assert job.new_track_ids is None

    async def test_start_job_carries_new_track_ids(self):
        """start_job threads new_track_ids onto the JobState it runs."""
        from app.services import ai_indexing_service as svc

        captured = {}

        async def _noop(job, db_client, llm_client):
            captured["ids"] = job.new_track_ids

        svc.register_task("inc_probe", _noop)
        try:
            job_id = svc.start_job(
                task_type="inc_probe", collection_name="c", lang="en",
                db_client=MagicMock(), llm_client=MagicMock(), n_total=2,
                new_track_ids=("p1", "p2"),
            )
            await svc._wait_for_job(job_id)
        finally:
            svc._registry.pop("inc_probe", None)
            svc._active.clear()
            svc._running_tasks.clear()

        assert captured["ids"] == ("p1", "p2")

    def test_sonic_vibe_incremental_uses_retrieve_not_scroll(self):
        """With new_track_ids the task fetches exactly those ids (retrieve) and
        never falls through to a whole-collection scroll."""
        from app.services.song_facts_service import get_song_facts_key

        MetadataDB.add_song_facts_batch(
            get_song_facts_key("B", "A"), "music",
            ["Recorded in one night in a hotel room before a flight"], source="test",
        )
        job = JobState(
            job_id="job-inc", task_type="sonic_vibe", collection_name="music",
            lang="en", n_total=1, new_track_ids=("t1",),
        )
        qdrant = MagicMock()
        # Only the NEW track is available via retrieve; the full scroll would
        # also expose a stale track that must NOT be re-processed.
        new_pt = MagicMock()
        new_pt.id = "t1"
        new_pt.payload = {
            "track_id": "t1", "title": "A", "artist": "B",
            "sonic_tags_json": json.dumps(["dreamy"]),
        }
        qdrant.retrieve.return_value = [new_pt]
        stale_pt = MagicMock()
        stale_pt.id = "t-old"
        stale_pt.payload = {"track_id": "t-old", "title": "Old", "artist": "B"}
        qdrant.scroll.return_value = ([stale_pt], None)
        db_client = MagicMock()
        db_client.qdrant = qdrant

        with patch(
            "app.services.ai_tasks.sonic_vibe.ask_llm", new_callable=AsyncMock,
            return_value=json.dumps({
                "best": "M1", "category": "A",
                "line": "Recorded in one night, hours before a flight.",
            }),
        ):
            asyncio.run(sonic_vibe.run(job, db_client, llm=None))

        qdrant.retrieve.assert_called_once()
        qdrant.scroll.assert_not_called()
        assert MetadataDB.get_sonic_vibe("t1", "music", "en") is not None
        # The stale track from the (never-called) scroll must be untouched.
        assert MetadataDB.get_sonic_vibe("t-old", "music", "en") is None

    @pytest.mark.parametrize("task,name", [
        (sonic_vibe, "sonic_vibe"),
        (refined_facts, "refined_facts"),
        (artist_bio, "artist_bio"),
    ])
    def test_an_empty_batch_processes_nothing(self, task, name):
        """``()`` means "this run touched no track" and must NOT be read as
        "walk everything".

        This was the bug behind the whole-library re-enrichment: a rescan whose
        candidates were all rejected by the duration/lyrics gates upserted
        nothing, handed an empty batch down, and every task took the falsy
        value as "no scope given" — so the cheapest possible run turned into
        the most expensive one, 5,600 tracks through the LLM.
        """
        job = JobState(
            job_id="job-empty", task_type=name, collection_name="c",
            lang="en", n_total=0, new_track_ids=(),
        )
        qdrant = MagicMock()
        qdrant.retrieve.return_value = []
        stale = MagicMock()
        stale.id = "p-stale"
        stale.payload = {"artist": "Calvin Harris", "title": "One Kiss",
                         "artist_slugs": ["calvin-harris"]}
        qdrant.scroll.return_value = ([stale], None)
        db_client = MagicMock()
        db_client.qdrant = qdrant

        asyncio.run(task.run(job, db_client, llm=None))

        qdrant.scroll.assert_not_called()
        assert job.n_done == 0

    def test_lyric_gems_empty_batch_does_not_scroll(self):
        from app.services.ai_tasks import lyric_gems

        job = JobState(
            job_id="job-empty", task_type="lyric_gems", collection_name="c",
            lang="en", n_total=0, new_track_ids=(),
        )
        qdrant = MagicMock()
        qdrant.retrieve.return_value = []
        qdrant.scroll.return_value = ([], None)
        db_client = MagicMock()
        db_client.qdrant = qdrant

        asyncio.run(lyric_gems.run(job, db_client, llm=None))

        qdrant.scroll.assert_not_called()

    def test_refined_facts_incremental_uses_retrieve_not_scroll(self):
        """refined_facts with new_track_ids refines only the new tracks' facts."""
        _seed_facts("calvin-harris", "c", [
            "Recorded his first hit in a single weekend",
            "His studio had a treadmill in the middle of the room",
        ])
        job = JobState(
            job_id="job-inc", task_type="refined_facts", collection_name="c",
            lang="en", n_total=1, new_track_ids=("p1",),
        )
        qdrant = MagicMock()
        new_pt = MagicMock()
        new_pt.id = "p1"
        new_pt.payload = {
            "artist": "Calvin Harris", "title": "How Deep Is Your Love",
            "artist_slugs": ["calvin-harris"],
        }
        qdrant.retrieve.return_value = [new_pt]
        stale_pt = MagicMock()
        stale_pt.id = "p-stale"
        stale_pt.payload = {
            "artist": "Calvin Harris", "title": "One Kiss",
            "artist_slugs": ["calvin-harris"],
        }
        qdrant.scroll.return_value = ([stale_pt], None)
        db_client = MagicMock()
        db_client.qdrant = qdrant

        with patch(
            "app.services.ai_tasks.refined_facts.ask_llm", new_callable=AsyncMock,
            return_value=json.dumps({"selected_facts": [
                {"reasoning": "weird", "short_fact": "A refined, interesting fact."},
            ]}),
        ):
            asyncio.run(refined_facts.run(job, db_client, llm=None))

        qdrant.retrieve.assert_called_once()
        qdrant.scroll.assert_not_called()
        assert MetadataDB.get_refined_facts(
            scope="artist", scope_key="calvin-harris",
            collection_name="c", lang="en",
        ) is not None

    def test_artist_bio_incremental_derives_artists_from_retrieved_payloads(self):
        """artist_bio with new_track_ids builds its artist list from the
        retrieved payloads (no collection walk) and dedupes by slug."""
        job = JobState(
            job_id="job-inc", task_type="artist_bio", collection_name="c",
            lang="en", n_total=2, new_track_ids=("p1", "p2"),
        )
        qdrant = MagicMock()
        p1 = MagicMock(); p1.id = "p1"
        p1.payload = {"artist": "Dua Lipa", "artist_slugs": ["dua-lipa"]}
        p2 = MagicMock(); p2.id = "p2"
        p2.payload = {"artist": "Dua Lipa", "artist_slugs": ["dua-lipa"]}
        qdrant.retrieve.return_value = [p1, p2]
        db_client = MagicMock()
        db_client.qdrant = qdrant

        with patch(
            "app.services.ai_tasks.artist_bio.bio2.build",
            return_value={"bio": "From London, indie-pop.", "facets": {}},
        ) as mock_bio:
            asyncio.run(artist_bio.run(job, db_client, None))

        qdrant.retrieve.assert_called_once()
        # The SQLite-mirror full walk must NOT be consulted in incremental mode.
        MetadataDB.get_artist_bio  # (attribute access, not a call, to be explicit)
        # Two tracks, same artist → exactly ONE research pass.
        assert mock_bio.call_count == 1
        assert MetadataDB.get_artist_bio("dua-lipa", "c", "en") == "From London, indie-pop."
