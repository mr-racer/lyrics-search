"""Unit tests for the fact retrieval stack: BM25, PRF, RRF and the visibility
gate that keeps a shared vector index from leaking across accounts."""

from __future__ import annotations

import pytest

from app.services import facts_index, facts_retrieval as R


# ── BM25 ─────────────────────────────────────────────────────────────────────


def _corpus(*texts: str) -> R._Bm25:
    return R._Bm25([R.content_tokens(t) for t in texts])


class TestBm25:
    def test_a_matching_document_outscores_a_silent_one(self):
        bm = _corpus(
            "Kanye West built the track around an Expo 83 break",
            "The album cover was photographed in Los Angeles",
        )
        scores = bm.score([("expo", 1.0)])
        assert scores[0] > 0
        assert scores[1] == 0

    def test_a_rare_term_outweighs_a_common_one(self):
        bm = _corpus(
            "producer Mike Dean mixed the record",
            "producer notes from the session",
            "producer credits on the sleeve",
        )
        rare = bm.score([("dean", 1.0)])
        common = bm.score([("producer", 1.0)])
        assert rare[0] > common[0]

    def test_a_cyrillic_name_finds_its_latin_spelling(self):
        """The whole cross-script problem in one assertion: the query is
        Russian, the source fact is English."""
        bm = _corpus("Michael Jackson wrote it thinking of Diana Ross",
                     "The video was shot on a soundstage")
        scores = bm.score([("диана", 1.0)])
        assert scores[0] > 0
        assert scores[1] == 0

    def test_unknown_terms_score_nothing_rather_than_raising(self):
        bm = _corpus("a fact about a record")
        assert bm.score([("nonexistentterm", 1.0)]) == [0.0]

    def test_length_normalisation_favours_the_tighter_document(self):
        short = "Expo 83 sample"
        padded = "Expo 83 sample " + " ".join(f"filler{i}" for i in range(60))
        bm = _corpus(short, padded)
        scores = bm.score([("expo", 1.0)])
        assert scores[0] > scores[1]

    def test_an_empty_corpus_is_harmless(self):
        bm = R._Bm25([])
        assert bm.score([("anything", 1.0)]) == []


class TestPseudoRelevanceFeedback:
    def test_terms_are_lifted_from_the_seed_documents(self):
        bm = _corpus(
            "Kanye found the Expo 83 break on a Chicago record run",
            "Completely unrelated text about camera lenses and tripods",
        )
        terms = dict(R._prf_terms(bm, [0]))
        assert any(t in terms for t in ("expo", "chicago", "kanye"))
        assert "tripods" not in terms

    def test_lifted_terms_weigh_less_than_the_users_own(self):
        bm = _corpus("Kanye found the Expo 83 break in Chicago")
        assert all(w == R.PRF_WEIGHT for _, w in R._prf_terms(bm, [0]))
        assert R.PRF_WEIGHT < 1.0

    def test_expansion_is_bounded(self):
        bm = _corpus(" ".join(f"word{i}" for i in range(200)))
        assert len(R._prf_terms(bm, [0])) <= R.PRF_TERMS

    def test_out_of_range_seeds_are_ignored(self):
        bm = _corpus("one short fact about a song")
        assert R._prf_terms(bm, [17, -3]) == []

    def test_prf_rescues_a_query_that_shares_no_words(self):
        """A Russian query against English facts scores zero on its own terms.
        Expansion from the top dense hit is what gives BM25 anything to say."""
        bm = _corpus(
            "Michael Jackson wrote it thinking of his mentor Diana Ross",
            "The sleeve was designed by a Los Angeles studio",
        )
        query_only = bm.score([(t, 1.0) for t in R.content_tokens(
            "он думал о своей наставнице")])
        assert max(query_only) == 0

        expanded = bm.score([(t, 1.0) for t in R.content_tokens(
            "он думал о своей наставнице")] + R._prf_terms(bm, [0]))
        assert expanded[0] > expanded[1]


class TestRrf:
    def test_agreement_between_rankings_wins(self):
        fused = R._rrf([2, 0, 1], [2, 1, 0])
        assert max(fused, key=fused.get) == 2

    def test_a_document_only_one_ranking_saw_still_places(self):
        """BM25 stays silent on most documents; a dense-only hit must survive."""
        fused = R._rrf([0, 1], [5])
        assert set(fused) == {0, 1, 5}
        # Top of the only list that saw it ties with top of the other list —
        # RRF ranks by position, and neither ranking outranks the other.
        assert fused[5] == fused[0] > fused[1]

    def test_appearing_in_both_rankings_beats_appearing_in_one(self):
        fused = R._rrf([0, 1], [1])
        assert fused[1] > fused[0]

    def test_empty_rankings_fuse_to_nothing(self):
        assert R._rrf([], []) == {}


# ── the visibility gate ──────────────────────────────────────────────────────


class TestVisibilityGate:
    def test_the_subjects_own_facts_never_need_a_lookup(self, monkeypatch):
        """The caller already has access to the thing it asked about."""
        called = []

        def _never(kind, slugs, collection_name):
            called.append(kind)
            return set()

        monkeypatch.setattr(
            "app.resources.metadata_db.MetadataDB.filter_visible_slugs", _never)
        hits = [{"kind": "song", "slug": "mine", "row_id": 1}]
        assert R._drop_invisible(hits, "acct_x", {"mine"}) == hits
        assert called == []

    def test_a_neighbour_from_another_account_is_dropped(self, monkeypatch):
        monkeypatch.setattr(
            "app.resources.metadata_db.MetadataDB.filter_visible_slugs",
            lambda kind, slugs, collection_name: {"visible"},
        )
        hits = [
            {"kind": "song", "slug": "visible", "row_id": 1},
            {"kind": "song", "slug": "someone-elses", "row_id": 2},
        ]
        out = R._drop_invisible(hits, "acct_x", set())
        assert [h["slug"] for h in out] == ["visible"]

    def test_a_broken_visibility_table_fails_closed(self, monkeypatch):
        """An unreadable table must not become "show everything from everyone"."""
        def _boom(kind, slugs, collection_name):
            raise RuntimeError("db is gone")

        monkeypatch.setattr(
            "app.resources.metadata_db.MetadataDB.filter_visible_slugs", _boom)
        hits = [{"kind": "song", "slug": "someone-elses", "row_id": 2}]
        assert R._drop_invisible(hits, "acct_x", set()) == []


class TestJoinTexts:
    def test_text_comes_from_sqlite_not_from_qdrant(self, monkeypatch):
        monkeypatch.setattr(
            "app.resources.metadata_db.MetadataDB.get_facts_by_ids",
            lambda kind, ids: {
                7: {"fact": "the story", "source": "songfacts",
                    "category": "", "slug": "a-song"},
            },
        )
        out = R._join_texts([{"kind": "song", "row_id": 7, "slug": "", "score": 0.9}])
        assert out[0]["text"] == "the story"
        assert out[0]["slug"] == "a-song"

    def test_a_hit_whose_row_vanished_is_dropped(self, monkeypatch):
        monkeypatch.setattr(
            "app.resources.metadata_db.MetadataDB.get_facts_by_ids",
            lambda kind, ids: {},
        )
        assert R._join_texts([{"kind": "song", "row_id": 7, "slug": "", "score": 0.9}]) == []


# ── point ids ────────────────────────────────────────────────────────────────


class TestPointIds:
    def test_song_and_artist_rows_with_the_same_id_do_not_collide(self):
        """song_facts.id and artist_facts.id are independent autoincrements."""
        assert facts_index.point_id("song", 1) != facts_index.point_id("artist", 1)

    def test_ids_are_stable_so_reindexing_overwrites(self):
        assert facts_index.point_id("song", 42) == facts_index.point_id("song", 42)


# ── the entry point degrades instead of raising ──────────────────────────────


class TestRetrieveDegrades:
    @pytest.mark.parametrize("query", ["", "   ", None])
    def test_an_empty_query_returns_nothing(self, query):
        assert R.retrieve(object(), collection_name="acct_x", query=query,
                          subject_slugs={"song": "s"}) == []

    def test_no_subject_slugs_returns_nothing(self, monkeypatch):
        monkeypatch.setattr(facts_index, "index_entity",
                            lambda *a, **k: 0)
        monkeypatch.setattr(facts_index, "search", lambda *a, **k: [])
        assert R.retrieve(object(), collection_name="acct_x",
                          query="what does this mean", subject_slugs={}) == []
