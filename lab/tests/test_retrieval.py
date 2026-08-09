"""Hybrid retrieval without any models loaded.

Everything here runs on a machine with no torch, which is the point: the
degradation path is not a theoretical branch, it is what the lab does before
the models finish downloading and what production would do if a model failed
to load. It must rank, not crash, and it must not silently return nothing.

A fake hub stands in where a scored signal is needed, so fusion and the
threshold can be tested deterministically.
"""

from lab.agent.config import AgentConfig
from lab.agent.retrieval import HybridRetriever, rrf
from lab.agent.retrieval.bm25 import BM25, tokenize
from lab.agent.retrieval.facts import FactsRetriever
from lab.agent.retrieval.types import Fact

DOCS = [
    "Kanye West interrupted Taylor Swift at the 2009 MTV Video Music Awards.",
    "Bohemian Rhapsody was written by Freddie Mercury for the album A Night at the Opera.",
    "The album Graduation was released in 2007 and sold very well.",
    "Radiohead formed in Abingdon, Oxfordshire in 1985.",
]


class _EmptyHub:
    """No models at all — every leg unavailable."""

    def __init__(self, cfg=None):
        self.cfg = cfg or AgentConfig()

    def encode_dense(self, texts, *, is_query=False):
        return None

    def encode_sparse(self, texts, *, is_query=False):
        return None

    def ce_probabilities(self, query, docs):
        return None


class _CEHub(_EmptyHub):
    """No embeddings, but a cross-encoder that scores by keyword overlap."""

    def __init__(self, cfg=None, probs=None):
        super().__init__(cfg)
        self.probs = probs
        self.seen: list[tuple[str, int]] = []

    def ce_probabilities(self, query, docs):
        self.seen.append((query, len(docs)))
        if self.probs is not None:
            return list(self.probs[:len(docs)])
        q = set(tokenize(query))
        return [len(q & set(tokenize(d))) / (len(q) or 1) for d in docs]


class TestBm25:
    def test_it_ranks_the_matching_document_first(self):
        scores = BM25(DOCS).scores("taylor swift")
        assert scores.index(max(scores)) == 0

    def test_an_unmatched_query_scores_everything_zero(self):
        assert set(BM25(DOCS).scores("квантовая механика")) == {0.0}

    def test_an_empty_corpus_does_not_divide_by_zero(self):
        assert BM25([]).scores("anything") == []


class TestFusion:
    def test_rrf_rewards_agreement_between_signals(self):
        fused = rrf({"a": [3, 1, 2], "b": [3, 2, 1]})
        assert max(fused, key=fused.get) == 3

    def test_weights_shift_the_outcome(self):
        fused = rrf({"a": [1, 0], "b": [0, 1]}, weights={"a": 5.0, "b": 0.1})
        assert max(fused, key=fused.get) == 1


class TestDegradation:
    def test_with_no_models_it_still_ranks_by_bm25(self):
        r = HybridRetriever(DOCS, hub=_EmptyHub())
        assert r.signals == ["bm25"]
        best = r.search("who wrote bohemian rhapsody")[0]
        assert r.docs[best.index].startswith("Bohemian Rhapsody")

    def test_without_a_cross_encoder_the_threshold_is_skipped_not_applied(self):
        """Dropping everything because the filter cannot run would turn a
        degraded ranking into no answer at all."""
        r = HybridRetriever(DOCS, hub=_EmptyHub())
        assert len(r.search("radiohead", min_prob=0.9)) == len(DOCS)

    def test_the_threshold_applies_once_a_cross_encoder_exists(self):
        r = HybridRetriever(DOCS, hub=_CEHub(probs=[0.9, 0.5, 0.1, 0.05]))
        kept = r.search("anything", min_prob=0.2)
        assert len(kept) == 2
        assert all(k.ce_prob >= 0.2 for k in kept)

    def test_the_cross_encoder_sees_its_own_query(self):
        """The retrieval query carries filters that help recall; the CE query
        is a statement about what a good passage says."""
        hub = _CEHub()
        HybridRetriever(DOCS, hub=hub).search("kanye 2009", ce_query="the VMA incident")
        assert hub.seen[-1][0] == "the VMA incident"

    def test_an_empty_query_returns_nothing(self):
        assert HybridRetriever(DOCS, hub=_EmptyHub()).search("  ") == []

    def test_extend_adds_documents_without_losing_the_old_ones(self):
        r = HybridRetriever(DOCS[:2], hub=_EmptyHub())
        assert r.extend(DOCS[2:]) == 2
        assert len(r) == 4
        best = r.search("radiohead oxfordshire")[0]
        assert "Radiohead" in r.docs[best.index]

    def test_extending_with_nothing_is_a_no_op(self):
        r = HybridRetriever(DOCS, hub=_EmptyHub())
        assert r.extend([]) == 0
        assert len(r) == len(DOCS)


class _Source:
    def __init__(self, facts):
        self.facts = facts

    def facts_for(self, kind, slug):
        return [f for f in self.facts if f.kind == kind and f.slug == slug]


class TestFactsRetriever:
    def _facts(self):
        return [
            Fact(row_id=1, kind="song", slug="kw-runaway",
                 text="The song samples Backseat Freestyle."),
            Fact(row_id=2, kind="song", slug="kw-runaway",
                 text="Runaway was performed at the 2010 MTV Video Music Awards."),
            Fact(row_id=3, kind="artist", slug="kanye-west",
                 text="Kanye West was born in Atlanta in 1977."),
            Fact(row_id=4, kind="artist", slug="other", text="Unrelated."),
        ]

    def test_the_pool_is_the_subject_and_nothing_else(self):
        r = FactsRetriever(_Source(self._facts()), hub=_EmptyHub())
        pool = r.pool(song_slug="kw-runaway", artist_slug="kanye-west")
        assert len(pool) == 3
        assert all(f.slug != "other" for f in pool)

    def test_duplicates_are_collapsed(self):
        """The same story reaches songfacts and Genius in near-identical words;
        a duplicate costs a slot in the answer's context for nothing."""
        facts = [Fact(row_id=1, kind="song", slug="s", text="Same  text."),
                 Fact(row_id=2, kind="song", slug="s", text="same text.")]
        r = FactsRetriever(_Source(facts), hub=_EmptyHub())
        assert len(r.pool(song_slug="s")) == 1

    def test_short_facts_are_not_filtered_out(self):
        """The old index skipped anything under 40 characters. Nothing is
        dropped before ranking now — the cross-encoder can score a stub low."""
        facts = [Fact(row_id=1, kind="artist", slug="a", text="Born 1977.")]
        r = FactsRetriever(_Source(facts), hub=_EmptyHub())
        assert len(r.pool(artist_slug="a")) == 1

    def test_a_subject_with_no_facts_returns_nothing(self):
        r = FactsRetriever(_Source([]), hub=_EmptyHub())
        assert r.retrieve("anything", song_slug="nope") == []

    def test_the_threshold_selects_and_does_not_cap(self):
        """Everything above the threshold goes to the model — no top-k."""
        facts = [Fact(row_id=i, kind="song", slug="s", text=f"fact {i}")
                 for i in range(6)]
        hub = _CEHub(probs=[0.9] * 6)
        r = FactsRetriever(_Source(facts), hub=hub)
        assert len(r.retrieve("q", song_slug="s")) == 6

    def test_ranking_carries_the_probability_back(self):
        facts = [Fact(row_id=1, kind="song", slug="s", text="a"),
                 Fact(row_id=2, kind="song", slug="s", text="b")]
        r = FactsRetriever(_Source(facts), hub=_CEHub(probs=[0.8, 0.3]))
        got = r.retrieve("q", song_slug="s", min_prob=0.5)
        assert len(got) == 1 and got[0].ce_prob == 0.8
