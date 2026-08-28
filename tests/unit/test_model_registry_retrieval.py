"""The two retrieval legs the assistant added to the registry.

What matters here is not that they load — a unit test with a stubbed torch
cannot prove that — but what they do when they do not. A leg that will not load
must be recorded, must not be re-attempted on every query, and must SAY SO.

Until 2026-08-28 "say so" meant returning ``None``, and that is the contract
this file used to pin. It was the wrong one: ``None`` is also what an empty
input returns, so a dead leg and an empty batch were indistinguishable at every
call site — and four of them each invented a different recovery, one of which
scored every Wikipedia candidate 1.0 and admitted the whole pool precisely when
nothing could judge it.

So the legs raise now. Degrading is still right in the retriever, and
``TestTheRetrieverStillDegrades`` is where that lives — as a decision made at a
named site and counted, rather than one that falls out of a bare ``None``.
"""

from __future__ import annotations

import sys
import types

import pytest

from app.resources.model_registry import ModelRegistry
from app.resources.models import (STATS, ModelEncodeFailed, ModelOOM,
                                  ModelUnavailable)


def _reset():
    ModelRegistry._sparse_model = None
    ModelRegistry._reranker = None
    ModelRegistry._breaker.reset()
    STATS.reset()


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset()
    yield
    _reset()


def _transformers(*, sparse=None, tokenizer=None, model=None, boom=False):
    """A fake ``transformers`` for the duration of one test.

    Injected into ``sys.modules`` because the registry imports it INSIDE the
    load functions — which is the rule that keeps this whole package importable
    on a machine with no GPU stack.
    """
    stub = types.ModuleType("transformers")

    def _from_pretrained(name, **kw):
        if boom:
            raise OSError(f"no such model: {name}")
        return sparse if sparse is not None else _Loadable()

    stub.AutoModel = types.SimpleNamespace(from_pretrained=_from_pretrained)
    stub.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda name, **kw: (_raise() if boom else
                                            (tokenizer or object())))
    stub.AutoModelForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda name, **kw: (_raise() if boom else
                                            (model or _Loadable())))
    return stub


def _raise():
    raise OSError("no such model")


def _raise_os(message: str):
    raise OSError(message)


class _Loadable:
    """Answers ``.to(device).eval()`` and counts how often it was built."""

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


class TestSparse:
    def test_it_loads_once_and_is_reused(self, monkeypatch):
        built = []

        def from_pretrained(name, **kw):
            built.append(name)
            return _Loadable()

        stub = _transformers()
        stub.AutoModel = types.SimpleNamespace(from_pretrained=from_pretrained)
        monkeypatch.setitem(sys.modules, "transformers", stub)

        first = ModelRegistry.load_sparse()
        assert ModelRegistry.load_sparse() is first
        assert built == [ModelRegistry.SPARSE_MODEL_NAME]

    def test_a_failed_load_is_recorded_and_not_retried(self, monkeypatch):
        """Re-attempting a missing model on every query turns one slow request
        into every request being slow."""
        attempts = []

        def from_pretrained(name, **kw):
            attempts.append(name)
            raise OSError("not found")

        stub = _transformers()
        stub.AutoModel = types.SimpleNamespace(from_pretrained=from_pretrained)
        monkeypatch.setitem(sys.modules, "transformers", stub)

        with pytest.raises(ModelUnavailable):
            ModelRegistry.load_sparse()
        with pytest.raises(ModelUnavailable):
            ModelRegistry.load_sparse()
        assert len(attempts) == 1
        assert "sparse" in ModelRegistry.retrieval_status()["failed"]

    def test_the_second_refusal_carries_the_original_reason(self, monkeypatch):
        """The breaker answers without touching the model, so it has to carry
        the reason forward — otherwise every log line after the first says
        nothing but "unavailable"."""
        stub = _transformers()
        stub.AutoModel = types.SimpleNamespace(
            from_pretrained=lambda name, **kw: _raise_os("weights are missing"))
        monkeypatch.setitem(sys.modules, "transformers", stub)

        with pytest.raises(ModelUnavailable):
            ModelRegistry.load_sparse()
        with pytest.raises(ModelUnavailable, match="weights are missing"):
            ModelRegistry.load_sparse()

    def test_a_breaker_that_expires_lets_the_leg_come_back(self, monkeypatch):
        """A load can fail because the card was full at that moment — the LLM
        next door moves its own footprint around — and that clears on its own.
        A permanent set meant one bad minute cost the leg until a restart."""
        calls = {"n": 0}

        def from_pretrained(name, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("CUDA out of memory")
            return _Loadable()

        stub = _transformers()
        stub.AutoModel = types.SimpleNamespace(from_pretrained=from_pretrained)
        monkeypatch.setitem(sys.modules, "transformers", stub)

        with pytest.raises(ModelUnavailable):
            ModelRegistry.load_sparse()
        ModelRegistry._breaker.reset("sparse")          # what the TTL does
        assert ModelRegistry.load_sparse() is not None
        assert calls["n"] == 2

    def test_encoding_without_the_model_raises(self, monkeypatch):
        """The failure the old ``None`` hid: this is a dead leg, not an empty
        batch, and the two must not read the same at the call site."""
        monkeypatch.setitem(sys.modules, "transformers", _transformers(boom=True))
        with pytest.raises(ModelUnavailable):
            ModelRegistry.encode_sparse(["a"])

    def test_an_empty_batch_never_touches_the_model(self, monkeypatch):
        def explode(*a, **kw):
            raise AssertionError("should not have loaded anything")

        stub = _transformers()
        stub.AutoModel = types.SimpleNamespace(from_pretrained=explode)
        monkeypatch.setitem(sys.modules, "transformers", stub)
        assert ModelRegistry.encode_sparse([]) is None


class TestReranker:
    def test_it_loads_once_and_is_reused(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "transformers", _transformers())
        first = ModelRegistry.load_reranker()
        assert first is not None
        assert ModelRegistry.load_reranker() is first

    def test_a_failed_load_raises_and_is_named_cross_encoder(self, monkeypatch):
        """One name for the leg everywhere. It was ``reranker`` in the old
        ``_failed`` set and ``cross_encoder`` in every counter and status field,
        which made the status impossible to read against the logs."""
        monkeypatch.setitem(sys.modules, "transformers", _transformers(boom=True))
        with pytest.raises(ModelUnavailable):
            ModelRegistry.load_reranker()
        with pytest.raises(ModelUnavailable):
            ModelRegistry.ce_probabilities("q", ["a", "b"])
        assert "cross_encoder" in ModelRegistry.retrieval_status()["failed"]

    def test_no_documents_means_no_call(self, monkeypatch):
        """An empty list is not a failure, so it must not become one now that
        the loader raises — and it must still not touch the model."""
        def explode(*a, **kw):
            raise AssertionError("should not have loaded anything")

        stub = _transformers()
        stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=explode)
        monkeypatch.setitem(sys.modules, "transformers", stub)
        assert ModelRegistry.ce_probabilities("q", []) == []


class TestTheSeamTheRetrieverUses:
    def test_the_hub_no_longer_swallows_a_missing_leg(self, monkeypatch):
        """The hub used to catch everything and answer ``None``. That is what
        made a dead leg indistinguishable from an empty corpus for a whole
        session; the decision belongs to the retriever, not to the seam."""
        from app.services.retrieval.hub import ModelHub

        monkeypatch.setitem(sys.modules, "transformers", _transformers(boom=True))
        hub = ModelHub()
        with pytest.raises(ModelUnavailable):
            hub.encode_sparse(["a"])
        with pytest.raises(ModelUnavailable):
            hub.ce_probabilities("q", ["a"])

    def test_the_hub_still_answers_an_empty_input_without_a_model(self, monkeypatch):
        from app.services.retrieval.hub import ModelHub

        monkeypatch.setitem(sys.modules, "transformers", _transformers(boom=True))
        hub = ModelHub()
        assert hub.encode_sparse([]) is None
        assert hub.ce_probabilities("q", []) == []

    def test_the_status_names_every_leg(self):
        status = ModelRegistry.retrieval_status()
        assert set(status) >= {"device", "dense", "sparse", "cross_encoder",
                               "failed", "encode_failures", "degradations"}


class TestTheRetrieverStillDegrades:
    """Raising moved the DECISION, it did not remove it.

    Ranking on two signals beats not answering, especially on a box whose card
    is shared with an LLM. What changed is that the choice is now made in one
    named place and counted, so a session that lost a leg is visible in
    ``GET /search/models/loaded`` and not only in how the answers read.
    """

    def test_a_dead_sparse_leg_does_not_take_the_retriever_down(self, monkeypatch):
        from app.services.retrieval.hybrid import HybridRetriever

        class _HalfHub:
            def encode_dense(self, texts, *, is_query=False):
                return None                      # no dense either: BM25 alone
            def encode_sparse(self, texts, *, is_query=False):
                raise ModelUnavailable("sparse", "load", "milco is not here")
            def ce_probabilities(self, query, docs):
                raise ModelUnavailable("cross_encoder", "load", "no reranker")

        retriever = HybridRetriever(["a red car", "a blue boat"], hub=_HalfHub())
        results = retriever.search("red")
        assert results, "BM25 alone must still answer"
        assert retriever.signals == ["bm25"]

    def test_losing_a_leg_is_counted_against_the_site_that_lost_it(self, monkeypatch):
        from app.services.retrieval.hybrid import HybridRetriever

        class _NoSparse:
            def encode_dense(self, texts, *, is_query=False):
                return None
            def encode_sparse(self, texts, *, is_query=False):
                raise ModelUnavailable("sparse", "load", "milco is not here")
            def ce_probabilities(self, query, docs):
                return [0.9] * len(docs)

        HybridRetriever(["a red car"], hub=_NoSparse())
        assert STATS.snapshot()["degradations"]["sparse/index"] == 1

    def test_an_unjudged_pack_is_returned_ranked_rather_than_empty(self):
        """``min_prob`` cannot be applied without a probability to compare, and
        dropping everything would be the wrong reading of "we could not judge"."""
        from app.services.retrieval.hybrid import HybridRetriever

        class _NoCE:
            def encode_dense(self, texts, *, is_query=False):
                return None
            def encode_sparse(self, texts, *, is_query=False):
                return None
            def ce_probabilities(self, query, docs):
                raise ModelEncodeFailed("cross_encoder", "score", "boom")

        retriever = HybridRetriever(["a red car", "a blue boat"], hub=_NoCE())
        results = retriever.search("red", min_prob=0.99)
        assert results, "an unjudged pack is ranked, not emptied"
        assert all(r.ce_prob is None for r in results)
        assert STATS.snapshot()["degradations"]["cross_encoder/search"] == 1


class _Rep(list):
    """One batch's sparse representation — one row per text it encoded.

    A list of the texts themselves so the row ORDER stays readable end to end:
    the leg encodes longest-first and permutes back, and a test that could not
    see rows could not catch that permutation going wrong.
    """

    def coalesce(self):
        return self


class _FakeMILCO:
    """A stand-in for the learned-sparse model that records every encode.

    ``oom_above`` is the whole point: the real model raises
    ``torch.OutOfMemoryError`` out of ``mlm_head`` when the
    ``[batch, tokens, 30522]`` logits will not fit, and how many texts were in
    that batch is exactly what decides it. Refusing anything larger reproduces
    that without a GPU.
    """

    def __init__(self, *, oom_above: int | None = None):
        self.oom_above = oom_above
        self.calls: list = []          # (n_texts, max_length, source_view)
        self.batches: list = []        # the texts of each batch, in order

    def to(self, device):
        return self

    def eval(self):
        return self

    def _encode(self, texts, *, max_length=None, source_view=False, **kw):
        import torch

        self.calls.append((len(texts), max_length, source_view))
        self.batches.append(list(texts))
        if self.oom_above is not None and len(texts) > self.oom_above:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return _Rep(texts)

    encode_document = _encode
    encode_query = _encode

    @property
    def batch_sizes(self) -> list:
        return [n for n, _, _ in self.calls]


def _install(monkeypatch, model):
    stub = _transformers()
    stub.AutoModel = types.SimpleNamespace(from_pretrained=lambda name, **kw: model)
    monkeypatch.setitem(sys.modules, "transformers", stub)
    return model


class TestTheSparseBudget:
    """MILCO projects EVERY token into the SPLADE-v3 vocabulary (30522 entries),
    so one batch costs ``batch x tokens x 30522 x 2`` bytes — and the mask
    multiply that follows keeps a second copy of it alive. Left unbounded, a
    960-token passage asked for 448 MiB in one allocation on a card with 130 MiB
    free, and the whole leg was lost for every document in the run.
    """

    def test_every_batch_is_truncated_to_the_sparse_budget(self, monkeypatch):
        """The cross-encoder that decides the final order already refuses to
        read past 512 tokens of the SAME passage. Terms mined from the tail
        beyond it are terms nothing downstream can score."""
        from app.resources.model_registry import SPARSE_MAX_LEN

        model = _install(monkeypatch, _FakeMILCO())
        assert ModelRegistry.encode_sparse(["a", "b", "c"]) is not None
        assert [ln for _, ln, _ in model.calls] == [SPARSE_MAX_LEN]

    def test_it_batches_at_the_sparse_budget_not_the_dense_one(self, monkeypatch):
        from app.resources.model_registry import SPARSE_BATCH

        model = _install(monkeypatch, _FakeMILCO())
        ModelRegistry.encode_sparse([f"doc {i}" for i in range(9)])
        assert max(model.batch_sizes) <= SPARSE_BATCH
        assert sum(model.batch_sizes) == 9

    def test_the_source_view_still_reaches_the_model(self, monkeypatch):
        """It is what makes the leg work on proper nouns in non-English text,
        which is most of what gets asked here — a truncation fix must not
        quietly drop it."""
        model = _install(monkeypatch, _FakeMILCO())
        ModelRegistry.encode_sparse(["a"])
        assert all(source_view for _, _, source_view in model.calls)


class TestLengthBucketing:
    """MILCO pads every batch to its longest member (``padding=True`` plus
    ``pad_to_multiple_of=32``), and the leg's cost is
    ``batch x tokens x 30522``. One 500-character passage sharing a batch with
    three one-liners therefore pays for four long passages. The retriever feeds
    exactly that mixture — web chunks next to one-sentence facts — so grouping
    by length is free memory, and the only thing it can break is row order.
    """

    def test_the_helper_round_trips_the_order(self):
        from app.resources.model_registry import _length_sorted_order

        texts = ["short", "a much longer passage here", "mid length", "x"]
        order, inverse = _length_sorted_order(texts)
        encoded = [texts[i] for i in order]                 # what the model sees
        restored = [encoded[inverse[i]] for i in range(len(texts))]
        assert restored == texts

    def test_the_longest_text_is_encoded_first(self):
        """Longest-first means the biggest batch is attempted immediately: if it
        fits, every later batch fits too, and if it does not, the shrink happens
        before the run has spent anything."""
        from app.resources.model_registry import _length_sorted_order

        texts = ["x", "xxxxxxxxxx", "xxx"]
        order, _ = _length_sorted_order(texts)
        assert texts[order[0]] == "xxxxxxxxxx"

    def test_batches_do_not_mix_long_texts_with_short_ones(self, monkeypatch):
        model = _install(monkeypatch, _FakeMILCO())
        texts = ["x" * 500, "x" * 480, "x" * 460, "x" * 440] + ["y"] * 4
        ModelRegistry.encode_sparse(texts)
        first, second = model.batches[0], model.batches[1]
        assert all(len(t) >= 440 for t in first)
        assert all(len(t) == 1 for t in second)

    def test_rows_come_back_in_the_callers_order(self, monkeypatch):
        """The retriever addresses ``self._sparse`` by document POSITION —
        ``_sparse_similarity`` index_selects into it with indices that came from
        the fused ranking. Rows left in encode order would silently score every
        passage against the wrong document."""
        _install(monkeypatch, _FakeMILCO())
        texts = ["tiny", "x" * 300, "medium length text", "z", "y" * 120]
        assert list(ModelRegistry.encode_sparse(texts)) == texts

    def test_the_order_survives_a_batch_shrink(self, monkeypatch):
        """The OOM path re-slices the reordered list mid-run; the permutation
        back has to account for that too."""
        _install(monkeypatch, _FakeMILCO(oom_above=2))
        texts = [f"{'x' * (i * 7)}-{i}" for i in range(9)]
        assert list(ModelRegistry.encode_sparse(texts)) == texts


class TestTheSparseLegSurvivesAnOOM:
    def test_an_oom_halves_the_batch_instead_of_losing_the_leg(self, monkeypatch):
        model = _install(monkeypatch, _FakeMILCO(oom_above=2))
        assert ModelRegistry.encode_sparse([f"d{i}" for i in range(8)]) is not None
        # 4 refused, then everything at 2 — and never 4 again: if the card is
        # tight now it is tight for the next batch too.
        assert model.batch_sizes == [4, 2, 2, 2, 2]

    def test_work_already_done_is_not_redone(self, monkeypatch):
        """Restarting the run at the smaller size would re-encode every batch
        that already succeeded — on 92 documents that is most of the work."""
        model = _install(monkeypatch, _FakeMILCO())
        calls = {"n": 0}
        real = model._encode

        def flaky(texts, **kw):
            calls["n"] += 1
            if calls["n"] == 3:
                import torch
                model.calls.append((len(texts), kw.get("max_length"), True))
                raise torch.OutOfMemoryError("CUDA out of memory")
            return real(texts, **kw)

        model.encode_document = flaky
        assert ModelRegistry.encode_sparse([f"d{i}" for i in range(12)]) is not None
        assert model.batch_sizes == [4, 4, 4, 2, 2]

    def test_an_oom_at_a_single_text_raises_model_oom(self, monkeypatch):
        """Down to one text and still refused means the card genuinely has no
        room. The retriever will rank on the remaining signals — but it decides
        that, having been told which leg it lost and why."""
        model = _install(monkeypatch, _FakeMILCO(oom_above=0))
        with pytest.raises(ModelOOM):
            ModelRegistry.encode_sparse(["a", "b"])
        assert model.batch_sizes == [2, 1]

    def test_a_recovered_oom_is_a_retry_not_a_failure(self, monkeypatch):
        _install(monkeypatch, _FakeMILCO(oom_above=2))
        ModelRegistry.encode_sparse([f"d{i}" for i in range(8)])
        status = ModelRegistry.retrieval_status()
        assert status["sparse_oom_retries"] == 1
        assert status["encode_failures"]["sparse"] == 0


class TestTheStatusStopsLyingAboutADeadLeg:
    """``sparse: True`` meant "the weights loaded", and stayed True through 29
    consecutive encode failures — the degradation that looks like "the answers
    got worse" and is invisible in the one place built to show it."""

    def test_a_failed_encode_is_counted(self, monkeypatch):
        _install(monkeypatch, _FakeMILCO(oom_above=0))
        with pytest.raises(ModelOOM):
            ModelRegistry.encode_sparse(["a"])
        status = ModelRegistry.retrieval_status()
        assert status["sparse"] is True          # the weights really are loaded
        assert status["encode_failures"]["sparse"] == 1

    def test_an_unexpected_failure_is_counted_too(self, monkeypatch):
        model = _install(monkeypatch, _FakeMILCO())
        model.encode_document = lambda *a, **kw: (_ for _ in ()).throw(
            ValueError("something else entirely"))
        with pytest.raises(ModelEncodeFailed):
            ModelRegistry.encode_sparse(["a"])
        assert ModelRegistry.retrieval_status()["encode_failures"]["sparse"] == 1

    def test_a_failed_rerank_is_counted(self, monkeypatch):
        class _Boom:
            def __call__(self, **kw):
                raise RuntimeError("no")

        stub = _transformers()
        stub.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda name, **kw: (lambda *a, **k: _Enc()))
        stub.AutoModelForSequenceClassification = types.SimpleNamespace(
            from_pretrained=lambda name, **kw: _Loadable())
        monkeypatch.setitem(sys.modules, "transformers", stub)

        with pytest.raises(ModelEncodeFailed):
            ModelRegistry.ce_probabilities("q", ["a"])
        assert ModelRegistry.retrieval_status()["encode_failures"][
            "cross_encoder"] == 1


class _Enc(dict):
    def to(self, device):
        return self
