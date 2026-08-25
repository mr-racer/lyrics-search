"""The two retrieval legs the assistant added to the registry.

What matters here is not that they load — a unit test with a stubbed torch
cannot prove that — but that they DEGRADE. A leg that will not load must be
recorded once, never retried, and must turn into ``None`` at the call site
rather than an exception halfway through a user's question. The alternative is
one missing model turning every request into a slow failure.
"""

from __future__ import annotations

import sys
import types

import pytest

from app.resources.model_registry import ModelRegistry


def _reset():
    ModelRegistry._sparse_model = None
    ModelRegistry._reranker = None
    ModelRegistry._failed = set()
    ModelRegistry._sparse_encode_failures = 0
    ModelRegistry._sparse_oom_retries = 0
    ModelRegistry._ce_encode_failures = 0


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

        assert ModelRegistry.load_sparse() is None
        assert ModelRegistry.load_sparse() is None
        assert len(attempts) == 1
        assert "sparse" in ModelRegistry.retrieval_status()["failed"]

    def test_encoding_without_the_model_is_none_not_an_exception(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "transformers", _transformers(boom=True))
        assert ModelRegistry.encode_sparse(["a"]) is None

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

    def test_a_failed_load_degrades_to_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "transformers", _transformers(boom=True))
        assert ModelRegistry.load_reranker() is None
        assert ModelRegistry.ce_probabilities("q", ["a", "b"]) is None
        assert "reranker" in ModelRegistry.retrieval_status()["failed"]

    def test_no_documents_means_no_call(self, monkeypatch):
        def explode(*a, **kw):
            raise AssertionError("should not have loaded anything")

        stub = _transformers()
        stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=explode)
        monkeypatch.setitem(sys.modules, "transformers", stub)
        assert ModelRegistry.ce_probabilities("q", []) is None


class TestTheSeamTheRetrieverUses:
    def test_the_hub_passes_a_missing_leg_through_as_none(self, monkeypatch):
        """``HybridRetriever`` reads None as "this signal does not exist" and
        ranks on the rest; anything else would take the request down."""
        from app.services.retrieval.hub import ModelHub

        monkeypatch.setitem(sys.modules, "transformers", _transformers(boom=True))
        hub = ModelHub()
        assert hub.encode_sparse(["a"]) is None
        assert hub.ce_probabilities("q", ["a"]) is None

    def test_the_status_names_every_leg(self):
        status = ModelRegistry.retrieval_status()
        assert set(status) >= {"device", "dense", "sparse", "cross_encoder",
                               "failed"}


class _Rep:
    """One batch's sparse representation. Answers ``.coalesce()``, nothing more."""

    def __init__(self, size: int):
        self.size = size

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

    def to(self, device):
        return self

    def eval(self):
        return self

    def _encode(self, texts, *, max_length=None, source_view=False, **kw):
        import torch

        self.calls.append((len(texts), max_length, source_view))
        if self.oom_above is not None and len(texts) > self.oom_above:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return _Rep(len(texts))

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

    def test_an_oom_at_a_single_text_degrades_to_none(self, monkeypatch):
        """Down to one text and still refused means the card genuinely has no
        room. Rank on the remaining signals rather than take the request down."""
        model = _install(monkeypatch, _FakeMILCO(oom_above=0))
        assert ModelRegistry.encode_sparse(["a", "b"]) is None
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
        ModelRegistry.encode_sparse(["a"])
        status = ModelRegistry.retrieval_status()
        assert status["sparse"] is True          # the weights really are loaded
        assert status["encode_failures"]["sparse"] == 1

    def test_an_unexpected_failure_is_counted_too(self, monkeypatch):
        model = _install(monkeypatch, _FakeMILCO())
        model.encode_document = lambda *a, **kw: (_ for _ in ()).throw(
            ValueError("something else entirely"))
        assert ModelRegistry.encode_sparse(["a"]) is None
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

        assert ModelRegistry.ce_probabilities("q", ["a"]) is None
        assert ModelRegistry.retrieval_status()["encode_failures"][
            "cross_encoder"] == 1


class _Enc(dict):
    def to(self, device):
        return self
