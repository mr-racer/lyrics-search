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


@pytest.fixture(autouse=True)
def _clean_registry():
    ModelRegistry._sparse_model = None
    ModelRegistry._reranker = None
    ModelRegistry._failed = set()
    yield
    ModelRegistry._sparse_model = None
    ModelRegistry._reranker = None
    ModelRegistry._failed = set()


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
