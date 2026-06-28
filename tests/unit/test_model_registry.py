"""Unit tests for app.resources.model_registry.

Consolidated from test_model_registry_concurrent.py and
test_model_registry_tiers.py.
"""

import threading

from app.resources.model_registry import TEXT_MODELS, ModelRegistry


class _FakeSentenceTransformer:
    """Records every instantiation so the test can assert call count."""
    instance_count = 0

    def __init__(self, name, device=None):
        type(self).instance_count += 1
        self.name = name

    def get_sentence_embedding_dimension(self):
        return 384


def _reset_registry():
    ModelRegistry._text_models.clear()
    if hasattr(ModelRegistry, "_load_locks"):
        ModelRegistry._load_locks.clear()


class TestConcurrent:
    """Concurrent get_text_model loads must NOT instantiate the same model twice."""

    def test_concurrent_loads_share_one_instance(self, monkeypatch):
        _reset_registry()
        _FakeSentenceTransformer.instance_count = 0
        monkeypatch.setattr(
            "app.resources.model_registry.SentenceTransformer",
            _FakeSentenceTransformer,
        )

        results: list[tuple] = []
        errors: list[BaseException] = []

        def _worker():
            try:
                results.append(ModelRegistry.get_text_model("fake/model"))
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        assert _FakeSentenceTransformer.instance_count == 1
        # All eight workers got the same triple
        first = results[0]
        assert all(r is first or r == first for r in results)
        # Triple shape: (model, vector_name, vector_dim)
        model, vector_name, vector_dim = first
        assert isinstance(model, _FakeSentenceTransformer)
        assert vector_name == "text_fake_model"
        assert vector_dim == 384

    def test_get_text_model_returns_cached_triple_without_reloading(self, monkeypatch):
        _reset_registry()
        _FakeSentenceTransformer.instance_count = 0
        monkeypatch.setattr(
            "app.resources.model_registry.SentenceTransformer",
            _FakeSentenceTransformer,
        )

        a = ModelRegistry.get_text_model("fake/model")
        b = ModelRegistry.get_text_model("fake/model")
        assert _FakeSentenceTransformer.instance_count == 1
        assert a == b

    def test_get_text_model_two_distinct_models_two_instances(self, monkeypatch):
        _reset_registry()
        _FakeSentenceTransformer.instance_count = 0
        monkeypatch.setattr(
            "app.resources.model_registry.SentenceTransformer",
            _FakeSentenceTransformer,
        )

        ModelRegistry.get_text_model("model-a")
        ModelRegistry.get_text_model("model-b")
        assert _FakeSentenceTransformer.instance_count == 2


class TestTiers:
    """The wizard's quality slider maps to exactly these three registry tiers
    (spec §2 — Fast / Balanced / Quality)."""

    def test_three_wizard_tiers_present(self):
        assert "jinaai/jina-embeddings-v2-small-en" in TEXT_MODELS   # Fast
        assert "intfloat/multilingual-e5-base" in TEXT_MODELS        # Balanced (new)
        assert "Qwen/Qwen3-Embedding-0.6B" in TEXT_MODELS            # Quality

    def test_tier_dims(self):
        assert TEXT_MODELS["jinaai/jina-embeddings-v2-small-en"]["dim"] == 512
        assert TEXT_MODELS["intfloat/multilingual-e5-base"]["dim"] == 768
        assert TEXT_MODELS["Qwen/Qwen3-Embedding-0.6B"]["dim"] == 1024
