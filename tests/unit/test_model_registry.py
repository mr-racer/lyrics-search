"""Unit tests for app.resources.model_registry.

Consolidated from test_model_registry_concurrent.py and
test_model_registry_tiers.py. Lifecycle tests (idle demote/unload, indexing
pin, CLAP single-load) cover the 2026-07 device policy.
"""

import threading
import time

import numpy as np

from app.resources.model_registry import (
    TEXT_IDLE_TO_CPU_SEC,
    TEXT_IDLE_UNLOAD_SEC,
    TEXT_MODELS,
    ModelRegistry,
)


class _FakeSentenceTransformer:
    """Records every instantiation so the test can assert call count."""
    instance_count = 0

    def __init__(self, name, device=None):
        type(self).instance_count += 1
        self.name = name
        self.device_moves: list[str] = []

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, sentences, **kw):
        return np.zeros(4, dtype=np.float32)

    def to(self, device):
        self.device_moves.append(str(device))
        return self


def _reset_registry():
    ModelRegistry._text_models.clear()
    if hasattr(ModelRegistry, "_load_locks"):
        ModelRegistry._load_locks.clear()
    if hasattr(ModelRegistry, "_text_state"):
        ModelRegistry._text_state.clear()


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


class TestLifecycle:
    """Device policy 2026-07: idle text models are demoted to CPU (60s) and
    unloaded (10 min); in-flight encodes and indexing pins block the reaper."""

    def _load_fake(self, monkeypatch, name="fake/model"):
        _reset_registry()
        _FakeSentenceTransformer.instance_count = 0
        monkeypatch.setattr(
            "app.resources.model_registry.SentenceTransformer",
            _FakeSentenceTransformer,
        )
        model, _, _ = ModelRegistry.get_text_model(name)
        return model

    def test_reap_unloads_after_idle_timeout(self, monkeypatch):
        self._load_fake(monkeypatch)
        st = ModelRegistry._state_for("fake/model")
        st.last_used = time.monotonic() - (TEXT_IDLE_UNLOAD_SEC + 1)
        ModelRegistry._reap_once()
        assert "fake/model" not in ModelRegistry.get_loaded_text_models()
        # Next use lazy-reloads (fresh instance).
        ModelRegistry.get_text_model("fake/model")
        assert _FakeSentenceTransformer.instance_count == 2

    def test_reap_keeps_fresh_model(self, monkeypatch):
        self._load_fake(monkeypatch)
        ModelRegistry._reap_once()  # last_used = just now
        assert "fake/model" in ModelRegistry.get_loaded_text_models()

    def test_inflight_encode_blocks_reap(self, monkeypatch):
        self._load_fake(monkeypatch)
        st = ModelRegistry._state_for("fake/model")
        st.last_used = time.monotonic() - (TEXT_IDLE_UNLOAD_SEC + 1)
        st.inflight = 1
        ModelRegistry._reap_once()
        assert "fake/model" in ModelRegistry.get_loaded_text_models()
        st.inflight = 0

    def test_indexing_pin_blocks_reap_until_released(self, monkeypatch):
        self._load_fake(monkeypatch)
        ModelRegistry.begin_indexing("fake/model")
        st = ModelRegistry._state_for("fake/model")
        st.last_used = time.monotonic() - (TEXT_IDLE_UNLOAD_SEC + 1)
        ModelRegistry._reap_once()
        assert "fake/model" in ModelRegistry.get_loaded_text_models()
        ModelRegistry.end_indexing("fake/model")
        st.last_used = time.monotonic() - (TEXT_IDLE_UNLOAD_SEC + 1)
        ModelRegistry._reap_once()
        assert "fake/model" not in ModelRegistry.get_loaded_text_models()

    def test_gpu_demote_after_short_idle(self, monkeypatch):
        model = self._load_fake(monkeypatch)
        st = ModelRegistry._state_for("fake/model")
        # Simulate an indexing run having promoted the model to the GPU.
        st.on_gpu = True
        st.last_used = time.monotonic() - (TEXT_IDLE_TO_CPU_SEC + 1)
        ModelRegistry._reap_once()
        assert st.on_gpu is False
        assert model.device_moves[-1] == "cpu"
        # Demoted but NOT unloaded (idle < unload threshold).
        assert "fake/model" in ModelRegistry.get_loaded_text_models()

    def test_encode_text_tracks_activity(self, monkeypatch):
        self._load_fake(monkeypatch)
        st = ModelRegistry._state_for("fake/model")
        st.last_used = time.monotonic() - (TEXT_IDLE_UNLOAD_SEC + 1)
        out = ModelRegistry.encode_text("fake/model", "hello")
        assert out is not None
        assert st.inflight == 0
        # Activity was stamped — the stale timestamp must be gone.
        assert time.monotonic() - st.last_used < 5.0


class _FakeClapModule:
    instance_count = 0

    def __init__(self, enable_fusion=False, amodel=""):
        type(self).instance_count += 1

    def load_ckpt(self, path):
        time.sleep(0.02)  # widen the race window for the concurrency test

    def eval(self):
        return self

    def to(self, device):
        assert str(device) == "cpu"  # CLAP is pinned to the CPU by policy
        return self


class TestClapSingleLoad:
    """Concurrent load_clap() calls must instantiate exactly one CLAP module."""

    def test_concurrent_load_clap_single_instance(self, monkeypatch, tmp_path):
        import types as _types
        weights = tmp_path / "w.pt"
        weights.write_bytes(b"stub")
        monkeypatch.setattr("app.resources.model_registry.CLAP_AVAILABLE", True)
        monkeypatch.setattr("app.resources.model_registry.CLAP_WEIGHTS_PATH", weights)
        monkeypatch.setattr(
            "app.resources.model_registry.laion_clap",
            _types.SimpleNamespace(CLAP_Module=_FakeClapModule),
            raising=False,
        )
        monkeypatch.setattr(ModelRegistry, "_clap_model", None)
        _FakeClapModule.instance_count = 0

        results, errors = [], []

        def _worker():
            try:
                results.append(ModelRegistry.load_clap())
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        assert _FakeClapModule.instance_count == 1
        assert all(r is results[0] for r in results)
        monkeypatch.setattr(ModelRegistry, "_clap_model", None)


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
