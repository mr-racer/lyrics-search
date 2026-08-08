"""Unit tests for app.resources.model_registry.

The 2026-08 policy: ONE text embedding model, loaded once in fp16 onto the GPU
and kept resident. No per-model dispatch, no idle reaper — so the lifecycle
tests that covered demote/unload are gone with the code they described.
"""

import threading
import time

import numpy as np

from app.resources.model_registry import (
    MAX_SEQ_LENGTH,
    QUERY_PREFIX,
    TEXT_MODEL_NAME,
    VECTOR_DIM,
    VECTOR_NAME,
    ModelRegistry,
)


class _FakeSentenceTransformer:
    """Records every instantiation so the test can assert call count."""
    instance_count = 0
    last_kwargs: dict = {}

    def __init__(self, name, device=None, **kwargs):
        type(self).instance_count += 1
        type(self).last_kwargs = {"device": device, **kwargs}
        self.name = name
        self.max_seq_length = 32768
        self.encoded: list = []

    def get_sentence_embedding_dimension(self):
        return VECTOR_DIM

    def encode(self, sentences, **kw):
        self.encoded.append(sentences)
        return np.zeros(4, dtype=np.float32)


def _reset_registry():
    ModelRegistry._text_model = None


def _install_fake(monkeypatch):
    _reset_registry()
    _FakeSentenceTransformer.instance_count = 0
    monkeypatch.setattr(
        "app.resources.model_registry.SentenceTransformer",
        _FakeSentenceTransformer,
    )


class TestLoading:
    def test_concurrent_loads_share_one_instance(self, monkeypatch):
        """Two threads racing on the first call must not build two models —
        a duplicate would double the resident VRAM for nothing."""
        _install_fake(monkeypatch)
        results: list[tuple] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def _worker():
            try:
                barrier.wait()
                results.append(ModelRegistry.get_text_model())
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        assert _FakeSentenceTransformer.instance_count == 1
        assert all(r is results[0] for r in results)
        _reset_registry()

    def test_returns_the_pinned_vector_name_and_dim(self, monkeypatch):
        _install_fake(monkeypatch)
        model, vector_name, dim = ModelRegistry.get_text_model()
        assert vector_name == VECTOR_NAME == "text"
        assert dim == VECTOR_DIM == 1024
        assert model.name == TEXT_MODEL_NAME
        _reset_registry()

    def test_vector_name_does_not_encode_the_model(self):
        """The old name was f"text_{model}" — renaming the model silently
        orphaned every existing collection."""
        assert "/" not in VECTOR_NAME
        assert "qwen" not in VECTOR_NAME.lower()

    def test_input_length_is_capped(self, monkeypatch):
        """The model's own config carries a 32768 window; left alone, a full
        lyric would be encoded whole."""
        _install_fake(monkeypatch)
        model, _, _ = ModelRegistry.get_text_model()
        assert model.max_seq_length == MAX_SEQ_LENGTH == 2048
        _reset_registry()

    def test_second_call_is_cached(self, monkeypatch):
        _install_fake(monkeypatch)
        a = ModelRegistry.get_text_model()
        b = ModelRegistry.get_text_model()
        assert a is b
        assert _FakeSentenceTransformer.instance_count == 1
        _reset_registry()

    def test_is_text_model_loaded_flips_on_load(self, monkeypatch):
        _install_fake(monkeypatch)
        assert ModelRegistry.is_text_model_loaded() is False
        ModelRegistry.get_text_model()
        assert ModelRegistry.is_text_model_loaded() is True
        _reset_registry()


class TestEncode:
    def test_query_side_gets_the_instruction_prefix(self, monkeypatch):
        """Qwen3-Embedding is asymmetric — mixing the sides up costs recall."""
        _install_fake(monkeypatch)
        ModelRegistry.encode_text("who produced this", is_query=True)
        model, _, _ = ModelRegistry.get_text_model()
        assert model.encoded[-1] == QUERY_PREFIX + "who produced this"
        _reset_registry()

    def test_document_side_is_left_bare(self, monkeypatch):
        _install_fake(monkeypatch)
        ModelRegistry.encode_text("a fact about a song")
        model, _, _ = ModelRegistry.get_text_model()
        assert model.encoded[-1] == "a fact about a song"
        _reset_registry()

    def test_prefix_applies_to_every_item_of_a_list(self, monkeypatch):
        _install_fake(monkeypatch)
        ModelRegistry.encode_text(["one", "two"], is_query=True)
        model, _, _ = ModelRegistry.get_text_model()
        assert model.encoded[-1] == [QUERY_PREFIX + "one", QUERY_PREFIX + "two"]
        _reset_registry()


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
