"""Unit tests for app.resources.model_registry.

The 2026-08 policy: ONE text embedding model, loaded once in fp16 onto the GPU
and kept resident. No per-model dispatch, no idle reaper — so the lifecycle
tests that covered demote/unload are gone with the code they described.
"""

import threading
import time

import numpy as np
import pytest
import torch

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

    prompts: dict = {}

    def __init__(self, name, device=None, **kwargs):
        type(self).instance_count += 1
        type(self).last_kwargs = {"device": device, **kwargs}
        self.name = name
        self.max_seq_length = 32768
        self.encoded: list = []
        self.encode_kwargs: list = []

    def get_sentence_embedding_dimension(self):
        return VECTOR_DIM

    def encode(self, sentences, **kw):
        self.encoded.append(sentences)
        self.encode_kwargs.append(kw)
        return np.zeros(4, dtype=np.float32)


class _FakeWithPrompts(_FakeSentenceTransformer):
    """What Octen actually is: an instruction on the query side, a single space
    on the document side, both shipped in the model's own config."""
    prompts = {"query": "Instruct: …\nQuery:", "document": " "}


def _reset_registry():
    ModelRegistry._text_model = None
    ModelRegistry._prompt_names = (None, None)


def _install_fake(monkeypatch, cls=_FakeSentenceTransformer):
    _reset_registry()
    cls.instance_count = 0
    monkeypatch.setattr("app.resources.model_registry.SentenceTransformer", cls)


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
        orphaned every existing collection. Swapping Qwen for Octen in 2026-08
        is exactly the event this guards against."""
        assert VECTOR_NAME == "text"
        assert TEXT_MODEL_NAME.split("/")[-1].lower() not in VECTOR_NAME.lower()

    def test_padding_is_on_the_left(self, monkeypatch):
        """Last-token pooling: right padding would pool a short text off its
        own padding instead of its final token."""
        _install_fake(monkeypatch)
        ModelRegistry.get_text_model()
        assert _FakeSentenceTransformer.last_kwargs["tokenizer_kwargs"] == {
            "padding_side": "left"}
        _reset_registry()

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

    def test_the_device_and_its_reason_are_reported(self, monkeypatch):
        """A silent CPU fallback on a GPU box is the failure that looks like
        success, so the choice has to be inspectable without the startup log."""
        _install_fake(monkeypatch)
        monkeypatch.setattr("app.resources.model_registry._resolve_device",
                            lambda: ("cpu", "no CUDA runtime visible"))
        ModelRegistry.get_text_model()
        assert ModelRegistry.text_device() == {
            "device": "cpu", "reason": "no CUDA runtime visible"}
        _reset_registry()

    def test_gpu_load_asks_for_fp16(self, monkeypatch):
        _install_fake(monkeypatch)
        monkeypatch.setattr("app.resources.model_registry._resolve_device",
                            lambda: ("cuda", "cuda:0 = fake"))
        ModelRegistry.get_text_model()
        kwargs = _FakeSentenceTransformer.last_kwargs
        assert kwargs["device"] == "cuda"
        assert kwargs["model_kwargs"]["torch_dtype"] is torch.float16
        _reset_registry()

    def test_cpu_load_stays_fp32(self, monkeypatch):
        """fp16 on the CPU is slower than fp32 and some ops have no half kernel."""
        _install_fake(monkeypatch)
        monkeypatch.setattr("app.resources.model_registry._resolve_device",
                            lambda: ("cpu", "FORCE_CPU is set"))
        ModelRegistry.get_text_model()
        assert "model_kwargs" not in _FakeSentenceTransformer.last_kwargs
        _reset_registry()

    def test_an_old_sentence_transformers_still_loads(self, monkeypatch):
        """No model_kwargs support: fp32 on the GPU still beats fp16 on the CPU."""
        class _NoKwargs(_FakeSentenceTransformer):
            def __init__(self, name, device=None, **kwargs):
                if kwargs:
                    raise TypeError("unexpected keyword argument 'model_kwargs'")
                super().__init__(name, device=device)

        _reset_registry()
        monkeypatch.setattr("app.resources.model_registry.SentenceTransformer", _NoKwargs)
        monkeypatch.setattr("app.resources.model_registry._resolve_device",
                            lambda: ("cuda", "cuda:0 = fake"))
        model, _, _ = ModelRegistry.get_text_model()
        assert model is not None
        _reset_registry()


    def test_a_loader_failure_is_not_disguised_as_a_kwargs_problem(self, monkeypatch):
        """A TypeError raised INSIDE the loader is not our kwargs' fault.

        sentence-transformers 6.0.0 feeds every module's own config.json to its
        constructor, and Octen ships ``2_Normalize/config.json`` with a key
        ``Normalize.__init__`` does not take. Retrying without model_kwargs only
        repeats that failure — while logging "model_kwargs unsupported", which
        blames the wrong thing and hides a total outage of the text model.
        """
        attempts: list = []

        class _BrokenLoader(_FakeSentenceTransformer):
            def __init__(self, name, device=None, **kwargs):
                attempts.append(kwargs)
                raise TypeError("Normalize.__init__() got an unexpected "
                                "keyword argument 'normalize_embeddings'")

        _reset_registry()
        monkeypatch.setattr("app.resources.model_registry.SentenceTransformer",
                            _BrokenLoader)
        monkeypatch.setattr("app.resources.model_registry._resolve_device",
                            lambda: ("cuda", "cuda:0 = fake"))
        with pytest.raises(TypeError, match="normalize_embeddings"):
            ModelRegistry.get_text_model()
        assert len(attempts) == 1, "a retry that cannot help must not be made"
        _reset_registry()


class TestEncodeWithModelPrompts:
    """The live model ships both prompts, so neither side is hand-rolled."""

    def test_query_side_uses_the_models_own_query_prompt(self, monkeypatch):
        _install_fake(monkeypatch, _FakeWithPrompts)
        ModelRegistry.encode_text("who produced this", is_query=True)
        model, _, _ = ModelRegistry.get_text_model()
        assert model.encoded[-1] == "who produced this"      # untouched text
        assert model.encode_kwargs[-1]["prompt_name"] == "query"
        _reset_registry()

    def test_document_side_uses_the_document_prompt(self, monkeypatch):
        """Not "bare": Octen's document prompt is a space, and leaving it off
        encodes documents differently from how the model was trained."""
        _install_fake(monkeypatch, _FakeWithPrompts)
        ModelRegistry.encode_text("a fact about a song")
        model, _, _ = ModelRegistry.get_text_model()
        assert model.encode_kwargs[-1]["prompt_name"] == "document"
        _reset_registry()

    def test_an_explicit_prompt_from_the_caller_wins(self, monkeypatch):
        """sentence-transformers raises when prompt and prompt_name both arrive."""
        _install_fake(monkeypatch, _FakeWithPrompts)
        ModelRegistry.encode_text("q", is_query=True, prompt="Custom: ")
        model, _, _ = ModelRegistry.get_text_model()
        assert "prompt_name" not in model.encode_kwargs[-1]
        assert model.encode_kwargs[-1]["prompt"] == "Custom: "
        _reset_registry()


class TestEncodeFallback:
    """A model carrying no prompts at all — the query side keeps the explicit
    instruction, the document side stays bare."""

    def test_query_side_gets_the_instruction_prefix(self, monkeypatch):
        _install_fake(monkeypatch)
        ModelRegistry.encode_text("who produced this", is_query=True)
        model, _, _ = ModelRegistry.get_text_model()
        assert model.encoded[-1] == QUERY_PREFIX + "who produced this"
        assert "prompt_name" not in model.encode_kwargs[-1]
        _reset_registry()

    def test_document_side_is_left_bare(self, monkeypatch):
        _install_fake(monkeypatch)
        ModelRegistry.encode_text("a fact about a song")
        model, _, _ = ModelRegistry.get_text_model()
        assert model.encoded[-1] == "a fact about a song"
        assert "prompt_name" not in model.encode_kwargs[-1]
        _reset_registry()

    def test_prefix_applies_to_every_item_of_a_list(self, monkeypatch):
        _install_fake(monkeypatch)
        ModelRegistry.encode_text(["one", "two"], is_query=True)
        model, _, _ = ModelRegistry.get_text_model()
        assert model.encoded[-1] == [QUERY_PREFIX + "one", QUERY_PREFIX + "two"]
        _reset_registry()

    def test_a_query_only_prompt_set_does_not_reach_the_document_side(self, monkeypatch):
        class _QueryOnly(_FakeSentenceTransformer):
            prompts = {"query": "Instruct: …"}

        _install_fake(monkeypatch, _QueryOnly)
        ModelRegistry.encode_text("a fact", is_query=False)
        model, _, _ = ModelRegistry.get_text_model()
        assert "prompt_name" not in model.encode_kwargs[-1]
        _reset_registry()


class _FakeClapModule:
    instance_count = 0
    last_device = "<unset>"

    def __init__(self, enable_fusion=False, amodel="", device=None):
        type(self).instance_count += 1
        # Mirrors laion_clap: device=None means "cuda:0 if it is there", and
        # create_model() does model.to(device) INSIDE the constructor.
        type(self).last_device = "cuda:0" if device is None else str(device)

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

    def test_clap_is_constructed_on_the_cpu(self, monkeypatch, tmp_path):
        """The device must be pinned in the CONSTRUCTOR, not after the fact.

        laion_clap's CLAP_Module defaults to cuda:0 whenever CUDA is visible
        and materialises the whole model there before returning, so a later
        ``.to("cpu")`` cannot save us: on a box whose VRAM is already spoken
        for by the text model and the LLM, the constructor itself raises
        "CUDA out of memory" and audio search dies.
        """
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
        _FakeClapModule.last_device = "<unset>"

        ModelRegistry.load_clap()

        assert _FakeClapModule.last_device == "cpu"
        monkeypatch.setattr(ModelRegistry, "_clap_model", None)
