"""Unit tests for SonicDescriptorService scaffold + prompt vocabulary loading."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Stub heavy optional deps that are not installed in the test env (Python 3.13).
sys.modules.setdefault("laion_clap", MagicMock())

import numpy as np

from app.services.sonic_descriptor_service import SonicDescriptorService


@pytest.fixture
def sample_vocab_file(tmp_path):
    vocab = {
        "version": 1,
        "groups": {
            "energy": ["punchy", "ambient"],
            "valence": ["sad", "happy"],
        },
    }
    p = tmp_path / "sonic_prompts.json"
    p.write_text(json.dumps(vocab))
    return p


def test_service_loads_vocab(sample_vocab_file):
    svc = SonicDescriptorService(prompt_vocab_path=sample_vocab_file)
    prompts = svc.load_prompt_vocab()
    assert prompts == ["punchy", "ambient", "sad", "happy"]


def test_service_default_top_k_is_five():
    svc = SonicDescriptorService()
    assert svc.top_k_tags == 5


def test_service_uses_explicit_top_k(sample_vocab_file):
    svc = SonicDescriptorService(prompt_vocab_path=sample_vocab_file, top_k_tags=3)
    assert svc.top_k_tags == 3


def test_prompt_embeddings_computed_via_clap_when_cache_missing(sample_vocab_file, tmp_path, monkeypatch):
    embeddings_path = tmp_path / "embeds.npy"
    svc = SonicDescriptorService(prompt_vocab_path=sample_vocab_file, embeddings_path=embeddings_path)

    fake_emb = np.random.rand(4, 512).astype(np.float32)
    fake_clap = MagicMock()
    fake_clap.get_text_embedding.return_value = fake_emb
    monkeypatch.setattr(
        "app.resources.model_registry.ModelRegistry.load_clap",
        lambda: fake_clap,
    )

    out = svc.load_or_compute_prompt_embeddings()
    assert out.shape == (4, 512)
    assert embeddings_path.exists()
    fake_clap.get_text_embedding.assert_called_once_with(["punchy", "ambient", "sad", "happy"], use_tensor=False)


def test_prompt_embeddings_loaded_from_cache(sample_vocab_file, tmp_path, monkeypatch):
    cached = np.random.rand(4, 512).astype(np.float32)
    embeddings_path = tmp_path / "embeds.npy"
    np.save(embeddings_path, cached)

    svc = SonicDescriptorService(prompt_vocab_path=sample_vocab_file, embeddings_path=embeddings_path)

    fake_clap = MagicMock()
    monkeypatch.setattr(
        "app.resources.model_registry.ModelRegistry.load_clap",
        lambda: fake_clap,
    )

    out = svc.load_or_compute_prompt_embeddings()
    np.testing.assert_array_equal(out, cached)
    fake_clap.get_text_embedding.assert_not_called()
