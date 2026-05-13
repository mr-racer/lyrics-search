"""Unit tests for SonicDescriptorService scaffold + prompt vocabulary loading."""

import json
from pathlib import Path

import pytest

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
