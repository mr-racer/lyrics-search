"""The wizard's quality slider maps to exactly these three registry tiers
(spec §2 — Fast / Balanced / Quality)."""
from app.resources.model_registry import TEXT_MODELS


def test_three_wizard_tiers_present():
    assert "jinaai/jina-embeddings-v2-small-en" in TEXT_MODELS   # Fast
    assert "intfloat/multilingual-e5-base" in TEXT_MODELS        # Balanced (new)
    assert "Qwen/Qwen3-Embedding-0.6B" in TEXT_MODELS            # Quality


def test_tier_dims():
    assert TEXT_MODELS["jinaai/jina-embeddings-v2-small-en"]["dim"] == 512
    assert TEXT_MODELS["intfloat/multilingual-e5-base"]["dim"] == 768
    assert TEXT_MODELS["Qwen/Qwen3-Embedding-0.6B"]["dim"] == 1024
