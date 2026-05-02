"""
Resources layer — singletons for models and database.

ModelRegistry:
- load_text_model(model_name) -> (model, vector_name, dim)
- load_clap() -> model
- list_text_models() -> dict of available models

DbClient:
- __enter__/__exit__
- lyrics_db property (LyricsDB instance)
"""

import gc
from pathlib import Path
import torch
from typing import Any, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise RuntimeError("Install: pip install sentence-transformers")

# CLAP imports
try:
    import laion_clap
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLAP_WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "music_audioset_epoch_15_esc_90.14.pt"

# Available text embedding models
TEXT_MODELS = {
    "jinaai/jina-embeddings-v2-small-en": {"dim": 512, "desc": "Lightweight model with CPU optimisatiobn"},
    "Qwen/Qwen3-Embedding-0.6B": {"dim": 1024, "desc": "Higher quality, slower"},
}


class ModelRegistry:
    """
    Singleton registry for models.
    - Text models (sentence-transformers) — multiple models cached by name
    - CLAP model (audio embeddings) — loaded once at startup
    """

    _text_models: dict[str, tuple[Any, str, int]] = {}
    _clap_model: Optional[Any] = None

    # ── Text models ──

    @classmethod
    def load_text_model(cls, model_name: str) -> tuple[Any, str, int]:
        """Load a text embedding model (cached)."""
        if model_name in cls._text_models:
            return cls._text_models[model_name]

        model = SentenceTransformer(model_name, device=DEVICE)
        dim = model.get_sentence_embedding_dimension()
        vector_name = f"text_{model_name.replace('/', '_')}"

        cls._text_models[model_name] = (model, vector_name, dim)
        return cls._text_models[model_name]

    @classmethod
    def get_text_model(cls, model_name: str = "jinaai/jina-embeddings-v2-small-en") -> Any:
        """Get a loaded text model by name (default: jinaai/jina-embeddings-v2-small-en)."""
        if model_name not in cls._text_models:
            raise RuntimeError(f"Text model '{model_name}' not loaded. Call load_text_model first.")
        return cls._text_models[model_name][0]

    @classmethod
    def get_text_model_config(cls, model_name: str = "jinaai/jina-embeddings-v2-small-en") -> tuple[str, int]:
        """Get vector_name and dim for a loaded text model."""
        if model_name not in cls._text_models:
            raise RuntimeError(f"Text model '{model_name}' not loaded.")
        return cls._text_models[model_name][1], cls._text_models[model_name][2]

    @classmethod
    def list_text_models(cls) -> dict[str, dict]:
        """Return catalog of available text embedding models."""
        return TEXT_MODELS

    @classmethod
    def get_loaded_text_models(cls) -> list[str]:
        """Return names of currently loaded text models."""
        return list(cls._text_models.keys())

    # ── CLAP ──

    @classmethod
    def load_clap(cls) -> Any:
        """Load CLAP model for audio embeddings (cached)."""
        if cls._clap_model is not None:
            return cls._clap_model

        if not CLAP_AVAILABLE:
            raise RuntimeError("CLAP not available: pip install laion-clap")

        cls._clap_model = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel='HTSAT-base'
        )
        if CLAP_WEIGHTS_PATH.exists():
            cls._clap_model.load_ckpt(str(CLAP_WEIGHTS_PATH))
        cls._clap_model.eval()
        cls._clap_model = cls._clap_model.to(DEVICE)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return cls._clap_model

    @classmethod
    def get_clap(cls) -> Any:
        """Get the loaded CLAP model."""
        if cls._clap_model is None:
            raise RuntimeError("CLAP model not loaded. Call load_clap() first.")
        return cls._clap_model

    @classmethod
    def is_clap_available(cls) -> bool:
        """Check if CLAP module is installed."""
        return CLAP_AVAILABLE
