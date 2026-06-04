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
import threading
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

import logging

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLAP_WEIGHTS_PATH = Path(__file__).parent.parent.parent / "weights" / "music_audioset_epoch_15_esc_90.14.pt"
CLAP_WEIGHTS_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"

# Available text embedding models
TEXT_MODELS = {
    "jinaai/jina-embeddings-v2-small-en": {"dim": 512, "desc": "Lightweight model with CPU optimisation"},
    "Qwen/Qwen3-Embedding-0.6B": {"dim": 1024, "desc": "Higher quality, slower"},
}


class ModelRegistry:
    """
    Singleton registry for models.
    - Text models (sentence-transformers) — multiple models cached by name
    - CLAP model (audio embeddings) — loaded once at startup
    """

    # Phase B: get_text_model is the canonical accessor — returns the
    # (model, vector_name, vector_dim) triple and lazy-loads on first call.
    # Concurrent first-loads of the same name serialise on a per-name lock so
    # only one SentenceTransformer(...) instantiation happens (no GPU-mem waste).
    _text_models: dict[str, tuple[Any, str, int]] = {}
    _load_locks: dict[str, threading.Lock] = {}
    _load_locks_master: threading.Lock = threading.Lock()
    _clap_model: Optional[Any] = None

    # ── Text models ──

    @classmethod
    def _lock_for(cls, model_name: str) -> threading.Lock:
        """Return the per-model load lock, creating it under the master lock."""
        lock = cls._load_locks.get(model_name)
        if lock is not None:
            return lock
        with cls._load_locks_master:
            # Double-check after acquiring the master lock.
            lock = cls._load_locks.get(model_name)
            if lock is None:
                lock = threading.Lock()
                cls._load_locks[model_name] = lock
            return lock

    @classmethod
    def get_text_model(cls, model_name: str) -> tuple[Any, str, int]:
        """Return ``(model, vector_name, vector_dim)`` for ``model_name``.

        Lazy-loads on first call. Subsequent calls return the cached triple.
        Thread-safe: concurrent first-loads of the same name serialise on a
        per-name lock so only one ``SentenceTransformer(...)`` is instantiated.
        """
        cached = cls._text_models.get(model_name)
        if cached is not None:
            return cached

        with cls._lock_for(model_name):
            cached = cls._text_models.get(model_name)
            if cached is not None:
                return cached
            model = SentenceTransformer(model_name, device=DEVICE)
            dim = model.get_sentence_embedding_dimension()
            vector_name = f"text_{model_name.replace('/', '_')}"
            triple = (model, vector_name, dim)
            cls._text_models[model_name] = triple
            return triple

    @classmethod
    def load_text_model(cls, model_name: str) -> tuple[Any, str, int]:
        """Back-compat alias for ``get_text_model``. Existing callers
        (lyrics_search_engine, library_service, search/chat routes) keep
        working and now get the per-model load lock for free."""
        return cls.get_text_model(model_name)

    @classmethod
    def get_text_model_config(cls, model_name: str) -> tuple[str, int]:
        """Get vector_name and dim for a text model (loads if needed)."""
        _, vector_name, dim = cls.get_text_model(model_name)
        return vector_name, dim

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

        # Ensure the music_audioset checkpoint is locally available. The version
        # of laion_clap pinned here only knows about the four base 630k-* models
        # via load_ckpt(model_id=...), and load_ckpt("filename.pt") does NOT
        # auto-download — it only resolves a local path. So we fetch the file
        # ourselves to ``weights/`` on first launch via torch.hub (which gives a
        # tqdm progress bar). For Docker setups, pre-stage the file at
        # ``weights/music_audioset_epoch_15_esc_90.14.pt`` to skip the download.
        if not CLAP_WEIGHTS_PATH.exists():
            CLAP_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                "[CLAP] weights not found at %s — downloading ~2.3 GB from %s",
                CLAP_WEIGHTS_PATH, CLAP_WEIGHTS_URL,
            )
            torch.hub.download_url_to_file(CLAP_WEIGHTS_URL, str(CLAP_WEIGHTS_PATH))
            logger.info("[CLAP] download complete: %s", CLAP_WEIGHTS_PATH)

        cls._clap_model = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel='HTSAT-base'
        )
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
