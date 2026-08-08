"""
Resources layer — singletons for models and database.

ModelRegistry:
- get_text_model() -> (model, VECTOR_NAME, VECTOR_DIM)
- encode_text(sentences, is_query=False, **kw) -> embeddings (use for ALL text encodes)
- load_clap() -> model

Device policy (2026-08):
- ONE text embedding model, ``TEXT_MODEL_NAME``, chosen once and for all. It is
  loaded in fp16 straight onto the GPU and stays there: the assistant's fact
  retrieval encodes on nearly every turn, so the old load/demote/unload dance
  bought latency and nothing else. There is no per-model dispatch left — the
  Qdrant vector is called ``text`` and no longer encodes a model name, so a
  future model swap is a re-embed either way (see
  ``scripts/migrate_dense.py``).
- CLAP lives on the CPU permanently: loaded once (startup preload), never
  moved, never unloaded. It does not compete with the text model for VRAM.
- ``FORCE_CPU=1`` puts the text model on the CPU in fp32 (fp16 on CPU is
  slower than fp32 for most ops, and unsupported for some).

DbClient:
- __enter__/__exit__
- lyrics_db property (LyricsDB instance)
"""

import gc
import os
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

_FORCE_CPU = os.environ.get("FORCE_CPU", "").strip().lower() in ("1", "true", "yes", "on")
_GPU_DEVICE = None if (_FORCE_CPU or not torch.cuda.is_available()) else torch.device("cuda")
CLAP_WEIGHTS_PATH = Path(__file__).parent.parent.parent / "weights" / "music_audioset_epoch_15_esc_90.14.pt"
CLAP_WEIGHTS_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"

# ── The one text embedding model ─────────────────────────────────────────────
# Multilingual on purpose: the assistant matches a Russian statement against
# English source facts, which the previous English-only default could not do at
# all. Everything downstream (Qdrant vector name, migration script, facts
# collection) is pinned to this choice.
TEXT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
VECTOR_NAME = "text"
VECTOR_DIM = 1024
# Set explicitly: the model's own config carries a 32768-token window, and a
# full lyric would then be encoded whole. Measured on 758 prod tracks, the
# longest deduplicated lyric is ~1900 tokens, so 2048 covers the library.
MAX_SEQ_LENGTH = 2048
# Qwen3-Embedding is asymmetric — queries take an instruction, documents do not.
QUERY_PREFIX = (
    "Instruct: Given a statement about music, retrieve passages that explain it\n"
    "Query: "
)


class ModelRegistry:
    """
    Singleton registry for models.
    - Text model (sentence-transformers) — one pinned model, loaded once in fp16
      onto the GPU and kept resident (see the module docstring)
    - CLAP model (audio embeddings) — loaded once, always on CPU, never unloaded
    """

    # Re-exported on the class so call sites read one name
    # (``ModelRegistry.VECTOR_NAME``) instead of importing module globals.
    TEXT_MODEL_NAME = TEXT_MODEL_NAME
    VECTOR_NAME = VECTOR_NAME
    VECTOR_DIM = VECTOR_DIM
    MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
    QUERY_PREFIX = QUERY_PREFIX

    _text_model: Optional[tuple[Any, str, int]] = None
    _text_lock: threading.Lock = threading.Lock()
    _clap_model: Optional[Any] = None
    _clap_lock: threading.Lock = threading.Lock()

    # ── Text model ──

    @classmethod
    def get_text_model(cls) -> tuple[Any, str, int]:
        """Return ``(model, VECTOR_NAME, VECTOR_DIM)``, loading on first call.

        Thread-safe: concurrent first calls serialise on ``_text_lock`` so only
        one ``SentenceTransformer(...)`` is ever instantiated — a duplicate load
        would double the VRAM footprint for nothing.
        """
        cached = cls._text_model
        if cached is not None:
            return cached

        with cls._text_lock:
            if cls._text_model is not None:
                return cls._text_model

            device = "cuda" if _GPU_DEVICE is not None else "cpu"
            # fp16 halves the resident footprint (~1.2 GB instead of ~2.4) and
            # costs nothing measurable on embeddings. On CPU it would be slower
            # than fp32, and some ops have no CPU half kernel — so fp32 there.
            model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else None
            model = SentenceTransformer(
                TEXT_MODEL_NAME, device=device,
                **({"model_kwargs": model_kwargs} if model_kwargs else {}),
            )
            model.max_seq_length = MAX_SEQ_LENGTH

            dim = model.get_sentence_embedding_dimension()
            if dim != VECTOR_DIM:
                # Loud, not fatal: a mismatch means every vector written from
                # here on disagrees with the collection schema, and silence
                # would surface as empty search results days later.
                logger.error(
                    "[ModelRegistry] '%s' reports dim=%d but VECTOR_DIM=%d — "
                    "collections were built for %d",
                    TEXT_MODEL_NAME, dim, VECTOR_DIM, VECTOR_DIM,
                )

            cls._text_model = (model, VECTOR_NAME, VECTOR_DIM)
            logger.info(
                "[ModelRegistry] text model '%s' loaded (dim=%d, %s, %s, max_seq=%d)",
                TEXT_MODEL_NAME, dim, device,
                "fp16" if model_kwargs else "fp32", MAX_SEQ_LENGTH,
            )
            return cls._text_model

    @classmethod
    def encode_text(cls, sentences, *, is_query: bool = False, **encode_kwargs):
        """Encode ``sentences`` — the ONE sanctioned way to run a text encode.

        ``is_query=True`` prepends the model's instruction prefix. Qwen3-Embedding
        is asymmetric: the query side takes an instruction, the document side does
        not, and mixing the two up costs real recall. Indexing never sets it;
        search and fact retrieval always do.
        """
        model, _, _ = cls.get_text_model()
        if is_query:
            if isinstance(sentences, str):
                sentences = QUERY_PREFIX + sentences
            else:
                sentences = [QUERY_PREFIX + s for s in sentences]
        return model.encode(sentences, **encode_kwargs)

    @classmethod
    def is_text_model_loaded(cls) -> bool:
        return cls._text_model is not None

    # ── CLAP ──

    @classmethod
    def load_clap(cls) -> Any:
        """Load CLAP model for audio embeddings (cached, thread-safe, CPU-only).

        Double-checked locking mirrors ``get_text_model``: concurrent first
        loads serialise on ``_clap_lock`` so only one ``CLAP_Module`` is ever
        instantiated (~2.3 GB — a duplicate load is an OOM risk). The cache
        attribute is published only AFTER the checkpoint is fully loaded, so a
        concurrent reader can never observe a half-initialised model, and a
        failed ``load_ckpt`` never leaves a broken instance permanently cached.

        CLAP is pinned to the CPU by design (device policy 2026-07): it stays
        resident forever and never competes with text models for VRAM.
        """
        if cls._clap_model is not None:
            return cls._clap_model

        if not CLAP_AVAILABLE:
            raise RuntimeError("CLAP not available: pip install laion-clap")

        with cls._clap_lock:
            if cls._clap_model is not None:
                return cls._clap_model

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

            model = laion_clap.CLAP_Module(
                enable_fusion=False,
                amodel='HTSAT-base'
            )
            model.load_ckpt(str(CLAP_WEIGHTS_PATH))
            model.eval()
            model = model.to("cpu")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            cls._clap_model = model
            logger.info("[CLAP] model loaded (cpu, resident)")
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
