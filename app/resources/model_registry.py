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
  slower than fp32 for most ops, and unsupported for some). Any OTHER reason
  for landing on the CPU is logged as a warning and reported by
  ``ModelRegistry.text_device()`` — a silent CPU fallback on a GPU box is the
  failure that looks like success.

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


def _resolve_device() -> tuple[str, str]:
    """``(device, reason)`` — decided at LOAD time, not at import time.

    Import happens very early (``app/__init__.py`` pulls this module in), and a
    CUDA probe that early can answer "no" in setups where the runtime is
    perfectly fine a moment later. More importantly: falling back to the CPU
    used to be silent, and a silent CPU fallback on a GPU box is the difference
    between 50 ms and several seconds per encode. Say which, and why.
    """
    if _FORCE_CPU:
        return "cpu", "FORCE_CPU is set"
    try:
        if torch.cuda.is_available():
            return "cuda", f"cuda:0 = {torch.cuda.get_device_name(0)}"
        return "cpu", ("torch.cuda.is_available() is False — no CUDA runtime "
                       "visible to this process (in Docker: is the container "
                       "started with GPU access?)")
    except Exception as e:  # noqa: BLE001 — a broken CUDA install must not crash startup
        return "cpu", f"CUDA probe raised {type(e).__name__}: {e}"


# What the model actually loaded onto, for /models/loaded and the logs.
_device_in_use: str | None = None
_device_reason: str = "not loaded yet"
CLAP_WEIGHTS_PATH = Path(__file__).parent.parent.parent / "weights" / "music_audioset_epoch_15_esc_90.14.pt"
CLAP_WEIGHTS_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"

# ── The one text embedding model ─────────────────────────────────────────────
# Multilingual on purpose: the assistant matches a Russian statement against
# English source facts, which the previous English-only default could not do at
# all. Everything downstream (Qdrant vector name, migration script, facts
# collection) is pinned to this choice.
TEXT_MODEL_NAME = "Octen/Octen-Embedding-0.6B"
VECTOR_NAME = "text"
VECTOR_DIM = 1024
# Set explicitly: the model's own config carries a 32768-token window, and a
# full lyric would then be encoded whole. Measured on 758 prod tracks, the
# longest deduplicated lyric is ~1900 tokens, so 2048 covers the library.
MAX_SEQ_LENGTH = 2048

# The model is asymmetric, and it ships BOTH sides of the pair in its own
# ``prompts`` config: an instruction for the query, a single space for the
# document. Address them by name rather than hand-rolling the query prefix and
# leaving the document bare — that was right for Qwen3-Embedding, whose
# document side genuinely takes nothing, and it silently costs recall here.
QUERY_PROMPT_NAME = "query"
DOCUMENT_PROMPT_NAME = "document"
# Fallback for a model that carries no prompts at all: instruction on the query
# side, nothing on the document side.
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
    # (query prompt name, document prompt name) — resolved once at load from
    # what the model actually carries. ``None`` on a side means "this model has
    # no prompt for it", which sends the query side to QUERY_PREFIX.
    _prompt_names: tuple[Optional[str], Optional[str]] = (None, None)
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

            global _device_in_use, _device_reason
            device, _device_reason = _resolve_device()
            if device == "cpu" and not _FORCE_CPU:
                # Loud on purpose: this is the failure that looks like success.
                logger.warning(
                    "[ModelRegistry] falling back to CPU — %s. Every search, "
                    "chat turn and fact lookup will encode on the CPU.",
                    _device_reason,
                )

            # fp16 halves the resident footprint (~1.2 GB instead of ~2.4) and
            # costs nothing measurable on embeddings. On CPU it would be slower
            # than fp32, and some ops have no CPU half kernel — so fp32 there.
            kwargs: dict[str, Any] = {
                # Decoder-derived embedders pool the LAST token, so padding has
                # to sit on the left or a short text in a batch is pooled off
                # its own padding.
                "tokenizer_kwargs": {"padding_side": "left"},
            }
            if device == "cuda":
                kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            try:
                model = SentenceTransformer(TEXT_MODEL_NAME, device=device, **kwargs)
            except TypeError:
                # Older sentence-transformers has neither kwarg. fp32 on the
                # GPU still beats fp16 on the CPU by a wide margin.
                logger.warning("[ModelRegistry] model_kwargs/tokenizer_kwargs "
                               "unsupported — loading fp32 on %s", device)
                model = SentenceTransformer(TEXT_MODEL_NAME, device=device)
            model.max_seq_length = MAX_SEQ_LENGTH
            cls._prompt_names = cls._resolve_prompt_names(model)
            _device_in_use = device

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
                "[ModelRegistry] text model '%s' loaded (dim=%d, device=%s [%s], "
                "%s, max_seq=%d, prompts=%s)",
                TEXT_MODEL_NAME, dim, device, _device_reason,
                "fp16" if "model_kwargs" in kwargs else "fp32", MAX_SEQ_LENGTH,
                cls._prompt_names,
            )
            return cls._text_model

    @staticmethod
    def _resolve_prompt_names(model: Any) -> tuple[Optional[str], Optional[str]]:
        """Which of the model's own prompts to use for each side of the pair.

        Read off the loaded model rather than assumed: a model that ships no
        ``prompts`` (or only a query one) must not be handed a ``prompt_name``
        sentence-transformers will reject.
        """
        prompts = getattr(model, "prompts", None) or {}
        return (
            QUERY_PROMPT_NAME if QUERY_PROMPT_NAME in prompts else None,
            DOCUMENT_PROMPT_NAME if DOCUMENT_PROMPT_NAME in prompts else None,
        )

    @classmethod
    def encode_text(cls, sentences, *, is_query: bool = False, **encode_kwargs):
        """Encode ``sentences`` — the ONE sanctioned way to run a text encode.

        The model is asymmetric, so each side gets its own prompt: the query
        side an instruction, the document side whatever the model was trained
        with (for Octen, a single space). Getting this backwards, or leaving one
        side bare because a previous model wanted that, costs real recall.
        Indexing passes ``is_query=False``; search and fact retrieval pass True.

        An explicit ``prompt``/``prompt_name`` from the caller wins — the two
        cannot be combined, and sentence-transformers raises when both arrive.
        """
        model, _, _ = cls.get_text_model()
        name = cls._prompt_names[0 if is_query else 1]
        caller_set = "prompt" in encode_kwargs or "prompt_name" in encode_kwargs
        if name is not None and not caller_set:
            encode_kwargs["prompt_name"] = name
        elif name is None and is_query and not caller_set:
            if isinstance(sentences, str):
                sentences = QUERY_PREFIX + sentences
            else:
                sentences = [QUERY_PREFIX + s for s in sentences]
        return model.encode(sentences, **encode_kwargs)

    @classmethod
    def is_text_model_loaded(cls) -> bool:
        return cls._text_model is not None

    @classmethod
    def text_device(cls) -> dict:
        """What the text model actually runs on, and why. Surfaced by
        ``GET /search/models/loaded`` so a silent CPU fallback is visible
        without reading the startup log."""
        return {"device": _device_in_use, "reason": _device_reason}

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
