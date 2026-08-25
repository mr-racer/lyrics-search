"""
Resources layer — singletons for models and database.

ModelRegistry:
- get_text_model() -> (model, VECTOR_NAME, VECTOR_DIM)
- encode_text(sentences, is_query=False, **kw) -> embeddings (use for ALL text encodes)
- encode_sparse(texts, is_query=False) -> coalesced sparse tensor | None
- ce_probabilities(query, docs) -> list[float] | None
- load_clap() -> model

Device policy (2026-08):
- ONE text embedding model, ``TEXT_MODEL_NAME``, chosen once and for all. It is
  loaded in fp16 straight onto the GPU and stays there: the assistant's fact
  retrieval encodes on nearly every turn, so the old load/demote/unload dance
  bought latency and nothing else. There is no per-model dispatch left — the
  Qdrant vector is called ``text`` and no longer encodes a model name, so a
  future model swap is a re-embed either way (see
  ``scripts/migrate_dense.py``).
- TWO more residents joined it in 2026-08 for the assistant's retrieval stack:
  a learned-sparse encoder (``SPARSE_MODEL_NAME``) and a cross-encoder
  (``RERANK_MODEL_NAME``). Both fp16 on the same device, both loaded once and
  never released — the assistant reranks on every turn, and the alternative
  (load on demand, free after) is the dance that was already removed once.
  Three residents come to ~3.6 GB, which is what the deployment budget allows.
  This registry is their ONLY owner: nothing else may instantiate them, or the
  same weights land on the card twice.
- Residency is only half the budget: what actually ran the card out of memory
  was TRANSIENT. Each leg therefore carries its own token ceiling and batch
  (``MAX_SEQ_LENGTH``, ``RERANK_MAX_LEN``/``RERANK_BATCH``,
  ``SPARSE_MAX_LEN``/``SPARSE_BATCH``) rather than a shared pair, because their
  peaks have very different shapes — the sparse leg's grows with the
  VOCABULARY, which is why it was the one that failed. The card is shared with
  an LLM whose free memory moves under us, so ``encode_sparse`` treats an
  allocator refusal as a signal to shrink, not as an error.
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
from typing import Any, Optional

# torch and sentence_transformers are imported INSIDE the functions that need
# them. This module is reached by ``app/resources/__init__.py``, so importing
# them here put the whole ML stack on the critical path of anything that merely
# touches ``app.*`` — including ``scripts/create_owner``, a SQLite-only
# bootstrap CLI that then could not run on a machine without torch installed.
# See CLAUDE.md: "Heavy/optional imports go inside functions".

# Bound on first load rather than at import. It stays a module-level NAME on
# purpose: the loader is exercised by swapping this attribute for a fake class,
# which needs something to swap.
SentenceTransformer: Any = None


def _sentence_transformer_cls():
    """The SentenceTransformer class, imported the first time one is built."""
    global SentenceTransformer
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
        except ImportError as e:
            raise RuntimeError("Install: pip install sentence-transformers") from e
        SentenceTransformer = _ST
    return SentenceTransformer

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
    import torch
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


def _release_cuda_cache() -> None:
    """Hand cached-but-unused blocks back before retrying a failed allocation.

    Never raises: this is called on the recovery path of an out-of-memory error,
    and a CPU-only box (or a broken CUDA install) must take the smaller batch
    rather than a second exception on top of the first.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


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

# ── The assistant's retrieval stack ──────────────────────────────────────────
# Learned sparse. It reads (expanded) TERMS where the dense model reads meaning,
# which is what lets the two disagree usefully — the near-duplicate check in
# ``services/retrieval/diversity.py`` requires both to agree before it drops a
# passage, and that guard is only worth anything while the signals fail
# differently.
SPARSE_MODEL_NAME = "omai-research/milco-650m"
# LexEcho source view. Matters for proper nouns in non-English text, which is
# most of what gets asked here.
SPARSE_SOURCE_VIEW = True
# The sparse leg's OWN budget — deliberately not the dense model's.
#
# MILCO projects every token into the SPLADE-v3 English vocabulary (30522
# entries) inside ``mlm_head``, and the mask multiply on the next line keeps a
# second copy of that tensor alive while ``logits.max(dim=1)`` reads both. One
# batch therefore costs ``2 x batch x tokens x 30522 x 2`` bytes, and NOTHING
# above capped it: ``milco.encode_text`` takes ``max_length`` but defaults it to
# the tokenizer's own ceiling, so a 960-token passage asked for 448 MiB in a
# single allocation. On a box where an LLM holds the rest of the card, that is
# the difference between ranking on three signals and ranking on two.
#
# 512 matches RERANK_MAX_LEN on purpose: it is the SAME passage, and the
# cross-encoder that decides the final order already refuses to read past it.
# Terms mined from the tail beyond that are terms nothing downstream can score.
SPARSE_MAX_LEN = 512
# Peak at this batch is ~120 MiB (twice that with the mask copy), which fits at
# the tightest moment measured on the deployment box. ENCODE_BATCH stays 8 for
# the dense model, which has never run out of memory: the two legs have very
# different shapes and sharing one number hid that for a while.
SPARSE_BATCH = 4

# Cross-encoder. Reads the query and the document TOGETHER and produces the
# number every threshold in the assistant is expressed in.
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
# Pairs are (question, passage) and a passage is at most ~1200 chars, so 512
# tokens covers it. Raising this costs quadratic attention for text the chunker
# already decided not to send.
RERANK_MAX_LEN = 512

ENCODE_BATCH = 8
RERANK_BATCH = 16


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
    SPARSE_MODEL_NAME = SPARSE_MODEL_NAME
    RERANK_MODEL_NAME = RERANK_MODEL_NAME

    _text_model: Optional[tuple[Any, str, int]] = None
    # (query prompt name, document prompt name) — resolved once at load from
    # what the model actually carries. ``None`` on a side means "this model has
    # no prompt for it", which sends the query side to QUERY_PREFIX.
    _prompt_names: tuple[Optional[str], Optional[str]] = (None, None)
    _text_lock: threading.Lock = threading.Lock()
    _clap_model: Optional[Any] = None
    _clap_lock: threading.Lock = threading.Lock()
    _sparse_model: Optional[Any] = None
    _sparse_lock: threading.Lock = threading.Lock()
    # (tokenizer, model)
    _reranker: Optional[tuple[Any, Any]] = None
    _rerank_lock: threading.Lock = threading.Lock()
    # A leg that tried to load and failed is not retried. A missing model is a
    # missing model, and re-attempting it on every query turns one slow request
    # into every request being slow.
    _failed: set = set()
    # Loading and ENCODING fail independently, and only the first one had a
    # name. A leg whose weights are resident but whose every encode dies still
    # reported itself as up, so hours of dense-only ranking looked exactly like
    # "the answers got worse". These are what ``retrieval_status`` says instead.
    _sparse_encode_failures: int = 0
    _sparse_oom_retries: int = 0
    _ce_encode_failures: int = 0

    # ── Text model ──

    @classmethod
    def get_text_model(cls) -> tuple[Any, str, int]:
        """Return ``(model, VECTOR_NAME, VECTOR_DIM)``, loading on first call.

        Thread-safe: concurrent first calls serialise on ``_text_lock`` so only
        one ``SentenceTransformer(...)`` is ever instantiated — a duplicate load
        would double the VRAM footprint for nothing.
        """
        import torch
        SentenceTransformer = _sentence_transformer_cls()
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
            except TypeError as e:
                # ONLY our own kwargs may send us down this path: an older
                # sentence-transformers has neither, and fp32 on the GPU still
                # beats fp16 on the CPU by a wide margin. A TypeError from
                # deeper in the loader is a different failure — the retry
                # cannot fix it, repeats it, and files it under a warning that
                # blames the wrong thing. Let that one out as itself.
                if not any(name in str(e) for name in kwargs):
                    raise
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

    # ── Retrieval stack: learned sparse + cross-encoder ──
    #
    # Both follow the text model's pattern (double-checked locking, fp16 on the
    # GPU) and both DEGRADE instead of raising: a leg that will not load is
    # recorded in ``_failed`` and reported by ``retrieval_status()``. The
    # retriever ranks on whatever signals it has, so a missing MILCO costs
    # ranking quality and a missing cross-encoder costs the thresholds — neither
    # costs the request.

    @classmethod
    def _shared_device(cls) -> str:
        """The device the retrieval models load onto.

        Whatever the text model landed on, so all three agree without probing
        CUDA three times. Resolved here when the text model has not loaded yet
        (the sparse leg can be the first thing a process touches).
        """
        global _device_in_use, _device_reason
        if _device_in_use is not None:
            return _device_in_use
        device, _device_reason = _resolve_device()
        _device_in_use = device
        return device

    @classmethod
    def load_sparse(cls) -> Optional[Any]:
        """The learned-sparse encoder, or None if it is unavailable."""
        import torch
        if cls._sparse_model is not None or "sparse" in cls._failed:
            return cls._sparse_model

        with cls._sparse_lock:
            if cls._sparse_model is not None or "sparse" in cls._failed:
                return cls._sparse_model
            device = cls._shared_device()
            try:
                from transformers import AutoModel

                kwargs: dict[str, Any] = {"trust_remote_code": True}
                if device == "cuda":
                    kwargs["torch_dtype"] = torch.float16
                model = AutoModel.from_pretrained(SPARSE_MODEL_NAME, **kwargs)
                cls._sparse_model = model.to(device).eval()
                logger.info("[ModelRegistry] sparse model '%s' loaded (device=%s, %s)",
                            SPARSE_MODEL_NAME, device,
                            "fp16" if device == "cuda" else "fp32")
            except Exception as e:  # noqa: BLE001 — a missing leg is a valid state
                cls._failed.add("sparse")
                logger.warning("[ModelRegistry] sparse model '%s' unavailable: %s",
                               SPARSE_MODEL_NAME, e, exc_info=True)
            return cls._sparse_model

    @classmethod
    def load_reranker(cls) -> Optional[tuple[Any, Any]]:
        """``(tokenizer, model)`` for the cross-encoder, or None."""
        import torch
        if cls._reranker is not None or "reranker" in cls._failed:
            return cls._reranker

        with cls._rerank_lock:
            if cls._reranker is not None or "reranker" in cls._failed:
                return cls._reranker
            device = cls._shared_device()
            try:
                from transformers import (AutoModelForSequenceClassification,
                                          AutoTokenizer)

                tokenizer = AutoTokenizer.from_pretrained(
                    RERANK_MODEL_NAME, trust_remote_code=True)
                model = AutoModelForSequenceClassification.from_pretrained(
                    RERANK_MODEL_NAME, trust_remote_code=True,
                    torch_dtype=(torch.float16 if device == "cuda"
                                 else torch.float32),
                ).to(device).eval()
                cls._reranker = (tokenizer, model)
                logger.info("[ModelRegistry] cross-encoder '%s' loaded (device=%s, %s)",
                            RERANK_MODEL_NAME, device,
                            "fp16" if device == "cuda" else "fp32")
            except Exception as e:  # noqa: BLE001
                cls._failed.add("reranker")
                logger.warning("[ModelRegistry] cross-encoder '%s' unavailable: %s",
                               RERANK_MODEL_NAME, e, exc_info=True)
            return cls._reranker

    @classmethod
    def encode_sparse(cls, texts: list, *, is_query: bool = False):
        """Learned-sparse vectors as one coalesced sparse tensor, or None.

        Asymmetric like the dense model: queries and documents go through
        different heads, so the side is named rather than inferred.

        Truncated at ``SPARSE_MAX_LEN`` and batched at ``SPARSE_BATCH`` — see
        those constants for why this leg gets its own two numbers instead of
        the dense model's.

        **An allocator failure shrinks the batch; it does not lose the leg.**
        The card is shared with an LLM whose free memory moves under us, so
        "there is no room right now" is a normal condition rather than an
        error. The batch ratchets DOWN and stays down for the rest of the call
        (tight now means tight for the next batch too), and the work already
        done is kept — on 92 documents, restarting would re-encode nearly all
        of it. Only a refusal at a single text gives up, because then the card
        genuinely has nothing left and the retriever is better off ranking on
        the signals it still has.
        """
        import torch
        model = cls.load_sparse()
        if model is None or not texts:
            return None
        encode = model.encode_query if is_query else model.encode_document

        reps: list = []
        size = SPARSE_BATCH
        i = 0
        while i < len(texts):
            try:
                with torch.no_grad():
                    reps.append(encode(texts[i:i + size],
                                       max_length=SPARSE_MAX_LEN,
                                       source_view=SPARSE_SOURCE_VIEW).coalesce())
                i += size
            except torch.OutOfMemoryError:
                # Halve what was ACTUALLY attempted, not the nominal size: on a
                # short tail ``size`` can already exceed the texts left, and
                # halving the nominal would re-send the identical batch and pay
                # a second failed forward for nothing.
                attempted = min(size, len(texts) - i)
                if attempted == 1:
                    cls._sparse_encode_failures += 1
                    logger.warning(
                        "[ModelRegistry] sparse encode out of memory even at one "
                        "text (%d of %d done) — this run ranks without the sparse "
                        "leg. Something else on the card grew; see "
                        "GET /search/models/loaded for the running count.",
                        i, len(texts))
                    return None
                size = attempted // 2
                cls._sparse_oom_retries += 1
                _release_cuda_cache()
                logger.info(
                    "[ModelRegistry] sparse encode hit OOM at %d of %d texts — "
                    "continuing at batch=%d", i, len(texts), size)
            except Exception:  # noqa: BLE001
                cls._sparse_encode_failures += 1
                logger.warning("[ModelRegistry] sparse encode failed", exc_info=True)
                return None
        return torch.cat(reps, dim=0).coalesce()

    @classmethod
    def ce_probabilities(cls, query: str, docs: list) -> Optional[list]:
        """``sigmoid(logit)`` per (query, doc) pair, or None when unavailable.

        A probability rather than a raw logit because every threshold in the
        assistant is one number compared against this: logits are not comparable
        between model families, probabilities roughly are.
        """
        import torch
        pair = cls.load_reranker()
        if pair is None or not docs:
            return None
        tokenizer, model = pair
        device = cls._shared_device()
        try:
            out: list[float] = []
            with torch.no_grad():
                for i in range(0, len(docs), RERANK_BATCH):
                    batch = docs[i:i + RERANK_BATCH]
                    enc = tokenizer([query] * len(batch), batch, padding=True,
                                    truncation=True, max_length=RERANK_MAX_LEN,
                                    return_tensors="pt").to(device)
                    logits = model(**enc).logits.float()
                    # 1-logit head (bge/jina/gte) vs the older 2-class one.
                    col = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
                    out.extend(torch.sigmoid(col).cpu().tolist())
            return out
        except Exception:  # noqa: BLE001
            cls._ce_encode_failures += 1
            logger.warning("[ModelRegistry] cross-encoder scoring failed",
                           exc_info=True)
            return None

    @classmethod
    def retrieval_status(cls) -> dict:
        """Which retrieval legs are actually up. Surfaced by
        ``GET /search/models/loaded`` so a degraded ranking is visible without
        reading the startup log — the failure that otherwise looks like
        'the answers got worse'.

        The booleans mean "the weights are resident" and nothing more, which is
        why the counters sit next to them: a leg can load perfectly and then
        fail every single encode, and for a while that read here as a healthy
        stack. ``encode_failures`` counts runs that lost the leg;
        ``sparse_oom_retries`` counts runs that kept it by shrinking the batch,
        which is a load signal rather than a fault.
        """
        return {
            "device": _device_in_use,
            "dense": cls._text_model is not None,
            "sparse": cls._sparse_model is not None,
            "cross_encoder": cls._reranker is not None,
            "failed": sorted(cls._failed),
            "encode_failures": {
                "sparse": cls._sparse_encode_failures,
                "cross_encoder": cls._ce_encode_failures,
            },
            "sparse_oom_retries": cls._sparse_oom_retries,
        }

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
        resident forever and never competes with text models for VRAM. The pin
        is applied in the CLAP_Module constructor — see the comment below.
        """
        import torch
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

            # device="cpu" is LOAD-BEARING, not a formality. Left at its
            # default (None), laion_clap resolves the device itself as
            # "cuda:0 if torch.cuda.is_available() else cpu" and create_model()
            # runs model.to(device) INSIDE the constructor — so the whole CLAP
            # stack is materialised in VRAM before this line returns. On a box
            # where the text model, the sparse/rerank pair and the LLM already
            # hold the card, that constructor is where audio search died with
            # "CUDA out of memory", and the .to("cpu") below never got its turn.
            model = laion_clap.CLAP_Module(
                enable_fusion=False,
                amodel='HTSAT-base',
                device="cpu",
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
