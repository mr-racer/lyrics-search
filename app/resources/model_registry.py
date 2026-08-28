"""
Resources layer — singletons for models and database.

ModelRegistry:
- get_text_model() -> (model, VECTOR_NAME, VECTOR_DIM)
- encode_text(sentences, is_query=False, **kw) -> embeddings (use for ALL text encodes)
- encode_sparse(texts, is_query=False) -> coalesced sparse tensor
- ce_probabilities(query, docs) -> list[float]
- load_clap() -> model

Every one of the text legs RAISES on failure (``app/resources/models/errors.py``).
They used to answer failure with ``None``, which read exactly like "encoded
nothing" at the call site; a leg could then be dead for hours while the only
symptom was worse answers. ``None`` and empty results now mean "you passed no
input" and nothing else. Degrading is still correct in the retriever, but it is
a decision written at a named site now, not the default that fell out of a bare
``None``.

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

# Pure python, no torch: this package is the shared vocabulary for failures and
# counters, and it is imported at module level on purpose — a caller that has to
# catch something must not need the ML stack to name it.
from .models import (STATS, CircuitBreaker, ModelEncodeFailed, ModelOOM,
                     ModelUnavailable)

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


def _length_sorted_order(texts: list) -> tuple[list, list]:
    """``(order, inverse)`` for encoding the longest texts first.

    ``order`` lists input positions longest-first; ``inverse[i]`` says where row
    ``i`` of the caller's list ended up, so the encoded rows can be permuted
    back before anyone sees them.

    Two reasons for longest-first rather than any grouping:

    * A padded batch costs its LONGEST member for every row it holds, so mixing
      a 500-character passage with three one-liners pays for four long ones.
      The retriever's corpus is exactly that mixture.
    * The biggest batch is then attempted first. If it fits, every later batch
      fits; if it does not, ``encode_sparse`` shrinks before the run has spent
      anything, instead of discovering the ceiling nine batches in.

    Character length rather than token count on purpose: tokenising twice to
    save padding would cost more than the padding does, and it is the same
    proxy ``sentence_transformers`` sorts by internally.
    """
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i] or ""))
    inverse = [0] * len(order)
    for position, original in enumerate(order):
        inverse[original] = position
    return order, inverse


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
# 8, not 16: the cross-encoder reads (question, passage) pairs through a
# 24-layer, 1024-wide XLM-R-large, so 16 x 512 tokens is ~8k tokens of
# activations in one forward. Halving the batch halves that peak and costs
# almost nothing in time — the FLOPs are identical either way, and 8 x 512
# still leaves the GEMMs far above the size where an Ampere card saturates.
RERANK_BATCH = 8


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
    # A leg that tried to load and failed is not retried STRAIGHT AWAY — a
    # retry on every query turns one slow request into every request being
    # slow. It is no longer permanent, though: a load can fail because the card
    # was full at that moment, and that clears on its own. See
    # ``models/breaker.py``.
    _breaker = CircuitBreaker()
    # Loading and ENCODING fail independently, and only the first one had a
    # name. A leg whose weights are resident but whose every encode dies still
    # reported itself as up, so hours of dense-only ranking looked exactly like
    # "the answers got worse". The counters that say so live in ``STATS`` now,
    # next to the DEGRADATION counts — how often a caller carried on without a
    # leg — which is the other half of the same question.

    # ── Text model ──

    @classmethod
    def get_text_model(cls) -> tuple[Any, str, int]:
        """Return ``(model, VECTOR_NAME, VECTOR_DIM)``, loading on first call.

        Thread-safe: concurrent first calls serialise on ``_text_lock`` so only
        one ``SentenceTransformer(...)`` is ever instantiated — a duplicate load
        would double the VRAM footprint for nothing.
        """
        import torch
        cached = cls._text_model
        if cached is not None:
            return cached
        cls._raise_if_breaker_open("dense", TEXT_MODEL_NAME)
        try:
            SentenceTransformer = _sentence_transformer_cls()
        except RuntimeError as e:
            cls._breaker.trip("dense", str(e))
            raise ModelUnavailable("dense", "load", str(e), cause=e) from e

        with cls._text_lock:
            if cls._text_model is not None:
                return cls._text_model
            cls._raise_if_breaker_open("dense", TEXT_MODEL_NAME)

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
                try:
                    model = SentenceTransformer(TEXT_MODEL_NAME, device=device,
                                                **kwargs)
                except TypeError as e:
                    # ONLY our own kwargs may send us down this path: an older
                    # sentence-transformers has neither, and fp32 on the GPU
                    # still beats fp16 on the CPU by a wide margin. A TypeError
                    # from deeper in the loader is a different failure — the
                    # retry cannot fix it, repeats it, and files it under a
                    # warning that blames the wrong thing. Let that one out.
                    if not any(name in str(e) for name in kwargs):
                        raise
                    logger.warning("[ModelRegistry] model_kwargs/tokenizer_kwargs"
                                   " unsupported — loading fp32 on %s", device)
                    model = SentenceTransformer(TEXT_MODEL_NAME, device=device)
            except Exception as e:  # noqa: BLE001 — re-raised as ModelUnavailable
                cls._breaker.trip("dense", f"{type(e).__name__}: {e}")
                logger.error("[ModelRegistry] text model '%s' would not load: %s",
                             TEXT_MODEL_NAME, e, exc_info=True)
                raise ModelUnavailable(
                    "dense", "load",
                    f"'{TEXT_MODEL_NAME}' would not load: {e}", cause=e) from e
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
        import torch
        try:
            return model.encode(sentences, **encode_kwargs)
        except torch.OutOfMemoryError as e:
            # Named apart from every other failure because it is the one the
            # caller can DO something about: ``encode_documents`` halves the
            # batch and carries on. Wrapping it into the generic failure would
            # turn a recoverable moment into a lost index run.
            raise ModelOOM("dense", "encode", str(e), cause=e) from e
        except Exception as e:  # noqa: BLE001 — re-raised as ModelEncodeFailed
            STATS.encode_failed("dense")
            logger.warning("[ModelRegistry] dense encode failed", exc_info=True)
            raise ModelEncodeFailed("dense", "encode", str(e), cause=e) from e

    @classmethod
    def encode_documents(cls, texts: list, *, progress=None,
                         is_query: bool = False):
        """Dense vectors for a WHOLE corpus, as one ``(n, VECTOR_DIM)`` array.

        ``is_query`` exists for the HTTP surface, which has to offer both sides
        of the asymmetric pair and wants the same batch ceiling and the same
        OOM shrink for each. Indexing leaves it at the default and reads on.

        The indexing pass's entry point. It exists because that pass used to
        call ``model.encode(texts, batch_size=32)`` directly, and the two things
        wrong with that were the two things this fixes.

        **The batch.** 32 texts at ``MAX_SEQ_LENGTH`` is 65k tokens in one
        forward. Whenever SDPA cannot take a memory-efficient path, attention
        materialises as ``batch x heads x tokens x tokens`` — 4.3 GiB at that
        size — and the card is shared with an LLM. ``ENCODE_BATCH`` is the
        number the rest of the dense leg already uses; there is no reason this
        one caller got its own, larger one.

        **The prompt.** Reaching past ``encode_text`` also skipped the document
        prompt. That was correct for Qwen3-Embedding, whose document side
        genuinely takes nothing, and it is wrong for Octen — and a corpus
        embedded one way cannot be searched by queries embedded the other.

        An allocator refusal shrinks the batch and keeps what is already done,
        as in ``encode_sparse``. It does NOT degrade to a partial result: a
        library written with missing vectors would look like a finished index
        while search quietly skipped those tracks, so the last resort here is
        to raise and be re-run, not to return less.
        """
        import numpy as np

        if not texts:
            return np.zeros((0, VECTOR_DIM), dtype=np.float32)

        chunks: list = []
        size = ENCODE_BATCH
        i = 0
        while i < len(texts):
            try:
                chunks.append(cls.encode_text(
                    texts[i:i + size], is_query=is_query,
                    batch_size=size, convert_to_numpy=True))
                i += size
                if progress:
                    progress(min(i, len(texts)))
            except ModelOOM:
                attempted = min(size, len(texts) - i)
                if attempted == 1:
                    STATS.encode_failed("dense")
                    logger.error(
                        "[ModelRegistry] dense encode out of memory at one text "
                        "(%d of %d done) — the card has nothing left, so this "
                        "index run stops rather than write a library with holes "
                        "in it", i, len(texts))
                    raise
                size = attempted // 2
                STATS.oom_retry("dense")
                _release_cuda_cache()
                logger.warning(
                    "[ModelRegistry] dense encode hit OOM at %d of %d texts — "
                    "continuing at batch=%d", i, len(texts), size)
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

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
    def load_sparse(cls) -> Any:
        """The learned-sparse encoder.

        Raises :class:`ModelUnavailable` when it will not load, and again —
        without re-attempting — for as long as the breaker stays open. It used
        to return ``None`` instead, which read identically to "encoded nothing"
        at the call site and is why a dead leg could rank an entire session
        without anyone noticing.
        """
        import torch
        if cls._sparse_model is not None:
            return cls._sparse_model
        cls._raise_if_breaker_open("sparse", SPARSE_MODEL_NAME)

        with cls._sparse_lock:
            if cls._sparse_model is not None:
                return cls._sparse_model
            cls._raise_if_breaker_open("sparse", SPARSE_MODEL_NAME)
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
            except Exception as e:  # noqa: BLE001 — re-raised as ModelUnavailable
                cls._breaker.trip("sparse", f"{type(e).__name__}: {e}")
                logger.warning("[ModelRegistry] sparse model '%s' unavailable: %s",
                               SPARSE_MODEL_NAME, e, exc_info=True)
                raise ModelUnavailable(
                    "sparse", "load",
                    f"'{SPARSE_MODEL_NAME}' unavailable: {e}", cause=e) from e
            return cls._sparse_model

    @classmethod
    def _raise_if_breaker_open(cls, leg: str, model_name: str) -> None:
        """Refuse fast while ``leg``'s breaker is open.

        Shared by all three loaders so the message is the same wherever it comes
        from — the caller is going to log it, and a leg that reads differently
        depending on which loader refused is a leg nobody can grep for.
        """
        if cls._breaker.is_open(leg):
            raise ModelUnavailable(
                leg, "load",
                f"'{model_name}': {cls._breaker.reason(leg) or 'load failed'} "
                "(not retried yet)")

    @classmethod
    def load_reranker(cls) -> tuple[Any, Any]:
        """``(tokenizer, model)`` for the cross-encoder.

        Raises :class:`ModelUnavailable` rather than returning ``None`` — see
        :meth:`load_sparse`. The breaker is keyed ``cross_encoder``, matching
        ``retrieval_status`` and the counters; the leg was called ``reranker``
        in the old ``_failed`` set and nowhere else, and one name is enough.
        """
        import torch
        if cls._reranker is not None:
            return cls._reranker
        cls._raise_if_breaker_open("cross_encoder", RERANK_MODEL_NAME)

        with cls._rerank_lock:
            if cls._reranker is not None:
                return cls._reranker
            cls._raise_if_breaker_open("cross_encoder", RERANK_MODEL_NAME)
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
            except Exception as e:  # noqa: BLE001 — re-raised as ModelUnavailable
                cls._breaker.trip("cross_encoder", f"{type(e).__name__}: {e}")
                logger.warning("[ModelRegistry] cross-encoder '%s' unavailable: %s",
                               RERANK_MODEL_NAME, e, exc_info=True)
                raise ModelUnavailable(
                    "cross_encoder", "load",
                    f"'{RERANK_MODEL_NAME}' unavailable: {e}", cause=e) from e
            return cls._reranker

    @classmethod
    def encode_sparse(cls, texts: list, *, is_query: bool = False):
        """Learned-sparse vectors as one coalesced sparse tensor.

        ``None`` ONLY for an empty ``texts``. A leg that will not load
        raises :class:`ModelUnavailable`, an allocator that refuses even a
        single text raises :class:`ModelOOM`, anything else raises
        :class:`ModelEncodeFailed`.

        Asymmetric like the dense model: queries and documents go through
        different heads, so the side is named rather than inferred.

        Truncated at ``SPARSE_MAX_LEN`` and batched at ``SPARSE_BATCH`` — see
        those constants for why this leg gets its own two numbers instead of
        the dense model's.

        Encoded LONGEST-FIRST and permuted back before returning: MILCO pads
        each batch to its longest member, so grouping by length is free memory
        on the retriever's mixed corpus (see ``_length_sorted_order``). The
        permutation is load-bearing — the retriever addresses these rows by
        document position, so leaving them in encode order would score
        passages against the wrong documents rather than fail.

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
        # Emptiness is checked BEFORE the load, not after. The load can now
        # raise, and an empty batch must not be the thing that discovers the leg
        # is down — callers pass empty lists on perfectly ordinary paths, and
        # ``None`` here still means "you gave me nothing", never "it broke".
        if not texts:
            return None
        model = cls.load_sparse()
        encode = model.encode_query if is_query else model.encode_document

        order, inverse = _length_sorted_order(texts)
        ordered = [texts[i] for i in order]

        reps: list = []
        size = SPARSE_BATCH
        i = 0
        while i < len(ordered):
            try:
                with torch.no_grad():
                    reps.append(encode(ordered[i:i + size],
                                       max_length=SPARSE_MAX_LEN,
                                       source_view=SPARSE_SOURCE_VIEW).coalesce())
                i += size
            except torch.OutOfMemoryError as e:
                # Halve what was ACTUALLY attempted, not the nominal size: on a
                # short tail ``size`` can already exceed the texts left, and
                # halving the nominal would re-send the identical batch and pay
                # a second failed forward for nothing.
                attempted = min(size, len(ordered) - i)
                if attempted == 1:
                    STATS.encode_failed("sparse")
                    logger.warning(
                        "[ModelRegistry] sparse encode out of memory even at one "
                        "text (%d of %d done) — the retriever will rank without "
                        "the sparse leg. Something else on the card grew; see "
                        "GET /search/models/loaded for the running count.",
                        i, len(ordered))
                    raise ModelOOM(
                        "sparse", "encode",
                        f"out of memory at a single text ({i} of {len(ordered)} "
                        f"done)", cause=e) from e
                size = attempted // 2
                STATS.oom_retry("sparse")
                _release_cuda_cache()
                logger.info(
                    "[ModelRegistry] sparse encode hit OOM at %d of %d texts — "
                    "continuing at batch=%d", i, len(ordered), size)
            except Exception as e:  # noqa: BLE001 — re-raised as ModelEncodeFailed
                STATS.encode_failed("sparse")
                logger.warning("[ModelRegistry] sparse encode failed", exc_info=True)
                raise ModelEncodeFailed("sparse", "encode", str(e), cause=e) from e

        combined = torch.cat(reps, dim=0).coalesce()
        if len(texts) > 1:
            # Back into the caller's order. Not cosmetic: the retriever holds
            # this tensor and index_selects into it with positions that came
            # from the fused ranking, so rows left sorted by length would score
            # every passage against a different document — silently, and only
            # visible as worse answers.
            combined = torch.index_select(
                combined, 0,
                torch.tensor(inverse, dtype=torch.long, device=combined.device),
            ).coalesce()
        return combined

    @classmethod
    def ce_probabilities(cls, query: str, docs: list) -> list:
        """``sigmoid(logit)`` per (query, doc) pair.

        An empty ``docs`` gives an empty list; everything else that goes
        wrong raises. The gates in ``bio_v2`` used to read the old ``None``
        as "score everything 1.0", which admitted the whole candidate pool
        precisely when nothing could judge it.

        A probability rather than a raw logit because every threshold in the
        assistant is one number compared against this: logits are not comparable
        between model families, probabilities roughly are.
        """
        import torch
        # Emptiness first, for the same reason as ``encode_sparse``: no
        # documents is a normal call, not a broken leg.
        if not docs:
            return []
        tokenizer, model = cls.load_reranker()
        device = cls._shared_device()
        try:
            out: list[float] = []
            # ``inference_mode`` rather than ``no_grad``: it additionally drops
            # version-counter and view tracking. Safe HERE specifically because
            # nothing survives the block — the scores leave as a plain list of
            # floats. The sparse leg deliberately keeps ``no_grad``: its tensor
            # escapes into the retriever, which holds and re-slices it, and
            # inference tensors carry restrictions that are not worth the
            # nothing this would save there (MILCO already sets its own grad
            # context internally, so the forward is unaffected either way).
            with torch.inference_mode():
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
        except torch.OutOfMemoryError as e:
            STATS.encode_failed("cross_encoder")
            logger.warning("[ModelRegistry] cross-encoder out of memory on %d "
                           "pairs", len(docs))
            raise ModelOOM("cross_encoder", "score", str(e), cause=e) from e
        except Exception as e:  # noqa: BLE001 — re-raised as ModelEncodeFailed
            STATS.encode_failed("cross_encoder")
            logger.warning("[ModelRegistry] cross-encoder scoring failed",
                           exc_info=True)
            raise ModelEncodeFailed("cross_encoder", "score", str(e),
                                    cause=e) from e

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
        counters = STATS.snapshot()
        failures = counters["encode_failures"]
        return {
            "device": _device_in_use,
            "dense": cls._text_model is not None,
            "sparse": cls._sparse_model is not None,
            "cross_encoder": cls._reranker is not None,
            # Legs whose breaker is currently open. Same key it had as a set,
            # because the route and its test already read that name — but it now
            # empties itself when the breaker expires, so a leg that reappears
            # here after a while really did fail again.
            "failed": cls._breaker.open_legs(),
            "encode_failures": {
                "sparse": failures.get("sparse", 0),
                "cross_encoder": failures.get("cross_encoder", 0),
                "dense": failures.get("dense", 0),
            },
            "sparse_oom_retries": counters["oom_retries"].get("sparse", 0),
            "oom_retries": counters["oom_retries"],
            # How often a caller CHOSE to carry on without a leg, keyed
            # ``leg/site``. The booleans above say what is loaded and the
            # failures say what broke; only this says what the user actually
            # got, which is the question the other two kept failing to answer.
            "degradations": counters["degradations"],
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
