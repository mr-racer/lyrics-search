"""
Resources layer — singletons for models and database.

ModelRegistry:
- get_text_model(model_name) -> (model, vector_name, dim)
- encode_text(model_name, sentences, **kw) -> embeddings (tracked; use for ALL text encodes)
- begin_indexing/end_indexing(model_name) — pin the text model to the GPU for a dense pass
- load_clap() -> model
- list_text_models() -> dict of available models

Device policy (2026-07):
- CLAP lives on the CPU permanently: loaded once (startup preload), never
  moved, never unloaded. It no longer competes with text models for VRAM,
  which removed the GPU juggling the indexing path used to do.
- Text embedders load on the CPU and are moved to the GPU only for the
  dense-encode stage of indexing (``begin_indexing``). A background reaper
  demotes an idle model back to the CPU after ``TEXT_IDLE_TO_CPU_SEC`` (60s)
  of no requests — waiting out in-flight encodes — and unloads it entirely
  after ``TEXT_IDLE_UNLOAD_SEC`` (10 min) of inactivity. The next search or
  indexing run lazy-loads it again.

DbClient:
- __enter__/__exit__
- lyrics_db property (LyricsDB instance)
"""

import gc
import os
import threading
import time
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
# GPU target for the indexing-time text encode; None → everything stays on CPU.
_GPU_DEVICE = None if (_FORCE_CPU or not torch.cuda.is_available()) else torch.device("cuda")
if _FORCE_CPU:
    logger.info("[ModelRegistry] FORCE_CPU set — using CPU for all models")
else:
    logger.info(
        "[ModelRegistry] CLAP pinned to CPU; text models use %s during indexing",
        _GPU_DEVICE or "cpu",
    )
CLAP_WEIGHTS_PATH = Path(__file__).parent.parent.parent / "weights" / "music_audioset_epoch_15_esc_90.14.pt"
CLAP_WEIGHTS_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"

# Text-model lifecycle timings (seconds). Env-tunable for ops experiments.
TEXT_IDLE_TO_CPU_SEC = float(os.environ.get("TEXT_IDLE_TO_CPU_SEC", "60"))
TEXT_IDLE_UNLOAD_SEC = float(os.environ.get("TEXT_IDLE_UNLOAD_SEC", "600"))
_REAPER_INTERVAL_SEC = 15.0

# Available text embedding models. Vector storage in Qdrant is keyed per
# model (vector_name/dim come from get_text_model), so adding an entry here
# needs no migration. NOTE: e5 models nominally want "query:"/"passage:"
# prefixes; we skip them (accepted simplification, wizard spec §3.2 —
# consistent with how Qwen is used without instruction prefixes).
TEXT_MODELS = {
    "jinaai/jina-embeddings-v2-small-en": {"dim": 512, "desc": "Lightweight model with CPU optimisation"},
    "intfloat/multilingual-e5-base": {"dim": 768, "desc": "Balanced, multilingual"},
    "Qwen/Qwen3-Embedding-0.6B": {"dim": 1024, "desc": "Higher quality, slower"},
}


class _TextModelState:
    """Concurrency + lifecycle bookkeeping for one loaded text model.

    ``cond`` guards every device move: encodes register in ``inflight`` under
    it, movers (begin_indexing / the reaper) only call ``.to()`` while holding
    it with ``inflight == 0`` — so a tensor is never mid-forward on a moving
    model. ``indexing`` pins the model to the GPU across a dense pass.
    """

    __slots__ = ("cond", "inflight", "indexing", "last_used", "on_gpu")

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.inflight = 0
        self.indexing = 0
        self.last_used = time.monotonic()
        self.on_gpu = False


class ModelRegistry:
    """
    Singleton registry for models.
    - Text models (sentence-transformers) — multiple models cached by name,
      CPU-resident, GPU only during indexing, idle-demoted/unloaded (see the
      module docstring for the full device policy)
    - CLAP model (audio embeddings) — loaded once, always on CPU, never unloaded
    """

    # Phase B: get_text_model is the canonical accessor — returns the
    # (model, vector_name, vector_dim) triple and lazy-loads on first call.
    # Concurrent first-loads of the same name serialise on a per-name lock so
    # only one SentenceTransformer(...) instantiation happens (no GPU-mem waste).
    _text_models: dict[str, tuple[Any, str, int]] = {}
    _text_state: dict[str, _TextModelState] = {}
    _load_locks: dict[str, threading.Lock] = {}
    _load_locks_master: threading.Lock = threading.Lock()
    _clap_model: Optional[Any] = None
    _clap_lock: threading.Lock = threading.Lock()
    _reaper_started: bool = False

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
    def _state_for(cls, model_name: str) -> _TextModelState:
        """Return the per-model lifecycle state, creating it under the master lock.

        State survives unloads on purpose: last_used/inflight bookkeeping must
        not reset when the reaper drops the weights.
        """
        st = cls._text_state.get(model_name)
        if st is not None:
            return st
        with cls._load_locks_master:
            st = cls._text_state.get(model_name)
            if st is None:
                st = _TextModelState()
                cls._text_state[model_name] = st
            return st

    @classmethod
    def get_text_model(cls, model_name: str) -> tuple[Any, str, int]:
        """Return ``(model, vector_name, vector_dim)`` for ``model_name``.

        Lazy-loads on first call (onto the CPU — GPU placement happens only
        via ``begin_indexing``). Subsequent calls return the cached triple.
        Thread-safe: concurrent first-loads of the same name serialise on a
        per-name lock so only one ``SentenceTransformer(...)`` is instantiated.

        Every call counts as activity for the idle reaper. Callers must NOT
        cache the returned model across requests — that would pin weights the
        reaper is supposed to free; go through ``encode_text`` instead.
        """
        st = cls._state_for(model_name)
        st.last_used = time.monotonic()

        cached = cls._text_models.get(model_name)
        if cached is not None:
            return cached

        with cls._lock_for(model_name):
            cached = cls._text_models.get(model_name)
            if cached is not None:
                return cached
            model = SentenceTransformer(model_name, device="cpu")
            dim = model.get_sentence_embedding_dimension()
            vector_name = f"text_{model_name.replace('/', '_')}"
            triple = (model, vector_name, dim)
            with st.cond:
                st.on_gpu = False
                st.last_used = time.monotonic()
            cls._text_models[model_name] = triple
            cls._ensure_reaper()
            logger.info("[ModelRegistry] text model '%s' loaded (dim=%d, cpu)", model_name, dim)
            return triple

    @classmethod
    def load_text_model(cls, model_name: str) -> tuple[Any, str, int]:
        """Back-compat alias for ``get_text_model``. Existing callers
        (lyrics_search_engine, library_service, search/chat routes) keep
        working and now get the per-model load lock for free."""
        return cls.get_text_model(model_name)

    @classmethod
    def encode_text(cls, model_name: str, sentences, **encode_kwargs):
        """Encode ``sentences`` with ``model_name`` — the ONE sanctioned way to
        run a text encode outside indexing.

        Registers the call as in-flight so the reaper never moves or unloads
        the model mid-forward, and stamps ``last_used`` so activity delays the
        idle demotion/unload.
        """
        model, _, _ = cls.get_text_model(model_name)
        st = cls._state_for(model_name)
        with st.cond:
            st.inflight += 1
            st.last_used = time.monotonic()
        try:
            return model.encode(sentences, **encode_kwargs)
        finally:
            with st.cond:
                st.inflight -= 1
                st.last_used = time.monotonic()
                st.cond.notify_all()

    @classmethod
    def begin_indexing(cls, model_name: str) -> Any:
        """Pin ``model_name`` for an indexing dense pass and move it to the GPU.

        Waits (bounded) for in-flight search encodes to drain before the device
        move so a running forward pass is never yanked between devices. Returns
        the model; the caller MUST pair this with ``end_indexing`` in a finally.
        No-op device-wise when no GPU is available / FORCE_CPU is set.
        """
        model, _, _ = cls.get_text_model(model_name)
        st = cls._state_for(model_name)
        with st.cond:
            st.indexing += 1
            st.last_used = time.monotonic()
            if _GPU_DEVICE is not None and not st.on_gpu:
                deadline = time.monotonic() + 30.0
                while st.inflight > 0 and time.monotonic() < deadline:
                    st.cond.wait(timeout=1.0)
                try:
                    model.to(_GPU_DEVICE)
                    st.on_gpu = True
                    logger.info("[ModelRegistry] text model '%s' → GPU (indexing)", model_name)
                except Exception:
                    logger.exception(
                        "[ModelRegistry] failed to move '%s' to GPU — dense pass stays on CPU",
                        model_name,
                    )
        return model

    @classmethod
    def end_indexing(cls, model_name: str) -> None:
        """Release the indexing pin taken by ``begin_indexing``.

        The model intentionally STAYS on the GPU here: back-to-back batches
        shouldn't thrash devices. The reaper demotes it to the CPU after
        ``TEXT_IDLE_TO_CPU_SEC`` of quiet.
        """
        st = cls._state_for(model_name)
        with st.cond:
            st.indexing = max(0, st.indexing - 1)
            st.last_used = time.monotonic()
            st.cond.notify_all()

    # ── Idle reaper ──

    @classmethod
    def _ensure_reaper(cls) -> None:
        """Start the daemon reaper thread once (lazily, on first model load)."""
        with cls._load_locks_master:
            if cls._reaper_started:
                return
            threading.Thread(
                target=cls._reaper_loop, name="model-registry-reaper", daemon=True,
            ).start()
            cls._reaper_started = True

    @classmethod
    def _reaper_loop(cls) -> None:
        while True:
            time.sleep(_REAPER_INTERVAL_SEC)
            try:
                cls._reap_once()
            except Exception:
                logger.exception("[ModelRegistry] reaper iteration failed")

    @classmethod
    def _reap_once(cls, now: float | None = None) -> None:
        """One reaper sweep: GPU→CPU after TEXT_IDLE_TO_CPU_SEC idle, full
        unload after TEXT_IDLE_UNLOAD_SEC. Skips models with in-flight encodes
        or an active indexing pin — never kills running work."""
        now = now if now is not None else time.monotonic()
        for name in list(cls._text_models.keys()):
            st = cls._state_for(name)
            with st.cond:
                if st.inflight > 0 or st.indexing > 0:
                    continue
                triple = cls._text_models.get(name)
                if triple is None:
                    continue
                idle = now - st.last_used
                model = triple[0]
                if st.on_gpu and idle >= TEXT_IDLE_TO_CPU_SEC:
                    try:
                        model.to("cpu")
                        st.on_gpu = False
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.info(
                            "[ModelRegistry] text model '%s' → CPU (%.0fs idle)", name, idle,
                        )
                    except Exception:
                        logger.exception("[ModelRegistry] GPU→CPU demotion failed for '%s'", name)
                        continue
                if idle >= TEXT_IDLE_UNLOAD_SEC:
                    cls._text_models.pop(name, None)
                    st.on_gpu = False
                    del model, triple
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info(
                        "[ModelRegistry] text model '%s' unloaded (%.0fs idle) — "
                        "will lazy-reload on next use", name, idle,
                    )

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
