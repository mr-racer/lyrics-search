"""LyricsSearchEngine — the canonical search-facing wrapper over a Qdrant collection.

Split from legacy ``search_engine.main.LyricsDB`` during Refactor 4. Holds only the
SEARCH responsibilities:

- Qdrant connection sanity check
- Lazy/eager loading of text + CLAP models (delegated to ModelRegistry)
- Hybrid dense + sparse (BM25) + CLAP-audio search via ``query_points``

INDEXING (fit, collection creation, batch upsert, payload building) lives in
``app.services.indexing_service`` (Refactor 5).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from qdrant_client import QdrantClient, models

from .qdrant_filters import build_filter

logger = logging.getLogger(__name__)


class LyricsSearchEngine:
    """Search-focused wrapper over one Qdrant collection.

    STATELESS DISPATCH (Phase B): ``search()`` accepts ``model_name`` and
    resolves the text model via ``ModelRegistry.get_text_model`` per call. It
    must NOT mutate ``self.model_name`` / ``self._model`` / ``self._vector_name``
    in the search path — that pattern caused concurrent searches from two
    accounts with different models to race (one thread's model leaked into the
    other's Qdrant query). If you reach for a ``self._model = ...`` assignment
    inside ``search()`` you are reintroducing the race. See
    docs/superpowers/plans/2026-06-02-phase-b-singleton-fixes.md.

    Model loading is lazy by default. The ``__init__`` ``model_name`` remains a
    per-engine default (used when ``search`` is called with ``model_name=None``)
    and the pre-load target for the indexing path (which calls ``engine.model``
    directly to encode batches). Pre-loaded models can be passed via ``model`` /
    ``model_clap`` to skip lazy loading entirely.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        model_name: str,
        include_clap: bool = False,
        lazy: bool = True,
        model: Any = None,
        model_clap: Any = None,
    ):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self._init_qdrant()

        self.model_name = model_name
        self.include_clap = include_clap
        self._model_lock = threading.Lock()

        # ── Text model: pre-loaded > eager > lazy ───────────────────────────
        if model is not None:
            self._model = model
            self._vector_name = f"text_{self.model_name.replace('/', '_')}"
            self._vector_dim = model.get_sentence_embedding_dimension()
            logger.info(
                "[LyricsSearchEngine] Using pre-loaded text model '%s' (dim=%d)",
                model_name, self._vector_dim,
            )
        elif not lazy:
            # Eager = warm the registry cache now, but do NOT pin the model on
            # self: the registry's idle reaper must stay able to unload it.
            from .model_registry import ModelRegistry
            self._model = None
            _, self._vector_name, self._vector_dim = ModelRegistry.load_text_model(self.model_name)
            logger.info(
                "[LyricsSearchEngine] Text model '%s' loaded eagerly via ModelRegistry (dim=%d)",
                model_name, self._vector_dim,
            )
        else:
            self._model = None
            self._vector_name = None
            self._vector_dim = None
            logger.info(
                "[LyricsSearchEngine] Text model '%s' — lazy load enabled", model_name,
            )

        # ── CLAP model: same pre-loaded > eager > lazy ──────────────────────
        if model_clap is not None:
            self._model_clap = model_clap
            logger.info("[LyricsSearchEngine] Using pre-loaded CLAP model")
        elif include_clap and not lazy:
            from .model_registry import ModelRegistry
            self._model_clap = ModelRegistry.load_clap()
            logger.info("[LyricsSearchEngine] CLAP loaded eagerly via ModelRegistry")
        else:
            self._model_clap = None
            if include_clap:
                logger.info("[LyricsSearchEngine] CLAP — lazy load enabled")

    # ── Lazy model accessors ────────────────────────────────────────────────

    @property
    def _model_config(self) -> tuple[str, int]:
        """Return (vector_name, vector_dim) — loads model if needed."""
        if self._vector_name is None:
            self._ensure_model()
        return self._vector_name, self._vector_dim

    @property
    def vector_name(self) -> str:
        return self._model_config[0]

    @property
    def vector_dim(self) -> int:
        return self._model_config[1]

    @property
    def model(self):
        """Return the text model for the engine's default ``model_name``.

        Resolved through ModelRegistry on EVERY access (a dict hit once
        loaded) instead of being cached on ``self`` — caching here would pin
        the weights and defeat the registry's idle unload. A ctor-supplied
        pre-loaded ``model`` (tests) is the only thing kept on the instance.
        """
        if self._model is not None:
            return self._model
        from .model_registry import ModelRegistry
        m, self._vector_name, self._vector_dim = ModelRegistry.get_text_model(self.model_name)
        return m

    @property
    def model_clap(self):
        """Return the CLAP model, loading lazily via ModelRegistry."""
        self._ensure_clap()
        return self._model_clap

    def _ensure_model(self) -> None:
        """Resolve vector metadata for the default model (no pinning)."""
        if self._vector_name is not None:
            return
        with self._model_lock:
            if self._vector_name is not None:
                return
            from .model_registry import ModelRegistry
            logger.info("[LyricsSearchEngine] Loading text model '%s' via ModelRegistry...", self.model_name)
            _, self._vector_name, self._vector_dim = ModelRegistry.load_text_model(self.model_name)
            logger.info("[LyricsSearchEngine] Text model loaded (dim=%d)", self._vector_dim)

    def _ensure_clap(self) -> None:
        if self._model_clap is not None or not self.include_clap:
            return
        with self._model_lock:
            if self._model_clap is not None:
                return
            from .model_registry import ModelRegistry
            logger.info("[LyricsSearchEngine] Loading CLAP via ModelRegistry...")
            self._model_clap = ModelRegistry.load_clap()
            logger.info("[LyricsSearchEngine] CLAP loaded")

    # ── Qdrant init ─────────────────────────────────────────────────────────

    def _init_qdrant(self) -> None:
        try:
            self.qdrant_client.get_collections()
        except Exception as e:
            raise ConnectionError(
                "Qdrant не запущен/не обнаружен. Пожалуйста, выключите VPN или перезапустите Docker"
            ) from e

    # ── Search ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 1,
        include_clap: bool = False,
        min_dense_score: float = 0.4,
        min_clap_score: float = 0.01,
        artist: str | None = None,
        album: str | None = None,
        title: str | None = None,
        genre: str | list[str] | None = None,
        year: int | None = None,
        year_ranges: list[str] | None = None,
        sonic_tags: list[str] | None = None,
        collection_name_override: str | None = None,
        model_name: str | None = None,
    ) -> list[models.ScoredPoint]:
        """Hybrid (dense + BM25) or CLAP search against the collection.

        ``model_name`` selects the text embedding model and is resolved via
        ``ModelRegistry.get_text_model`` per call (stateless dispatch); falls
        back to ``self.model_name`` when not supplied. The engine does NOT cache
        the resolved model on ``self`` — keeping search stateless makes
        concurrent calls with different models safe.

        When ``include_clap=True`` the query is converted to a CLAP text embedding
        and matched against the ``clap`` audio vectors (CLAP is the single global
        audio model, so it is not per-account). Otherwise the query is
        sentence-encoded + matched against the dense lyric vectors, fused (RRF)
        with sparse BM25 results over the metadata+lyrics document.
        """
        col = collection_name_override or self.collection_name

        query_filter = build_filter(
            artist=artist, album=album, title=title, genre=genre,
            year=year, year_ranges=year_ranges, sonic_tags=sonic_tags,
        )

        if not include_clap:
            # Resolve the text model per call — stateless dispatch, no self.* write.
            # encode_text (not raw model.encode) so the registry's in-flight
            # tracking keeps the idle reaper from moving/unloading mid-encode.
            from .model_registry import ModelRegistry
            effective_model_name = model_name or self.model_name
            _, vector_name, _vector_dim = ModelRegistry.get_text_model(effective_model_name)
            query_vector = ModelRegistry.encode_text(effective_model_name, query).tolist()
            results = self.qdrant_client.query_points(
                collection_name=col,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using=vector_name,
                        limit=15,
                        score_threshold=min_dense_score,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=models.Document(text=query, model="Qdrant/bm25"),
                        using="bm25",
                        limit=25,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            clap_vector = self.model_clap.get_text_embedding([query])[0].tolist()
            results = self.qdrant_client.query_points(
                collection_name=col,
                prefetch=[
                    models.Prefetch(
                        query=clap_vector,
                        using="clap",
                        limit=15,
                        score_threshold=min_clap_score,
                        filter=query_filter,
                    )
                ],
                limit=limit,
                with_payload=True,
            )

        return results.points
