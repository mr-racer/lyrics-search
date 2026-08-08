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

    There is exactly one text embedding model in this application
    (``ModelRegistry.TEXT_MODEL_NAME``) and one dense vector name
    (``ModelRegistry.VECTOR_NAME``), so this class no longer dispatches on a
    model: the per-call ``model_name`` plumbing that used to live here existed
    only to keep two accounts on two different models from racing, and there is
    now nothing to race over.

    Model loading stays lazy — ``ModelRegistry`` owns it and caches it process
    wide. A pre-loaded ``model`` / ``model_clap`` may still be injected by tests.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        include_clap: bool = False,
        lazy: bool = True,
        model: Any = None,
        model_clap: Any = None,
    ):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self._init_qdrant()

        self.include_clap = include_clap
        self._model_lock = threading.Lock()
        self._model = model

        if model is None and not lazy:
            # Eager = warm the registry cache now, but do NOT pin the model on
            # self; the registry is the single owner of the weights.
            from .model_registry import ModelRegistry
            ModelRegistry.get_text_model()

        # ── CLAP model: pre-loaded > eager > lazy ───────────────────────────
        if model_clap is not None:
            self._model_clap = model_clap
        elif include_clap and not lazy:
            from .model_registry import ModelRegistry
            self._model_clap = ModelRegistry.load_clap()
        else:
            self._model_clap = None

    # ── Model accessors ─────────────────────────────────────────────────────

    @property
    def vector_name(self) -> str:
        from .model_registry import ModelRegistry
        return ModelRegistry.VECTOR_NAME

    @property
    def vector_dim(self) -> int:
        from .model_registry import ModelRegistry
        return ModelRegistry.VECTOR_DIM

    @property
    def model(self):
        """The text model. Resolved through ModelRegistry on every access (a
        cheap attribute hit once loaded); a ctor-supplied model (tests) wins."""
        if self._model is not None:
            return self._model
        from .model_registry import ModelRegistry
        return ModelRegistry.get_text_model()[0]

    @property
    def model_clap(self):
        """Return the CLAP model, loading lazily via ModelRegistry."""
        self._ensure_clap()
        return self._model_clap

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
    ) -> list[models.ScoredPoint]:
        """Hybrid (dense + BM25) or CLAP search against the collection.

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
            # is_query=True: this side of the pair takes the instruction prefix.
            from .model_registry import ModelRegistry
            vector_name = ModelRegistry.VECTOR_NAME
            query_vector = ModelRegistry.encode_text(query, is_query=True).tolist()
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
