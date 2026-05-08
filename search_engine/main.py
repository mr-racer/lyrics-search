from __future__ import annotations

from .utils import (
    load_model, prepare_metadata, build_filter,
    build_text_for_embedding, _encode_clap, load_model_clap
)
from file_processor.utils import get_metadata

import asyncio
import gc
import logging
import threading
import uuid
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from pathlib import Path

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


class LyricsDB:
    """Manage a Qdrant collection with hybrid dense and sparse (BM25) lyric embeddings.

    Model loading is lazy by default (lazy=True) — the text model and CLAP are loaded
    on first actual use (search/fit), not in __init__.  This allows the FastAPI server
    to start instantly and preload models in the background.

    A pre-loaded model can be passed via +model+ / +model_clap+ to skip lazy loading
    entirely (useful when ModelRegistry already cached the model).
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        model_name: str,
        include_clap: bool = False,
        lazy: bool = True,
        model = None,
        model_clap = None,
    ):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self._init_qdrant()

        self.model_name = model_name
        self.include_clap = include_clap

        # Pre-loaded model takes priority
        if model is not None:
            self._model = model
            # Derive vector config from the model
            vd = model.get_sentence_embedding_dimension()
            self._vector_name = f"text_{self.model_name.replace('/', '_')}"
            self._vector_dim = vd
            logger.info("[LyricsDB] Using pre-loaded text model '%s' (dim=%d)",
                        model_name, vd)
        elif not lazy:
            self._model, vn, vd = load_model(self.model_name)
            self._vector_name = vn
            self._vector_dim = vd
            logger.info("[LyricsDB] Text model '%s' loaded eagerly (dim=%d)",
                        model_name, vd)
        else:
            self._model = None
            self._vector_name = None
            self._vector_dim = None
            logger.info("[LyricsDB] Text model '%s' — lazy load enabled", model_name)

        self._model_lock = threading.Lock()

        if model_clap is not None:
            self._model_clap = model_clap
            logger.info("[LyricsDB] Using pre-loaded CLAP model")
        elif include_clap and not lazy:
            self._model_clap = load_model_clap()
            logger.info("[LyricsDB] CLAP loaded eagerly")
        else:
            self._model_clap = None
            if include_clap:
                logger.info("[LyricsDB] CLAP — lazy load enabled")

    # ── Lazy model accessors ─────────────────────────────────────────────────

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
        """Return the text model, loading lazily if needed."""
        self._ensure_model()
        return self._model

    @property
    def model_clap(self):
        """Return the CLAP model, loading lazily if needed."""
        self._ensure_clap()
        return self._model_clap

    def _ensure_model(self):
        """Load text model on first access (thread-safe).

        Checks ModelRegistry cache first (populated by background preload),
        falls back to loading directly via load_model().
        """
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return  # double-check

            # Try ModelRegistry cache first (avoids duplicate GPU memory)
            try:
                from app.resources.model_registry import ModelRegistry
                cached = ModelRegistry._text_models.get(self.model_name)
                if cached is not None:
                    self._model = cached[0]
                    self._vector_name = cached[1]
                    self._vector_dim = cached[2]
                    logger.info("[LyricsDB] Text model '%s' loaded from ModelRegistry cache",
                                self.model_name)
                    return
            except (ImportError, ModuleNotFoundError):
                pass  # app not available (running outside FastAPI)

            # Fallback: load directly
            logger.info("[LyricsDB] Lazy-loading text model '%s' ...", self.model_name)
            self._model, self._vector_name, self._vector_dim = load_model(self.model_name)
            logger.info("[LyricsDB] Text model loaded (dim=%d)", self._vector_dim)

    def _ensure_clap(self):
        """Load CLAP model on first access (thread-safe).

        Checks ModelRegistry cache first (populated by background preload),
        falls back to loading directly via load_model_clap().
        """
        if self._model_clap is not None:
            return
        with self._model_lock:
            if self._model_clap is not None:
                return
            if not self.include_clap:
                return  # CLAP not needed

            # Try ModelRegistry cache first
            try:
                from app.resources.model_registry import ModelRegistry
                cached = ModelRegistry._clap_model
                if cached is not None:
                    self._model_clap = cached
                    logger.info("[LyricsDB] CLAP loaded from ModelRegistry cache")
                    return
            except (ImportError, ModuleNotFoundError):
                pass  # app not available

            # Fallback: load directly
            logger.info("[LyricsDB] Lazy-loading CLAP ...")
            self._model_clap = load_model_clap()
            logger.info("[LyricsDB] CLAP loaded")

    # ── Qdrant init ──────────────────────────────────────────────────────────

    def _init_qdrant(self):
        try:
            self.qdrant_client.get_collections()
        except Exception as e:
            raise ConnectionError(
                "Qdrant не запущен/не обнаружен. Пожалуйста, выключите VPN или перезапустите Docker"
            ) from e



    def _create_collection(self, clap_paths: list):
        collections = self.qdrant_client.get_collections().collections
        exists = any(c.name == str(self.collection_name) for c in collections)
        if exists:
            self.qdrant_client.delete_collection(self.collection_name)

        vectors_config = {
            self.vector_name: models.VectorParams(
                size=self.vector_dim,
                distance=models.Distance.COSINE,
            ),
        }
        if clap_paths:
            vectors_config["clap"] = models.VectorParams(
                size=512,
                distance=models.Distance.COSINE,
            )

        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        print(f"Коллекция {self.collection_name} была успешно создана")


    def _upsert_in_batches(
        self,
        data: list[dict],
        text_vecs: np.ndarray,
        clap_map: dict = {},
        batch_size: int = 32,
    ):
        matched = 0
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i : i + batch_size]
            vecs  = text_vecs[i : i + batch_size]

            points = []
            for song_info, vec in zip(batch, vecs):
                vector = {
                    "bm25": models.Document(text=build_text_for_embedding(song_info), model="Qdrant/bm25"),
                    self.vector_name: vec,
                }
                if clap_map:
                    key = (song_info.get("artist", "").strip().lower(),
                           song_info.get("title", "").strip().lower())
                    clap_vec = clap_map.get(key)
                    if clap_vec is not None:
                        vector["clap"] = clap_vec
                        matched += 1

                points.append(models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vector,
                    payload={
                        "lyrics":     song_info["lyrics"],
                        "title":      song_info["title"],
                        "artist":     song_info["artist"],
                        "album":      song_info["album"],
                        "year":       song_info.get("year"),
                        "year_range": song_info.get("year_range"),
                        "genre":      song_info.get("genre"),
                        "duration":   song_info.get("duration"),
                        "file_path":  song_info.get("file_path"),
                        "cover_art_path": song_info.get("cover_art_path"),
                    },
                ))

            self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

        if clap_map:
            logger.info("[LyricsDB] CLAP vectors attached to %d / %d points", matched, len(data))

    def fit(self, data: list[dict], path: str | None = None, collection_name: str | None = None):
        _saved = self.collection_name
        if collection_name:
            self.collection_name = collection_name
        try:
            self._fit_impl(data, path)
        finally:
            self.collection_name = _saved

    def fit_with_progress(self, data: list[dict], path: str | None = None, collection_name: str | None = None,
                           progress_callback: callable | None = None):
        """Variant of fit() that reports progress at each encoding stage.

        Args:
            progress_callback: sync callable(stage, current, total, message) invoked from a worker thread.
        """
        _saved = self.collection_name
        if collection_name:
            self.collection_name = collection_name
        try:
            self._fit_impl(data, path, progress_callback=progress_callback)
        finally:
            self.collection_name = _saved

    def _fit_impl(self, data: list[dict], path: str | None = None, progress_callback: callable | None = None):
        prepared_data = prepare_metadata(data)
        filtered = [s for s in prepared_data if len(s["lyrics"].split()) < 1500]

        # CLAP paths: из path-аргумента ИЛИ из file_path в метаданных
        if path:
            paths = [
                p for p in Path(path).rglob("*")
                if p.suffix.lower() in (".flac", ".m4a", ".mp3")
            ]
        else:
            paths = [
                s.get("file_path")
                for s in filtered
                if s.get("file_path") and Path(s["file_path"]).suffix.lower() in (".flac", ".m4a", ".mp3")
            ]

        self._create_collection(clap_paths=paths)
        total = len(filtered)

        # Pass 1: encode all lyrics at once (more efficient than per-batch)
        if progress_callback:
            progress_callback("lyrics", 0, total, "Encoding lyrics...")
        text_vecs = self.model.encode(
            [s["lyrics"] for s in filtered],
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        if progress_callback:
            progress_callback("lyrics", total, total, "Lyrics encoding done")

        # Vacate GPU before loading CLAP — remember original device to restore after
        text_device = next(self.model.parameters()).device
        if torch.cuda.is_available() and text_device.type == "cuda":
            self.model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        try:
            # Pass 2: CLAP audio embeddings (GPU now free)
            if paths:
                if progress_callback:
                    progress_callback("audio", 0, total, "Encoding audio (CLAP)...")
                clap_map = _encode_clap(filtered, self.model_clap if self.model_clap else None)
                if progress_callback:
                    progress_callback("audio", total, total, "CLAP encoding done")
            else:
                clap_map = {}
        finally:
            # Restore text model to original device so subsequent searches stay on GPU
            if text_device.type == "cuda" and torch.cuda.is_available():
                self.model.to(text_device)

        # Upsert (сетевой запрос — модель на CPU не мешает)
        self._upsert_in_batches(filtered, text_vecs, clap_map or None)
        print("Тексты песен были успешно проиндексированы в DB")


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
        year_range: str | None = None,
        collection_name_override: str | None = None,
    ) -> list[models.ScoredPoint]:
        col = collection_name_override or self.collection_name

        query_filter = build_filter(
            artist=artist,
            album=album,
            title=title,
            genre=genre,
            year=year,
            year_range=year_range
        )

        if not include_clap:
            query_vector = self.model.encode(query).tolist()

            results = self.qdrant_client.query_points(
                collection_name=col,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using=self.vector_name,
                        limit=15,
                        score_threshold=min_dense_score,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=models.Document(
                            text=query,
                            model="Qdrant/bm25",
                        ),
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
                # query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )

        return results.points
