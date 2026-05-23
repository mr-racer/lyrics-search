from __future__ import annotations

import gc
import logging
import uuid
from pathlib import Path

import numpy as np
import torch
from qdrant_client import models
from tqdm.auto import tqdm

from app.resources.lyrics_search_engine import LyricsSearchEngine
from .utils import (
    build_text_for_embedding, prepare_metadata, _encode_clap,
)

logger = logging.getLogger(__name__)


def _build_payload_for_upsert(song_info: dict, slug: str | None = None) -> dict:
    """Build the Qdrant payload for a single track.

    When ``slug`` is provided and a matching row exists in MetadataDB,
    the payload is enriched with ``sonic_tags`` (list[str]) so that
    SearchFilters can later match on them.
    """
    from app.resources.metadata_db import MetadataDB
    sonic_tags: list[str] = []
    if slug:
        try:
            desc = MetadataDB.get_sonic_descriptor(slug)
            if desc:
                tags_obj = desc.get("tags") or []
                # tags_obj is a list of {"tag": str, "score": float} or list[str].
                sonic_tags = [
                    (t["tag"] if isinstance(t, dict) else t)
                    for t in tags_obj
                ]
        except Exception:
            # Sonic-Descriptor data is optional — don't block indexing.
            pass
    return {
        "lyrics":         song_info["lyrics"],
        "title":          song_info["title"],
        "artist":         song_info["artist"],
        "album":          song_info["album"],
        "year":           song_info.get("year"),
        "year_range":     song_info.get("year_range"),
        "genre":          song_info.get("genre"),
        "duration":       song_info.get("duration"),
        "duration_range": song_info.get("duration_range"),
        "file_path":      song_info.get("file_path"),
        "cover_art_path": song_info.get("cover_art_path"),
        "producer":       song_info.get("producer"),
        "label":          song_info.get("label"),
        "samples":        song_info.get("samples"),
        "sampled_by":     song_info.get("sampled_by"),
        "bitrate_kbps":   song_info.get("bitrate_kbps"),
        "sonic_tags":     sonic_tags,
    }


class LyricsDB(LyricsSearchEngine):
    """Manage a Qdrant collection with hybrid dense and sparse (BM25) lyric embeddings.

    NOTE (Refactor 4): All search logic now lives in LyricsSearchEngine (superclass).
    This subclass only retains indexing methods (fit, _create_collection,
    _upsert_in_batches) which will move to app.services.indexing_service in Refactor 5.

    Model loading is lazy by default (lazy=True) — the text model and CLAP are loaded
    on first actual use (search/fit), not in __init__.  This allows the FastAPI server
    to start instantly and preload models in the background.

    A pre-loaded model can be passed via +model+ / +model_clap+ to skip lazy loading
    entirely (useful when ModelRegistry already cached the model).
    """

    # __init__, model accessors, _ensure_model, _ensure_clap, _init_qdrant,
    # and search are all inherited from LyricsSearchEngine.

    # ── Indexing (moves to IndexingService in Refactor 5) ───────────────────

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

                # Derive slug matching metadata_db.ensure_song convention:
                # _slugify(artist) + "-" + _slugify(title)
                slug = None
                try:
                    from app.resources.metadata_db import _slugify
                    _artist = (song_info.get("artist") or "").strip()
                    _title  = (song_info.get("title")  or "").strip()
                    if _artist and _title:
                        slug = _slugify(_artist) + "-" + _slugify(_title)
                except Exception:
                    pass

                points.append(models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vector,
                    payload=_build_payload_for_upsert(song_info, slug=slug),
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

                def _clap_cb(c, t):
                    if progress_callback:
                        progress_callback("audio", c, t, "Encoding audio (CLAP)...")

                clap_map = _encode_clap(
                    filtered,
                    self.model_clap if self.model_clap else None,
                    progress_callback=_clap_cb,
                )
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
