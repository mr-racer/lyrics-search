"""Indexing service — orchestrates folder scan → metadata + lyrics + cover → Qdrant upsert.

Replaces legacy ``FileProcessor`` (file_processor.main) and the indexing-side methods
of legacy ``LyricsDB`` (``fit``, ``_create_collection``, ``_upsert_in_batches``, ``_fit_impl``).
All search responsibilities live in ``app.resources.lyrics_search_engine.LyricsSearchEngine``.

Architecture:

    folder_path
        │
        ▼
    scan_folder() ─── delegates to app.indexing.folder_scanner.scan_and_enrich_folder
        │
        ▼
    dict[str, track_dict] (metadata + lyrics + file_path)
        │
        ▼
    IndexingService.fit(data, ...) ─┬─ prepare_metadata (qdrant_payload)
                                    ├─ encode text + CLAP
                                    ├─ _create_collection (drop + recreate target)
                                    └─ _upsert_in_batches (with sonic_tags enrichment)
"""

from __future__ import annotations

import gc
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from qdrant_client import QdrantClient, models
from tqdm.auto import tqdm

from app.indexing.folder_scanner import scan_and_enrich_folder
from app.resources.clap_features import _encode_clap
from app.resources.lyrics_search_engine import LyricsSearchEngine
from app.resources.metadata_db import MetadataDB, _slugify
from app.resources.qdrant_payload import build_text_for_embedding, prepare_metadata

logger = logging.getLogger(__name__)


# ─── Folder scan wrapper ─────────────────────────────────────────────────────

def scan_folder(
    folder_path: str,
    better_lyrics_quality: bool = False,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Walk a folder, read audio metadata, fetch lyrics, return a dict of processed tracks.

    Thin wrapper around ``app.indexing.folder_scanner.scan_and_enrich_folder`` so that
    consumers (LibraryService) can import a single ``indexing_service`` module without
    crossing into the lower-level ``app.indexing`` package.
    """
    return scan_and_enrich_folder(
        music_folder=folder_path,
        better_lyrics_quality=better_lyrics_quality,
        progress_callback=progress_callback,
    )


# ─── Qdrant payload builder ──────────────────────────────────────────────────

def _build_payload_for_upsert(song_info: dict, slug: str | None = None) -> dict:
    """Construct the Qdrant payload dict for one track.

    When ``slug`` is provided and a row exists in MetadataDB, the payload is
    enriched with ``sonic_tags: list[str]`` so SearchFilters can match on them
    (Sonic Descriptor data is optional — failures don't block indexing).
    """
    sonic_tags: list[str] = []
    if slug:
        try:
            desc = MetadataDB.get_sonic_descriptor(slug)
            if desc:
                tags_obj = desc.get("tags") or []
                sonic_tags = [
                    (t["tag"] if isinstance(t, dict) else t)
                    for t in tags_obj
                ]
        except Exception:
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


# ─── IndexingService ─────────────────────────────────────────────────────────

class IndexingService:
    """Owns the Qdrant collection lifecycle for batch indexing.

    Wraps a ``LyricsSearchEngine`` instance for model + Qdrant access, but adds:
    - ``_create_collection`` — drop+recreate the target collection with the right
      vector schema (text + bm25 + optional clap)
    - ``_upsert_in_batches`` — chunked upsert with payload enrichment
    - ``fit`` / ``fit_with_progress`` — top-level entry that runs both passes
      (text encode + CLAP encode) and uploads

    ``data`` accepted by ``fit`` / ``fit_with_progress`` is a ``dict[str, dict]``
    keyed by ``"Artist — Title"`` — the same shape produced by
    ``scan_folder`` / ``FileProcessor.process_folder`` and consumed by
    ``app.resources.qdrant_payload.prepare_metadata``.
    """

    def __init__(self, engine: LyricsSearchEngine):
        self.engine = engine

    # Convenience pass-through so legacy call sites see a single object.
    @property
    def collection_name(self) -> str:
        return self.engine.collection_name

    @collection_name.setter
    def collection_name(self, v: str) -> None:
        self.engine.collection_name = v

    def _create_collection(self, clap_paths: list) -> None:
        client = self.engine.qdrant_client
        coll = str(self.engine.collection_name)
        existing = client.get_collections().collections
        if any(c.name == coll for c in existing):
            client.delete_collection(coll)

        vectors_config = {
            self.engine.vector_name: models.VectorParams(
                size=self.engine.vector_dim,
                distance=models.Distance.COSINE,
            ),
        }
        if clap_paths:
            vectors_config["clap"] = models.VectorParams(
                size=512,
                distance=models.Distance.COSINE,
            )

        client.create_collection(
            collection_name=coll,
            vectors_config=vectors_config,
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )
        logger.info("[IndexingService] Collection '%s' created", coll)

    def _upsert_in_batches(
        self,
        data: list[dict],
        text_vecs: np.ndarray,
        clap_map: Optional[dict] = None,
        clap_chunks_map: Optional[dict] = None,
        batch_size: int = 32,
    ) -> None:
        client = self.engine.qdrant_client
        coll = self.engine.collection_name
        matched = 0
        chunks_attached = 0

        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i: i + batch_size]
            vecs = text_vecs[i: i + batch_size]
            points = []
            for song_info, vec in zip(batch, vecs):
                vector = {
                    "bm25": models.Document(
                        text=build_text_for_embedding(song_info),
                        model="Qdrant/bm25",
                    ),
                    self.engine.vector_name: vec,
                }
                key = (
                    song_info.get("artist", "").strip().lower(),
                    song_info.get("title", "").strip().lower(),
                )
                if clap_map:
                    clap_vec = clap_map.get(key)
                    if clap_vec is not None:
                        vector["clap"] = clap_vec
                        matched += 1

                # Derive slug for sonic-tag enrichment lookup
                slug = None
                artist = (song_info.get("artist") or "").strip()
                title = (song_info.get("title") or "").strip()
                if artist and title:
                    slug = _slugify(artist) + "-" + _slugify(title)

                payload = _build_payload_for_upsert(song_info, slug=slug)
                if clap_chunks_map:
                    chunk_list = clap_chunks_map.get(key)
                    if chunk_list:
                        payload["clap_chunks"] = [c.tolist() for c in chunk_list]
                        chunks_attached += 1

                points.append(models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vector,
                    payload=payload,
                ))

            client.upsert(collection_name=coll, points=points)

        if clap_map:
            logger.info(
                "[IndexingService] CLAP vectors attached to %d / %d points "
                "(per-chunk lists attached to %d)",
                matched, len(data), chunks_attached,
            )

    def fit(
        self,
        data: dict,
        path: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Synchronous fit — drops + recreates collection, encodes, uploads.

        Args:
            data: ``dict[str, dict]`` keyed by ``"Artist — Title"`` (the format
                  returned by ``scan_folder`` / ``FileProcessor.process_folder``).
            path: Optional explicit folder path used to locate CLAP audio files.
                  When ``None``, ``file_path`` values inside ``data`` are used.
            collection_name: Override target collection for this run only.
        """
        _saved = self.engine.collection_name
        if collection_name:
            self.engine.collection_name = collection_name
        try:
            self._fit_impl(data, path)
        finally:
            self.engine.collection_name = _saved

    def fit_with_progress(
        self,
        data: dict,
        path: str | None = None,
        collection_name: str | None = None,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """``fit`` variant that reports stage progress via ``progress_callback(stage, current, total, message)``.

        Args:
            data: ``dict[str, dict]`` keyed by ``"Artist — Title"``.
            path: Optional explicit folder path for CLAP audio file discovery.
            collection_name: Override target collection for this run only.
            progress_callback: Sync callable ``(stage: str, current: int, total: int, message: str) → None``.
                               ``stage`` is ``"lyrics"`` (dense encoding) or ``"audio"`` (CLAP encoding).
        """
        _saved = self.engine.collection_name
        if collection_name:
            self.engine.collection_name = collection_name
        try:
            self._fit_impl(data, path, progress_callback=progress_callback)
        finally:
            self.engine.collection_name = _saved

    def _fit_impl(
        self,
        data: dict,
        path: str | None = None,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        prepared = prepare_metadata(data)
        filtered = [s for s in prepared if len(s["lyrics"].split()) < 1500]

        # CLAP paths: explicit override OR from track metadata
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

        # Pass 1: encode all lyrics at once
        if progress_callback:
            progress_callback("lyrics", 0, total, "Encoding lyrics...")
        text_vecs = self.engine.model.encode(
            [s["lyrics"] for s in filtered],
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        if progress_callback:
            progress_callback("lyrics", total, total, "Lyrics encoding done")

        # Vacate GPU before loading CLAP — remember original device to restore after.
        text_device = next(self.engine.model.parameters()).device
        if torch.cuda.is_available() and text_device.type == "cuda":
            self.engine.model.to("cpu")
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

                clap_map, clap_chunks_map = _encode_clap(
                    filtered,
                    self.engine.model_clap if self.engine.model_clap else None,
                    progress_callback=_clap_cb,
                )
                if progress_callback:
                    progress_callback("audio", total, total, "CLAP encoding done")
            else:
                clap_map = {}
                clap_chunks_map = {}
        finally:
            if text_device.type == "cuda" and torch.cuda.is_available():
                self.engine.model.to(text_device)

        # Upsert (network IO — CPU model is fine)
        self._upsert_in_batches(
            filtered, text_vecs, clap_map or None, clap_chunks_map or None,
        )
        logger.info("[IndexingService] Indexing complete: %d tracks", total)
