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
from typing import Callable, Optional

import numpy as np
import torch
from qdrant_client import models
from tqdm.auto import tqdm

from app.indexing.folder_scanner import scan_and_enrich_folder
from app.resources.clap_features import (
    AXIS_NAMES,
    _encode_clap,
    axes_for_clap_vectors,
    axis_version,
    compute_axis_text_embeddings,
)
from app.resources.lyrics_search_engine import LyricsSearchEngine
from app.resources.metadata_db import MetadataDB, _slugify
from app.resources.qdrant_payload import build_text_for_embedding, prepare_metadata
from app.services.artist_split import split_artists, artist_slugs as _artist_slugs

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
    artists = split_artists(song_info.get("artist") or "")
    slugs = _artist_slugs(song_info.get("artist") or "")
    return {
        "lyrics":              song_info["lyrics"],
        "title":               song_info["title"],
        "artist":              song_info["artist"],
        "artists":             artists,
        "artist_slugs":        slugs,
        "primary_artist_slug": slugs[0] if slugs else None,
        "album":               song_info["album"],
        "year":                song_info.get("year"),
        "year_range":          song_info.get("year_range"),
        "genre":               song_info.get("genre"),
        "duration":            song_info.get("duration"),
        "duration_range":      song_info.get("duration_range"),
        "file_path":           song_info.get("file_path"),
        "cover_art_path":      song_info.get("cover_art_path"),
        "producer":            song_info.get("producer"),
        "label":               song_info.get("label"),
        "samples":             song_info.get("samples"),
        "sampled_by":          song_info.get("sampled_by"),
        "bitrate_kbps":        song_info.get("bitrate_kbps"),
        "sonic_tags":          sonic_tags,
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
        # Keyword index over the multi-valued artist_slugs so the artist page
        # can filter server-side by participant (MatchValue per array element).
        try:
            client.create_payload_index(
                collection_name=coll,
                field_name="artist_slugs",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            logger.warning("[IndexingService] artist_slugs index already exists or creation failed: %s", e)
        logger.info("[IndexingService] Collection '%s' created", coll)

    def _upsert_in_batches(
        self,
        data: list[dict],
        text_vecs: np.ndarray,
        clap_map: Optional[dict] = None,
        clap_chunks_map: Optional[dict] = None,
        batch_size: int = 32,
        sonic_axes_map: Optional[dict] = None,
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
                if sonic_axes_map:
                    axes = sonic_axes_map.get(key)
                    if axes:
                        payload["sonic_axes"] = axes

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

    def index_uploads(
        self,
        account_id: str,
        upload_rows: list[dict],
        *,
        progress_callback: Optional[Callable] = None,
    ) -> dict[str, str]:
        """Index a list of already-uploaded files (server mode).

        Each row in ``upload_rows`` is a ``pending_uploads`` record produced by
        the multipart upload endpoint; the file already lives at
        ``row['storage_path']`` (``media/<account_id>/audio/<sha>.<ext>``).

        We reuse the folder-scan pipeline unchanged: ``process_file`` builds the
        same per-track ``song_info`` dict the folder flow produces (so slugging
        flows through ``_build_payload_for_upsert`` — see two_divergent_slugify
        memo), then ``fit_with_progress`` runs it against
        ``collection_name = f"acct_{account_id}"``. Returns ``{upload_id: track_id}``
        for every row that landed in Qdrant; updates ``pending_uploads.status``
        per row (indexing → done/failed).
        """
        from app.indexing.cover_art import save_cover_art
        from app.indexing.folder_scanner import process_file
        from app.resources.metadata_db import MetadataDB

        # Mark all rows 'indexing' so the status endpoint reflects in-progress
        # work even if the heavy pipeline takes minutes.
        for r in upload_rows:
            MetadataDB.update_pending_upload_status(r["upload_id"], status="indexing")

        collection_name = f"acct_{account_id}"
        data: dict[str, dict] = {}
        upload_by_key: dict[str, str] = {}   # "Artist — Title" -> upload_id

        # This loop is the slow pre-embedding part of an upload job (per-file
        # tag read + ONLINE lyrics fetch), so it must report progress — "scan"
        # maps to the LYRICS stage in LibraryService._on_index_progress.
        # Without these calls the wizard's SSE stream (and tqdm) stays silent
        # until embedding starts.
        total_rows = len(upload_rows)
        if progress_callback:
            progress_callback("scan", 0, total_rows, "Чтение метаданных и поиск текстов...")

        for i, row in enumerate(
            tqdm(upload_rows, desc="[index_uploads] metadata+lyrics"), start=1,
        ):
            file_path = Path(row["storage_path"])
            label = row.get("original_filename") or file_path.name
            if not file_path.exists():
                MetadataDB.update_pending_upload_status(
                    row["upload_id"], status="failed",
                    error=f"file missing on disk: {file_path}",
                )
            else:
                try:
                    info = process_file(file_path, False)
                    if not info or not info.get("title") or not info.get("artist"):
                        MetadataDB.update_pending_upload_status(
                            row["upload_id"], status="failed",
                            error="missing title/artist in metadata tags",
                        )
                        info = None
                    else:
                        if not info.get("lyrics"):
                            info["lyrics"] = "No lyrics were found :("
                        info["file_path"] = str(file_path)
                        # The folder flow extracts embedded art in
                        # _metadata_to_tracks; uploads skipped that step, so
                        # every server-mode track rendered the no-cover
                        # equalizer fallback.
                        try:
                            info["cover_art_path"] = save_cover_art(
                                str(file_path), row["sha256"][:16],
                            )
                        except Exception as e:
                            logger.warning(
                                "[index_uploads] cover extraction failed for %s: %s",
                                file_path, e,
                            )
                            info["cover_art_path"] = None
                        key = f"{info['artist']} — {info['title']}"
                        data[key] = info
                        upload_by_key[key] = row["upload_id"]
                        label = key
                except Exception as e:
                    logger.exception("[index_uploads] metadata read failed for %s", file_path)
                    MetadataDB.update_pending_upload_status(
                        row["upload_id"], status="failed", error=str(e),
                    )
            if progress_callback:
                progress_callback("scan", i, total_rows, f"[{i}/{total_rows}] {label}")

        if not data:
            logger.info("[index_uploads] nothing to index for account=%s", account_id)
            return {}

        # Run the same pipeline as the folder-scan flow. fit_with_progress sets
        # (and restores) engine.collection_name around _fit_impl, so the upsert
        # lands in acct_<account_id> without mutating the shared engine's default.
        self.fit_with_progress(
            data, path=None, collection_name=collection_name,
            progress_callback=progress_callback,
        )

        # After upsert, scroll the collection to pick up freshly created point ids
        # and update pending_uploads.track_id. We match by (artist, title) which
        # is unique within a single batch — same caveat as scan_folder.
        try:
            client = self.engine.qdrant_client
            offset = None
            id_by_key: dict[str, str] = {}
            while True:
                points, next_offset = client.scroll(
                    collection_name=collection_name,
                    offset=offset,
                    limit=500,
                    with_payload=["title", "artist"],
                    with_vectors=False,
                )
                for p in points:
                    pl = p.payload or {}
                    k = f"{(pl.get('artist') or '').strip()} — {(pl.get('title') or '').strip()}"
                    if k in upload_by_key and k not in id_by_key:
                        id_by_key[k] = str(p.id)
                if next_offset is None or not points:
                    break
                offset = next_offset

            out: dict[str, str] = {}
            for key, upload_id in upload_by_key.items():
                track_id = id_by_key.get(key)
                if track_id:
                    MetadataDB.update_pending_upload_status(
                        upload_id, status="done", track_id=track_id,
                    )
                    out[upload_id] = track_id
                else:
                    MetadataDB.update_pending_upload_status(
                        upload_id, status="failed",
                        error="track did not appear in Qdrant after upsert",
                    )
            return out
        except Exception as e:
            logger.exception("[index_uploads] post-upsert resolve failed")
            for upload_id in upload_by_key.values():
                MetadataDB.update_pending_upload_status(
                    upload_id, status="failed", error=str(e),
                )
            return {}

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

            # Sonic axes (Stream RecSys): project pooled CLAP vectors onto the
            # 6-axis prompt space while the CLAP model is still loaded. Raw
            # scores go into the payload; z-scoring happens at read time.
            sonic_axes_map = self._compute_sonic_axes(clap_map)
        finally:
            if text_device.type == "cuda" and torch.cuda.is_available():
                self.engine.model.to(text_device)

        # Upsert (network IO — CPU model is fine)
        self._upsert_in_batches(
            filtered, text_vecs, clap_map or None, clap_chunks_map or None,
            sonic_axes_map=sonic_axes_map or None,
        )

        # Indexing drops + recreates the collection, so this batch IS the whole
        # collection — its mean/std are the collection's normalisation stats.
        self._persist_axis_norm_stats(sonic_axes_map)
        logger.info("[IndexingService] Indexing complete: %d tracks", total)

    def _compute_sonic_axes(self, clap_map: dict) -> dict:
        """Map ``(artist, title) → {axis: raw_score}`` for every CLAP-encoded track.

        Axis failures must never block indexing — returns {} on any error.
        """
        if not clap_map:
            return {}
        try:
            model_clap = self.engine.model_clap
            if not model_clap:
                from app.resources.model_registry import ModelRegistry
                model_clap = ModelRegistry.load_clap()
            text_emb = compute_axis_text_embeddings(model_clap)
            keys = list(clap_map)
            axes_dicts = axes_for_clap_vectors(
                np.stack([clap_map[k] for k in keys]), text_emb,
            )
            logger.info("[IndexingService] sonic axes computed for %d tracks", len(keys))
            return dict(zip(keys, axes_dicts))
        except Exception:
            logger.exception("[IndexingService] sonic axes failed — indexing continues without them")
            return {}

    def _persist_axis_norm_stats(self, sonic_axes_map: dict) -> None:
        """Write per-collection axis mean/std to collection_settings.

        ``ddof=1`` matches the notebook's pandas ``.std()``; with n=1 that is
        NaN — sanitised to 0.0 (readers must guard zero-std anyway).
        """
        if not sonic_axes_map:
            return
        try:
            arr = np.array(
                [[d[a] for a in AXIS_NAMES] for d in sonic_axes_map.values()],
                dtype=float,
            )
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(len(AXIS_NAMES))
            std = np.nan_to_num(std, nan=0.0)
            stats = {
                "version": axis_version(),
                "n": int(arr.shape[0]),
                "mean": {a: float(v) for a, v in zip(AXIS_NAMES, mean)},
                "std": {a: float(v) for a, v in zip(AXIS_NAMES, std)},
            }
            MetadataDB.set_axis_norm_stats(str(self.engine.collection_name), stats)
            logger.info(
                "[IndexingService] axis_norm_stats persisted for '%s' (n=%d)",
                self.engine.collection_name, stats["n"],
            )
        except Exception:
            logger.exception("[IndexingService] failed to persist axis_norm_stats")
