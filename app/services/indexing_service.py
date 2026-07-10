"""Indexing service — orchestrates folder scan → metadata + lyrics + cover → Qdrant upsert.

Replaces legacy ``FileProcessor`` (file_processor.main) and the indexing-side methods
of legacy ``LyricsDB`` (``fit``, ``_create_collection``, ``_upsert_in_batches``, ``_fit_impl``).
All search responsibilities live in ``app.resources.lyrics_search_engine.LyricsSearchEngine``.

Architecture:

    dict[str, track_dict] (metadata + lyrics + file_path)
        │
        ▼
    IndexingService.fit(data, ...) ─┬─ prepare_metadata (qdrant_payload)
                                    ├─ encode CLAP + text (clap-first; see _fit_impl)
                                    ├─ _create_collection (drop + recreate target)
                                    └─ _upsert_in_batches (with sonic_tags enrichment)

The concurrent folder/upload path lives in ``app.services.index_pipeline``; this
module exposes the reusable encode/upsert stages it composes. ``index_uploads``
remains as a synchronous batch-index entry point (used by maintenance scripts).
"""

from __future__ import annotations

import gc
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from qdrant_client import models
from tqdm.auto import tqdm

from app.resources.qdrant_utils import scroll_all
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

# Parallel workers for the server-mode upload scan (per-file tag read + ONLINE
# lyrics fetch). Mirrors scan_and_enrich_folder's default so uploaded files
# fetch lyrics concurrently instead of one-at-a-time. The work is network-bound,
# so threads (not processes) are the right tool; the lyrics APIs' rate limits
# are respected by lyrics_fetchers' per-request time.sleep inside each thread.
_UPLOAD_SCAN_WORKERS = 8


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
        "audio_codec":         song_info.get("audio_codec"),
        "track_number":        song_info.get("track_number"),
        "disc_number":         song_info.get("disc_number"),
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
            # Collect (point_id, payload) for SQLite upsert
            sqlite_rows: list[tuple[str, dict]] = []
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

                point_id = uuid.uuid4().hex
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                ))
                sqlite_rows.append((point_id, payload))

            client.upsert(collection_name=coll, points=points)

            # Mirror payload to SQLite track_metadata (best-effort)
            try:
                MetadataDB.upsert_track_metadata_bulk(coll, sqlite_rows)
            except Exception:
                logger.warning("[IndexingService] SQLite bulk upsert failed — non-fatal")

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
        indexed_sink: Optional[dict] = None,
    ) -> dict[str, str]:
        """Index a list of already-uploaded files (server mode).

        ``indexed_sink``: optional out-dict. When provided, it is updated with the
        ``"Artist — Title" -> song_info`` map of every track that was indexed, so
        the caller (the upload runner) can run the FACTS/enrichment stage on
        exactly this batch's artists and songs. The return value is unchanged.

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

        # Feature #2: build ONE Yandex enrichment client for this batch (account
        # token if linked, else anonymous), reused across all files. Best-effort —
        # if the yandex package/credentials are unavailable, enrichment no-ops.
        enrich_client = None
        try:
            from app.services.yandex.enrichment import client_for_account
            enrich_client = client_for_account(account_id)
        except Exception:
            logger.debug("[index_uploads] enrichment client unavailable", exc_info=True)

        # The pre-embedding scan (per-file tag read + ONLINE lyrics fetch) is the
        # slow part of an upload job, and it is network-bound — so we fan it out
        # across a thread pool (mirroring scan_and_enrich_folder) instead of
        # processing one file at a time. Each worker does only the pure
        # network/disk work and returns a result; DB status writes, the data dict
        # and progress reporting all happen in THIS thread as futures complete, so
        # no locking is needed despite the parallelism.
        def _scan_one(row: dict) -> tuple[dict, dict | None, str | None]:
            """Read tags, fetch lyrics and extract cover art for one uploaded file.

            Pure w.r.t. shared state: returns ``(row, info, error)`` and lets the
            caller serialize DB writes + progress. ``info`` is None (with a
            non-None ``error``) when the row should be marked failed.
            """
            file_path = Path(row["storage_path"])
            if not file_path.exists():
                return row, None, f"file missing on disk: {file_path}"
            try:
                info = process_file(file_path, False, enrich_client=enrich_client)
            except Exception as e:
                logger.exception("[index_uploads] metadata read failed for %s", file_path)
                return row, None, str(e)
            if not info or not info.get("title") or not info.get("artist"):
                return row, None, "missing title/artist in metadata tags"
            if not info.get("lyrics"):
                info["lyrics"] = "No lyrics were found :("
            info["file_path"] = str(file_path)
            # The folder flow extracts embedded art in _metadata_to_tracks;
            # uploads skipped that step, so every server-mode track rendered the
            # no-cover equalizer fallback.
            try:
                info["cover_art_path"] = save_cover_art(
                    str(file_path), row["sha256"][:16],
                    meta=info, yandex_client=enrich_client,
                )
            except Exception as e:
                logger.warning(
                    "[index_uploads] cover extraction failed for %s: %s", file_path, e,
                )
                info["cover_art_path"] = None
            return row, info, None

        # Progress must be reported — "scan" maps to the LYRICS stage in
        # LibraryService._on_index_progress; without it the wizard's SSE stream
        # (and tqdm) stays silent until embedding starts.
        total_rows = len(upload_rows)
        if progress_callback:
            progress_callback("scan", 0, total_rows, "Чтение метаданных и поиск текстов...")

        completed = 0
        workers = min(_UPLOAD_SCAN_WORKERS, total_rows) or 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_scan_one, row): row for row in upload_rows}
            for future in tqdm(
                as_completed(futures), total=total_rows,
                desc="[index_uploads] metadata+lyrics",
            ):
                row, info, error = future.result()
                completed += 1
                if error is not None:
                    MetadataDB.update_pending_upload_status(
                        row["upload_id"], status="failed", error=error,
                    )
                    label = row.get("original_filename") or Path(row["storage_path"]).name
                else:
                    key = f"{info['artist']} — {info['title']}"
                    data[key] = info
                    upload_by_key[key] = row["upload_id"]
                    label = key
                if progress_callback:
                    progress_callback(
                        "scan", completed, total_rows, f"[{completed}/{total_rows}] {label}",
                    )

        # Hand the indexed batch back to the caller (for the FACTS stage) before
        # the early-return so an empty batch simply leaves the sink empty.
        if indexed_sink is not None:
            indexed_sink.update(data)

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
            id_by_key: dict[str, str] = {}
            for p in scroll_all(
                client, collection_name, batch_size=500,
                with_payload=["title", "artist"], with_vectors=False,
            ):
                pl = p.payload or {}
                k = f"{(pl.get('artist') or '').strip()} — {(pl.get('title') or '').strip()}"
                if k in upload_by_key and k not in id_by_key:
                    id_by_key[k] = str(p.id)

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

    # ─── Reusable encode/upsert stages (shared by _fit_impl + IndexPipeline) ──

    def prepare(self, data: dict, path: str | None = None) -> tuple[list[dict], list]:
        """Normalize + filter tracks and collect CLAP audio paths.

        Returns ``(filtered, clap_paths)``. ``filtered`` is the canonical per-track
        list (fresh dicts from ``prepare_metadata``) that EVERY downstream stage
        keys off — encode_clap, encode_dense and upsert all read the SAME objects,
        so the pipeline can write fetched lyrics into these dicts between the CLAP
        and dense passes without any (artist, title) key drift.
        """
        prepared = prepare_metadata(data)
        filtered = [s for s in prepared if len(s["lyrics"].split()) < 1500]

        # CLAP paths: explicit folder override OR from per-track metadata.
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
        return filtered, paths

    def encode_clap(
        self,
        clap_tracks: list[dict],
        *,
        progress_callback: Optional[Callable] = None,
    ) -> tuple[dict, dict, dict]:
        """CLAP audio pass → ``(clap_map, clap_chunks_map, sonic_axes_map)``.

        Needs only ``file_path``/``artist``/``title`` per track — NOT lyrics — so the
        IndexPipeline runs this concurrently with the online-lyrics fetch. Vacates
        the text model from the GPU first (text + CLAP can't share VRAM); the
        original device is stashed on ``self._text_device`` and restored by
        ``encode_dense``. On a CLAP error the text model is restored to the GPU
        before re-raising, so a failed index never strands live search on the CPU.

        Sonic axes (Stream RecSys) are projected from the pooled CLAP vectors while
        the CLAP model is still loaded; raw scores go to the payload, z-scoring is
        done at read time.
        """
        text_device = next(self.engine.model.parameters()).device
        self._text_device = text_device

        has_audio = any(
            t.get("file_path") and Path(t["file_path"]).suffix.lower() in (".flac", ".m4a", ".mp3")
            for t in clap_tracks
        )
        if not has_audio:
            return {}, {}, {}

        moved = False
        if torch.cuda.is_available() and text_device.type == "cuda":
            self.engine.model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            moved = True

        total = len(clap_tracks)

        def _clap_cb(c, t):
            if progress_callback:
                progress_callback("audio", c, t, "Encoding audio (CLAP)...")

        try:
            if progress_callback:
                progress_callback("audio", 0, total, "Encoding audio (CLAP)...")
            clap_map, clap_chunks_map = _encode_clap(
                clap_tracks,
                self.engine.model_clap if self.engine.model_clap else None,
                progress_callback=_clap_cb,
            )
            if progress_callback:
                progress_callback("audio", total, total, "CLAP encoding done")
            sonic_axes_map = self._compute_sonic_axes(clap_map)
            return clap_map, clap_chunks_map, sonic_axes_map
        except Exception:
            if moved and torch.cuda.is_available():
                try:
                    self.engine.model.to(text_device)
                except Exception:
                    logger.warning("[IndexingService] failed to restore text model to GPU after CLAP error")
            raise

    def encode_dense(
        self,
        filtered: list[dict],
        *,
        progress_callback: Optional[Callable] = None,
    ) -> np.ndarray:
        """Dense lyrics pass → ``text_vecs``. Restores the text model to its GPU
        device first (``encode_clap`` may have vacated it). Reads ``s["lyrics"]``,
        so a caller that fetches lyrics after CLAP must have written them into the
        same ``filtered`` dicts before calling this.
        """
        text_device = getattr(self, "_text_device", None)
        if text_device is not None and text_device.type == "cuda" and torch.cuda.is_available():
            self.engine.model.to(text_device)

        total = len(filtered)
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
        return text_vecs

    # Public wrappers so IndexPipeline composes the stages without reaching into
    # the "_"-prefixed internals.
    def create_collection(self, clap_paths: list) -> None:
        self._create_collection(clap_paths=clap_paths)

    def upsert(
        self,
        data: list[dict],
        text_vecs: np.ndarray,
        clap_map: Optional[dict] = None,
        clap_chunks_map: Optional[dict] = None,
        sonic_axes_map: Optional[dict] = None,
    ) -> None:
        self._upsert_in_batches(
            data, text_vecs, clap_map, clap_chunks_map, sonic_axes_map=sonic_axes_map,
        )

    def finalize_norms_and_prune(self, sonic_axes_map: Optional[dict]) -> None:
        """Persist per-collection axis norm stats + prune orphaned track refs."""
        self._persist_axis_norm_stats(sonic_axes_map)
        self._prune_orphaned_track_refs()

    def _fit_impl(
        self,
        data: dict,
        path: str | None = None,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        filtered, paths = self.prepare(data, path)
        self._create_collection(clap_paths=paths)

        # Clear old SQLite track_metadata before upserting new data
        try:
            MetadataDB.clear_track_metadata(str(self.engine.collection_name))
        except Exception:
            logger.warning("[IndexingService] failed to clear old track_metadata — non-fatal")

        # CLAP first (audio-only), then dense — they can't share the GPU. The
        # IndexPipeline overlaps these with the network; here (sequential) only the
        # order matters: encode_clap vacates the text model, encode_dense restores
        # it before encoding lyrics.
        clap_map, clap_chunks_map, sonic_axes_map = self.encode_clap(
            filtered, progress_callback=progress_callback,
        )
        text_vecs = self.encode_dense(filtered, progress_callback=progress_callback)

        # Upsert (network IO — CPU model is fine)
        self._upsert_in_batches(
            filtered, text_vecs, clap_map or None, clap_chunks_map or None,
            sonic_axes_map=sonic_axes_map or None,
        )

        # Indexing drops + recreates the collection, so this batch IS the whole
        # collection — its mean/std are the collection's normalisation stats.
        self._persist_axis_norm_stats(sonic_axes_map)

        # Fresh uuid4 point ids orphan the old reactions/playback events in
        # SQLite — drop those so «Поток» stops surfacing empty «—» 404 tracks.
        self._prune_orphaned_track_refs()
        logger.info("[IndexingService] Indexing complete: %d tracks", len(filtered))

    def _prune_orphaned_track_refs(self) -> None:
        """Drop SQLite reactions/events pointing at now-deleted point ids.

        Best-effort: a re-index already succeeded by the time this runs, so a
        prune failure must never fail the index — it only leaves stale rows the
        stream's per-pool resolve guard still filters out.
        """
        coll = str(self.engine.collection_name)
        client = self.engine.qdrant_client
        try:
            live_ids: set[str] = {
                str(p.id)
                for p in scroll_all(
                    client, coll, batch_size=1000,
                    with_payload=False, with_vectors=False,
                )
            }
            removed = MetadataDB.prune_orphaned_tracks(coll, live_ids)
            if any(removed.values()):
                logger.info(
                    "[IndexingService] pruned orphaned track refs in %s: %s",
                    coll, removed,
                )
        except Exception:
            logger.exception("[IndexingService] orphan prune failed — non-fatal")

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
