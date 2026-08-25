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

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import numpy as np
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
from app.services.artist_split import (
    split_artists,
    artist_slugs as _artist_slugs,
    parse_title_feat,
    tag_feat_slugs,
    _resolved_slug,
)

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
    # Feat-in-title extraction: «Bangarang (ft. Sirah)» credits Sirah even
    # though the artist tag says only «Skrillex». «(with X)» co-leads join the
    # mains right after the tag artists; feats append last + are marked in
    # feat_artist_slugs so serializers can label their role. The raw `title`
    # payload field stays untouched — it is baked into the text embeddings.
    parsed_title = parse_title_feat(song_info.get("title") or "")
    # Tag-level feats first («Drake feat. Future» in the artist tag), then
    # title-level ones — feat_artist_slugs marks their role for serializers.
    _tag_feats = tag_feat_slugs(song_info.get("artist") or "")
    feat_artist_slugs: list[str] = [s for s in slugs if s in _tag_feats]
    for name, is_feat in (
        [(n, False) for n in parsed_title.with_names]
        + [(n, True) for n in parsed_title.feat_names]
    ):
        slug = _resolved_slug(name)
        if not slug or slug in slugs:
            continue
        artists.append(name)
        slugs.append(slug)
        if is_feat:
            feat_artist_slugs.append(slug)
    title_display = (
        parsed_title.clean_title
        if parsed_title.clean_title != " ".join((song_info.get("title") or "").split())
        else None
    )
    # primary_artist_slug drives album grouping (library_service, catalog
    # search) — it must be the ALBUM's artist, not just whoever is credited
    # on this one track, or a compilation / a "feat." track fragments its
    # album across multiple "artists". Prefer the file's Album Artist tag
    # (TPE2 / aART / albumartist); fall back to the track artist when absent.
    album_artist_slugs = _artist_slugs(song_info.get("album_artist") or "")
    primary_artist_slug = (
        album_artist_slugs[0] if album_artist_slugs
        else (slugs[0] if slugs else None)
    )
    return {
        "lyrics":              song_info["lyrics"],
        "title":               song_info["title"],
        "title_display":       title_display,
        "artist":              song_info["artist"],
        "artists":             artists,
        "artist_slugs":        slugs,
        "feat_artist_slugs":   feat_artist_slugs,
        "primary_artist_slug": primary_artist_slug,
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

MAX_LYRICS_WORDS = 1500  # a text longer than this is a transcript, not a song


def explain_rejections(tracks: dict) -> dict:
    """``{reason: count}`` for the tracks ``prepare`` would silently drop.

    The two gates below are hard and quiet: a track over ``MAX_DURATION`` or
    with more than ``MAX_LYRICS_WORDS`` of text never reaches Qdrant, and the
    path diff that offered it has no idea, so the next rescan offers it again.
    Reporting the reason is what breaks that loop — the user can then shorten
    the cap, split the file, or accept the skip, instead of pressing "add
    music" a fourth time.
    """
    from app.resources.qdrant_payload import MAX_DURATION

    reasons: dict[str, int] = {}
    for meta in (tracks or {}).values():
        duration = meta.get("duration")
        if not isinstance(duration, (int, float)):
            reasons["no_duration"] = reasons.get("no_duration", 0) + 1
        elif duration > MAX_DURATION:
            reasons["too_long"] = reasons.get("too_long", 0) + 1
        elif len((meta.get("lyrics") or "").split()) >= MAX_LYRICS_WORDS:
            reasons["lyrics_too_long"] = reasons.get("lyrics_too_long", 0) + 1
    return reasons


class IndexingService:
    """Owns the Qdrant collection lifecycle for batch indexing.

    Wraps a ``LyricsSearchEngine`` instance for model + Qdrant access, but adds:
    - ``_create_collection`` — drop+recreate the target collection with the right
      vector schema (text + bm25 + optional clap)
    - ``ensure_collection`` — the additive counterpart: create only when absent,
      so "add music" extends a library instead of replacing it
    - ``_upsert_in_batches`` — chunked upsert with payload enrichment
    - ``fit`` / ``fit_with_progress`` — top-level entry that runs both passes
      (text encode + CLAP encode) and uploads

    ``data`` accepted by ``fit`` / ``fit_with_progress`` is a ``dict[str, dict]``
    keyed by ``"Artist — Title"`` — the same shape produced by
    ``scan_folder`` / ``FileProcessor.process_folder`` and consumed by
    ``app.resources.qdrant_payload.prepare_metadata``.
    """

    def __init__(
        self,
        engine: LyricsSearchEngine,
        *,
        collection_name: str | None = None,
    ):
        """``collection_name`` is snapshotted PER INSTANCE.

        The engine is shared across concurrent indexing jobs (semaphore allows
        2), so job-specific state must never be written onto it — two accounts
        indexing at once used to race on ``engine.collection_name`` and could
        upsert into each other's collection. Instances are cheap; create one per
        run with an explicit target.
        """
        self.engine = engine
        self.collection_name = str(collection_name or engine.collection_name)
        # Set by ensure_collection when appending into a collection that was
        # built without a 'clap' vector — Qdrant can't add one after creation.
        self._skip_clap = False

    def _vector_params(self) -> tuple[str, int]:
        """``(vector_name, vector_dim)`` — constants now, one model app-wide.
        Read off the engine so test fakes can still supply their own."""
        return self.engine.vector_name, self.engine.vector_dim

    def _collection_exists(self) -> bool:
        try:
            existing = self.engine.qdrant_client.get_collections().collections
        except Exception:
            logger.exception("[IndexingService] get_collections failed for '%s'", self.collection_name)
            raise
        return any(c.name == self.collection_name for c in existing)

    def _create_collection(self, clap_paths: list) -> None:
        """Drop the target collection if it exists, then build it from scratch.

        DESTRUCTIVE — this is the first-index / full-rebuild path. Adding music
        to a library that already has tracks must go through
        :meth:`ensure_collection` instead.
        """
        if self._collection_exists():
            self.engine.qdrant_client.delete_collection(self.collection_name)
        self._build_collection(clap_paths)

    def ensure_collection(self, clap_paths: list) -> None:
        """Create the collection only when it is missing — the append entry point.

        A collection built WITHOUT a ``clap`` vector (CLAP disabled at first
        index) cannot gain one later: Qdrant fixes the named-vector set at
        creation time. Rather than fail the whole append, note it and let the
        upsert go text-only.
        """
        if not self._collection_exists():
            self._build_collection(clap_paths)
            return
        if clap_paths and not self._collection_has_clap():
            logger.warning(
                "[IndexingService] '%s' has no 'clap' vector — appending without audio vectors",
                self.collection_name,
            )
            self._skip_clap = True

    def _collection_has_clap(self) -> bool:
        """True unless the existing collection is provably clap-less.

        Any probe failure answers True: a false negative would silently drop
        audio vectors from a perfectly good collection, which is far worse than
        letting the upsert raise a clear Qdrant error.
        """
        try:
            vectors = self.engine.qdrant_client.get_collection(
                self.collection_name,
            ).config.params.vectors
            return "clap" in vectors
        except Exception:
            logger.debug(
                "[IndexingService] clap-vector probe failed for '%s' — assuming present",
                self.collection_name, exc_info=True,
            )
            return True

    def _build_collection(self, clap_paths: list) -> None:
        client = self.engine.qdrant_client
        coll = self.collection_name

        vector_name, vector_dim = self._vector_params()
        vectors_config = {
            vector_name: models.VectorParams(
                size=vector_dim,
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
        """Upsert points + mirror their payloads into SQLite.

        ``clap_chunks_map`` is accepted (the pipeline still computes it) but no
        longer persisted: the per-chunk vectors used to be written into the
        payload as ``clap_chunks``, nothing ever read them back, and at ~100 KB
        of JSON per track they dominated every payload transfer — a 150-hit
        CLAP search returned ~20 MB. See qdrant_utils.PAYLOAD_EXCLUDE_HEAVY.
        """
        client = self.engine.qdrant_client
        coll = self.collection_name
        vector_name, _ = self._vector_params()
        matched = 0

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
                    vector_name: vec,
                }
                key = (
                    song_info.get("artist", "").strip().lower(),
                    song_info.get("title", "").strip().lower(),
                )
                if clap_map and not self._skip_clap:
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
                if sonic_axes_map:
                    axes = sonic_axes_map.get(key)
                    if axes:
                        payload["sonic_axes"] = axes

                # Canonical dashed UUID: Qdrant normalizes any UUID form to the
                # dashed representation in every read (scroll/search/retrieve),
                # so the SQLite mirror must store the same form — an undashed
                # .hex here would give the two stores different keys for the
                # same track.
                point_id = str(uuid.uuid4())
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
                "[IndexingService] CLAP vectors attached to %d / %d points",
                matched, len(data),
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
                             Instance-local — the shared engine is not touched.
        """
        if collection_name:
            self.collection_name = str(collection_name)
        self._fit_impl(data, path)

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
                             Instance-local — the shared engine is not touched.
            progress_callback: Sync callable ``(stage: str, current: int, total: int, message: str) → None``.
                               ``stage`` is ``"lyrics"`` (dense encoding) or ``"audio"`` (CLAP encoding).
        """
        if collection_name:
            self.collection_name = str(collection_name)
        self._fit_impl(data, path, progress_callback=progress_callback)

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

        # Run the same pipeline as the folder-scan flow. fit_with_progress
        # retargets THIS instance at acct_<account_id>; the shared engine's
        # default collection is never mutated.
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
        filtered = [s for s in prepared if len(s["lyrics"].split()) < MAX_LYRICS_WORDS]

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
        IndexPipeline runs this concurrently with the online-lyrics fetch.

        Device policy 2026-07: CLAP is pinned to the CPU by ModelRegistry, so
        the old GPU juggling (vacate text model → CLAP → restore) is gone —
        the text model can hold the GPU for the whole indexing run while CLAP
        encodes audio on the CPU in parallel.

        Sonic axes (Stream RecSys) are projected from the pooled CLAP vectors while
        the CLAP model is still loaded; raw scores go to the payload, z-scoring is
        done at read time.
        """
        has_audio = any(
            t.get("file_path") and Path(t["file_path"]).suffix.lower() in (".flac", ".m4a", ".mp3")
            for t in clap_tracks
        )
        if not has_audio:
            return {}, {}, {}

        total = len(clap_tracks)

        def _clap_cb(c, t):
            if progress_callback:
                progress_callback("audio", c, t, "Encoding audio (CLAP)...")

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

    def encode_dense(
        self,
        filtered: list[dict],
        *,
        progress_callback: Optional[Callable] = None,
    ) -> np.ndarray:
        """Dense lyrics pass → ``text_vecs``. Reads ``s["lyrics"]``, so a caller
        that fetches lyrics after CLAP must have written them into the same
        ``filtered`` dicts before calling this.

        Goes through ``ModelRegistry.encode_documents`` rather than touching the
        model: that is what bounds the batch and what applies the document side
        of Octen's prompt pair. This function used to call
        ``self.engine.model.encode(..., batch_size=32)`` and did neither — 65k
        tokens in one forward on a card shared with an LLM, embedded with a bare
        document side. See that method for both stories.
        """
        from app.resources.model_registry import ModelRegistry

        total = len(filtered)
        if progress_callback:
            progress_callback("lyrics", 0, total, "Encoding lyrics...")

        def _progress(done: int) -> None:
            if progress_callback:
                progress_callback("lyrics", done, total, "Encoding lyrics...")

        texts = [s["lyrics"] for s in filtered]
        text_vecs = ModelRegistry.encode_documents(texts, progress=_progress)

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
            MetadataDB.clear_track_metadata(self.collection_name)
        except Exception:
            logger.warning("[IndexingService] failed to clear old track_metadata — non-fatal")

        # CLAP (audio-only, CPU) first, then dense (GPU) — sequential here; the
        # IndexPipeline variant overlaps them with the network lane.
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
        coll = self.collection_name
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

    def persist_axis_norm_stats_from_mirror(self) -> None:
        """Recompute axis norm stats over the WHOLE collection (append path).

        ``_persist_axis_norm_stats`` derives mean/std from the batch it is
        handed — correct only when the batch IS the collection (the
        drop-and-rebuild path). On an append the batch is a handful of tracks,
        so writing its stats would replace the library's distribution with
        noise (``n=5``). The SQLite mirror already carries ``sonic_axes`` per
        track, so recompute from there: one indexed read, no Qdrant I/O.

        Best-effort — an append that already upserted must not fail here.
        """
        try:
            axes_map = {}
            for point_id, payload in MetadataDB.get_light_points(self.collection_name):
                axes = (payload or {}).get("sonic_axes")
                if axes and all(a in axes for a in AXIS_NAMES):
                    axes_map[point_id] = axes
        except Exception:
            logger.exception(
                "[IndexingService] axis-stat mirror read failed for '%s'",
                self.collection_name,
            )
            return
        if not axes_map:
            logger.info(
                "[IndexingService] no sonic_axes in the mirror for '%s' — norm stats left as-is",
                self.collection_name,
            )
            return
        self._persist_axis_norm_stats(axes_map)

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
            MetadataDB.set_axis_norm_stats(self.collection_name, stats)
            logger.info(
                "[IndexingService] axis_norm_stats persisted for '%s' (n=%d)",
                self.collection_name, stats["n"],
            )
        except Exception:
            logger.exception("[IndexingService] failed to persist axis_norm_stats")
