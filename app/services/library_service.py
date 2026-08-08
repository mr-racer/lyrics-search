"""Library service — indexing tracks from a folder."""

import asyncio
import hashlib
import logging
import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from ..domain.models import (
    AlbumSummary,
    AlbumTrack,
    ArtistRef,
    LibraryAlbumsResponse,
)
from ..resources.metadata_db import MetadataDB
from ..resources.model_registry import ModelRegistry
from ..resources.db_client import DbClient
from .artist_facts_service import fetch_facts_for_artists
from .artist_split import (
    split_artists, normalize_artist_name, primary_artist, artist_slugs,
    artist_refs_for_track,
    display_title_for_track,
)
from .genius_facts_service import fetch_genius_facts_for_songs
from .index_pipeline import IndexPipeline
from .job_tracker import JobTracker, IndexStage, IndexStatus
from .similarity_service import analyze_collection
from .song_facts_service import fetch_facts_for_songs
from .sonic_descriptor_service import SonicDescriptorService
from .track_credits_service import aggregate_labels, label_key

logger = logging.getLogger(__name__)

# Phase B (spec §6.2): global cap on concurrent indexing jobs across ALL
# accounts. Indexing is heavy (CLAP encoding is GPU-bound, lyrics fetch is
# I/O-bound) — two accounts in parallel is fine, ten would saturate the box.
# The per-account slot (_active_jobs) prevents an account double-starting;
# this semaphore bounds total parallelism. Env-tunable, default 2. Bounded so
# an over-release in a buggy finally raises ValueError instead of silently
# leaking capacity.
MAX_PARALLEL_INDEXING_JOBS = int(os.environ.get("MAX_PARALLEL_INDEXING_JOBS", "2"))
_INDEX_SEMAPHORE = threading.BoundedSemaphore(MAX_PARALLEL_INDEXING_JOBS)


def _slug_of_artist(name: str) -> str:
    """Canonical, Cyrillic-safe slug for a single artist name.

    Routes through ``artist_slugs`` (the same path used at index time and by the
    artist page), so the slug matches what ``track_artist_slugs`` stores and what
    ``GET /artists/{slug}`` resolves to. The old implementation used
    ``[^a-z0-9]+`` which stripped every non-ASCII character, turning any Cyrillic
    name into an empty slug — the Russian artist page then never opened.
    """
    slugs = artist_slugs(name)
    return slugs[0] if slugs else ""


def _album_artist_credit(primary_raw: str, feat_raws: list[str]):
    """Derive an album's display credit from its raw artist tags.

    ``primary_raw`` is the album's majority raw ``artist`` tag — which may itself
    be a collaboration like "Calvin Harris, Dua Lipa". ``feat_raws`` are the
    other distinct raw tags on the album. Returns
    ``(primary_name, primary_slug, feat_refs)`` where the primary is the LEADING
    participant of the majority tag and ``feat_refs`` is every OTHER participant
    across all tags (deduped, canonical slugs).

    This splits a single collaboration tag into primary + feat instead of
    treating the whole "A, B" string as one un-clickable primary that resolves
    to a non-existent combined slug.
    """
    primary_name = primary_artist(primary_raw) or (primary_raw or "").strip() or "—"
    primary_slug = _slug_of_artist(primary_name)
    seen = {primary_slug} if primary_slug else set()
    feats: list[ArtistRef] = []
    for raw in [primary_raw, *feat_raws]:
        for name in split_artists(raw):
            slug = _slug_of_artist(name)
            if not slug or slug in seen:
                continue
            seen.add(slug)
            feats.append(ArtistRef(name=name, slug=slug))
    return primary_name, primary_slug, feats


def _album_summary_from_sqlite(a: dict) -> AlbumSummary:
    """Build an AlbumSummary from a get_library_albums_from_sqlite aggregate,
    splitting collaboration tags into a primary + feat credit (canonical slugs)."""
    primary_name, primary_slug, feats = _album_artist_credit(
        a["primary_artist"], a.get("feat_artists") or [],
    )
    return AlbumSummary(
        album_title=a["album"],
        primary_artist=primary_name,
        primary_artist_slug=primary_slug,
        feat_artists=feats,
        year=a["year"],
        year_range=a["year_range"],
        cover_art_path=a["cover_art_path"],
        track_count=a["track_count"],
        duration_seconds=int(sum(t["duration"] or 0 for t in a["tracks"])),
        top_genres=a["top_genres"],
        labels=aggregate_labels(a.get("labels_raw") or []),
        tracks=[
            AlbumTrack(
                track_id=t["track_id"],
                title=t["title"],
                artist=t["artist"],
                duration=t["duration"],
                year=t["year"],
                cover_art_path=t["cover_art_path"],
            )
            for t in a["tracks"]
        ],
    )


def _label_peak_hour(hour: int, lang: str) -> str:
    if 5 <= hour <= 11:
        return "утра" if lang == "ru" else "mornings"
    if 12 <= hour <= 17:
        return "дня" if lang == "ru" else "afternoons"
    if 18 <= hour <= 22:
        return "вечера" if lang == "ru" else "evenings"
    return "ночи" if lang == "ru" else "late nights"


class LibraryService:
    """Index music files, extract metadata + lyrics, and upsert to Qdrant."""

    def __init__(
        self,
        search_service=None,
        db_client: Optional[DbClient] = None,
        sonic_descriptor_service: Optional[SonicDescriptorService] = None,
    ):
        """
        Args:
            search_service: SearchService instance for indexing tracks into Qdrant.
            db_client: DbClient instance for Qdrant access (needed for similarity analysis).
            sonic_descriptor_service: Optional SonicDescriptorService for per-track
                tag/class computation hooked into the AUDIO stage.
        """
        self.search_service = search_service
        self.db_client = db_client
        self.sonic_descriptor_service = sonic_descriptor_service
        self._job_tracker = JobTracker()
        # Phase B: per-account active-job map (replaces the single _current_job_id
        # slot, which rejected B's indexing while A indexed). Lock guards the dict;
        # maps account_id → in-flight job_id.
        self._active_jobs: dict[str, str] = {}
        self._active_jobs_lock = threading.Lock()

    # ── Per-account indexing queue (Phase B) ───────────────────────────────

    def try_start_job(self, account_id: str, job_id: str) -> bool:
        """Atomically claim the indexing slot for ``account_id``.

        Returns True if the slot was free and is now claimed by ``job_id``;
        False if this account already has an in-flight job. Other accounts are
        unaffected — a compare-and-swap under the lock, so 50 racing threads for
        the same account yield exactly one winner.
        """
        with self._active_jobs_lock:
            if account_id in self._active_jobs:
                return False
            self._active_jobs[account_id] = job_id
            return True

    def finish_job(self, account_id: str) -> None:
        """Release the indexing slot for ``account_id`` (idempotent)."""
        with self._active_jobs_lock:
            self._active_jobs.pop(account_id, None)

    def is_account_indexing(self, account_id: str) -> bool:
        with self._active_jobs_lock:
            return account_id in self._active_jobs

    def get_account_job_id(self, account_id: str) -> Optional[str]:
        with self._active_jobs_lock:
            return self._active_jobs.get(account_id)

    # ── Server-mode upload indexing (Phase C) ──────────────────────────────

    def enqueue_upload_indexing(
        self, *, account_id: str, upload_ids: list[str],
        lang: str = "ru",
    ) -> str:
        """Server-mode batch-commit entry point. Returns the JobTracker job_id.

        Reuses Phase B's per-account queue: a second commit for the SAME account
        while one is RUNNING raises 409; different accounts run concurrently,
        bounded by the global MAX_PARALLEL_INDEXING_JOBS semaphore inside the
        background runner.
        """
        from fastapi import HTTPException

        existing = self.get_account_job_id(account_id)
        if existing:
            cur = self._job_tracker.get_job(existing)
            if cur and cur.overall_status == IndexStatus.RUNNING:
                raise HTTPException(
                    status_code=409,
                    detail=f"indexing already in progress for this account (job={existing})",
                )
            self.finish_job(account_id)  # stale entry left by a crashed job

        collection_name = f"acct_{account_id}"
        job = self._job_tracker.create_job(
            folder_path=f"<uploads:{len(upload_ids)}>",
            collection_name=collection_name,
        )
        if not self.try_start_job(account_id=account_id, job_id=job.job_id):
            self._job_tracker.remove_completed_job(job.job_id)
            raise HTTPException(
                status_code=409,
                detail="indexing already in progress for this account",
            )
        job.overall_status = IndexStatus.RUNNING
        asyncio.create_task(
            self._run_upload_indexing_job(job, account_id, upload_ids, lang)
        )
        return job.job_id

    async def _run_upload_indexing_job(
        self, job, account_id: str, upload_ids: list[str],
        lang: str = "ru",
    ):
        """Background runner for server-mode uploads — mirrors _run_indexing_job."""
        await asyncio.to_thread(_INDEX_SEMAPHORE.acquire)
        try:
            rows = []
            for uid in upload_ids:
                row = MetadataDB.get_pending_upload(uid)
                if row and row["account_id"] == account_id:
                    rows.append(row)
            if not rows:
                job.overall_status = IndexStatus.FAILED
                job.error_message = "no valid upload ids"
                await self._notify_progress(job, {
                    "overall_status": IndexStatus.FAILED.value, "error": job.error_message,
                })
                return
            if self.db_client is None:
                job.overall_status = IndexStatus.FAILED
                job.error_message = "DB client unavailable"
                await self._notify_progress(job, {
                    "overall_status": IndexStatus.FAILED.value, "error": job.error_message,
                })
                return

            engine = self.db_client.search_engine

            # Same sanitize + batch-model selection as the folder flow
            # (_run_indexing_job): treat stringified-null junk as None. The
            # batch model is passed EXPLICITLY through the pipeline — the
            # shared engine is never mutated, so two accounts indexing in
            # parallel (semaphore=2) can't leak models/collections into each
            # other's job. Warm the registry cache up front.
            # Warm cache OFF the event loop: a cold SentenceTransformer load
            # is tens of seconds of blocking I/O + torch init, and this
            # coroutine runs on the loop — inline it and every request
            # (login included) freezes until the weights are in RAM.
            await asyncio.to_thread(ModelRegistry.get_text_model)

            collection_name = f"acct_{account_id}"
            loop = asyncio.get_running_loop()

            def _pipeline_progress(stage, current, total, message, **kw):
                # scan → LYRICS, lyrics → DENSE, audio → AUDIO (see _on_index_progress).
                # Thread-safe: the pipeline fires this from its GPU/lyrics executors.
                asyncio.run_coroutine_threadsafe(
                    self._on_index_progress(job, stage, current, total, message), loop,
                )

            # Build the Yandex enrichment client once (account token if linked, else
            # anonymous) for the metadata backfill during tag-read.
            enrich_client = None
            try:
                from app.services.yandex.enrichment import client_for_account
                enrich_client = client_for_account(account_id)
            except Exception:
                logger.debug("[upload] enrichment client unavailable", exc_info=True)

            # Stage 0 — local tag-read (NO online lyrics; embedded/Yandex text is
            # read here and skips the network). Marks rows 'indexing', extracts
            # covers, fails unidentifiable rows.
            tracks, upload_by_key = await self._tagread_upload_rows(
                account_id, rows, enrich_client, _pipeline_progress,
            )
            if not tracks:
                logger.info("[upload] nothing indexable for account=%s", account_id)
                job.overall_status = IndexStatus.COMPLETED
                await self._notify_progress(job, {
                    "overall_status": IndexStatus.COMPLETED.value,
                    "message": "Нет треков для индексации",
                })
                return

            # FACTS / bio / images run CONCURRENTLY with the encode pipeline — they
            # need only artist+title (already in `tracks`), so they overlap the
            # lyrics fetch and CLAP/dense GPU work instead of trailing them.
            facts_task = asyncio.create_task(
                self._fetch_facts_batch(job, collection_name, tracks, lang),
                name="upload-facts",
            )

            # Encode pipeline: lyrics fetch ‖ (CLAP → dense) → upsert.
            pipeline = IndexPipeline(engine)
            _, track_ids = await pipeline.run(
                tracks, collection_name, better_lyrics_quality=False,
                progress=_pipeline_progress, resolve_track_ids=True,
            )

            # Wait for FACTS before AI (the AI tasks consume facts/bio as input).
            try:
                producer_label_by_song = await facts_task
            except Exception:
                logger.exception("[enrich] upload FACTS task failed (tracks already indexed)")
            else:
                self._apply_producer_label(producer_label_by_song, track_ids, collection_name)


            # Stamp pending_uploads.track_id from the resolved point ids.
            self._apply_upload_track_ids(upload_by_key, track_ids)

            # AI tasks (after facts + upsert); awaited so COMPLETED gates player entry.
            try:
                await self._run_ai_tasks(collection_name, len(tracks), lang, job=job)
            except Exception:
                logger.exception("[enrich] upload AI tasks failed (tracks already indexed)")

            job.overall_status = IndexStatus.COMPLETED
            await self._notify_progress(job, {
                "overall_status": IndexStatus.COMPLETED.value,
                "message": f"Загружено {len(tracks)} треков",
            })
        except Exception as e:
            logger.exception("[LibraryService] upload indexing job %s failed", job.job_id)
            job.overall_status = IndexStatus.FAILED
            job.error_message = str(e)
            await self._notify_progress(job, {
                "overall_status": IndexStatus.FAILED.value, "error": str(e),
            })
        finally:
            _INDEX_SEMAPHORE.release()
            self.finish_job(account_id=account_id)

    # ── Yandex Music import (download phase → existing upload indexing) ──────

    def enqueue_yandex_import(
        self, *, account_id: str, sources: list, lang: str = "ru",
    ) -> str:
        """Start a Yandex import job (download phase). Returns the JobTracker job_id.

        ``sources`` is a list of selected sources, each ``"likes"`` or
        ``{"kind": <int>}`` — they are merged (deduped by track id) into one job.

        Mirrors ``enqueue_upload_indexing``'s per-account slot semantics: a second
        import (or upload) for the SAME account while one is RUNNING raises 409.
        When the download phase finishes, the runner releases the slot and hands
        the downloaded files to the normal ``enqueue_upload_indexing`` flow.
        """
        from fastapi import HTTPException

        existing = self.get_account_job_id(account_id)
        if existing:
            cur = self._job_tracker.get_job(existing)
            if cur and cur.overall_status == IndexStatus.RUNNING:
                raise HTTPException(
                    status_code=409,
                    detail=f"indexing already in progress for this account (job={existing})",
                )
            self.finish_job(account_id)  # stale entry left by a crashed job

        collection_name = f"acct_{account_id}"
        job = self._job_tracker.create_job(
            folder_path=f"<yandex:{len(sources)} sources>",
            collection_name=collection_name,
        )
        if not self.try_start_job(account_id=account_id, job_id=job.job_id):
            self._job_tracker.remove_completed_job(job.job_id)
            raise HTTPException(
                status_code=409,
                detail="indexing already in progress for this account",
            )
        job.overall_status = IndexStatus.RUNNING
        asyncio.create_task(
            self._run_yandex_import_job(job, account_id, sources, lang)
        )
        return job.job_id

    async def _run_yandex_import_job(self, job, account_id: str, sources: list, lang: str = "ru"):
        """Download phase runner: download Yandex tracks, then chain into upload indexing."""
        from app.services.yandex import importer

        await asyncio.to_thread(_INDEX_SEMAPHORE.acquire)
        upload_ids: list[str] = []
        report: dict = {}
        download_failed = False
        try:
            loop = asyncio.get_running_loop()

            def _progress(done: int, total: int, message: str):
                asyncio.run_coroutine_threadsafe(
                    self._notify_progress(job, {
                        "overall_status": IndexStatus.RUNNING.value,
                        "stage": "download", "current": done, "total": total,
                        "message": message,
                    }),
                    loop,
                )

            upload_ids, report = await loop.run_in_executor(
                None,
                lambda: importer.download_sources(account_id, sources, on_progress=_progress),
            )
            # Persist the skipped-tracks report on the job so a status endpoint can
            # surface it (it also lives in yandex_imports for durability).
            job.yandex_report = report
        except importer.YandexNotLinkedError:
            download_failed = True
            job.overall_status = IndexStatus.FAILED
            job.error_message = "yandex account not linked"
            await self._notify_progress(job, {
                "overall_status": IndexStatus.FAILED.value, "error": job.error_message,
            })
        except Exception as e:
            download_failed = True
            logger.exception("[LibraryService] yandex import download phase failed")
            job.overall_status = IndexStatus.FAILED
            job.error_message = str(e)
            await self._notify_progress(job, {
                "overall_status": IndexStatus.FAILED.value, "error": str(e),
            })
        finally:
            # Release BEFORE chaining: enqueue_upload_indexing claims the same
            # per-account slot + semaphore for the indexing phase.
            _INDEX_SEMAPHORE.release()
            self.finish_job(account_id=account_id)

        if download_failed:
            return

        # Chain into the EXISTING upload-indexing flow FIRST so we can hand its
        # job_id to the frontend in the download job's completion event — letting
        # the import UI switch deterministically from the download bar to the
        # indexing wizard with no polling race (spec §3.2 of the frontend design).
        skipped_n = len(report.get("skipped", []))
        indexing_job_id = None
        if upload_ids:
            try:
                indexing_job_id = self.enqueue_upload_indexing(
                    account_id=account_id, upload_ids=upload_ids, lang=lang,
                )
            except Exception:
                logger.exception(
                    "[LibraryService] failed to start indexing after yandex download "
                    "(files are in pending_uploads; user can retry batch-commit)",
                )

        job.overall_status = IndexStatus.COMPLETED
        # Persist the handoff on the job (not just in the transient event) so a
        # LATE SSE subscriber — one that connects after the download already
        # finished — still gets indexing_job_id in the stream's initial
        # get_progress_summary() snapshot. Without this, a fast/deduped download
        # races the EventSource: the live completion event is delivered to zero
        # subscribers and the late connector sees completed WITHOUT the handoff,
        # navigating away while indexing is still running in the background.
        if indexing_job_id:
            job.indexing_job_id = indexing_job_id
        completion = {
            "overall_status": IndexStatus.COMPLETED.value,
            "message": (
                f"Скачано {report.get('downloaded', 0)} треков"
                + (f", пропущено {skipped_n}" if skipped_n else "")
            ),
            "yandex_report": report,
        }
        if indexing_job_id:
            completion["indexing_job_id"] = indexing_job_id
        await self._notify_progress(job, completion)

    async def _tagread_upload_rows(self, account_id: str, rows: list, enrich_client, progress):
        """Local tag-read (NO online lyrics) for upload rows.

        Returns ``("Artist — Title" -> meta, "Artist — Title" -> upload_id)``.
        Embedded/Yandex lyrics are read here (so they skip the pipeline's network
        lane); online lyrics for the rest are fetched inside IndexPipeline. Marks
        every row 'indexing', extracts cover art, and fails rows with no
        title/artist or a missing file. Tag reads fan out across a thread pool —
        the per-file work (mutagen + m4a optimize + cover) is I/O-bound.
        """
        from concurrent.futures import ThreadPoolExecutor
        from app.indexing.cover_art import save_cover_art
        from app.indexing.folder_scanner import read_tags_only

        for r in rows:
            MetadataDB.update_pending_upload_status(r["upload_id"], status="indexing")

        total = len(rows)
        if progress:
            progress("scan", 0, total, "Чтение метаданных и обложек...")

        def _read_one(row):
            file_path = Path(row["storage_path"])
            if not file_path.exists():
                return row, None, f"file missing on disk: {file_path}"
            try:
                info = read_tags_only(file_path, enrich_client=enrich_client)
            except Exception as e:
                logger.exception("[upload] tag read failed for %s", file_path)
                return row, None, str(e)
            if not info or not info.get("title") or not info.get("artist"):
                return row, None, "missing title/artist in metadata tags"
            try:
                info["cover_art_path"] = save_cover_art(
                    str(file_path), row["sha256"][:16],
                    meta=info, yandex_client=enrich_client,
                )
            except Exception as e:
                logger.warning("[upload] cover extraction failed for %s: %s", file_path, e)
                info["cover_art_path"] = None
            return row, info, None

        loop = asyncio.get_running_loop()
        workers = min(8, total) or 1
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="upl-tagread") as ex:
            results = await asyncio.gather(*[
                loop.run_in_executor(ex, _read_one, row) for row in rows
            ])

        data: dict[str, dict] = {}
        upload_by_key: dict[str, str] = {}
        for row, info, error in results:
            if error is not None:
                MetadataDB.update_pending_upload_status(
                    row["upload_id"], status="failed", error=error,
                )
                continue
            key = f"{info['artist']} — {info['title']}"
            data[key] = info
            upload_by_key[key] = row["upload_id"]
        return data, upload_by_key

    def _apply_upload_track_ids(self, upload_by_key: dict, track_ids: dict) -> None:
        """Stamp pending_uploads.track_id from the pipeline's resolved point ids."""
        for key, upload_id in upload_by_key.items():
            tid = track_ids.get(key)
            if tid:
                MetadataDB.update_pending_upload_status(
                    upload_id, status="done", track_id=tid,
                )
            else:
                MetadataDB.update_pending_upload_status(
                    upload_id, status="failed",
                    error="track did not appear in Qdrant after upsert",
                )

    async def _fetch_facts_batch(self, job, collection_name: str, indexed_data: dict, lang: str = "ru") -> dict:
        """FACTS stage shared by the upload and folder flows — song/artist facts +
        AudioDB biography + artist images. Launched CONCURRENTLY with the encode
        pipeline (it needs only artist+title), so it overlaps the lyrics fetch and
        the GPU passes. Collab tags are split into individual artists ('DNCE, Nicki
        Minaj' → two artist lookups); song facts use the primary artist.

        Returns the Genius producer/label mapping (``"{artist} — {title}" ->
        (producer, label)``) collected along the way — no Qdrant point id
        exists yet at this point, so the caller applies it once track ids are
        resolved after the encode pipeline's upsert (see
        ``_apply_producer_label``).
        """
        if not indexed_data:
            return {}

        from app.services.audiodb_service import fetch_audiodb_for_artists

        # Split raw artist tags into individual participants, normalize and dedupe.
        # A track with "Calvin Harris, Dua Lipa" yields two separate artists
        # for facts/AudioDB lookup, each stored under its own slug.
        raw_artists = {
            (info.get("artist") or "").strip()
            for info in indexed_data.values()
            if (info.get("artist") or "").strip()
        }
        unique_artists = sorted({
            norm_name
            for raw in raw_artists
            for name in split_artists(raw)
            if (norm_name := normalize_artist_name(name))
        })
        unique_songs = sorted({
            ((info.get("artist") or "").strip(), (info.get("title") or "").strip())
            for info in indexed_data.values()
            if (info.get("artist") or "").strip() and (info.get("title") or "").strip()
        })

        # ── Stage FACTS ──────────────────────────────────────────────────────
        stage_facts = job.stages[IndexStage.FACTS]
        stage_facts.status = IndexStatus.RUNNING
        stage_facts.started_at = time.time()
        # Each artist contributes 2 units (songfacts + audiodb); each song 2
        # (songfacts + genius).
        facts_total = len(unique_artists) * 2 + len(unique_songs) * 2
        stage_facts.total = facts_total
        stage_facts.message = "Поиск фактов..."
        await self._notify_progress(job, {
            "stage": IndexStage.FACTS.value,
            "stage_status": IndexStatus.RUNNING.value,
            "message": stage_facts.message, "current": 0, "total": facts_total,
        })
        logger.info(
            "[enrich] FACTS stage start: %d artists, %d songs (collection=%s)",
            len(unique_artists), len(unique_songs), collection_name,
        )

        facts_progress = {"artists": 0, "songs": 0, "audiodb": 0, "genius": 0}
        facts_found = {"artists": 0, "songs": 0, "audiodb": 0, "genius": 0}

        def _make_cb(bucket: str):
            def _cb(current: int, total: int, label: str, found: bool):
                facts_progress[bucket] = current
                if found:
                    facts_found[bucket] += 1
                combined = sum(facts_progress.values())
                stage_facts.current = combined
                stage_facts.found = sum(facts_found.values())
                stage_facts.not_found = max(0, facts_total - stage_facts.found)
                stage_facts.message = f"Факты: {label}"
                asyncio.create_task(self._notify_progress(job, {
                    "stage": IndexStage.FACTS.value,
                    "current": combined, "total": facts_total,
                    "message": stage_facts.message,
                    "found": stage_facts.found, "not_found": stage_facts.not_found,
                }))
            return _cb

        # No wall-clock timeout: all fetchers are idempotent (cached artists/
        # songs are skipped), so a large library enriches fully instead of
        # being cut off mid-alphabet after N seconds and never backfilled on
        # later runs. All four sources run CONCURRENTLY — none depends on
        # another's output.
        producer_label_by_song: dict = {}

        async def _run_genius() -> None:
            nonlocal producer_label_by_song
            producer_label_by_song = await fetch_genius_facts_for_songs(
                unique_songs, collection_name,
                progress_callback=_make_cb("genius"),
            )

        if facts_total:
            tasks = []
            if unique_artists or unique_songs:
                tasks.append(fetch_facts_for_artists(
                    unique_artists, collection_name,
                    progress_callback=_make_cb("artists"),
                ))
                tasks.append(fetch_facts_for_songs(
                    unique_songs, collection_name,
                    progress_callback=_make_cb("songs"),
                ))
            if unique_artists:
                tasks.append(fetch_audiodb_for_artists(
                    unique_artists, collection_name,
                    progress_callback=_make_cb("audiodb"),
                ))
            if unique_songs:
                tasks.append(_run_genius())
            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    logger.warning("[enrich] facts fetch failed: %s", e)

        stage_facts.status = IndexStatus.COMPLETED
        stage_facts.current = facts_total
        stage_facts.completed_at = time.time()
        stage_facts.found = sum(facts_found.values())
        stage_facts.not_found = max(0, facts_total - stage_facts.found)
        stage_facts.message = (
            f"Факты: {stage_facts.found} найдено из {facts_total}"
            if facts_total else "Нет данных"
        )
        await self._notify_progress(job, {
            "stage": IndexStage.FACTS.value,
            "stage_status": IndexStatus.COMPLETED.value,
            "current": facts_total, "total": facts_total,
            "message": stage_facts.message,
            "found": stage_facts.found, "not_found": stage_facts.not_found,
        })
        logger.info(
            "[enrich] FACTS stage done: %d/%d found (collection=%s)",
            stage_facts.found, facts_total, collection_name,
        )
        return producer_label_by_song

    def _apply_producer_label(
        self, producer_label_by_song: dict, track_ids: dict, collection_name: str,
    ) -> None:
        """Write Genius producer/label onto track_metadata once ids are known.

        ``track_ids`` and ``producer_label_by_song`` are both keyed by
        ``"{artist} — {title}"`` (see ``IndexPipeline._resolve_track_ids`` and
        ``genius_facts_service.fetch_genius_facts_for_songs``).
        """
        for key, (producer, label) in producer_label_by_song.items():
            track_id = track_ids.get(key)
            if track_id:
                MetadataDB.update_track_producer_label(collection_name, track_id, producer, label)

    async def _run_ai_tasks(
        self, collection_name: str, n_total: int, lang: str = "ru", job=None,
    ) -> None:
        """Auto AI-indexing (see ``task_types`` below) for a
        just-indexed batch — only when the LLM is reachable. Awaited by the runner
        AFTER facts + upsert, so COMPLETED gates player entry (the frontend relies
        on it), and so the AI tasks see the facts/bio they consume.

        ``job``: optional JobTracker job. When given, per-task guru progress is
        published into the SAME SSE stream the core stages use (``ai_stages`` key
        on each event + the late-subscriber snapshot via get_progress_summary),
        so the frontend never has to poll /library/ai-index/status and can't
        confuse this run with a previous one's rows.
        """
        from app.services import ai_indexing_service
        from app.services.llm_client import (
            is_llm_available, resolve_base_url, resolve_model,
        )

        if not await is_llm_available():
            logger.info(
                "[enrich] LLM unreachable — skipping auto AI-indexing (collection=%s)",
                collection_name,
            )
            return

        base_url = resolve_base_url()
        model = resolve_model()
        logger.info(
            "[enrich] LLM reachable (base_url=%s model=%s) — auto AI-indexing %s",
            base_url, model, collection_name,
        )

        # Order matters: the tasks run back-to-back so a local LLM serves one at
        # a time. fact_relations + lyric_gems sit between refined_facts and
        # artist_bio — all the text/fact mining groups together, and the slow
        # web-search bio stays last. fact_relations replaces the old inline hook
        # in song_facts_service (invisible, uncountable); lyric_gems needs the
        # tracks to be in Qdrant already, which is true here (after upsert) but
        # not during the FACTS stage.
        task_types = (
            "sonic_vibe", "refined_facts", "fact_relations", "lyric_gems", "artist_bio",
        )
        ai_stages: dict = {
            tt: {"status": "pending", "n_done": 0, "n_total": 0} for tt in task_types
        }

        async def _publish() -> None:
            if job is None:
                return
            job.ai_stages = ai_stages  # snapshot for late SSE subscribers
            try:
                await self._notify_progress(job, {"ai_stages": ai_stages})
            except Exception:
                logger.debug("[enrich] ai_stages notify failed", exc_info=True)

        def _pull_row(task_type: str) -> None:
            """Refresh one task's entry from its ai_indexing_jobs row."""
            try:
                row = MetadataDB.get_latest_ai_job(collection_name, task_type)
            except Exception:
                return
            if not row:
                return
            ai_stages[task_type] = {
                "status": row["status"],
                "n_done": int(row["n_done"] or 0),
                "n_total": int(row["n_total"] or 0),
            }

        await _publish()
        for task_type in task_types:
            try:
                job_id = ai_indexing_service.start_job(
                    task_type=task_type,
                    collection_name=collection_name,
                    lang=lang,
                    db_client=self.db_client,
                    llm_client=None,
                    n_total=n_total,
                    llm_base_url=base_url,
                    llm_model=model,
                    bio_source=("web" if task_type == "artist_bio" else "facts"),
                )
                waiter = asyncio.create_task(ai_indexing_service.wait_for_job(job_id))
                while not waiter.done():
                    await asyncio.wait({waiter}, timeout=2.0)
                    _pull_row(task_type)
                    await _publish()
                _pull_row(task_type)
                await _publish()
                logger.info(
                    "[enrich] AI task '%s' finished (collection=%s)",
                    task_type, collection_name,
                )
            except ValueError as e:
                logger.warning("[enrich] AI task '%s' skipped: %s", task_type, e)
                ai_stages[task_type]["status"] = "skipped"
                await _publish()
            except Exception:
                logger.exception(
                    "[enrich] AI task '%s' failed (collection=%s)",
                    task_type, collection_name,
                )
                ai_stages[task_type]["status"] = "failed"
                await _publish()

    async def index_folder(
        self,
        folder_path: str,
        collection_name: str,
        better_lyrics_quality: bool = False,
        enhance_by_musicbrainz: bool = False,
        account_id: str = "default",
    ) -> dict:
        """
        Index all audio files in folder with progress tracking.
        Returns dict with job_id for tracking progress via SSE.

        ``collection_name`` is REQUIRED — no silent default. Every collection
        this service creates must be the caller's derived account collection
        (``acct_<user.id>``); a fallback name here once produced a library
        invisible to every derived route (playback history, stream, profile
        all read ``acct_*`` while the music sat in the default-named
        collection).

        ``account_id``: per-account indexing slot (Phase B). Defaults to
        ``"default"`` so callers that don't thread an identity through still
        share a single bucket; the /library/index route passes ``current_user.id``.
        Two distinct accounts may index concurrently; a second job for the SAME
        account is rejected.
        """
        logger.info(
            "[LibraryService] index_folder called: account=%s folder=%s collection=%s",
            account_id, folder_path, collection_name,
        )

        # Reject if THIS account is already indexing (other accounts unaffected).
        existing = self.get_account_job_id(account_id)
        if existing:
            current_job = self._job_tracker.get_job(existing)
            if current_job and current_job.overall_status == IndexStatus.RUNNING:
                logger.warning(
                    "[LibraryService] account=%s already indexing job=%s, rejecting",
                    account_id, existing,
                )
                return {
                    "status": "failed",
                    "message": "Indexing already in progress for this account",
                    "job_id": existing,
                }
            # Stale entry (a prior job finished/failed without finally cleanup).
            self.finish_job(account_id)

        # Create the job, then atomically claim the slot. If the claim races and
        # loses, drop the freshly-created job and bail.
        job = self._job_tracker.create_job(folder_path, collection_name)
        if not self.try_start_job(account_id=account_id, job_id=job.job_id):
            self._job_tracker.remove_completed_job(job.job_id)
            return {
                "status": "failed",
                "message": "Indexing already in progress for this account",
                "job_id": self.get_account_job_id(account_id),
            }
        job.overall_status = IndexStatus.RUNNING
        logger.info("[LibraryService] Job created: %s (account=%s)", job.job_id, account_id)

        # Start indexing in background task to allow SSE progress updates
        task = asyncio.create_task(
            self._run_indexing_job(
                job, better_lyrics_quality, enhance_by_musicbrainz, account_id
            )
        )
        logger.info("[LibraryService] Background task created: %s", task)

        return {
            "status": "started",
            "job_id": job.job_id,
            "message": "Indexing started",
        }

    async def _run_indexing_job(
        self, job, better_lyrics_quality: bool,
        enhance_by_musicbrainz: bool = False, account_id: str = "default",
    ):
        """Execute the indexing process with progress updates."""
        folder_path = job.folder_path
        collection_name = job.collection_name
        logger.info("[LibraryService] _run_indexing_job START: job=%s, folder=%s, collection=%s",
                    job.job_id, folder_path, collection_name)

        # Phase B (spec §6.2): gate heavy indexing on the global semaphore so N
        # accounts starting on the same minute don't all saturate GPU/network.
        # Acquired in a worker thread because BoundedSemaphore.acquire() blocks —
        # to_thread keeps the event loop responsive while this job waits its turn.
        await asyncio.to_thread(_INDEX_SEMAPHORE.acquire)
        try:
            # Cold load = tens of seconds of blocking work; keep it off the
            # event loop or every request (login included) hangs meanwhile.
            await asyncio.to_thread(ModelRegistry.get_text_model)

            loop = asyncio.get_event_loop()

            # ── Stage LYRICS: tag-read (online lyrics fetched by the pipeline) ──
            logger.info("[LibraryService] Stage LYRICS: reading tags + covers")
            stage_lyrics = job.stages[IndexStage.LYRICS]
            stage_lyrics.status = IndexStatus.RUNNING
            stage_lyrics.started_at = time.time()
            stage_lyrics.message = "Чтение тегов..."

            # rglob walks the whole tree synchronously — on a big/network
            # folder that's seconds of blocked event loop, so off-thread it.
            audio_files = await asyncio.to_thread(
                lambda: [
                    p for p in Path(folder_path).rglob("*")
                    if p.suffix.lower() in (".flac", ".m4a", ".mp3")
                ]
            )
            stage_lyrics.total = len(audio_files)
            await self._notify_progress(job, {
                "stage": IndexStage.LYRICS.value,
                "stage_status": IndexStatus.RUNNING.value,
                "message": f"Найдено {len(audio_files)} файлов",
                "current": 0, "total": len(audio_files),
            })

            # Local tag-read (NO online lyrics): embedded text is read here and
            # skips the network; covers are extracted now (the indexing payload
            # needs them). The IndexPipeline then fetches online lyrics for the
            # rest — overlapped with CLAP/dense — and drives the LYRICS stage to
            # COMPLETED via its "scan" progress (see _on_index_progress).
            processed_files = await asyncio.to_thread(self._tagread_folder, audio_files)
            track_count = len(processed_files)
            logger.info("[LibraryService] Tag-read done, %d identifiable tracks", track_count)
            stage_lyrics.message = f"Прочитано {track_count} треков"
            await self._notify_progress(job, {
                "stage": IndexStage.LYRICS.value,
                "message": stage_lyrics.message, "current": 0, "total": track_count,
            })

            # ── Stage FACTS: launched CONCURRENTLY with encoding ───────────────
            # Facts/bio/images need only artist+title (already read), so they run
            # in parallel with the lyrics fetch + CLAP/dense and are awaited after
            # the encode pipeline (before ANALYSIS). _fetch_facts_batch owns the
            # FACTS stage lifecycle (RUNNING → COMPLETED) + progress.
            logger.info("[LibraryService] Stage FACTS: launching concurrent fetch")
            facts_task = asyncio.create_task(
                self._fetch_facts_batch(job, collection_name, processed_files),
                name="folder-facts",
            )

            # ── Stage METADATA: MusicBrainz enrichment ────────────────────────
            logger.info("[LibraryService] Stage METADATA: MusicBrainz enrichment")
            stage_meta = job.stages[IndexStage.METADATA]

            # CURRENTLY MUSICBRAINZ API IS DISABLED DUE TO GIVING UNSTABLE PARSING RESULTS

            # if enhance_by_musicbrainz and processed_files:
            #     stage_meta.status = IndexStatus.RUNNING
            #     stage_meta.started_at = time.time()
            #     stage_meta.total = track_count
            #     stage_meta.message = "Обогащение метаданных (MusicBrainz)..."

            #     await self._notify_progress(job, {
            #         "stage": IndexStage.METADATA.value,
            #         "stage_status": IndexStatus.RUNNING.value,
            #         "message": stage_meta.message,
            #         "current": 0,
            #         "total": track_count,
            #     })

            #     mb_error = None
            #     mb_found = 0
            #     mb_not_found = 0

            #     async def on_mb_progress(current: int, total: int, label: str, found: int = 0, not_found: int = 0):
            #         nonlocal mb_found, mb_not_found
            #         stage_meta.current = current
            #         stage_meta.message = label
            #         mb_found = found
            #         mb_not_found = not_found
            #         eta = job.calculate_eta_seconds(IndexStage.METADATA)
            #         await self._notify_progress(job, {
            #             "stage": IndexStage.METADATA.value,
            #             "current": current,
            #             "total": total,
            #             "message": label,
            #             "eta_seconds": eta,
            #             "found": found,
            #             "not_found": not_found,
            #         })

            #     try:
            #         await asyncio.to_thread(
            #             self._enrich_with_musicbrainz,
            #             processed_files,
            #             enhance_by_musicbrainz,
            #             lambda c, t, label, f, nf: asyncio.run_coroutine_threadsafe(
            #                 on_mb_progress(c, t, label, f, nf), loop
            #             ),
            #         )
            #     except Exception as e:
            #         logger.warning("[LibraryService] MusicBrainz enrichment failed (non-critical): %s", e)
            #         mb_error = str(e)

            #     stage_meta.status = IndexStatus.COMPLETED
            #     stage_meta.current = stage_meta.current
            #     stage_meta.completed_at = time.time()
            #     stage_meta.found = mb_found
            #     stage_meta.not_found = mb_not_found
            #     if mb_error:
            #         stage_meta.message = f"Ошибка: {mb_error}"
            #     else:
            #         stage_meta.message = f"Метаданные: {mb_found} найдено из {track_count}"

            #     await self._notify_progress(job, {
            #         "stage": IndexStage.METADATA.value,
            #         "stage_status": IndexStatus.COMPLETED.value,
            #         "current": stage_meta.current,
            #         "total": track_count,
            #         "message": stage_meta.message,
            #         "found": mb_found,
            #         "not_found": mb_not_found,
            #         "stage_error": mb_error,
            #     })
            # else:

            stage_meta.status = IndexStatus.COMPLETED
            stage_meta.message = "Пропущено"

            await self._notify_progress(job, {
                "stage": IndexStage.METADATA.value,
                "stage_status": IndexStatus.COMPLETED.value,
                "message": stage_meta.message,
            })

            # ── Stage DENSE + AUDIO: encode via the concurrent pipeline ────────
            # processed_files already carries covers + numeric duration (tag-read),
            # so it feeds IndexPipeline directly — no TrackMetadata round-trip. The
            # pipeline fetches online lyrics (its "scan" stage → LYRICS) overlapped
            # with CLAP (→ AUDIO) then dense (→ DENSE).
            logger.info("[LibraryService] Stage DENSE/AUDIO: encoding via IndexPipeline")
            track_count = len(processed_files)

            stage_dense = job.stages[IndexStage.DENSE]
            stage_dense.status = IndexStatus.RUNNING
            stage_dense.started_at = time.time()
            stage_dense.total = track_count
            stage_dense.message = "Кодирование текстов (dense)..."

            stage_audio = job.stages[IndexStage.AUDIO]
            stage_audio.status = IndexStatus.RUNNING
            stage_audio.started_at = time.time()
            stage_audio.total = track_count
            stage_audio.message = "CLAP-кодирование аудио..."

            await self._notify_progress(job, {
                "stage": IndexStage.DENSE.value,
                "stage_status": IndexStatus.RUNNING.value,
                "message": stage_dense.message, "current": 0, "total": track_count,
            })
            await self._notify_progress(job, {
                "stage": IndexStage.AUDIO.value,
                "stage_status": IndexStatus.RUNNING.value,
                "message": stage_audio.message, "current": 0, "total": track_count,
            })

            track_ids: dict = {}
            if self.db_client and processed_files:
                # The shared engine is never mutated, so parallel jobs can't
                # cross-contaminate; the collection is per-run.
                engine = self.db_client.search_engine
                # warm cache off-loop (см. комментарий выше)
                await asyncio.to_thread(ModelRegistry.get_text_model)

                loop = asyncio.get_event_loop()

                def _pipeline_cb(stage, current, total, message, **kw):
                    # scan → LYRICS (online fetch), lyrics → DENSE, audio → AUDIO.
                    # Thread-safe: fired from the pipeline's GPU/lyrics executors.
                    asyncio.run_coroutine_threadsafe(
                        self._on_index_progress(job, stage, current, total, message), loop,
                    )

                pipeline = IndexPipeline(engine)
                # resolve_track_ids=True: needed to apply Genius producer/label
                # (collected by the concurrent FACTS stage) onto track_metadata
                # once point ids exist post-upsert — see _apply_producer_label.
                _, track_ids = await pipeline.run(
                    processed_files, collection_name,
                    better_lyrics_quality=better_lyrics_quality,
                    progress=_pipeline_cb, resolve_track_ids=True,
                )
                logger.info("[LibraryService] IndexPipeline.run done")

                # ── Sonic Descriptor hook: per-track tags + class for new tracks.
                # Scrolls the collection once; wrapped so a bug never fails the index.
                try:
                    if self.sonic_descriptor_service is not None and self.db_client is not None:
                        await asyncio.to_thread(
                            self._run_sonic_descriptor_hook,
                            collection_name=collection_name,
                        )
                except Exception as e:
                    logger.warning(
                        "[LibraryService] sonic_descriptor hook failed for collection %s: %s",
                        collection_name, e,
                    )
            else:
                logger.warning("[LibraryService] Skipping indexing: db_client=%s, tracks=%d",
                              self.db_client is not None, len(processed_files))
                # No pipeline ran → close LYRICS (the pipeline's "scan" usually
                # completes it) plus DENSE/AUDIO so no stage is stuck RUNNING.
                for stage in (IndexStage.LYRICS, IndexStage.DENSE, IndexStage.AUDIO):
                    sp = job.stages[stage]
                    sp.status = IndexStatus.COMPLETED
                    sp.completed_at = time.time()

            # FACTS were launched concurrently with encoding — await before ANALYSIS
            # so the job doesn't report COMPLETED while facts are still writing.
            try:
                producer_label_by_song = await facts_task
            except Exception:
                logger.exception("[LibraryService] folder FACTS task failed (tracks indexed)")
            else:
                self._apply_producer_label(producer_label_by_song, track_ids, collection_name)

            # ── Stage ANALYSIS: Similarity analysis ───────────────────────────
            logger.info("[LibraryService] Stage ANALYSIS: similarity analysis")
            await self._notify_progress(job, {
                "stage": IndexStage.ANALYSIS.value,
                "stage_status": IndexStatus.RUNNING.value,
                "message": "Анализ схожих и разных треков...",
                "current": 0,
            })

            stage_analysis = job.stages[IndexStage.ANALYSIS]
            stage_analysis.status = IndexStatus.RUNNING
            stage_analysis.started_at = time.time()
            stage_analysis.total = 1
            stage_analysis.message = "Анализ схожих и разных треков..."

            try:
                if self.db_client and processed_files:
                    await analyze_collection(
                        qdrant_client=self.db_client.qdrant,
                        collection_name=collection_name,
                        progress_callback=lambda *a, **kw: self._on_analysis_progress(job, *a, **kw),
                    )
                    stage_analysis.status = IndexStatus.COMPLETED
                    stage_analysis.current = 1
                    stage_analysis.completed_at = time.time()
                    stage_analysis.message = "Анализ завершён"

                    await self._notify_progress(job, {
                        "stage": IndexStage.ANALYSIS.value,
                        "stage_status": IndexStatus.COMPLETED.value,
                        "current": 1,
                        "message": "Анализ завершён",
                    })
                else:
                    logger.warning("[LibraryService] Skipping analysis: db_client=%s", self.db_client is not None)
                    stage_analysis.status = IndexStatus.COMPLETED
            except Exception as e:
                logger.error("[LibraryService] Analysis failed: %s", e, exc_info=True)
                stage_analysis.status = IndexStatus.COMPLETED

            # Persist which text model this collection was indexed with, so that
            # future searches load the matching model automatically and don't
            # accidentally hit Qdrant with a vector_name that doesn't exist.

            # Mark job as completed
            job.overall_status = IndexStatus.COMPLETED
            logger.info("[LibraryService] Job %s COMPLETED", job.job_id)
            await self._notify_progress(job, {
                "overall_status": IndexStatus.COMPLETED.value,
                "message": f"Индексация завершена! {track_count} треков",
            })

        except Exception as e:
            logger.error("[LibraryService] Job %s FAILED: %s", job.job_id, e, exc_info=True)
            job.overall_status = IndexStatus.FAILED
            job.error_message = str(e)
            await self._notify_progress(job, {
                "overall_status": IndexStatus.FAILED.value,
                "error": str(e),
            })
        finally:
            _INDEX_SEMAPHORE.release()
            self.finish_job(account_id=account_id)
            logger.info(
                "[LibraryService] _run_indexing_job FINALLY, released semaphore + account=%s job=%s",
                account_id, job.job_id,
            )

    async def _notify_progress(self, job, data: dict):
        """Send progress update to all subscribers."""
        await job.notify_subscribers(data)

    async def _on_index_progress(self, job, stage: str, current: int, total: int, message: str):
        """Callback from SearchService for indexing progress.

        Maps internal stage keys ("scan" → LYRICS for the upload flow's
        metadata+lyrics pass, "lyrics" → DENSE, "audio" → AUDIO) and marks
        stages COMPLETED as soon as their encoding finishes. Phase B: the active
        ``job`` is passed in explicitly (was looked up via the now-removed global
        ``_current_job_id``) so concurrent per-account jobs don't cross wires.
        """
        if job is None:
            return

        # Map callback stage keys to IndexStage enum
        if stage == "scan":
            index_stage = IndexStage.LYRICS
        elif stage == "lyrics":
            index_stage = IndexStage.DENSE
        elif stage == "audio":
            index_stage = IndexStage.AUDIO
        else:
            return

        sp = job.stages[index_stage]
        # The upload flow has no explicit stage bootstrap (the folder flow sets
        # RUNNING/started_at before scanning) — promote on first event so the
        # wizard renders the stage as active and ETA math has a start time.
        if sp.status == IndexStatus.PENDING:
            sp.status = IndexStatus.RUNNING
            sp.started_at = sp.started_at or time.time()
        sp.current = current
        sp.total = total
        sp.message = message

        # Mark COMPLETED immediately when encoding finishes
        if current >= total and total > 0:
            sp.status = IndexStatus.COMPLETED
            sp.completed_at = time.time()

        eta = job.calculate_eta_seconds(index_stage)
        await self._notify_progress(job, {
            "stage": index_stage.value,
            "current": current,
            "total": total,
            "message": message,
            "eta_seconds": eta,
        })

    def _run_sonic_descriptor_hook(self, collection_name: str) -> None:
        """Per-track Sonic Descriptor pass over a freshly indexed collection.

        Scrolls the Qdrant collection once, pulling each point's ``audio`` (CLAP)
        vector and ``slug`` payload, then invokes
        :meth:`SonicDescriptorService.index_track_descriptor` per track. This emulates
        a "per-track post-upsert hook" given that per-track upserts happen inside
        ``LyricsDB._upsert_in_batches`` (not directly reachable from here).

        Runs in a worker thread (see ``asyncio.to_thread`` caller).
        """
        import numpy as np

        if self.sonic_descriptor_service is None or self.db_client is None:
            return

        qdrant = self.db_client.qdrant
        audio_vector_name = "clap"
        offset = None
        n_processed = 0
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=500,
                with_payload=["slug", "title", "artist"],
                with_vectors=[audio_vector_name],
            )
            if not points:
                break
            for p in points:
                vec = (p.vector or {}).get(audio_vector_name) if isinstance(p.vector, dict) else None
                if vec is None:
                    continue
                payload = p.payload or {}
                # Derive a stable song slug from artist+title (matches MetadataDB.ensure_song format).
                artist = (payload.get("artist") or "").strip()
                title = (payload.get("title") or "").strip()
                if not artist or not title:
                    continue
                song_slug = "-".join(artist.lower().split()) + "-" + "-".join(title.lower().split())
                try:
                    # Ensure songs row exists (idempotent); SongFacts may have skipped this track,
                    # in which case upsert_sonic_descriptor's UPDATE would silently no-op.
                    from app.resources.metadata_db import MetadataDB
                    MetadataDB.ensure_song(artist=artist, title=title, collection_name=collection_name)
                    self.sonic_descriptor_service.index_track_descriptor(
                        collection=collection_name,
                        slug=song_slug,
                        audio_vector=np.asarray(vec, dtype=np.float32),
                    )
                    n_processed += 1
                except Exception as e:
                    logger.warning(
                        "[LibraryService] sonic_descriptor hook failed for %s: %s",
                        song_slug, e,
                    )
            if next_offset is None:
                break
            offset = next_offset
        logger.info(
            "[LibraryService] sonic_descriptor hook processed %d tracks in %s",
            n_processed, collection_name,
        )

    async def _on_analysis_progress(self, job, stage: IndexStage, current: int, total: int, message: str):
        """Callback from SimilarityService for analysis progress.

        Phase B: ``job`` is bound explicitly by the caller (see the lambda in
        ``_run_indexing_job``) instead of resolved via the removed global
        ``_current_job_id``.
        """
        if job is None:
            return
        job.stages[IndexStage.ANALYSIS].current = current
        job.stages[IndexStage.ANALYSIS].total = total
        job.stages[IndexStage.ANALYSIS].message = message

        await self._notify_progress(job, {
            "stage": IndexStage.ANALYSIS.value,
            "current": current,
            "total": total,
            "message": message,
        })

    async def get_status(self, account_id: str = "default") -> dict:
        """Return current indexing status for ``account_id`` (Phase B)."""
        job_id = self.get_account_job_id(account_id)
        if not job_id:
            return {
                "indexing_in_progress": False,
                "current_job_id": None,
            }

        job = self._job_tracker.get_job(job_id)
        if not job:
            return {
                "indexing_in_progress": False,
                "current_job_id": None,
            }

        return self._job_tracker.get_progress_summary(job)

    @classmethod
    def get_albums(
        cls, *, qdrant_client, collection_name: str,
        sort: str = "alphabetical",
        label: Optional[str] = None,
    ) -> LibraryAlbumsResponse:
        """Group all tracks in the collection by album_title, derive primary
        artist via majority vote, return AlbumSummary list.

        ``label`` narrows the result to albums where at least one track
        carries that record label (matched via ``label_key`` normalization).

        Tries SQLite first (fast indexed query). Falls back to Qdrant scroll
        if SQLite tables are empty (pre-backfill deploy). The fallback carries
        no labels (the light payload projection omits them) — acceptable, as
        it only triggers when track_metadata is empty, i.e. no labels exist.
        """
        # Shared ordering for both the SQLite fast-path and the Qdrant fallback,
        # so the `sort` query param behaves identically regardless of source.
        def _year_for_sort(a: "AlbumSummary") -> int:
            if a.year:
                return a.year
            if a.year_range:
                try:
                    return int(a.year_range.split("—")[0])
                except (ValueError, IndexError):
                    return 0
            return 0

        def _apply_sort(albums: list) -> list:
            if sort == "year_desc":
                albums.sort(key=lambda a: -_year_for_sort(a))
            elif sort == "year_asc":
                albums.sort(key=lambda a: _year_for_sort(a) or 9999)
            elif sort == "track_count_desc":
                albums.sort(key=lambda a: -a.track_count)
            else:
                albums.sort(key=lambda a: a.album_title.lower())
            return albums

        def _apply_label_filter(albums: list) -> list:
            if not label:
                return albums
            wanted = label_key(label)
            return [a for a in albums if any(label_key(l) == wanted for l in a.labels)]

        # ── Fast path: read from SQLite ──
        try:
            sqlite_albums = MetadataDB.get_library_albums_from_sqlite(collection_name)
            if sqlite_albums:
                logger.info(
                    "[LibraryService] get_albums: %d albums from SQLite (%s)",
                    len(sqlite_albums), collection_name,
                )
                return LibraryAlbumsResponse(
                    albums=_apply_sort(_apply_label_filter([
                        _album_summary_from_sqlite(a) for a in sqlite_albums
                    ])),
                    collection_name=collection_name,
                    qdrant_available=True,
                )
        except Exception:
            logger.warning(
                "[LibraryService] SQLite album lookup failed, falling back to Qdrant",
                exc_info=True,
            )

        # ── Fallback: scroll Qdrant (pre-backfill or error) ──
        try:
            cols = qdrant_client.get_collections().collections
        except Exception:
            return LibraryAlbumsResponse(
                albums=[], collection_name=collection_name, qdrant_available=False,
            )
        if not any(c.name == collection_name for c in cols):
            return LibraryAlbumsResponse(
                albums=[], collection_name=collection_name, qdrant_available=True,
            )

        # Group tracks by album_title (case-insensitive key, preserves first-seen casing).
        from app.resources.qdrant_utils import light_points
        groups: dict[str, dict] = {}
        try:
            for tid, pl in light_points(qdrant_client, collection_name):
                album = (pl.get("album") or "").strip()
                if not album:
                    continue
                key = album.lower()
                g = groups.setdefault(key, {
                    "display_title": album,
                    "artist_counter": Counter(),
                    "year_counter": Counter(),
                    "genre_counter": Counter(),
                    "tracks": [],
                    "total_duration": 0.0,
                    "first_cover": None,
                })
                artist = (pl.get("artist") or "").strip()
                g["artist_counter"][artist] += 1
                yr = pl.get("year")
                try:
                    yi = int(yr) if yr is not None else None
                except (TypeError, ValueError):
                    yi = None
                if yi:
                    g["year_counter"][yi] += 1
                genre = (pl.get("genre") or "").strip()
                if genre:
                    g["genre_counter"][genre] += 1
                dur_raw = pl.get("duration") or 0
                try:
                    dur_float = float(dur_raw)
                except (TypeError, ValueError):
                    dur_float = None
                if dur_float is not None:
                    g["total_duration"] += dur_float
                if g["first_cover"] is None:
                    g["first_cover"] = pl.get("cover_art_path")
                g["tracks"].append(AlbumTrack(
                    track_id=tid,
                    title=pl.get("title") or "—",
                    artist=artist or "—",
                    duration=dur_float,
                    year=yi,
                    cover_art_path=pl.get("cover_art_path"),
                ))
        except Exception:
            logger.exception("get_albums: aggregation aborted on collection=%s", collection_name)

        albums = []
        for key, g in groups.items():
            artists = g["artist_counter"]
            primary_raw = sorted(
                artists.items(), key=lambda kv: (-kv[1], kv[0])
            )[0][0] if artists else "—"
            feat_raws = [
                a for a, _ in sorted(
                    [(a, c) for a, c in artists.items() if a != primary_raw],
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ]
            primary_name, primary_slug, feat_refs = _album_artist_credit(
                primary_raw, feat_raws,
            )
            year_range = None
            year = None
            if g["year_counter"]:
                ys = list(g["year_counter"].elements())
                ymin, ymax = min(ys), max(ys)
                if ymin == ymax:
                    year = ymin
                else:
                    year_range = f"{ymin}—{ymax}"
            albums.append(AlbumSummary(
                album_title=g["display_title"],
                primary_artist=primary_name,
                primary_artist_slug=primary_slug,
                feat_artists=feat_refs,
                year=year,
                year_range=year_range,
                cover_art_path=g["first_cover"],
                track_count=len(g["tracks"]),
                duration_seconds=int(g["total_duration"]),
                top_genres=[gn for gn, _ in g["genre_counter"].most_common(3)],
                tracks=g["tracks"],
            ))

        return LibraryAlbumsResponse(
            albums=_apply_sort(_apply_label_filter(albums)),
            collection_name=collection_name,
            qdrant_available=True,
        )

    @classmethod
    def get_liked_songs(
        cls, *, qdrant_client, collection_name: str,
    ):
        """Return all tracks the user has marked 'like' in this collection,
        ordered newest-liked first. Tracks whose Qdrant payload has been
        evicted (e.g. re-index churn) are silently skipped."""
        from app.domain.models import LikedSongTrack, LikedSongsResponse
        from app.services._payload_coerce import coerce_float, coerce_year
        pairs = MetadataDB.get_liked_track_ids_with_updated_at(collection_name)
        if not pairs:
            return LikedSongsResponse(tracks=[], collection_name=collection_name)

        ids = [tid for tid, _ in pairs]
        liked_at_by_id = {tid: ts for tid, ts in pairs}

        try:
            points = qdrant_client.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            points = []

        tracks = []
        for p in points:
            pl = p.payload or {}
            tracks.append(LikedSongTrack(
                track_id=str(p.id),
                title=pl.get("title") or "—",
                title_display=display_title_for_track(pl),
                artist=pl.get("artist") or "—",
                album=pl.get("album"),
                year=coerce_year(pl.get("year")),
                duration=coerce_float(pl.get("duration")),
                cover_art_path=pl.get("cover_art_path"),
                genre=pl.get("genre"),
                liked_at=liked_at_by_id.get(str(p.id), ""),
                artist_refs=artist_refs_for_track(pl),
            ))

        # Preserve like-order: re-sort by liked_at DESC (Qdrant.retrieve may not preserve)
        tracks.sort(key=lambda t: t.liked_at, reverse=True)
        return LikedSongsResponse(tracks=tracks, collection_name=collection_name)

    @classmethod
    def get_rediscover(cls, *, qdrant_client, collection_name: str):
        """Pick a long-unplayed track to resurface. Never-played tracks win;
        otherwise the oldest-played ones (random among the top-N gap)."""
        import random
        from app.domain.models import HomeTrack, RediscoverResponse
        from app.services._payload_coerce import coerce_float, coerce_year

        # Fast path: track ids from the SQLite mirror; fall back to an id-only
        # Qdrant scroll when the mirror is empty (pre-backfill).
        library_ids: list[str] = MetadataDB.get_track_ids_for_collection(collection_name)
        if not library_ids:
            offset = None
            while True:
                try:
                    points, offset = qdrant_client.scroll(
                        collection_name=collection_name, limit=256, offset=offset,
                        with_payload=False, with_vectors=False,
                    )
                except Exception:
                    break
                library_ids.extend(str(p.id) for p in points)
                if offset is None or not points:
                    break
        if not library_ids:
            return RediscoverResponse(collection_name=collection_name)

        recency = MetadataDB.get_play_recency_map(collection_name)
        never = [tid for tid in library_ids if tid not in recency]

        if never:
            chosen = random.choice(never)
            last_played, never_played = None, True
        else:
            ordered = sorted(library_ids, key=lambda t: recency.get(t, ""))
            pool = ordered[: min(20, len(ordered))]
            chosen = random.choice(pool)
            last_played, never_played = recency.get(chosen), False

        # Chosen track's card metadata: SQLite mirror first, Qdrant retrieve as
        # fallback (pre-backfill / mirror miss).
        row = MetadataDB.get_track_by_id(collection_name, chosen)
        if row is None:
            try:
                pts = qdrant_client.retrieve(
                    collection_name=collection_name, ids=[chosen],
                    with_payload=True, with_vectors=False,
                )
            except Exception:
                pts = []
            if not pts:
                return RediscoverResponse(collection_name=collection_name)
            row = pts[0].payload or {}
            row["track_id"] = str(pts[0].id)

        track = HomeTrack(
            track_id=str(row.get("track_id") or chosen),
            title=row.get("title") or "—",
            title_display=display_title_for_track(row),
            artist=row.get("artist") or "—",
            album=row.get("album"),
            year=coerce_year(row.get("year")),
            duration=coerce_float(row.get("duration")),
            cover_art_path=row.get("cover_art_path"),
            genre=row.get("genre"),
            artist_refs=artist_refs_for_track(row),
        )
        return RediscoverResponse(
            track=track, last_played=last_played, never_played=never_played,
            collection_name=collection_name,
        )

    @classmethod
    def get_listening_stats(
        cls, *, qdrant_client, collection_name: str, lang: str = "en",
        tz_offset_minutes: int = 0,
    ):
        """Aggregate listening summary for the Library overhaul UI.

        Combines:
          - Total seconds listened (sum of played_sec, all events).
          - Top track: most non-skipped plays; payload joined from Qdrant.
          - Top artist: per-artist sum of non-skipped plays via DB+payload join.
          - Peak hour: hour-of-day with most non-skipped plays, localised label.
            Bucketed in the user's local time via ``tz_offset_minutes``.
        """
        from app.domain.models import (
            ListeningStatsResponse, TopTrackBrief, TopArtistBrief, PeakHour,
        )
        total_sec, since = MetadataDB.get_listening_total(collection_name)
        top_track_row = MetadataDB.get_top_played_track(collection_name)
        peak_hour_int = MetadataDB.get_peak_hour(collection_name, tz_offset_minutes)

        top_track = None
        top_artist = None
        if top_track_row:
            tid, plays = top_track_row
            try:
                points = qdrant_client.retrieve(
                    collection_name=collection_name, ids=[tid],
                    with_payload=True, with_vectors=False,
                )
            except Exception:
                points = []
            if points:
                pl = points[0].payload or {}
                top_track = TopTrackBrief(
                    track_id=tid,
                    title=pl.get("title") or "—",
                    artist=pl.get("artist") or "—",
                    play_count=plays,
                    cover_art_path=pl.get("cover_art_path"),
                )

        # Top artist — aggregate non-skipped plays per artist by joining DB + payload
        counts_by_id = MetadataDB.get_play_counts_by_track(collection_name)
        if counts_by_id:
            try:
                points = qdrant_client.retrieve(
                    collection_name=collection_name,
                    ids=list(counts_by_id.keys()),
                    with_payload=["artist", "primary_artist_slug"],
                    with_vectors=False,
                )
            except Exception:
                points = []
            artist_counts: dict[str, int] = {}
            # Per artist, remember the slug of their single most-played track so
            # we can look up the artist's cached AudioDB photo for the avatar.
            artist_best_slug: dict[str, tuple[int, str | None]] = {}
            for p in points:
                pl = p.payload or {}
                a = pl.get("artist") or ""
                if not a:
                    continue
                c = counts_by_id.get(str(p.id), 0)
                artist_counts[a] = artist_counts.get(a, 0) + c
                best = artist_best_slug.get(a)
                if best is None or c > best[0]:
                    artist_best_slug[a] = (c, pl.get("primary_artist_slug"))
            if artist_counts:
                top_artist_name = max(artist_counts.items(), key=lambda kv: kv[1])[0]
                slug = (
                    artist_best_slug.get(top_artist_name, (0, None))[1]
                    or _slug_of_artist(top_artist_name)
                )
                ad = MetadataDB.get_artist_audiodb(slug, collection_name) or {}
                top_artist = TopArtistBrief(
                    name=top_artist_name,
                    slug=slug,
                    play_count=artist_counts[top_artist_name],
                    image=ad.get("thumb_path") or ad.get("cutout_path"),
                )

        peak_hour = None
        if peak_hour_int is not None:
            label = _label_peak_hour(peak_hour_int, lang)
            peak_hour = PeakHour(hour=peak_hour_int, label=label)

        return ListeningStatsResponse(
            total_seconds_listened=int(total_sec),
            since=since,
            top_track=top_track,
            top_artist=top_artist,
            peak_hour=peak_hour,
        )

    @classmethod
    def get_rhythm(
        cls, *, qdrant_client, collection_name: str, lang: str = "en",
        tz_offset_minutes: int = 0,
    ):
        """Listening rhythm for the stats tab: per-day calendar, 24h histogram,
        streaks and the busiest day. All bucketed in the user's local time."""
        from datetime import date as _date, datetime, timezone, timedelta
        from app.domain.models import (
            RhythmResponse, RhythmDay, BusiestDay, TopTrackBrief,
        )

        day_rows = MetadataDB.get_plays_by_local_day(collection_name, tz_offset_minutes)
        by_hour = MetadataDB.get_plays_by_local_hour(collection_name, tz_offset_minutes)
        days = [RhythmDay(date=d, count=n) for d, n in day_rows]
        counts_by_date = {d: n for d, n in day_rows}

        # Best streak: longest run of consecutive calendar days with ≥1 play.
        streak_best = 0
        if counts_by_date:
            parsed = sorted(_date.fromisoformat(d) for d in counts_by_date)
            run = 1 if parsed else 0
            streak_best = run
            for i in range(1, len(parsed)):
                run = run + 1 if (parsed[i] - parsed[i - 1]).days == 1 else 1
                streak_best = max(streak_best, run)

        # Current streak: consecutive days ending at the user's local today
        # (with a one-day grace, so "haven't played yet today" doesn't reset it).
        local_today = (
            datetime.now(timezone.utc) + timedelta(minutes=int(tz_offset_minutes))
        ).date()
        streak_current = 0
        anchor = local_today if local_today.isoformat() in counts_by_date else (
            local_today - timedelta(days=1)
        )
        probe = anchor
        while probe.isoformat() in counts_by_date:
            streak_current += 1
            probe = probe - timedelta(days=1)

        # Busiest day + its top track (payload joined from Qdrant, like top_track).
        busiest = None
        if day_rows:
            bd, bc = max(day_rows, key=lambda kv: kv[1])
            top_track = None
            top = MetadataDB.get_top_track_on_local_day(
                collection_name, bd, tz_offset_minutes,
            )
            if top:
                tid, plays = top
                try:
                    pts = qdrant_client.retrieve(
                        collection_name=collection_name, ids=[tid],
                        with_payload=["title", "artist"], with_vectors=False,
                    )
                except Exception:
                    pts = []
                if pts:
                    pl = pts[0].payload or {}
                    top_track = TopTrackBrief(
                        track_id=tid,
                        title=pl.get("title") or "—",
                        artist=pl.get("artist") or "—",
                        play_count=plays,
                    )
            busiest = BusiestDay(date=bd, count=bc, top_track=top_track)

        return RhythmResponse(
            days=days, by_hour=by_hour,
            streak_current=streak_current, streak_best=streak_best,
            busiest_day=busiest,
        )

    @classmethod
    def get_weekly_pulse(
        cls, *, collection_name: str, tz_offset_minutes: int = 0,
    ):
        """Light "this week" summary for the home page: seconds listened, top
        genre and discoveries (first-time-heard tracks) for the current
        calendar week (Monday..now). Accounts with little recent activity
        simply get sparse/zero values."""
        from app.domain.models import WeeklyPulseResponse

        seconds, top_genre, discoveries, daily = (
            MetadataDB.get_weekly_listening_summary(
                collection_name, tz_offset_minutes,
            )
        )
        return WeeklyPulseResponse(
            seconds_listened=int(seconds), top_genre=top_genre,
            discoveries=discoveries,
            daily_seconds=[int(s) for s in daily],
        )

    @classmethod
    def get_engagement(
        cls, *, qdrant_client, collection_name: str, lang: str = "en",
    ):
        """Honest-mirror engagement: overall completion + the tracks you
        replay-and-finish ("loved") vs launch-often-but-skip ("guilty").

        "Loved" = the top 5 tracks you most often play through to the end,
        ranked by finish count (a track finished only once doesn't qualify).
        "Guilty" = the top 5 you bail on fastest: ranked by how few seconds you
        typically hear before skipping, then by how often. A track must have
        been skipped early at least 3 times, and only counts as a quick bail if
        you usually drop it inside 10 seconds."""
        from app.domain.models import EngagementResponse, EngagementTrack

        # rows: (track_id, plays, comp, finish_count, skip_count, avg_skip_sec)
        rows = MetadataDB.get_engagement_detail_stats(collection_name)
        overall = MetadataDB.get_overall_completion(collection_name)

        loved = sorted(
            [r for r in rows if r[3] >= 2],
            key=lambda r: (r[3], r[2] or 0.0), reverse=True,
        )[:5]
        guilty = sorted(
            [r for r in rows if r[4] >= 3 and r[5] is not None and r[5] < 10.0],
            key=lambda r: (r[5], -r[4]),
        )[:5]

        # Join display fields from Qdrant (no lyrics — light payload only).
        want = {r[0] for r in loved} | {r[0] for r in guilty}
        payloads: dict = {}
        if want:
            try:
                pts = qdrant_client.retrieve(
                    collection_name=collection_name, ids=list(want),
                    with_payload=["title", "artist", "cover_art_path"],
                    with_vectors=False,
                )
            except Exception:
                pts = []
            for p in pts:
                payloads[str(p.id)] = p.payload or {}

        def _mk(r):
            pl = payloads.get(r[0], {})
            return EngagementTrack(
                track_id=r[0],
                title=pl.get("title") or "—",
                artist=pl.get("artist") or "—",
                cover_art_path=pl.get("cover_art_path"),
                completion=round(r[2], 3) if r[2] is not None else 0.0,
                plays=r[1],
                finish_count=r[3],
                skip_count=r[4],
                skip_seconds=round(r[5], 1) if r[5] is not None else None,
            )

        return EngagementResponse(
            overall_completion=round(overall, 3) if overall is not None else 0.0,
            loved=[_mk(r) for r in loved],
            guilty=[_mk(r) for r in guilty],
        )

    # ── Helpers ──

    # def _enrich_with_musicbrainz(
    #     self,
    #     processed_files: dict,
    #     enhance: bool,
    #     progress_callback=None,
    # ):
    #     """Enrich track metadata via MusicBrainz API.

    #     Args:
    #         processed_files: dict keyed by "artist — title", values are metadata dicts.
    #         enhance: if True, MB data replaces local; if False, MB is fallback only.
    #         progress_callback: optional callable(current, total, label, found, not_found).
    #     """
    #     from musicbraniz_search import MusicBrainzLookup

    #     mb = MusicBrainzLookup("MusiX", "1.0", "https://musix.local")
    #     total = len(processed_files)
    #     enriched = 0
    #     not_enriched = 0
    #     idx = 0

    #     for key, info in processed_files.items():
    #         title = info.get("title", "").strip()
    #         artist = info.get("artist", "").strip()
    #         idx += 1
    #         if not title or not artist:
    #             not_enriched += 1
    #             logger.info(
    #                 "[LibraryService] MB enrichment %d/%d: ✗ %s — %s (missing title/artist)",
    #                 idx, total, artist or "?", title or "?",
    #             )
    #             if progress_callback:
    #                 progress_callback(idx, total, f"{artist} — {title}", enriched, not_enriched)
    #             continue

    #         try:
    #             rec_id = mb.resolve_recording_id(title, artist)
    #             if not rec_id:
    #                 not_enriched += 1
    #                 logger.info(
    #                     "[LibraryService] MB enrichment %d/%d: ✗ %s — %s (no MB match)",
    #                     idx, total, artist, title,
    #                 )
    #                 if progress_callback:
    #                     progress_callback(idx, total, f"{artist} — {title}", enriched, not_enriched)
    #                 continue

    #             # Year — merge with existing based on flag
    #             mb_year = mb.get_recording_year(rec_id)
    #             local_year = info.get("year")
    #             if mb_year:
    #                 info["year"] = mb_year if enhance else (local_year or mb_year)

    #             # Producer
    #             producers = mb.get_recording_producers(rec_id)
    #             if producers:
    #                 info["producer"] = ", ".join(producers) if enhance or not info.get("producer") else info["producer"]

    #             # Labels
    #             labels = mb.get_recording_labels(rec_id)
    #             if labels:
    #                 label_name = labels[0].get("name", "")
    #                 if label_name:
    #                     info["label"] = label_name if enhance else (info.get("label") or label_name)

    #             # Samples
    #             samples = mb.get_recording_samples(rec_id)
    #             if samples:
    #                 sample_strs = [
    #                     f"{s.get('artist', '?')} — {s.get('title', '?')}"
    #                     for s in samples if s.get("title") or s.get("artist")
    #                 ]
    #                 info["samples"] = sample_strs if enhance or not info.get("samples") else info["samples"]

    #             sampled_by = mb.get_recording_sampled_by(rec_id)
    #             if sampled_by:
    #                 sampled_strs = [
    #                     f"{s.get('artist', '?')} — {s.get('title', '?')}"
    #                     for s in sampled_by if s.get("title") or s.get("artist")
    #                 ]
    #                 info["sampled_by"] = sampled_strs if enhance or not info.get("sampled_by") else info["sampled_by"]

    #             enriched += 1
    #             logger.info(
    #                 "[LibraryService] MB enrichment %d/%d: ✓ %s — %s (enriched=%d, not=%d)",
    #                 idx, total, artist, title, enriched, not_enriched,
    #             )
    #         except Exception as e:
    #             logger.warning(
    #                 "[LibraryService] MB enrichment %d/%d: ✗ %s — %s (error: %s — %s)",
    #                 idx, total, artist, title, type(e).__name__, e,
    #             )
    #             not_enriched += 1

    #         if progress_callback:
    #             progress_callback(idx, total, f"{artist} — {title}", enriched, not_enriched)

    def _tagread_folder(self, audio_files, *, enrich_client=None) -> dict:
        """Tag-read every file (NO online lyrics) + extract covers → key→meta dict.

        The folder-flow counterpart of the upload flow's ``_tagread_upload_rows``:
        defers the online lyrics fetch to IndexPipeline, reads embedded text here
        (so it skips the network), and extracts covers (the indexing payload needs
        them). Returns ``"Artist — Title" -> meta`` with a numeric ``duration``
        (default 0) so ``prepare_metadata`` keeps the track.
        Runs in a worker thread (``asyncio.to_thread`` caller); fans file reads out
        across an inner pool since each is I/O-bound (mutagen + m4a optimize).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from app.indexing.cover_art import save_cover_art
        from app.indexing.folder_scanner import read_tags_only

        def _read(fp):
            info = read_tags_only(fp, enrich_client=enrich_client)
            if not info:
                return None
            try:
                info["duration"] = int(info.get("duration") or 0)
            except (TypeError, ValueError):
                info["duration"] = 0
            track_id = self._compute_track_id(info.get("file_path") or "")
            if info.get("file_path") and track_id:
                try:
                    info["cover_art_path"] = save_cover_art(
                        info["file_path"], track_id,
                        meta=info, yandex_client=enrich_client,
                    )
                except Exception as e:
                    logger.warning("[LibraryService] cover extraction failed for %s: %s",
                                   info["file_path"], e)
                    info["cover_art_path"] = None
            return info

        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_read, fp): fp for fp in audio_files}
            for fut in as_completed(futs):
                try:
                    info = fut.result()
                except Exception:
                    logger.exception("[LibraryService] tag-read failed for %s", futs[fut])
                    info = None
                if info:
                    results.setdefault(f"{info['artist']} — {info['title']}", info)
        return results

    @staticmethod
    def _compute_track_id(file_path: str) -> str:
        """Stable ID from file path (SHA256, first 16 chars)."""
        if file_path:
            return hashlib.sha256(file_path.encode()).hexdigest()[:16]
        else:
            logger.warning("[LibraryService] Error while resolving file path")
            return None

    @classmethod
    def list_distinct_artist_slugs(cls, *, qdrant_client, collection_name: str):
        """Return a deterministically-sorted list of (slug, name) for every
        distinct artist in the collection (slug-deduped).

        Tries SQLite first (fast indexed query). Falls back to Qdrant scroll
        if SQLite tables are empty (pre-backfill deploy).
        """
        # ── Fast path: read from SQLite ──
        try:
            sqlite_result = MetadataDB.get_distinct_artist_slugs_from_sqlite(collection_name)
            if sqlite_result:
                logger.info(
                    "[LibraryService] list_distinct_artist_slugs: %d artists from SQLite (%s)",
                    len(sqlite_result), collection_name,
                )
                return [(a["slug"], a["name"]) for a in sqlite_result]
        except Exception:
            logger.warning(
                "[LibraryService] SQLite artist slug lookup failed, falling back to Qdrant",
                exc_info=True,
            )

        # ── Fallback: scroll Qdrant (pre-backfill or error) ──
        from app.services.artist_split import artist_slugs, split_artists
        from app.resources.qdrant_utils import light_points
        seen: dict[str, str] = {}  # slug -> name (first seen)
        for _tid, payload in light_points(qdrant_client, collection_name):
            name = (payload.get("artist") or "").strip()
            if not name:
                continue
            slugs = payload.get("artist_slugs")
            names = payload.get("artists")
            if not slugs or not names:
                slugs = artist_slugs(name)
                names = split_artists(name)
            for i, slug in enumerate(slugs):
                display = names[i] if i < len(names) else slug
                seen.setdefault(slug, display)
        return sorted(seen.items())
