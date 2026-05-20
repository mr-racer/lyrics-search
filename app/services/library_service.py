"""Library service — indexing tracks from a folder."""

import asyncio
import hashlib
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..domain.models import (
    AlbumSummary,
    AlbumTrack,
    ArtistRef,
    LibraryAlbumsResponse,
    TrackMetadata,
)
from ..resources.metadata_db import MetadataDB
from ..resources.model_registry import ModelRegistry
from ..resources.db_client import DbClient
from ..existing.folder_processor import FileProcessor
from .artist_facts_service import fetch_facts_for_artists
from .job_tracker import JobTracker, IndexStage, IndexStatus
from .similarity_service import analyze_collection
from .song_facts_service import fetch_facts_for_songs
from .sonic_descriptor_service import SonicDescriptorService
from ._WIP_musicbraniz_search import MusicBrainzLookup

logger = logging.getLogger(__name__)


def _slugify_artist_for_album(name: str) -> str:
    """Same slug logic frontend uses (lowercase, non-alnum → '-', strip)."""
    return re.sub(r'[^a-z0-9]+', '-', (name or '').lower()).strip('-')


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
        self._current_job_id = None

    async def index_folder(
        self,
        folder_path: str,
        collection_name: str = "music_explorer",
        better_lyrics_quality: bool = False,
        text_model: Optional[str] = None,
        enhance_by_musicbrainz: bool = False,
    ) -> dict:
        """
        Index all audio files in folder with progress tracking.
        Returns dict with job_id for tracking progress via SSE.
        """
        logger.info("[LibraryService] index_folder called: folder=%s, collection=%s, text_model=%s",
                    folder_path, collection_name, text_model)

        # Check if indexing is already in progress
        if self._current_job_id:
            current_job = self._job_tracker.get_job(self._current_job_id)
            if current_job and current_job.overall_status == IndexStatus.RUNNING:
                logger.warning("[LibraryService] Indexing already in progress, rejecting new request")
                return {
                    "status": "failed",
                    "message": "Indexing already in progress",
                    "job_id": self._current_job_id,
                }

        # Create new job
        job = self._job_tracker.create_job(folder_path, collection_name)
        self._current_job_id = job.job_id
        job.overall_status = IndexStatus.RUNNING
        logger.info("[LibraryService] Job created: %s", job.job_id)

        # Start indexing in background task to allow SSE progress updates
        task = asyncio.create_task(
            self._run_indexing_job(job, better_lyrics_quality, text_model, enhance_by_musicbrainz)
        )
        logger.info("[LibraryService] Background task created: %s", task)

        return {
            "status": "started",
            "job_id": job.job_id,
            "message": "Indexing started",
        }

    async def _run_indexing_job(
        self, job, better_lyrics_quality: bool, text_model: Optional[str] = None,
        enhance_by_musicbrainz: bool = False,
    ):
        """Execute the indexing process with progress updates."""
        folder_path = job.folder_path
        collection_name = job.collection_name
        logger.info("[LibraryService] _run_indexing_job START: job=%s, folder=%s, collection=%s",
                    job.job_id, folder_path, collection_name)

        try:
            # Sanitize text_model: treat literal strings "null"/"undefined"/"" the
            # same as None. This guards against legacy frontend localStorage where
            # `localStorage.setItem('text_model', null)` stringified the value.
            if text_model in (None, "", "null", "undefined", "None"):
                text_model = None

            # Use already-loaded text model (don't reload during indexing).
            if text_model:
                logger.info("[LibraryService] Getting text model: %s (cached if already loaded)", text_model)
                ModelRegistry.load_text_model(text_model)
            else:
                logger.info("[LibraryService] No text_model specified, using default")

            loop = asyncio.get_event_loop()

            # ── Stage LYRICS: scan files, read tags, fetch lyrics ─────────────
            logger.info("[LibraryService] Stage LYRICS: scanning and fetching lyrics")
            stage_lyrics = job.stages[IndexStage.LYRICS]
            stage_lyrics.status = IndexStatus.RUNNING
            stage_lyrics.started_at = time.time()
            stage_lyrics.message = "Поиск текстов песен..."

            await self._notify_progress(job, {
                "stage": IndexStage.LYRICS.value,
                "stage_status": IndexStatus.RUNNING.value,
                "message": stage_lyrics.message,
                "current": 0,
            })

            # Scan for total count
            audio_files = [
                p for p in Path(folder_path).rglob("*")
                if p.suffix.lower() in (".flac", ".m4a", ".mp3")
            ]
            stage_lyrics.total = len(audio_files)

            await self._notify_progress(job, {
                "stage": IndexStage.LYRICS.value,
                "total": len(audio_files),
                "message": f"Найдено {len(audio_files)} файлов",
            })

            logger.info("[LibraryService] Starting FileProcessor...")
            processor = FileProcessor()

            async def on_lyrics_progress(current: int, total: int, message: str, details: dict = None):
                stage_lyrics.current = current
                stage_lyrics.total = total
                stage_lyrics.message = message
                if details:
                    if details.get("found") is not None:
                        stage_lyrics.found = details["found"]
                    if details.get("not_found") is not None:
                        stage_lyrics.not_found = details["not_found"]
                eta = details.get("eta_seconds") or job.calculate_eta_seconds(IndexStage.LYRICS)
                notify_data = {
                    "stage": IndexStage.LYRICS.value,
                    "current": current,
                    "total": total,
                    "message": message,
                    "eta_seconds": eta,
                }
                await self._notify_progress(job, notify_data)

            processed_files = await asyncio.to_thread(
                processor.process_folder,
                music_folder=folder_path,
                better_lyrics_quality=better_lyrics_quality,
                progress_callback=lambda c, t, m, d=None: asyncio.run_coroutine_threadsafe(
                    on_lyrics_progress(c, t, m, d), loop
                ),
            )

            track_count = len(processed_files)
            logger.info("[LibraryService] FileProcessor done, processed %d tracks", track_count)
            stage_lyrics.status = IndexStatus.COMPLETED
            stage_lyrics.current = track_count
            stage_lyrics.completed_at = time.time()
            stage_lyrics.message = f"Обработано {track_count} треков"

            await self._notify_progress(job, {
                "stage": IndexStage.LYRICS.value,
                "stage_status": IndexStatus.COMPLETED.value,
                "current": stage_lyrics.total,
                "message": stage_lyrics.message,
            })

            # ── Stage FACTS: SongFacts (artists + songs) ──────────────────────
            logger.info("[LibraryService] Stage FACTS: fetching song facts")
            unique_artists = sorted({
                info.get("artist", "").strip()
                for info in processed_files.values()
                if info.get("artist", "").strip()
            })
            unique_songs = sorted({
                (info.get("artist", "").strip(), info.get("title", "").strip())
                for info in processed_files.values()
                if info.get("artist", "").strip() and info.get("title", "").strip()
            })
            # Each unique artist contributes TWO units of work: songfacts + audiodb.
            facts_total = len(unique_artists) * 2 + len(unique_songs)

            stage_facts = job.stages[IndexStage.FACTS]
            stage_facts.status = IndexStatus.RUNNING
            stage_facts.started_at = time.time()
            stage_facts.total = facts_total
            stage_facts.message = "Поиск фактов..."

            await self._notify_progress(job, {
                "stage": IndexStage.FACTS.value,
                "stage_status": IndexStatus.RUNNING.value,
                "message": stage_facts.message,
                "current": 0,
                "total": facts_total,
            })

            # Check if all facts are cached — skip if so
            facts_all_cached = True
            for artist in unique_artists:
                from .artist_facts_service import get_cached_facts
                if not get_cached_facts(collection_name, artist):
                    facts_all_cached = False
                    break
            if facts_all_cached:
                for _, song in unique_songs:
                    from .song_facts_service import get_cached_song_facts
                    if not get_cached_song_facts(collection_name, _, song):
                        facts_all_cached = False
                        break

            # Shared state across facts branches + the unconditional audiodb step.
            # Declared here so the audiodb block (which runs regardless of the
            # facts cache short-circuit) can update progress + record errors.
            facts_progress = {"artists": 0, "songs": 0, "audiodb": 0}
            facts_found = {"artists": 0, "songs": 0, "audiodb": 0}
            facts_state = {"error": None}
            artist_facts_result: Dict[str, str] = {}
            song_facts_result: Dict[str, str] = {}
            audiodb_result: Dict[str, Any] = {}

            def on_audiodb_progress(current: int, total: int, label: str, found: bool):
                facts_progress["audiodb"] = current
                if found:
                    facts_found["audiodb"] += 1
                combined = sum(facts_progress.values())
                stage_facts.current = combined
                stage_facts.found = sum(facts_found.values())
                stage_facts.not_found = max(0, facts_total - stage_facts.found)
                stage_facts.message = f"AudioDB: {label}"
                eta = job.calculate_eta_seconds(IndexStage.FACTS)
                asyncio.create_task(self._notify_progress(job, {
                    "stage": IndexStage.FACTS.value,
                    "current": combined,
                    "total": facts_total,
                    "message": stage_facts.message,
                    "found": stage_facts.found,
                    "not_found": stage_facts.not_found,
                    "eta_seconds": eta,
                }))

            if facts_all_cached and facts_total > 0:
                logger.info("[LibraryService] All facts cached, skipping FACTS fetches")
                # Stage stays RUNNING; audiodb still runs unconditionally below.
                stage_facts.message = "Факты из кеша"
                await self._notify_progress(job, {
                    "stage": IndexStage.FACTS.value,
                    "current": 0,
                    "total": facts_total,
                    "message": stage_facts.message,
                })
            elif facts_total == 0:
                # No artists/songs at all — nothing for audiodb either.
                stage_facts.status = IndexStatus.COMPLETED
                stage_facts.message = "Нет данных"

                await self._notify_progress(job, {
                    "stage": IndexStage.FACTS.value,
                    "stage_status": IndexStatus.COMPLETED.value,
                    "message": stage_facts.message,
                })
            else:
                # Launch facts fetches with progress callbacks
                def on_artist_facts_progress(current: int, total: int, label: str, found: bool):
                    facts_progress["artists"] = current
                    if found:
                        facts_found["artists"] += 1
                    combined = sum(facts_progress.values())
                    stage_facts.current = combined
                    stage_facts.found = sum(facts_found.values())
                    stage_facts.not_found = max(0, facts_total - stage_facts.found)
                    stage_facts.message = f"Факты: {label}"
                    eta = job.calculate_eta_seconds(IndexStage.FACTS)
                    asyncio.create_task(self._notify_progress(job, {
                        "stage": IndexStage.FACTS.value,
                        "current": combined,
                        "total": facts_total,
                        "message": stage_facts.message,
                        "found": stage_facts.found,
                        "not_found": stage_facts.not_found,
                        "eta_seconds": eta,
                    }))

                def on_song_facts_progress(current: int, total: int, label: str, found: bool):
                    facts_progress["songs"] = current
                    if found:
                        facts_found["songs"] += 1
                    combined = sum(facts_progress.values())
                    stage_facts.current = combined
                    stage_facts.found = sum(facts_found.values())
                    stage_facts.not_found = max(0, facts_total - stage_facts.found)
                    stage_facts.message = f"Факты: {label}"
                    eta = job.calculate_eta_seconds(IndexStage.FACTS)
                    asyncio.create_task(self._notify_progress(job, {
                        "stage": IndexStage.FACTS.value,
                        "current": combined,
                        "total": facts_total,
                        "message": stage_facts.message,
                        "found": stage_facts.found,
                        "not_found": stage_facts.not_found,
                        "eta_seconds": eta,
                    }))

                try:
                    facts_task = asyncio.create_task(
                        fetch_facts_for_artists(
                            unique_artists, collection_name,
                            progress_callback=on_artist_facts_progress,
                        ),
                        name="artist-facts",
                    )
                    song_facts_task = asyncio.create_task(
                        fetch_facts_for_songs(
                            unique_songs, collection_name,
                            progress_callback=on_song_facts_progress,
                        ),
                        name="song-facts",
                    )
                    logger.info("[LibraryService] Launched facts fetch for %d artists, %d songs",
                                len(unique_artists), len(unique_songs))

                    # Await facts (runs parallel to encoding, cached to disk)
                    try:
                        artist_facts_result = await asyncio.wait_for(facts_task, timeout=60)
                        logger.info("[LibraryService] Artist facts fetched: %d found", len(artist_facts_result))
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning("[LibraryService] Artist facts fetch timed out or failed: %s", e)
                        facts_task.cancel()
                        if not facts_state["error"]:
                            facts_state["error"] = str(e)

                    try:
                        song_facts_result = await asyncio.wait_for(song_facts_task, timeout=120)
                        logger.info("[LibraryService] Song facts fetched: %d found", len(song_facts_result))
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning("[LibraryService] Song facts fetch timed out or failed: %s", e)
                        song_facts_task.cancel()
                        if not facts_state["error"]:
                            facts_state["error"] = str(e)
                except Exception as e:
                    logger.warning("[LibraryService] Facts stage failed (non-critical): %s", e)
                    facts_state["error"] = str(e)

            # AudioDB enrichment runs independently of the facts cache check —
            # idempotent per-artist via audiodb_fetched_at, so re-running on a
            # fully-cached collection just confirms no new work is needed.
            if unique_artists:
                from app.services.audiodb_service import fetch_audiodb_for_artists
                audiodb_task = asyncio.create_task(
                    fetch_audiodb_for_artists(
                        unique_artists, collection_name,
                        progress_callback=on_audiodb_progress,
                    ),
                    name="audiodb-enrich",
                )
                try:
                    audiodb_result = await asyncio.wait_for(audiodb_task, timeout=300)
                    logger.info("[LibraryService] AudioDB enrichment fetched: %d artists",
                                sum(1 for v in (audiodb_result or {}).values() if v))
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning("[LibraryService] AudioDB enrichment timed out or failed: %s", e)
                    audiodb_task.cancel()
                    audiodb_result = {}
                    if not facts_state["error"]:
                        facts_state["error"] = str(e)

            # Emit a single COMPLETED notify covering facts + audiodb (unless
            # facts_total == 0, which already marked the stage complete above).
            if facts_total > 0:
                audiodb_found_count = sum(1 for v in (audiodb_result or {}).values() if v)
                facts_found_total = (
                    len(artist_facts_result) + len(song_facts_result) + audiodb_found_count
                )
                facts_not_found = max(0, facts_total - facts_found_total)

                stage_facts.status = IndexStatus.COMPLETED
                stage_facts.current = facts_total
                stage_facts.completed_at = time.time()
                stage_facts.found = facts_found_total
                stage_facts.not_found = facts_not_found
                if facts_state["error"]:
                    stage_facts.message = f"Частично: {facts_state['error']}"
                elif facts_all_cached and audiodb_found_count == 0:
                    stage_facts.message = "Факты из кеша"
                else:
                    stage_facts.message = (
                        f"Факты: {facts_found_total} найдено из {facts_total}"
                    )

                await self._notify_progress(job, {
                    "stage": IndexStage.FACTS.value,
                    "stage_status": IndexStatus.COMPLETED.value,
                    "current": stage_facts.current,
                    "total": facts_total,
                    "message": stage_facts.message,
                    "found": facts_found_total,
                    "not_found": facts_not_found,
                    "stage_error": facts_state["error"],
                })

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

            # ── Convert to TrackMetadata ──────────────────────────────────────
            logger.info("[LibraryService] Converting to TrackMetadata...")
            tracks = self._metadata_to_tracks(processed_files)
            logger.info("[LibraryService] Converted %d tracks to TrackMetadata", len(tracks))

            # ── Stage DENSE + AUDIO: vector encoding ──────────────────────────
            logger.info("[LibraryService] Stage DENSE/AUDIO: vector encoding")

            stage_dense = job.stages[IndexStage.DENSE]
            stage_dense.status = IndexStatus.RUNNING
            stage_dense.started_at = time.time()
            stage_dense.total = len(tracks)
            stage_dense.message = "Кодирование текстов (dense)..."

            stage_audio = job.stages[IndexStage.AUDIO]
            stage_audio.status = IndexStatus.RUNNING
            stage_audio.started_at = time.time()
            stage_audio.total = len(tracks)
            stage_audio.message = "CLAP-кодирование аудио..."

            await self._notify_progress(job, {
                "stage": IndexStage.DENSE.value,
                "stage_status": IndexStatus.RUNNING.value,
                "message": stage_dense.message,
                "current": 0,
                "total": len(tracks),
            })
            await self._notify_progress(job, {
                "stage": IndexStage.AUDIO.value,
                "stage_status": IndexStatus.RUNNING.value,
                "message": stage_audio.message,
                "current": 0,
                "total": len(tracks),
            })

            if self.search_service and tracks:
                logger.info("[LibraryService] Starting SearchService.index_tracks_with_progress...")
                await self.search_service.index_tracks_with_progress(
                    tracks,
                    collection_name=collection_name,
                    progress_callback=self._on_index_progress,
                    text_model=text_model,
                )
                logger.info("[LibraryService] SearchService.index_tracks_with_progress done")

                # ── Sonic Descriptor hook: per-track tags + class for freshly indexed tracks.
                # Scrolls the collection once and invokes ``index_track_descriptor`` per point.
                # Wrapped in try/except so indexing never fails if Sonic Descriptor has a bug.
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
                logger.warning("[LibraryService] Skipping indexing: search_service=%s, tracks=%d",
                              self.search_service is not None, len(tracks))
                # Mark both as completed if skipped
                for stage in (IndexStage.DENSE, IndexStage.AUDIO):
                    sp = job.stages[stage]
                    sp.status = IndexStatus.COMPLETED
                    sp.completed_at = time.time()

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
                if self.db_client and tracks:
                    await analyze_collection(
                        qdrant_client=self.db_client.qdrant,
                        collection_name=collection_name,
                        progress_callback=self._on_analysis_progress,
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
            try:
                from app.resources.metadata_db import MetadataDB
                MetadataDB.init()
                MetadataDB.set_collection_text_model(collection_name, text_model)
                logger.info("[LibraryService] persisted collection_settings: %s → text_model=%s",
                            collection_name, text_model or "(default)")
            except Exception as e:
                logger.warning("[LibraryService] failed to persist collection_settings: %s", e)

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
            self._current_job_id = None
            logger.info("[LibraryService] _run_indexing_job FINALLY, cleared current_job_id")

    async def _notify_progress(self, job, data: dict):
        """Send progress update to all subscribers."""
        await job.notify_subscribers(data)

    async def _on_index_progress(self, stage: str, current: int, total: int, message: str):
        """Callback from SearchService for indexing progress.

        Maps internal stage keys ("lyrics" → DENSE, "audio" → AUDIO) and marks
        stages COMPLETED as soon as their encoding finishes.
        """
        if self._current_job_id:
            job = self._job_tracker.get_job(self._current_job_id)
            if not job:
                return

            # Map callback stage keys to IndexStage enum
            if stage == "lyrics":
                index_stage = IndexStage.DENSE
            elif stage == "audio":
                index_stage = IndexStage.AUDIO
            else:
                return

            sp = job.stages[index_stage]
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

    async def _on_analysis_progress(self, stage: IndexStage, current: int, total: int, message: str):
        """Callback from SimilarityService for analysis progress."""
        if self._current_job_id:
            job = self._job_tracker.get_job(self._current_job_id)
            if job:
                job.stages[IndexStage.ANALYSIS].current = current
                job.stages[IndexStage.ANALYSIS].total = total
                job.stages[IndexStage.ANALYSIS].message = message

                await self._notify_progress(job, {
                    "stage": IndexStage.ANALYSIS.value,
                    "current": current,
                    "total": total,
                    "message": message,
                })

    async def get_status(self) -> dict:
        """Return current indexing status."""
        if not self._current_job_id:
            return {
                "indexing_in_progress": False,
                "current_job_id": None,
            }
        
        job = self._job_tracker.get_job(self._current_job_id)
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
    ) -> LibraryAlbumsResponse:
        """Group all tracks in the collection by album_title, derive primary
        artist via majority vote, return AlbumSummary list."""
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

        # Group tracks by album_title (case-insensitive key, preserves first-seen casing)
        groups: dict[str, dict] = {}
        offset = None
        try:
            while True:
                results, next_offset = qdrant_client.scroll(
                    collection_name=collection_name,
                    offset=offset,
                    limit=250,
                    with_payload=[
                        "album", "artist", "title", "year", "duration",
                        "genre", "cover_art_path",
                    ],
                    with_vectors=False,
                )
                for pt in results:
                    pl = pt.payload or {}
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
                    dur = pl.get("duration") or 0
                    try:
                        g["total_duration"] += float(dur)
                    except (TypeError, ValueError):
                        pass
                    if g["first_cover"] is None:
                        g["first_cover"] = pl.get("cover_art_path")
                    g["tracks"].append(AlbumTrack(
                        track_id=str(pt.id) if hasattr(pt, "id") else "",
                        title=pl.get("title") or "—",
                        artist=artist or "—",
                        duration=dur or None,
                        year=yi,
                        cover_art_path=pl.get("cover_art_path"),
                    ))
                if next_offset is None or not results:
                    break
                offset = next_offset
        except Exception:
            pass  # partial result is acceptable

        albums = []
        for key, g in groups.items():
            artists = g["artist_counter"]
            primary_artist = sorted(
                artists.items(), key=lambda kv: (-kv[1], kv[0])
            )[0][0] if artists else "—"
            feat = sorted(
                [(a, c) for a, c in artists.items() if a != primary_artist],
                key=lambda kv: (-kv[1], kv[0])
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
                primary_artist=primary_artist,
                primary_artist_slug=_slugify_artist_for_album(primary_artist),
                feat_artists=[
                    ArtistRef(name=a, slug=_slugify_artist_for_album(a))
                    for a, _ in feat
                ],
                year=year,
                year_range=year_range,
                cover_art_path=g["first_cover"],
                track_count=len(g["tracks"]),
                duration_seconds=int(g["total_duration"]),
                top_genres=[gn for gn, _ in g["genre_counter"].most_common(3)],
                tracks=g["tracks"],
            ))

        # Sort
        def _year_for_sort(a):
            if a.year:
                return a.year
            if a.year_range:
                try:
                    return int(a.year_range.split("—")[0])
                except (ValueError, IndexError):
                    return 0
            return 0

        if sort == "year_desc":
            albums.sort(key=lambda a: -_year_for_sort(a))
        elif sort == "year_asc":
            albums.sort(key=lambda a: _year_for_sort(a) or 9999)
        elif sort == "track_count_desc":
            albums.sort(key=lambda a: -a.track_count)
        else:
            albums.sort(key=lambda a: a.album_title.lower())

        return LibraryAlbumsResponse(
            albums=albums, collection_name=collection_name, qdrant_available=True,
        )

    @classmethod
    def get_liked_songs(
        cls, *, qdrant_client, collection_name: str,
    ):
        """Return all tracks the user has marked 'like' in this collection,
        ordered newest-liked first. Tracks whose Qdrant payload has been
        evicted (e.g. re-index churn) are silently skipped."""
        from app.domain.models import LikedSongTrack, LikedSongsResponse
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
                artist=pl.get("artist") or "—",
                album=pl.get("album"),
                year=int(pl["year"]) if pl.get("year") and str(pl["year"]).isdigit() else None,
                duration=pl.get("duration"),
                cover_art_path=pl.get("cover_art_path"),
                genre=pl.get("genre"),
                liked_at=liked_at_by_id.get(str(p.id), ""),
            ))

        # Preserve like-order: re-sort by liked_at DESC (Qdrant.retrieve may not preserve)
        tracks.sort(key=lambda t: t.liked_at, reverse=True)
        return LikedSongsResponse(tracks=tracks, collection_name=collection_name)

    @classmethod
    def get_listening_stats(
        cls, *, qdrant_client, collection_name: str, lang: str = "en",
    ):
        """Aggregate listening summary for the Library overhaul UI.

        Combines:
          - Total seconds listened (sum of played_sec, all events).
          - Top track: most non-skipped plays; payload joined from Qdrant.
          - Top artist: per-artist sum of non-skipped plays via DB+payload join.
          - Peak hour: hour-of-day with most non-skipped plays, localised label.
        """
        from app.domain.models import (
            ListeningStatsResponse, TopTrackBrief, TopArtistBrief, PeakHour,
        )
        total_sec, since = MetadataDB.get_listening_total(collection_name)
        top_track_row = MetadataDB.get_top_played_track(collection_name)
        peak_hour_int = MetadataDB.get_peak_hour(collection_name)

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
                )

        # Top artist — aggregate non-skipped plays per artist by joining DB + payload
        counts_by_id = MetadataDB.get_play_counts_by_track(collection_name)
        if counts_by_id:
            try:
                points = qdrant_client.retrieve(
                    collection_name=collection_name,
                    ids=list(counts_by_id.keys()),
                    with_payload=["artist"],
                    with_vectors=False,
                )
            except Exception:
                points = []
            artist_counts: dict[str, int] = {}
            for p in points:
                a = (p.payload or {}).get("artist") or ""
                if a:
                    artist_counts[a] = artist_counts.get(a, 0) + counts_by_id.get(str(p.id), 0)
            if artist_counts:
                top_artist_name = max(artist_counts.items(), key=lambda kv: kv[1])[0]
                top_artist = TopArtistBrief(
                    name=top_artist_name,
                    slug=_slugify_artist_for_album(top_artist_name),
                    play_count=artist_counts[top_artist_name],
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

    def _metadata_to_tracks(self, metadata: dict) -> List[TrackMetadata]:
        """Convert FileProcessor metadata dict to TrackMetadata list.

        FileProcessor returns items with:
          - year: int | None  (not a range string)
          - duration: int     (raw seconds, not "MM:SS")
        """
        from file_processor.utils import save_cover_art

        tracks = []
        for key, info in metadata.items():
            file_path = info.get("file_path", "")
            track_id = self._compute_track_id(file_path)

            # year is already int|None from file_processor
            year = info.get("year")
            if isinstance(year, str):
                # defensive: if somehow a string slipped in, parse first digits
                try:
                    year = int(str(year).split("-")[0])
                except (ValueError, AttributeError):
                    year = None

            # duration is already int (seconds) from file_processor
            raw_duration = info.get("duration", 0)
            if isinstance(raw_duration, (int, float)):
                duration_sec = float(raw_duration)
            else:
                # defensive: parse "MM:SS" string if somehow that's what we got
                duration_sec = self._parse_duration(str(raw_duration))

            # Extract and save cover art
            cover_art_path = None
            if file_path and track_id:
                try:
                    cover_art_path = save_cover_art(file_path, track_id)
                except Exception as e:
                    logger.warning("[LibraryService] Cover art extraction failed for %s: %s", file_path, e)

            track = TrackMetadata(
                track_id=track_id,
                title=info.get("title", ""),
                artist=info.get("artist", ""),
                album=info.get("album"),
                year=year,
                genre=info.get("genre"),
                duration_sec=duration_sec,
                file_path=file_path,
                lyrics=info.get("lyrics"),
                cover_art_path=cover_art_path,
                producer=info.get("producer"),
                label=info.get("label"),
                samples=info.get("samples"),
                sampled_by=info.get("sampled_by"),
            )

            if track.file_path:
                tracks.append(track)

        return tracks

    @staticmethod
    def _compute_track_id(file_path: str) -> str:
        """Stable ID from file path (SHA256, first 16 chars)."""
        if file_path:
            return hashlib.sha256(file_path.encode()).hexdigest()[:16]
        else:
            logger.warning("[LibraryService] Error while resolving file path")
            return None

    @staticmethod
    def _parse_duration(duration_str: str) -> float:
        """Parse 'MM:SS' or 'HH:MM:SS' to seconds."""
        if not duration_str:
            return 0.0
        parts = duration_str.split(":")
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except (ValueError, IndexError):
            pass
        return 0.0
