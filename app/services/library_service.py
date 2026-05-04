"""Library service — indexing tracks from a folder."""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Optional

from ..domain.models import TrackMetadata, IndexProgress
from ..resources.model_registry import ModelRegistry
from ..resources.db_client import DbClient
from ..existing.folder_processor import FileProcessor
from .job_tracker import JobTracker, IndexStage, IndexStatus
from .similarity_service import analyze_collection

logger = logging.getLogger(__name__)


class LibraryService:
    """Index music files, extract metadata + lyrics, and upsert to Qdrant."""

    def __init__(self, search_service=None, db_client: Optional[DbClient] = None):
        """
        Args:
            search_service: SearchService instance for indexing tracks into Qdrant.
            db_client: DbClient instance for Qdrant access (needed for similarity analysis).
        """
        self.search_service = search_service
        self.db_client = db_client
        self._job_tracker = JobTracker()
        self._current_job_id = None

    async def index_folder(
        self,
        folder_path: str,
        collection_name: str = "music_explorer",
        better_lyrics_quality: bool = False,
        text_model: Optional[str] = None,
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
        task = asyncio.create_task(self._run_indexing_job(job, better_lyrics_quality, text_model))
        logger.info("[LibraryService] Background task created: %s", task)

        return {
            "status": "started",
            "job_id": job.job_id,
            "message": "Indexing started",
        }

    async def _run_indexing_job(self, job, better_lyrics_quality: bool, text_model: Optional[str] = None):
        """Execute the indexing process with progress updates."""
        folder_path = job.folder_path
        collection_name = job.collection_name
        logger.info("[LibraryService] _run_indexing_job START: job=%s, folder=%s, collection=%s",
                    job.job_id, folder_path, collection_name)

        try:
            # Stage 1: Metadata fetching
            logger.info("[LibraryService] Stage 1: Metadata fetching")
            await self._notify_progress(job, {
                "stage": IndexStage.METADATA.value,
                "stage_status": IndexStatus.RUNNING.value,
                "message": "Сканирование папки...",
                "current": 0,
            })

            stage_meta = job.stages[IndexStage.METADATA]
            stage_meta.status = IndexStatus.RUNNING
            stage_meta.started_at = time.time()
            stage_meta.message = "Сканирование папки и чтение метаданных..."

            # Use already-loaded text model (don't reload during indexing)
            # The model should be loaded at startup via ModelRegistry
            if text_model:
                logger.info("[LibraryService] Getting text model: %s (cached if already loaded)", text_model)
                ModelRegistry.load_text_model(text_model)
            else:
                logger.info("[LibraryService] No text_model specified, using default")

            # Process files - scan first to get total count
            logger.info("[LibraryService] Scanning folder for audio files: %s", folder_path)
            audio_files = [
                p for p in Path(folder_path).rglob("*")
                if p.suffix.lower() in (".flac", ".m4a", ".mp3")
            ]
            stage_meta.total = len(audio_files)
            logger.info("[LibraryService] Found %d audio files", len(audio_files))

            await self._notify_progress(job, {
                "stage": IndexStage.METADATA.value,
                "total": len(audio_files),
                "message": f"Найдено {len(audio_files)} файлов",
            })

            # Process files with FileProcessor (run in thread pool to avoid blocking)
            logger.info("[LibraryService] Starting FileProcessor...")
            processor = FileProcessor()
            loop = asyncio.get_event_loop()

            # Progress callback for file processing (metadata + lyrics fetching)
            async def on_file_progress(current: int, total: int, message: str, details: dict = None):
                stage_meta.current = current
                stage_meta.total = total
                stage_meta.message = message
                extra: dict = {}
                if details:
                    extra = {
                        "lyrics_found":        details.get("found", 0),
                        "lyrics_not_found":    details.get("not_found", 0),
                        "lyrics_rate":         details.get("rate_per_min"),
                        "lyrics_eta":          details.get("eta_seconds"),
                        "lyrics_last_track":   details.get("last_track"),
                        "lyrics_last_success": details.get("last_success"),
                    }
                await self._notify_progress(job, {
                    "stage": IndexStage.METADATA.value,
                    "current": current,
                    "total": total,
                    "message": message,
                    **extra,
                })

            processed_files = await asyncio.to_thread(
                processor.process_folder,
                music_folder=folder_path,
                better_lyrics_quality=better_lyrics_quality,
                progress_callback=lambda c, t, m, d=None: asyncio.run_coroutine_threadsafe(
                    on_file_progress(c, t, m, d), loop
                ),  # fire-and-forget — no .result() to avoid deadlock
            )

            track_count = len(processed_files)
            logger.info("[LibraryService] FileProcessor done, processed %d tracks", track_count)
            stage_meta.status = IndexStatus.COMPLETED
            stage_meta.current = track_count
            stage_meta.completed_at = time.time()
            stage_meta.message = f"Обработано {track_count} треков"

            await self._notify_progress(job, {
                "stage": IndexStage.METADATA.value,
                "stage_status": IndexStatus.COMPLETED.value,
                "current": track_count,
                "message": stage_meta.message,
            })

            # Convert to TrackMetadata list
            logger.info("[LibraryService] Converting to TrackMetadata...")
            tracks = self._metadata_to_tracks(processed_files)
            logger.info("[LibraryService] Converted %d tracks to TrackMetadata", len(tracks))

            # Stage 2: Lyrics indexing
            logger.info("[LibraryService] Stage 2: Lyrics indexing")
            await self._notify_progress(job, {
                "stage": IndexStage.LYRICS.value,
                "stage_status": IndexStatus.RUNNING.value,
                "message": "Кодирование текстов песен...",
                "current": 0,
            })

            stage_lyrics = job.stages[IndexStage.LYRICS]
            stage_lyrics.status = IndexStatus.RUNNING
            stage_lyrics.started_at = time.time()
            stage_lyrics.total = len(tracks)
            stage_lyrics.message = "Индексация текстов в векторную БД..."

            # Stage 3: Index into Qdrant via SearchService (includes both lyrics and audio)
            if self.search_service and tracks:
                logger.info("[LibraryService] Starting SearchService.index_tracks_with_progress...")
                # Index with progress callbacks
                await self.search_service.index_tracks_with_progress(
                    tracks,
                    collection_name=collection_name,
                    progress_callback=self._on_index_progress,
                )
                logger.info("[LibraryService] SearchService.index_tracks_with_progress done")
            else:
                logger.warning("[LibraryService] Skipping indexing: search_service=%s, tracks=%d",
                              self.search_service is not None, len(tracks))

            # Mark lyrics/audio stages as completed
            job.stages[IndexStage.LYRICS].status = IndexStatus.COMPLETED
            job.stages[IndexStage.LYRICS].completed_at = time.time()
            job.stages[IndexStage.AUDIO].status = IndexStatus.COMPLETED
            job.stages[IndexStage.AUDIO].completed_at = time.time()

            # Stage 4: Similarity analysis
            logger.info("[LibraryService] Stage 4: Similarity analysis")
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
                # Don't fail the whole job — analysis is optional
                stage_analysis.status = IndexStatus.COMPLETED

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
        """Callback from SearchService for indexing progress."""
        if self._current_job_id:
            job = self._job_tracker.get_job(self._current_job_id)
            if job:
                # Update the appropriate stage
                if stage == "lyrics":
                    job.stages[IndexStage.LYRICS].current = current
                    job.stages[IndexStage.LYRICS].total = total
                    job.stages[IndexStage.LYRICS].message = message
                elif stage == "audio":
                    job.stages[IndexStage.AUDIO].current = current
                    job.stages[IndexStage.AUDIO].total = total
                    job.stages[IndexStage.AUDIO].message = message

                await self._notify_progress(job, {
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "message": message,
                })

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

    # ── Helpers ──

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
                except Exception:
                    pass  # cover art is optional, don't fail indexing

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
            print('Error while resolving file path')
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
