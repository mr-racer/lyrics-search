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
from .artist_facts_service import fetch_facts_for_artists
from .job_tracker import JobTracker, IndexStage, IndexStatus
from .similarity_service import analyze_collection
from .song_facts_service import fetch_facts_for_songs

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

            # Enrich metadata via MusicBrainz (optional, runs in thread pool)
            if enhance_by_musicbrainz and processed_files:
                try:
                    await asyncio.to_thread(
                        self._enrich_with_musicbrainz, processed_files, enhance_by_musicbrainz
                    )
                except Exception as e:
                    logger.warning("[LibraryService] MusicBrainz enrichment skipped: %s", e)

            # Extract unique artists and start fetching facts in background
            unique_artists = sorted({
                info.get("artist", "").strip()
                for info in processed_files.values()
                if info.get("artist", "").strip()
            })
            facts_task = asyncio.create_task(
                fetch_facts_for_artists(unique_artists, collection_name),
                name="artist-facts",
            )
            logger.info("[LibraryService] Launched facts fetch for %d artists", len(unique_artists))

            # Extract unique songs and start fetching song facts in background
            unique_songs = sorted({
                (info.get("artist", "").strip(), info.get("title", "").strip())
                for info in processed_files.values()
                if info.get("artist", "").strip() and info.get("title", "").strip()
            })
            song_facts_task = asyncio.create_task(
                fetch_facts_for_songs(unique_songs, collection_name),
                name="song-facts",
            )
            logger.info("[LibraryService] Launched song facts fetch for %d songs", len(unique_songs))

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

            stage_lyrics = job.stages[IndexStage.LYRICS]
            stage_lyrics.status = IndexStatus.RUNNING
            stage_lyrics.started_at = time.time()
            stage_lyrics.total = len(tracks)
            stage_lyrics.message = "Индексация текстов в векторную БД..."

            # CLAP encoding runs inside fit() alongside text encoding without its own
            # progress callback — mark AUDIO as RUNNING here so the UI doesn't show
            # it as idle while the GPU is busy on it.
            stage_audio = job.stages[IndexStage.AUDIO]
            stage_audio.status = IndexStatus.RUNNING
            stage_audio.started_at = time.time()
            stage_audio.total = len(tracks)
            stage_audio.message = "CLAP-кодирование аудио..."

            # Notify AFTER updating stage states so the snapshot captures RUNNING
            await self._notify_progress(job, {
                "stage": IndexStage.LYRICS.value,
                "stage_status": IndexStatus.RUNNING.value,
                "message": stage_lyrics.message,
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

            # Stage 3: Index into Qdrant via SearchService (includes both lyrics and audio)
            if self.search_service and tracks:
                logger.info("[LibraryService] Starting SearchService.index_tracks_with_progress...")
                # Index with progress callbacks
                await self.search_service.index_tracks_with_progress(
                    tracks,
                    collection_name=collection_name,
                    progress_callback=self._on_index_progress,
                    text_model=text_model,
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

            # Await facts (runs parallel to encoding, cached to disk)
            try:
                await asyncio.wait_for(facts_task, timeout=60)
                logger.info("[LibraryService] Artist facts fetched")
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("[LibraryService] Artist facts fetch timed out or failed: %s", e)
                facts_task.cancel()

            try:
                await asyncio.wait_for(song_facts_task, timeout=120)
                logger.info("[LibraryService] Song facts fetched")
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("[LibraryService] Song facts fetch timed out or failed: %s", e)
                song_facts_task.cancel()

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
                index_stage = IndexStage.LYRICS if stage == "lyrics" else IndexStage.AUDIO if stage == "audio" else None
                if index_stage:
                    sp = job.stages[index_stage]
                    sp.current = current
                    sp.total = total
                    sp.message = message

                eta = job.calculate_eta_seconds(index_stage) if index_stage else None
                await self._notify_progress(job, {
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "message": message,
                    "eta_seconds": eta,
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

    def _enrich_with_musicbrainz(self, processed_files: dict, enhance: bool):
        """Enrich track metadata via MusicBrainz API.

        Args:
            processed_files: dict keyed by "artist — title", values are metadata dicts.
            enhance: if True, MB data replaces local; if False, MB is fallback only.
        """
        from musicbraniz_search import MusicBrainzLookup

        mb = MusicBrainzLookup("MusiX", "1.0", "https://musix.local")
        total = len(processed_files)
        enriched = 0

        for key, info in processed_files.items():
            title = info.get("title", "").strip()
            artist = info.get("artist", "").strip()
            if not title or not artist:
                continue

            rec_id = mb.resolve_recording_id(title, artist)
            if not rec_id:
                continue

            # Year — merge with existing based on flag
            mb_year = mb.get_recording_year(rec_id)
            local_year = info.get("year")
            if mb_year:
                info["year"] = mb_year if enhance else (local_year or mb_year)
                enriched += 1

            # Producer
            producers = mb.get_recording_producers(rec_id)
            if producers:
                info["producer"] = ", ".join(producers) if enhance or not info.get("producer") else info["producer"]

            # Labels
            labels = mb.get_recording_labels(rec_id)
            if labels:
                label_name = labels[0].get("name", "")
                if label_name:
                    info["label"] = label_name if enhance else (info.get("label") or label_name)

            # Samples
            samples = mb.get_recording_samples(rec_id)
            if samples:
                sample_strs = [
                    f"{s.get('artist', '?')} — {s.get('title', '?')}"
                    for s in samples if s.get("title") or s.get("artist")
                ]
                info["samples"] = sample_strs if enhance or not info.get("samples") else info["samples"]

            sampled_by = mb.get_recording_sampled_by(rec_id)
            if sampled_by:
                sampled_strs = [
                    f"{s.get('artist', '?')} — {s.get('title', '?')}"
                    for s in sampled_by if s.get("title") or s.get("artist")
                ]
                info["sampled_by"] = sampled_strs if enhance or not info.get("sampled_by") else info["sampled_by"]

            logger.info(
                "[LibraryService] MB enrichment %d/%d (enriched=%d): %s — %s",
                enriched + 1, total, enriched, artist, title,
            )

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
