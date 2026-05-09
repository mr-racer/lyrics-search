"""Library service — indexing tracks from a folder."""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Optional

from ..domain.models import TrackMetadata
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
            # Use already-loaded text model (don't reload during indexing)
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
            facts_total = len(unique_artists) + len(unique_songs)

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

            if facts_all_cached and facts_total > 0:
                logger.info("[LibraryService] All facts cached, skipping FACTS stage")
                stage_facts.status = IndexStatus.COMPLETED
                stage_facts.current = facts_total
                stage_facts.completed_at = time.time()
                stage_facts.message = "Факты из кеша"

                await self._notify_progress(job, {
                    "stage": IndexStage.FACTS.value,
                    "stage_status": IndexStatus.COMPLETED.value,
                    "current": facts_total,
                    "total": facts_total,
                    "message": stage_facts.message,
                })
            elif facts_total == 0:
                stage_facts.status = IndexStatus.COMPLETED
                stage_facts.message = "Нет данных"

                await self._notify_progress(job, {
                    "stage": IndexStage.FACTS.value,
                    "stage_status": IndexStatus.COMPLETED.value,
                    "message": stage_facts.message,
                })
            else:
                # Launch facts fetches with progress callbacks
                facts_progress = {"artists": 0, "songs": 0}
                facts_state = {"error": None}

                async def on_artist_facts_progress(current: int, total: int, label: str):
                    facts_progress["artists"] = current
                    combined = facts_progress["artists"] + facts_progress["songs"]
                    stage_facts.current = combined
                    stage_facts.message = f"Факты: {label}"
                    eta = job.calculate_eta_seconds(IndexStage.FACTS)
                    await self._notify_progress(job, {
                        "stage": IndexStage.FACTS.value,
                        "current": combined,
                        "total": facts_total,
                        "message": stage_facts.message,
                        "eta_seconds": eta,
                    })

                async def on_song_facts_progress(current: int, total: int, label: str):
                    facts_progress["songs"] = current
                    combined = facts_progress["artists"] + facts_progress["songs"]
                    stage_facts.current = combined
                    stage_facts.message = f"Факты: {label}"
                    eta = job.calculate_eta_seconds(IndexStage.FACTS)
                    await self._notify_progress(job, {
                        "stage": IndexStage.FACTS.value,
                        "current": combined,
                        "total": facts_total,
                        "message": stage_facts.message,
                        "eta_seconds": eta,
                    })

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
                        await asyncio.wait_for(facts_task, timeout=60)
                        logger.info("[LibraryService] Artist facts fetched")
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning("[LibraryService] Artist facts fetch timed out or failed: %s", e)
                        facts_task.cancel()
                        if not facts_state["error"]:
                            facts_state["error"] = str(e)

                    try:
                        await asyncio.wait_for(song_facts_task, timeout=120)
                        logger.info("[LibraryService] Song facts fetched")
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning("[LibraryService] Song facts fetch timed out or failed: %s", e)
                        song_facts_task.cancel()
                        if not facts_state["error"]:
                            facts_state["error"] = str(e)
                except Exception as e:
                    logger.warning("[LibraryService] Facts stage failed (non-critical): %s", e)
                    facts_state["error"] = str(e)

                stage_facts.status = IndexStatus.COMPLETED
                stage_facts.current = facts_progress["artists"] + facts_progress["songs"]
                stage_facts.completed_at = time.time()
                if facts_state["error"]:
                    stage_facts.message = f"Частично: {facts_state['error']}"
                else:
                    stage_facts.message = f"Факты: {stage_facts.current}/{facts_total} обработано"

                await self._notify_progress(job, {
                    "stage": IndexStage.FACTS.value,
                    "stage_status": IndexStatus.COMPLETED.value,
                    "current": stage_facts.current,
                    "total": facts_total,
                    "message": stage_facts.message,
                    "stage_error": facts_state["error"],
                })

            # ── Stage METADATA: MusicBrainz enrichment ────────────────────────
            logger.info("[LibraryService] Stage METADATA: MusicBrainz enrichment")
            stage_meta = job.stages[IndexStage.METADATA]

            if enhance_by_musicbrainz and processed_files:
                stage_meta.status = IndexStatus.RUNNING
                stage_meta.started_at = time.time()
                stage_meta.total = track_count
                stage_meta.message = "Обогащение метаданных (MusicBrainz)..."

                await self._notify_progress(job, {
                    "stage": IndexStage.METADATA.value,
                    "stage_status": IndexStatus.RUNNING.value,
                    "message": stage_meta.message,
                    "current": 0,
                    "total": track_count,
                })

                mb_error = None

                async def on_mb_progress(current: int, total: int, label: str):
                    stage_meta.current = current
                    stage_meta.message = label
                    eta = job.calculate_eta_seconds(IndexStage.METADATA)
                    await self._notify_progress(job, {
                        "stage": IndexStage.METADATA.value,
                        "current": current,
                        "total": total,
                        "message": label,
                        "eta_seconds": eta,
                    })

                try:
                    await asyncio.to_thread(
                        self._enrich_with_musicbrainz,
                        processed_files,
                        enhance_by_musicbrainz,
                        lambda c, t, l: asyncio.run_coroutine_threadsafe(
                            on_mb_progress(c, t, l), loop
                        ),
                    )
                except Exception as e:
                    logger.warning("[LibraryService] MusicBrainz enrichment failed (non-critical): %s", e)
                    mb_error = str(e)

                stage_meta.status = IndexStatus.COMPLETED
                stage_meta.current = stage_meta.current
                stage_meta.completed_at = time.time()
                if mb_error:
                    stage_meta.message = f"Ошибка: {mb_error}"
                else:
                    stage_meta.message = "Метаданные обогащены"

                await self._notify_progress(job, {
                    "stage": IndexStage.METADATA.value,
                    "stage_status": IndexStatus.COMPLETED.value,
                    "current": stage_meta.current,
                    "total": track_count,
                    "message": stage_meta.message,
                    "stage_error": mb_error,
                })
            else:
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

    def _enrich_with_musicbrainz(
        self,
        processed_files: dict,
        enhance: bool,
        progress_callback=None,
    ):
        """Enrich track metadata via MusicBrainz API.

        Args:
            processed_files: dict keyed by "artist — title", values are metadata dicts.
            enhance: if True, MB data replaces local; if False, MB is fallback only.
            progress_callback: optional callable(current, total, label).
        """
        from musicbraniz_search import MusicBrainzLookup

        mb = MusicBrainzLookup("MusiX", "1.0", "https://musix.local")
        total = len(processed_files)
        enriched = 0
        idx = 0

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

            idx += 1
            if progress_callback:
                progress_callback(idx, total, f"{artist} — {title}")

            logger.info(
                "[LibraryService] MB enrichment %d/%d (enriched=%d): %s — %s",
                idx, total, enriched, artist, title,
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
