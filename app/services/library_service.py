"""Library service — indexing tracks from a folder."""

import asyncio
import hashlib
from pathlib import Path
from typing import List, Optional

from ..domain.models import TrackMetadata, IndexProgress
from ..resources.model_registry import ModelRegistry
from ..existing.folder_processor import FileProcessor


class LibraryService:
    """Index music files, extract metadata + lyrics, and upsert to Qdrant."""

    def __init__(self, search_service=None):
        """
        Args:
            search_service: SearchService instance for indexing tracks into Qdrant.
        """
        self.search_service = search_service
        self._indexing_in_progress = False
        self._indexed_count = 0

    async def index_folder(
        self,
        folder_path: str,
        better_lyrics_quality: bool = False,
        text_model: Optional[str] = None,
    ) -> dict:
        """
        Index all audio files in folder.
        Returns dict with status, count, and message.
        """
        if self._indexing_in_progress:
            return {"status": "failed", "message": "Indexing already in progress"}

        self._indexing_in_progress = True

        try:
            # Load text model if specified
            if text_model:
                ModelRegistry.load_text_model(text_model)

            # Process files with FileProcessor (run in thread pool to avoid blocking)
            processor = FileProcessor()
            processed_files = await asyncio.to_thread(
                processor.process_folder,
                music_folder=folder_path,
                better_lyrics_quality=better_lyrics_quality,
            )

            track_count = len(processed_files)
            self._indexed_count = track_count

            # Convert to TrackMetadata list
            tracks = self._metadata_to_tracks(processed_files)

            # Index into Qdrant via SearchService
            if self.search_service and tracks:
                await self.search_service.index_tracks(tracks)

            return {
                "status": "completed",
                "count": track_count,
                "message": f"Indexed {track_count} tracks",
            }
        except Exception as e:
            return {
                "status": "failed",
                "count": 0,
                "message": str(e),
            }
        finally:
            self._indexing_in_progress = False

    async def get_status(self) -> dict:
        """Return current indexing status."""
        return {
            "indexing_in_progress": self._indexing_in_progress,
            "indexed_count": self._indexed_count,
        }

    # ── Helpers ──

    def _metadata_to_tracks(self, metadata: dict) -> List[TrackMetadata]:
        """Convert FileProcessor metadata dict to TrackMetadata list.

        FileProcessor returns items with:
          - year: int | None  (not a range string)
          - duration: int     (raw seconds, not "MM:SS")
        """
        tracks = []
        for key, info in metadata.items():
            # FileProcessor doesn't add file_path to the result dict;
            # the key is "Artist — Title", so we fall back to it.
            file_path = info.get("file_path", key)
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
            )
            tracks.append(track)
        return tracks

    @staticmethod
    def _compute_track_id(file_path: str) -> str:
        """Stable ID from file path (SHA256, first 16 chars)."""
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]

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
