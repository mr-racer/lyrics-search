"""Search service — unified interface for text/audio/hybrid search."""

import torch
from typing import List, Optional, Literal

from ..domain.models import TrackMetadata, TrackHit, SearchRequest, SearchResponse, SearchFilters
from ..existing.qdrant_db import LyricsDB
from ..resources.model_registry import ModelRegistry


class SearchService:
    """
    Unified search supporting:
    - text: dense + BM25 fusion via LyricsDB
    - audio: CLAP text embedding search
    - hybrid: dense + BM25 + CLAP RRF fusion
    """

    def __init__(self, lyrics_db: LyricsDB):
        self.lyrics_db = lyrics_db

    async def search(
        self,
        query: str,
        mode: Literal["text", "audio", "hybrid"] = "text",
        text_model: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
        limit: int = 10,
    ) -> List[TrackHit]:
        """Unified search dispatching to mode-specific handlers."""
        if mode == "audio":
            return await self._search_audio(query, filters, limit)
        elif mode == "hybrid":
            return await self._search_hybrid(query, filters, limit)
        else:  # text
            return await self._search_text(query, filters, limit)

    # ── Text search (dense + BM25) ──

    async def _search_text(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> List[TrackHit]:
        """Text-based search using dense + BM25 fusion."""
        qdrant_filter = self._build_qdrant_filter(filters)
        filter_kwargs = self._extract_filter_kwargs(filters)

        results = self.lyrics_db.search(
            query=query,
            limit=limit * 2,
            min_dense_score=0.3,
            include_clap=False,
            **filter_kwargs,
        )

        return self._points_to_hits(results[:limit], matched_on="lyrics")

    # ── Audio search (CLAP) ──

    async def _search_audio(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> List[TrackHit]:
        """Audio-based search using CLAP text embedding."""
        clap_model = ModelRegistry.get_clap()
        clap_vector = clap_model.get_text_embedding([query])[0].tolist()

        filter_kwargs = self._extract_filter_kwargs(filters)

        results = self.lyrics_db.search(
            query=query,
            limit=limit * 2,
            include_clap=True,
            min_clap_score=0.01,
            **filter_kwargs,
        )

        hits = self._points_to_hits(results[:limit], matched_on="audio")
        return hits

    # ── Hybrid search (dense + BM25 + CLAP) ──

    async def _search_hybrid(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> List[TrackHit]:
        """Hybrid search combining text and audio embeddings."""
        filter_kwargs = self._extract_filter_kwargs(filters)

        results = self.lyrics_db.search(
            query=query,
            limit=limit * 2,
            include_clap=True,
            min_dense_score=0.3,
            min_clap_score=0.01,
            **filter_kwargs,
        )

        hits = self._points_to_hits(results[:limit], matched_on="hybrid")
        return hits

    # ── Legacy alias (backward compat) ──

    async def search_lyrics(
        self,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 10,
    ) -> List[TrackHit]:
        """Alias for _search_text for backward compatibility."""
        return await self._search_text(query, filters, limit)

    # ── Helpers ──

    def _extract_filter_kwargs(self, filters: Optional[SearchFilters]) -> dict:
        """Extract keyword args for LyricsDB.search() from SearchFilters."""
        if not filters:
            return {}
        return {
            "artist": filters.artist,
            "album": filters.album,
            "genre": filters.genre,
        }

    def _build_qdrant_filter(self, filters: Optional[SearchFilters]) -> dict | None:
        """Build raw Qdrant filter dict (for future use)."""
        if not filters or all(v is None for v in [filters.artist, filters.album, filters.genre]):
            return None

        conditions = []
        if filters.artist:
            conditions.append({"key": "artist", "match": {"value": filters.artist}})
        if filters.album:
            conditions.append({"key": "album", "match": {"value": filters.album}})
        if filters.genre:
            conditions.append({"key": "genre", "match": {"value": filters.genre}})

        return {"must": conditions} if conditions else None

    def _points_to_hits(self, points, matched_on: str = "lyrics") -> List[TrackHit]:
        """Convert Qdrant ScoredPoint list to TrackHit list."""
        hits = []
        for point in points:
            payload = point.payload or {}

            # Year: stored as int OR as "YYYY-YYYY" range string
            raw_year = payload.get("year")
            if raw_year is None:
                raw_year = payload.get("year_range")
            try:
                year = int(str(raw_year).split("-")[0]) if raw_year is not None else None
            except (ValueError, TypeError):
                year = None

            # Duration: stored as numeric seconds
            raw_dur = payload.get("duration", 0)
            try:
                duration_sec = float(raw_dur) if raw_dur else 0.0
            except (ValueError, TypeError):
                duration_sec = 0.0

            # Lyrics snippet for LLM context (first ~400 chars; not sent to frontend)
            raw_lyrics: str = payload.get("lyrics") or ""
            snippet: str | None = None
            if raw_lyrics.strip():
                flat = raw_lyrics.replace("\n", " ").strip()
                snippet = flat[:400] + ("…" if len(flat) > 400 else "")

            track = TrackMetadata(
                track_id=str(point.id),
                title=payload.get("title", "Unknown"),
                artist=payload.get("artist", "Unknown"),
                album=payload.get("album"),
                year=year,
                genre=payload.get("genre"),
                duration_sec=duration_sec,
                file_path=payload.get("file_path", ""),
                lyrics=None,  # not sent to frontend — use snippet for LLM
            )
            hits.append(TrackHit(
                track=track,
                score=float(point.score),
                matched_on=matched_on,
                snippet=snippet,
            ))
        return hits

    async def index_tracks(self, tracks: List[TrackMetadata]) -> None:
        """Index tracks into Qdrant.

        Note: LyricsDB.fit() calls prepare_metadata() which expects a dict keyed by
        "Artist — Title". Duration must be raw seconds (int/float), not "MM:SS" string,
        because prepare_metadata() computes IQR buckets from numeric durations.
        """
        if not tracks:
            return

        # Build dict as prepare_metadata() expects: {"Artist — Title": {...}}
        data: dict[str, dict] = {}
        for track in tracks:
            key = f"{track.artist} — {track.title}"
            data[key] = {
                "title": track.title,
                "artist": track.artist,
                "album": track.album,
                "year": track.year,
                "genre": track.genre,
                # duration must be numeric seconds for prepare_metadata() IQR bucketing
                "duration": int(track.duration_sec) if track.duration_sec else 0,
                "lyrics": track.lyrics or "",
                "file_path": track.file_path,
            }

        self.lyrics_db.fit(data, path=None)
