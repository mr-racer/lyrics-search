"""Search service — unified interface for text/audio/hybrid search."""

import asyncio
import logging
from typing import List, Optional, Literal, Dict

from ..domain.models import TrackMetadata, TrackHit, SearchFilters, ScoreBreakdown
from ..existing.qdrant_db import LyricsDB
from ..resources.model_registry import ModelRegistry
from ..resources.metadata_db import MetadataDB
from .artist_facts_service import load_all_facts_for_collection, _slugify as _slugify_artist
from .song_facts_service import load_all_song_facts_for_collection, get_song_facts_key

logger = logging.getLogger(__name__)


class SearchService:
    """
    Unified search supporting:
    - text: dense + BM25 fusion via LyricsDB
    - audio: CLAP text embedding → search only by 'clap' vector
    - hybrid: parallel dense+BM25 + CLAP, merged with 0.5/0.5 score weighting
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
        collection_name: Optional[str] = None,
    ) -> List[TrackHit]:
        """Unified search dispatching to mode-specific handlers."""
        if mode == "audio":
            return await self._search_audio(query, filters, limit, collection_name)
        elif mode == "hybrid":
            return await self._search_hybrid(query, filters, limit, collection_name)
        else:  # text
            return await self._search_text(query, filters, limit, collection_name)

    # ── Text search (dense + BM25) ──

    async def _search_text(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
        collection_name: Optional[str] = None,
    ) -> List[TrackHit]:
        """Text-based search using dense + BM25 fusion."""
        col = collection_name or self.lyrics_db.collection_name
        await self._ensure_model_for_collection(col)

        filter_kwargs = self._extract_filter_kwargs(filters)
        active = {k: v for k, v in filter_kwargs.items() if v is not None}
        logger.info(
            "[SearchService] _search_text: query=%r  collection=%s  filters=%s",
            query, col, active or "(none)",
        )

        results = self.lyrics_db.search(
            query=query,
            limit=limit * 2,
            min_dense_score=0.3,
            include_clap=False,
            collection_name_override=collection_name,
            **filter_kwargs,
        )

        logger.info("[SearchService] _search_text: Qdrant returned %d points", len(results))
        return self._points_to_hits(results[:limit], matched_on="lyrics", collection_name=collection_name)

    async def _ensure_model_for_collection(self, collection_name: str) -> None:
        """Ensure LyricsDB uses the correct text model for the given collection.

        Reads the collection's vector config from Qdrant to find which text
        embedding model was used during indexing, then switches LyricsDB
        to match it.
        """
        try:
            info = self.lyrics_db.qdrant_client.get_collection(collection_name)
            vec_names = list(info.config.params.vectors.keys())
        except Exception:
            return

        # Find text vector name (not 'bm25', not 'clap')
        text_vec = None
        for vn in vec_names:
            if vn in ('bm25', 'clap'):
                continue
            if vn.startswith('text_'):
                text_vec = vn
                break
        if text_vec is None:
            # Fallback: pick first non-special vector
            for vn in vec_names:
                if vn not in ('bm25', 'clap'):
                    text_vec = vn
                    break

        if text_vec is None or text_vec == self.lyrics_db._vector_name:
            return

        # Derive model name from vector name (text_{org}_{model} -> org/model)
        vec_without_prefix = text_vec[len('text_'):]
        parts = vec_without_prefix.split('_', 1)
        if len(parts) == 2:
            model_name = f"{parts[0]}/{parts[1]}"
        else:
            model_name = vec_without_prefix

        # Try to load from ModelRegistry
        if model_name in ModelRegistry._text_models:
            logger.info("[SearchService] Auto-switching model for collection '%s': %s",
                        collection_name, model_name)
            model, vector_name, vector_dim = ModelRegistry.load_text_model(model_name)
            self.lyrics_db.model_name = model_name
            self.lyrics_db._model = model
            self.lyrics_db._vector_name = vector_name
            self.lyrics_db._vector_dim = vector_dim

    # ── Audio search (CLAP only) ──

    async def _search_audio(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
        collection_name: Optional[str] = None,
    ) -> List[TrackHit]:
        """Audio-based search: encode query with CLAP, search Qdrant by 'clap' vector only."""
        logger.info("[SearchService] _search_audio: query='%s', limit=%d, collection=%s",
                    query, limit, collection_name)

        if not ModelRegistry.is_clap_available():
            raise RuntimeError("Audio search unavailable: CLAP module not installed (pip install laion-clap)")

        try:
            clap_model = await asyncio.get_running_loop().run_in_executor(
                None, ModelRegistry.get_clap
            )
        except RuntimeError:
            logger.info("[SearchService] CLAP not yet loaded — loading now (lazy)")
            try:
                clap_model = await asyncio.get_running_loop().run_in_executor(
                    None, ModelRegistry.load_clap
                )
            except Exception as e:
                logger.error("[SearchService] CLAP model failed to load: %s", e)
                raise RuntimeError(f"Audio search unavailable: CLAP model could not be loaded — {e}") from e
        
        try:
            clap_vector = clap_model.get_text_embedding([query])[0].tolist()
            logger.debug("[SearchService] CLAP text embedding generated, dim=%d", len(clap_vector))
        except Exception as e:
            logger.error("[SearchService] Failed to generate CLAP embedding: %s", e)
            return []

        col = collection_name or self.lyrics_db.collection_name
        qdrant_filter = self._build_qdrant_filter_models(filters)
        
        logger.info("[SearchService] Querying Qdrant collection='%s', using='clap', limit=%d",
                    col, limit * 2)

        try:
            results = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.lyrics_db.qdrant_client.query_points(
                    collection_name=col,
                    query=clap_vector,
                    using="clap",
                    limit=limit * 2,
                    query_filter=qdrant_filter,
                    with_payload=True,
                ),
            )
        except Exception as e:
            logger.error("[SearchService] Qdrant query failed: %s", e, exc_info=True)
            raise

        logger.info("[SearchService] Qdrant returned %d points", len(results.points) if results else 0)
        
        hits = self._points_to_hits(results.points[:limit], matched_on="audio", collection_name=collection_name)
        # filtered_hits = [h for h in hits if h.score >= 0.1]
        filtered_hits = hits
        
        logger.info("[SearchService] After score filter (>=0.1): %d hits", len(filtered_hits))
        if filtered_hits:
            logger.debug("[SearchService] Top hit: '%s' by '%s' (score=%.3f)",
                        filtered_hits[0].track.title,
                        filtered_hits[0].track.artist,
                        filtered_hits[0].score)
        
        return filtered_hits

    # ── Hybrid search (dense+BM25 0.5 + CLAP 0.5) ──

    async def _search_hybrid(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
        collection_name: Optional[str] = None,
    ) -> List[TrackHit]:
        """Hybrid search: parallel text (dense+BM25) + CLAP, merge with 0.5/0.5 weighting.
        
        If audio search fails, gracefully degrades to text-only results.
        """
        text_task = asyncio.create_task(
            self._search_text(query, filters, limit * 2, collection_name)
        )
        audio_task = asyncio.create_task(
            self._search_audio(query, filters, limit * 2, collection_name)
        )
        text_result, audio_result = await asyncio.gather(
            text_task, audio_task, return_exceptions=True
        )

        # Handle exceptions gracefully
        if isinstance(audio_result, Exception):
            logger.warning("Audio search failed in hybrid mode, using text-only fallback: %s",
                           audio_result)
            audio_hits: List[TrackHit] = []
        else:
            audio_hits = audio_result

        if isinstance(text_result, Exception):
            logger.error("Text search failed in hybrid mode: %s", text_result)
            raise

        text_hits = text_result

        # Merge by track_id with weighted scores
        merged = self._merge_hits(text_hits, audio_hits, text_weight=0.5, clap_weight=0.5)
        return merged[:limit]

    @staticmethod
    def _merge_hits(
        text_hits: List[TrackHit],
        clap_hits: List[TrackHit],
        text_weight: float = 0.5,
        clap_weight: float = 0.5,
    ) -> List[TrackHit]:
        """Merge two hit lists by track_id, applying weighted score fusion.

        Tracks appearing in both lists get combined score:
            final_score = normalized_text_score * text_weight + normalized_clap_score * clap_weight

        Scores are min-max normalized within each list before fusion to ensure
        fair weighting across different vector spaces.

        Each returned hit carries a ``score_breakdown`` with per-modality
        normalized scores (``None`` when a track was absent from that modality)
        and the weights formula used, enabling consumers to explain ranking.
        """
        def _normalize_scores(hits: List[TrackHit]) -> List[tuple[TrackHit, float]]:
            """Return (hit, normalized_score) pairs; single-item list gets score 1.0."""
            if not hits:
                return []
            scores = [h.score for h in hits]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s > min_s else 1.0
            return [(h, (h.score - min_s) / range_s) for h in hits]

        norm_text = _normalize_scores(text_hits)
        norm_clap = _normalize_scores(clap_hits)

        # score_map[tid] = {
        #   "track": TrackMetadata,
        #   "text_norm": float | None,   # normalized text score
        #   "audio_norm": float | None,  # normalized audio score
        #   "lead": TrackHit,            # first hit seen (for lyrics/facts)
        # }
        score_map: Dict[str, dict] = {}

        for hit, norm_score in norm_text:
            tid = hit.track.track_id
            score_map[tid] = {
                "track": hit.track,
                "text_norm": norm_score,
                "audio_norm": None,
                "lead": hit,
            }

        for hit, norm_score in norm_clap:
            tid = hit.track.track_id
            if tid in score_map:
                score_map[tid]["audio_norm"] = norm_score
            else:
                score_map[tid] = {
                    "track": hit.track,
                    "text_norm": None,
                    "audio_norm": norm_score,
                    "lead": hit,
                }

        weights_dict = {"text_dense": text_weight, "audio": clap_weight}

        # Build final list, sorted by merged score descending
        merged: List[TrackHit] = []
        for entry in score_map.values():
            lead = entry["lead"]
            text_present = entry["text_norm"] is not None
            audio_present = entry["audio_norm"] is not None
            # Use `is not None` rather than `or 0.0` so a legitimate 0.0
            # (e.g. the lowest-scoring track after min-max normalization) is
            # not confused with "absent modality" when reading the breakdown.
            t_score = entry["text_norm"] if text_present else 0.0
            a_score = entry["audio_norm"] if audio_present else 0.0
            final = t_score * text_weight + a_score * clap_weight

            breakdown = ScoreBreakdown(
                text_dense_score=entry["text_norm"],
                audio_score=entry["audio_norm"],
                final_score=final,
                weights=weights_dict,
            )

            # matched_on reflects which modalities actually contributed.
            if text_present and audio_present:
                matched_on = "hybrid"
            elif text_present:
                matched_on = "lyrics"
            else:
                matched_on = "audio"

            merged.append(TrackHit(
                track=lead.track,
                score=final,
                matched_on=matched_on,
                lyrics=lead.lyrics,
                artist_facts=lead.artist_facts,
                song_facts=lead.song_facts,
                score_breakdown=breakdown,
            ))

        merged.sort(key=lambda h: h.score, reverse=True)
        return merged

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
            "year_range": filters.year_range,
        }

    def _build_qdrant_filter_models(self, filters: Optional[SearchFilters]):
        """Build proper qdrant_client.models.Filter object for query_points()."""
        from qdrant_client import models
        
        if not filters or all(v is None for v in [filters.artist, filters.album, filters.genre]):
            return None

        conditions = []
        if filters.artist:
            conditions.append(models.FieldCondition(
                key="artist", match=models.MatchValue(value=filters.artist)
            ))
        if filters.album:
            conditions.append(models.FieldCondition(
                key="album", match=models.MatchValue(value=filters.album)
            ))
        if filters.genre:
            conditions.append(models.FieldCondition(
                key="genre", match=models.MatchValue(value=filters.genre)
            ))

        return models.Filter(must=conditions) if conditions else None

    def _points_to_hits(
        self,
        points,
        matched_on: str = "lyrics",
        collection_name: Optional[str] = None,
    ) -> List[TrackHit]:
        """Convert Qdrant ScoredPoint list to TrackHit list."""
        facts_cache: Dict[str, str] = {}
        song_facts_cache: Dict[str, str] = {}

        if collection_name:
            # Determine whether AI features are enabled for this collection.
            # When enabled, refined facts (from AI Indexing) take precedence
            # over originals. When disabled, only original facts are shown.
            ai_enabled = True  # default — pre-AI-flag world had facts always on
            try:
                MetadataDB.init()
                ai_enabled = MetadataDB.get_collection_ai_enabled(collection_name)
            except Exception:
                pass  # graceful fallback to True

            facts_cache = load_all_facts_for_collection(
                collection_name, ai_enabled=ai_enabled,
            )
            song_facts_cache = load_all_song_facts_for_collection(
                collection_name, ai_enabled=ai_enabled,
            )

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

            # Lyrics for LLM context (first ~400 chars; not sent to frontend)
            raw_lyrics: str = payload.get("lyrics") or ""
            lyrics: str | None = None
            if raw_lyrics.strip():
                lyrics = raw_lyrics.replace("\n", " ").strip()

            # Facts from cache
            artist = payload.get("artist", "Unknown")
            artist_slug = _slugify_artist(artist)
            artist_facts = facts_cache.get(artist_slug)

            song_title = payload.get("title", "Unknown")
            song_key = get_song_facts_key(artist, song_title)
            song_facts = song_facts_cache.get(song_key)

            track = TrackMetadata(
                track_id=str(point.id),
                title=payload.get("title", "Unknown"),
                artist=artist,
                album=payload.get("album"),
                year=year,
                genre=payload.get("genre"),
                duration_sec=duration_sec,
                file_path=payload.get("file_path", ""),
                cover_art_path=payload.get("cover_art_path"),
                lyrics=raw_lyrics or None,
                producer=payload.get("producer"),
                label=payload.get("label"),
                samples=payload.get("samples"),
                sampled_by=payload.get("sampled_by"),
                bitrate_kbps=payload.get("bitrate_kbps"),
            )
            hits.append(TrackHit(
                track=track,
                score=float(point.score),
                matched_on=matched_on,
                lyrics=lyrics,
                artist_facts=artist_facts,
                song_facts=song_facts,
            ))
        return hits

    async def index_tracks_with_progress(
        self,
        tracks: List[TrackMetadata],
        collection_name: Optional[str] = None,
        progress_callback=None,
        text_model: Optional[str] = None,
    ) -> None:
        """Index tracks into Qdrant with progress callbacks.

        Args:
            tracks: List of tracks to index
            collection_name: Qdrant collection name
            progress_callback: Async callback(stage, current, total, message) for progress updates
            text_model: Override text embedding model for this indexing run.
                        Must already be loaded via ModelRegistry.load_text_model().
        """
        if not tracks:
            logger.warning("[SearchService] index_tracks_with_progress: no tracks to index")
            return

        logger.info("[SearchService] index_tracks_with_progress: %d tracks, collection=%s, text_model=%s",
                    len(tracks), collection_name, text_model)

        if text_model and text_model != self.lyrics_db.model_name:
            logger.info("[SearchService] Switching LyricsDB text model: %s -> %s",
                        self.lyrics_db.model_name, text_model)
            model, vector_name, vector_dim = ModelRegistry.load_text_model(text_model)
            self.lyrics_db.model_name = text_model
            self.lyrics_db._model = model
            self.lyrics_db._vector_name = vector_name
            self.lyrics_db._vector_dim = vector_dim

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
                "duration": int(track.duration_sec) if track.duration_sec else 0,
                "lyrics": track.lyrics or "",
                "file_path": track.file_path,
                "cover_art_path": track.cover_art_path,
                "producer": track.producer,
                "label": track.label,
                "samples": track.samples,
                "sampled_by": track.sampled_by,
            }

        # Report progress: lyrics encoding started
        if progress_callback:
            await progress_callback("lyrics", 0, len(tracks), "Кодирование текстов песен...")

        # Bridge async callback into sync context inside run_in_executor
        loop = asyncio.get_event_loop()
        def _sync_cb(stage, current, total, message):
            if progress_callback:
                asyncio.run_coroutine_threadsafe(
                    progress_callback(stage, current, total, message), loop
                )

        # Call fit_with_progress in executor (sync op, may take minutes)
        logger.info("[SearchService] Calling lyrics_db.fit_with_progress() in executor...")
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.lyrics_db.fit_with_progress(
                    data,
                    path=None,
                    collection_name=collection_name,
                    progress_callback=_sync_cb if progress_callback else None,
                ),
            )
            logger.info("[SearchService] lyrics_db.fit_with_progress() completed successfully")
        except Exception as e:
            logger.error("[SearchService] lyrics_db.fit_with_progress() failed: %s", e, exc_info=True)
            raise
