"""Concurrent indexing pipeline — overlap the lyrics fetch with GPU encoding.

The slow parts of an index are network-bound (online lyrics fetch) and GPU-bound
(CLAP + dense). They barely contend for resources, yet the legacy flow ran them
back-to-back. ``IndexPipeline.run`` overlaps them:

    network lane : online lyrics  (only tracks without embedded text)
    GPU lane     : CLAP (audio-only → starts immediately) → dense (after lyrics)

CLAP needs only the audio file, so it runs from t=0 in parallel with the fetch;
dense needs the lyrics, so it waits on ``lyrics_done``. CLAP and dense are
sequenced on a single-worker GPU executor — they can't share VRAM.

Facts / artist-bio / images are NOT in here: they need only ``artist``+``title``
(known before this runs), so the caller launches them as a third concurrent task
(see ``LibraryService``). This module stays focused on the encode→upsert path and
takes only the search engine.

The caller passes the canonical per-track ``filtered`` list implicitly via
``tracks`` (tag-read dicts keyed ``"Artist — Title"``); ``IndexingService.prepare``
turns it into the one list that CLAP, dense and upsert all key off, so fetched
lyrics written between the passes never cause (artist, title) key drift.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from app.indexing.folder_scanner import fetch_online_lyrics
from app.resources.metadata_db import MetadataDB
from app.resources.qdrant_utils import scroll_all

from .indexing_service import IndexingService

logger = logging.getLogger(__name__)

# Network-bound lyrics fetch — fan out like scan_and_enrich_folder. The lyrics
# APIs' rate limits are respected by each fetcher's internal per-request sleep.
_LYRICS_WORKERS = 8


class IndexPipeline:
    """Run CLAP/dense encoding concurrently with the online-lyrics fetch."""

    def __init__(self, engine, *, model_name: str | None = None):
        """``model_name`` — text embedding model for THIS batch (None → the
        engine's default). Passed through to a per-run IndexingService instead
        of being written onto the shared engine (two concurrent jobs used to
        race on ``engine.model_name`` / ``engine.collection_name``)."""
        self.engine = engine
        self.model_name = model_name

    async def run(
        self,
        tracks: dict[str, dict],
        collection_name: str,
        *,
        better_lyrics_quality: bool = False,
        progress: Optional[Callable] = None,
        resolve_track_ids: bool = False,
    ) -> tuple[int, dict[str, str]]:
        """Index a tag-read batch into ``collection_name``.

        Args:
            tracks: ``"Artist — Title" -> meta`` (tags already read; ``meta["lyrics"]``
                is the embedded text or empty when an online fetch is still needed).
            collection_name: target Qdrant collection (drop + recreate).
            better_lyrics_quality: pass-through to the lyrics fetcher (1 worker upstream).
            progress: thread-safe ``(stage, current, total, message, **kw)`` callback.
                Stages: ``"scan"`` (lyrics fetch), ``"audio"`` (CLAP), ``"lyrics"`` (dense).
            resolve_track_ids: when True, scroll the collection after upsert and
                return ``{"Artist — Title": point_id}`` (the upload flow needs it to
                stamp ``pending_uploads.track_id``).

        Returns ``(upserted_count, track_ids_by_key)``.
        """
        loop = asyncio.get_running_loop()
        # Per-run service: collection + model are instance-local, so parallel
        # jobs for different accounts can't cross-contaminate via the engine.
        indexing = IndexingService(
            self.engine, collection_name=collection_name, model_name=self.model_name,
        )

        filtered, clap_paths = await loop.run_in_executor(None, indexing.prepare, tracks)
        if not filtered:
            logger.info("[IndexPipeline] nothing to index for %s", collection_name)
            return 0, {}

        total = len(filtered)
        need_lyrics = [e for e in filtered if not (e.get("lyrics") or "").strip()]
        have = total - len(need_lyrics)
        logger.info(
            "[IndexPipeline] %d tracks: %d with embedded lyrics, %d need online fetch (%s)",
            total, have, len(need_lyrics), collection_name,
        )

        lyrics_done = asyncio.Event()
        # better_lyrics_quality enables Musixmatch upstream — serialize to 1 worker
        # to respect its rate limit (mirrors scan_and_enrich_folder).
        max_lyrics_workers = 1 if better_lyrics_quality else _LYRICS_WORKERS
        lyrics_pool = ThreadPoolExecutor(
            max_workers=min(max_lyrics_workers, max(1, len(need_lyrics))),
            thread_name_prefix="idx-lyrics",
        )
        gpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="idx-gpu")

        def _emit(stage, cur, tot, msg, **kw):
            if progress:
                try:
                    progress(stage, cur, tot, msg, **kw)
                except Exception:
                    logger.debug("[IndexPipeline] progress emit failed", exc_info=True)

        def _fetch_into(entry: dict):
            try:
                entry["lyrics"] = fetch_online_lyrics(entry, better_lyrics_quality)
            except Exception:
                logger.debug("[IndexPipeline] lyrics fetch failed for %s — %s",
                             entry.get("artist"), entry.get("title"), exc_info=True)
                entry["lyrics"] = entry.get("lyrics") or ""

        async def network_lane():
            # Tracks with embedded/Yandex lyrics skip the fetch — they're already
            # "done" toward the scan progress total.
            try:
                _emit("scan", have, total, f"Тексты из метаданных: {have}/{total}")
                done = have
                if need_lyrics:
                    futs = [loop.run_in_executor(lyrics_pool, _fetch_into, e) for e in need_lyrics]
                    for fut in asyncio.as_completed(futs):
                        await fut
                        done += 1
                        _emit("scan", done, total, f"[{done}/{total}] поиск текстов")
                _emit("scan", total, total, "Тексты готовы")
            finally:
                # Always release the GPU lane, even if a fetch raised — failed
                # fetches leave the existing (possibly empty) lyrics in place.
                lyrics_done.set()

        def _gpu_cb(stage, cur, tot, msg):
            _emit(stage, cur, tot, msg)

        async def gpu_lane():
            clap_map, chunks, axes = await loop.run_in_executor(
                gpu_pool, lambda: indexing.encode_clap(filtered, progress_callback=_gpu_cb),
            )
            await lyrics_done.wait()
            text_vecs = await loop.run_in_executor(
                gpu_pool, lambda: indexing.encode_dense(filtered, progress_callback=_gpu_cb),
            )
            return text_vecs, clap_map, chunks, axes

        try:
            (text_vecs, clap_map, chunks, axes), _ = await asyncio.gather(
                gpu_lane(), network_lane(),
            )
        finally:
            lyrics_pool.shutdown(wait=False)
            gpu_pool.shutdown(wait=False)

        def _create_and_upsert():
            # The IndexingService instance was constructed with this run's
            # collection_name — no shared-engine mutation, safe under parallel jobs.
            indexing.create_collection(clap_paths)
            try:
                MetadataDB.clear_track_metadata(str(collection_name))
            except Exception:
                logger.warning("[IndexPipeline] clear_track_metadata failed — non-fatal")
            indexing.upsert(
                filtered, text_vecs, clap_map or None, chunks or None,
                sonic_axes_map=axes or None,
            )
            indexing.finalize_norms_and_prune(axes or None)

        await loop.run_in_executor(None, _create_and_upsert)
        logger.info("[IndexPipeline] upserted %d tracks into %s", total, collection_name)

        track_ids: dict[str, str] = {}
        if resolve_track_ids:
            track_ids = await loop.run_in_executor(
                None, self._resolve_track_ids, collection_name, filtered,
            )
        return total, track_ids

    def _resolve_track_ids(self, collection_name: str, filtered: list[dict]) -> dict[str, str]:
        """Map ``"Artist — Title" -> point id`` (first match per key) post-upsert."""
        want = {
            f"{(e.get('artist') or '').strip()} — {(e.get('title') or '').strip()}"
            for e in filtered
        }
        out: dict[str, str] = {}
        try:
            for p in scroll_all(
                self.engine.qdrant_client, collection_name, batch_size=500,
                with_payload=["title", "artist"], with_vectors=False,
            ):
                pl = p.payload or {}
                k = f"{(pl.get('artist') or '').strip()} — {(pl.get('title') or '').strip()}"
                if k in want and k not in out:
                    out[k] = str(p.id)
        except Exception:
            logger.exception("[IndexPipeline] track-id resolve failed for %s", collection_name)
        return out
