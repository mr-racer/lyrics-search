"""Artist Bio task — one-paragraph bio per artist via web search + LLM.

For each distinct artist in the collection, runs a pydantic_ai agent loop
(search → evaluate → refine) via llm_web_search.web_research_bio and
persists the result to MetadataDB.

Cache key is (artist_slug, collection, lang).
"""

from __future__ import annotations

import logging

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service
from app.services.artist_split import (
    artist_slugs, name_for_slug,
)
from app.services.llm_web_search import web_research_bio

logger = logging.getLogger(__name__)


async def run(job, db_client, llm) -> None:
    """Iterate distinct artists in the collection; bio each one via web search."""
    qdrant = db_client.qdrant
    seen_artist_slugs: set[str] = set()
    n_done = 0
    n_failed = 0
    n_skipped = 0
    offset = None

    while True:
        points, offset = qdrant.scroll(
            collection_name=job.collection_name,
            limit=64,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for pt in points:
            p = pt.payload or {}

            # Prefer artist_slugs from payload (split at index time); fallback
            # to splitting the raw "artist" tag ourselves — uses artist_slugs()
            # which includes alias resolution, matching the indexing path.
            slugs = p.get("artist_slugs") or []
            if not slugs:
                raw = (p.get("artist") or "").strip()
                if raw:
                    slugs = artist_slugs(raw)

            for artist_slug in slugs:
                if not artist_slug or artist_slug in seen_artist_slugs:
                    continue
                seen_artist_slugs.add(artist_slug)

                # Display name for search/logging — prefer the canonical name
                # stored alongside this slug, fall back to the raw artist tag.
                artist_name = name_for_slug(p.get("artist"), artist_slug)
                if not artist_name:
                    artist_name = (p.get("artist") or "").strip() or artist_slug

                audiodb_data = MetadataDB.get_artist_audiodb(artist_slug, job.collection_name)
                seed_bio = (audiodb_data or {}).get("audiodb_bio")

                logger.info(
                    "[artist_bio] searching web for: %s (slug=%s, seed_bio=%s)",
                    artist_name, artist_slug,
                    "yes" if seed_bio else "no",
                )
                try:
                    bio = await web_research_bio(
                        artist_name=artist_name,
                        lang=job.lang,
                        base_url=job.llm_base_url,
                        model_name=job.llm_model,
                        seed_bio=seed_bio,
                    )
                except Exception as e:
                    logger.warning(
                        "[artist_bio] web search failed for %s: %s", artist_name, e,
                    )
                    n_failed += 1
                    MetadataDB.update_ai_job(job_id=job.job_id, n_failed=n_failed)
                    continue

                if not bio:
                    logger.warning("[artist_bio] empty result for %s", artist_name)
                    n_skipped += 1
                    MetadataDB.update_ai_job(job_id=job.job_id, n_skipped=n_skipped)
                    continue

                MetadataDB.set_artist_bio(artist_slug, job.collection_name, job.lang, bio)
                n_done += 1
                MetadataDB.update_ai_job(job_id=job.job_id, n_done=n_done)

        if offset is None:
            break

    logger.info(
        "[artist_bio] done: %d written, %d skipped, %d failed — collection=%s",
        n_done, n_skipped, n_failed, job.collection_name,
    )


# Register at import time
ai_indexing_service.register_task("artist_bio", run)
