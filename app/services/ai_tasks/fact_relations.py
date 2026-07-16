"""Fact-relations backfill task — extract producers/samples from EXISTING facts.

The inline hook in ``song_facts_service`` runs the GLiNER2+LLM pipeline for
facts fetched from now on; this task backfills songs whose facts were fetched
before the pipeline existed. It walks
``MetadataDB.get_songs_needing_relations`` (songs visible to the collection
that have English facts but no ``producers``/``samples_json`` yet) and runs the
same pipeline per song, so the two paths share exactly one extraction/LLM code
path (``fact_relations.process_song_facts_async``).
"""
from __future__ import annotations

import logging

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service
from app.services.fact_relations import process_song_facts_async

logger = logging.getLogger(__name__)


async def run(job, db_client, llm) -> None:
    """Backfill producer/sample relations for songs in the job's collection."""
    songs = MetadataDB.get_songs_needing_relations(job.collection_name)

    n_done = 0
    n_failed = 0
    n_skipped = 0

    for slug, title, artist_slug in songs:
        try:
            facts = MetadataDB.get_song_facts(slug, job.collection_name)
        except Exception:
            facts = []

        if not facts:
            n_skipped += 1
        else:
            try:
                await process_song_facts_async(
                    slug, facts, title or "", artist_slug or "", MetadataDB,
                )
                n_done += 1
            except Exception as e:
                logger.warning("[fact_relations] backfill failed for %s: %s", slug, e)
                n_failed += 1

        MetadataDB.update_ai_job(
            job_id=job.job_id, n_done=n_done, n_failed=n_failed, n_skipped=n_skipped,
        )

    if n_done == 0 and n_skipped > 0 and n_failed == 0:
        logger.warning(
            "[fact_relations] all %d songs skipped — no facts in collection %s. "
            "Run facts indexing first.",
            n_skipped, job.collection_name,
        )


# Register at import time
ai_indexing_service.register_task("fact_relations", run)
