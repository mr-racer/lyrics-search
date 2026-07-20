"""Fact-relations task — extract producers/samples from a collection's facts.

This is the ONLY path that runs the GLiNER2+LLM relation pipeline. It used to
share the job with an inline per-song hook in ``song_facts_service``, but that
hook was invisible to the progress UI and impossible to count, so it was
removed; ``library_service._run_ai_tasks`` runs this task right after the FACTS
stage instead, with its own progress bar.

It walks ``MetadataDB.get_songs_needing_relations`` (songs visible to the
collection that have English facts but no ``producers``/``samples_json`` yet)
and runs ``fact_relations.process_song_facts_async`` per song. Being keyed on
"no relations yet" makes it idempotent: a re-run only picks up songs added
since, plus any whose facts arrived outside indexing.
"""
from __future__ import annotations

import logging

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service
from app.services.fact_relations import process_song_facts_async

logger = logging.getLogger(__name__)


async def run(job, db_client, llm) -> None:
    """Extract producer/sample relations for songs in the job's collection."""
    songs = MetadataDB.get_songs_needing_relations(job.collection_name)
    # The starter sizes every task by track count; ours is per SONG, so correct
    # the denominator before the progress bar renders it.
    MetadataDB.update_ai_job(job_id=job.job_id, n_total=len(songs))

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
