"""Producer-credits task — extract producers from a collection's facts.

This is the ONLY path that runs the GLiNER2+LLM relation pipeline. It used to
share the job with an inline per-song hook in ``song_facts_service``, but that
hook was invisible to the progress UI and impossible to count, so it was
removed; ``library_service._run_ai_tasks`` runs this task right after the FACTS
stage instead, with its own progress bar.

**Sampling links are not this task's business any more.** They come out of
``ai_tasks/refined_facts``, which owns ``sample_links`` and the
``songs.samples_json`` read cache built from it. This task used to extract them
too and, running later in the auto pipeline, overwrote what facts_v2 had just
written.

It walks ``MetadataDB.get_songs_needing_producers`` (songs visible to the
collection that have English facts but no ``producers`` yet) and runs
``fact_relations.process_song_facts_async`` per song. Being keyed on "no
producers yet" makes it idempotent: a re-run only picks up songs added since,
plus any whose facts arrived outside indexing.

The collection-wide context (library index + artist aliases) is built ONCE per
run and shared by every song, and is still handed to the pipeline for call
compatibility even though only the sampling gates ever read it.
"""
from __future__ import annotations

import asyncio
import logging

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service
from app.services.fact_relations import process_song_facts_async
from app.services.fact_relations.gates import SongContext
from app.services.fact_relations.library_index import LibraryIndex, artist_key

logger = logging.getLogger(__name__)


def _artist_keys(artist_slug: str, aliases: dict[str, str], members: dict[str, str]):
    """Every name the performer answers to, normalized.

    Aliases and band members both count: "Ye" naming Kanye and Bad Meets Evil
    naming Eminem are the performer talking about themselves, not a sample.
    """
    keys = {artist_key(artist_slug or "")}
    for name, slug in (*aliases.items(), *members.items()):
        if slug == artist_slug:
            keys.add(artist_key(name))
    return frozenset(k for k in keys if k)


def _build_context(collection_name: str):
    """``(library_index, aliases, members)`` shared by every song of the run."""
    tracks = MetadataDB.get_tracks_slim(collection_name)
    songs = MetadataDB.get_songs_for_collection(collection_name)
    index = LibraryIndex(tracks, songs)
    try:
        aliases = MetadataDB.get_artist_aliases_map()
        members = MetadataDB.get_artist_members_map()
    except Exception:
        logger.warning("[fact_relations] alias lookup failed — self-check on slug only",
                       exc_info=True)
        aliases, members = {}, {}
    return index, aliases, members


async def run(job, db_client, llm) -> None:
    """Extract producer credits for songs in the job's collection."""
    songs = MetadataDB.get_songs_needing_producers(job.collection_name)
    # The starter sizes every task by track count; ours is per SONG, so correct
    # the denominator before the progress bar renders it.
    MetadataDB.update_ai_job(job_id=job.job_id, n_total=len(songs))

    index, aliases, members = await asyncio.to_thread(_build_context, job.collection_name)

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
            own = index.resolve(title, (artist_slug or "").replace("-", " ")) or {}
            ctx = SongContext(
                slug=slug,
                title=title or "",
                artist=artist_slug or "",
                album=own.get("album") or "",
                year=own.get("year"),
                collection_name=job.collection_name,
                artist_keys=_artist_keys(artist_slug or "", aliases, members),
                library=index,
            )
            try:
                await process_song_facts_async(
                    slug, facts, title or "", artist_slug or "", MetadataDB, ctx=ctx,
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
