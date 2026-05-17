"""Artist Bio task — one-paragraph bio per artist via LLM.

Iterates the collection once, building a set of distinct artists from the
Qdrant payload; for each artist with at least one fact in `artist_facts`,
calls the LLM with all their facts as input and persists the result.

Mirrors refined_facts.py in structure: empty _SYSTEM_PROMPT → RuntimeError
(operator must fill it). Cache key is (artist_slug, collection, lang).
"""

from __future__ import annotations

import logging

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service
from app.services.artist_facts_service import _slugify as _slugify_artist
from app.services.llm_client import ask_llm

logger = logging.getLogger(__name__)

# OPERATOR-FILLED. Empty by design so the operator can shape tone/length
# (e.g. "Write a 2-3 sentence biographical paragraph from the facts below.
# Lead with origin + genre, keep it journalistic, no clichés.").
_SYSTEM_PROMPT = ""


def _build_user_prompt(*, artist_name: str, facts: list[str], lang: str) -> str:
    body = "\n".join(f"- {f}" for f in facts)
    return (
        f"Artist: {artist_name}\n"
        f"Known facts:\n{body}\n\n"
        f"Write the bio in {lang}."
    )


async def run(job, db_client, llm) -> None:
    """Iterate distinct artists in the collection's tracks; bio each one once."""
    if not _SYSTEM_PROMPT.strip():
        raise RuntimeError(
            "artist_bio: _SYSTEM_PROMPT is empty — set it in "
            "app/services/ai_tasks/artist_bio.py before running this task."
        )

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
            artist_name = (p.get("artist") or "").strip()
            if not artist_name:
                continue
            artist_slug = _slugify_artist(artist_name)
            if artist_slug in seen_artist_slugs:
                continue
            seen_artist_slugs.add(artist_slug)

            try:
                facts = MetadataDB.get_artist_facts(artist_slug, job.collection_name)
            except Exception:
                facts = []
            if not facts:
                n_skipped += 1
                MetadataDB.update_ai_job(job_id=job.job_id, n_skipped=n_skipped)
                continue

            user_prompt = _build_user_prompt(
                artist_name=artist_name, facts=facts, lang=job.lang,
            )
            try:
                raw = await ask_llm(
                    user_prompt,
                    system_prompt=_SYSTEM_PROMPT,
                    temperature=0.4,
                    base_url=job.llm_base_url,
                    model=job.llm_model,
                )
            except Exception as e:
                logger.warning("[artist_bio] LLM error for %s: %s", artist_slug, e)
                n_failed += 1
                MetadataDB.update_ai_job(job_id=job.job_id, n_failed=n_failed)
                continue

            bio = (raw or "").strip()
            if bio:
                MetadataDB.set_artist_bio(artist_slug, job.collection_name, job.lang, bio)
                n_done += 1
            else:
                n_failed += 1
            MetadataDB.update_ai_job(
                job_id=job.job_id, n_done=n_done, n_failed=n_failed,
            )

        if offset is None:
            break

    if n_done == 0 and n_skipped > 0 and n_failed == 0:
        logger.warning(
            "[artist_bio] all %d artists skipped — no artist_facts in "
            "collection %s. Run facts indexing first.",
            n_skipped, job.collection_name,
        )


# Register at import time
ai_indexing_service.register_task("artist_bio", run)
