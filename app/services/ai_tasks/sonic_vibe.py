"""Sonic Vibe task — one-sentence atmospheric description per track via LLM.

Reads sonic_tags (Plan 1 prompt-probing output) + Qdrant payload + cached
facts; calls the project's ask_llm helper (OpenAI-compatible); persists the
validated phrase in the sonic_vibes table (PK track_id+collection+lang).

The llm parameter in run() is accepted for framework compatibility but unused
at runtime — all LLM calls go through ask_llm() which resolves connection
config from environment variables.
"""

from __future__ import annotations

import json
import logging

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service
from app.services.llm_client import ask_llm
from app.services.song_facts_service import get_song_facts_key

logger = logging.getLogger(__name__)

MAX_PHRASE_CHARS = 160

_SYSTEM_PROMPT = (
    "You write one-sentence atmospheric descriptions of music tracks for a "
    "listening UI. The sentence must read like a serif-italic pull-quote in a "
    "music magazine: evocative, concrete, no clichés. Max ~120 characters. "
    "No emoji, no quotes, no track or artist names. "
    "Respond ONLY in {lang_name}; do not mix languages."
)

_LANG_NAMES = {"ru": "Russian", "en": "English"}


def _build_user_prompt(
    *, tags: list[str], payload: dict, facts: list[str], lang: str,
) -> str:
    decade = None
    year = payload.get("year")
    if isinstance(year, int) and year > 0:
        decade = f"{(year // 10) * 10}s"
    facts_compact = "; ".join(facts[:2]) if facts else ""
    return (
        f"Track sonic descriptors: {', '.join(tags)}\n"
        f"Genre/era hint: {decade or ''} {facts_compact}".strip()
        + "\nOne sentence atmospheric description:"
    )


def _validate(phrase: str) -> str:
    """Trim, drop wrapping quotes, enforce length cap."""
    phrase = (phrase or "").strip().strip('"').strip("'")
    while len(phrase) > 1 and phrase[-1] == phrase[-2] and phrase[-1] in ".!?":
        phrase = phrase[:-1]
    if len(phrase) > MAX_PHRASE_CHARS:
        phrase = phrase[:MAX_PHRASE_CHARS].rstrip() + "…"
    return phrase


async def run(job, db_client, llm) -> None:
    """Iterate the collection's tracks, generate vibes for those with inputs.

    Skip rule: if a track has neither sonic_tags_json nor any song facts,
    skip it (no LLM call). Also skip if a vibe is already cached for this
    (track, collection, lang).

    The ``llm`` parameter is accepted for framework compatibility but unused —
    the module-level ask_llm() is called directly so it can be patched in tests.
    """
    qdrant = db_client.qdrant
    lang = job.lang if job.lang in _LANG_NAMES else "en"
    system = _SYSTEM_PROMPT.format(lang_name=_LANG_NAMES[lang])

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
            tid = str(pt.id)
            p = pt.payload or {}

            # Cache hit counts as "done" — the track is already processed
            # from a previous run, even though no LLM call happened now.
            if MetadataDB.get_sonic_vibe(tid, job.collection_name, lang):
                n_done += 1
                MetadataDB.update_ai_job(job_id=job.job_id, n_done=n_done)
                continue

            # Gather inputs.
            tags: list[str] = []
            tags_raw = p.get("sonic_tags_json")
            if tags_raw:
                try:
                    tags = json.loads(tags_raw)
                except Exception:
                    tags = []

            # Resolve song-fact slug from artist+title in the Qdrant payload.
            # Indexing does NOT write a precomputed song_slug to the payload;
            # facts live in SQLite keyed by get_song_facts_key(artist, title)
            # (e.g. "dua-lipa-break-my-heart"). Reading p["song_slug"] used
            # to silently return None and pretend every track had no facts.
            facts: list[str] = []
            artist_name = (p.get("artist") or "").strip()
            title_text = (p.get("title") or "").strip()
            if artist_name and title_text:
                song_slug = get_song_facts_key(artist_name, title_text)
                try:
                    facts = MetadataDB.get_song_facts(song_slug, job.collection_name)
                except Exception:
                    facts = []

            if not tags and not facts:
                # Nothing to feed the model — count as skipped (NOT done) so
                # the UI can tell when a job completed without any real work.
                n_skipped += 1
                MetadataDB.update_ai_job(job_id=job.job_id, n_skipped=n_skipped)
                continue

            user = _build_user_prompt(tags=tags, payload=p, facts=facts, lang=lang)

            try:
                raw = await ask_llm(
                    user,
                    system_prompt=system,
                    temperature=0.7,
                )
            except Exception as e:
                logger.warning("[sonic_vibe] LLM error on %s: %s", tid, e)
                n_failed += 1
                MetadataDB.update_ai_job(job_id=job.job_id, n_failed=n_failed)
                continue

            phrase = _validate(raw or "")
            if phrase:
                MetadataDB.set_sonic_vibe(tid, job.collection_name, lang, phrase)
            else:
                n_failed += 1
            n_done += 1
            MetadataDB.update_ai_job(
                job_id=job.job_id, n_done=n_done, n_failed=n_failed,
            )

        if offset is None:
            break

    if n_done == 0 and n_skipped > 0 and n_failed == 0:
        # The whole collection lacks the inputs sonic_vibe needs — surface
        # this as a job-level note so the UI can show why nothing happened.
        logger.warning(
            "[sonic_vibe] all %d tracks skipped — no sonic_tags_json and no "
            "song_facts in collection %s. Run sonic-prompt-probing and/or "
            "facts indexing first.", n_skipped, job.collection_name,
        )


# Register at import time
ai_indexing_service.register_task("sonic_vibe", run)
