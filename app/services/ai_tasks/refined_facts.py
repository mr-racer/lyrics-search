"""Refined Facts task — batch-filter and shorten existing song/artist facts via LLM.

The system prompt is intentionally left empty — the operator fills it in
based on their preferred filtering criteria (skip uninteresting bio
chronology, keep unusual/specific facts, shorten everything to a sharp
sentence). The user message format and JSON response format are FIXED;
do not change them without updating _parse_llm_response.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service
from app.services.llm_client import ask_llm
from app.services.song_facts_service import get_song_facts_key
# Use the SAME slugify that artist_facts_service used when saving to DB,
# otherwise a name like "Guns N' Roses" resolves to a different slug here
# than in storage and we miss its facts.
from app.services.artist_facts_service import _slugify as _slugify_artist

logger = logging.getLogger(__name__)

REFINED_FACTS_BATCH_SIZE = 5  # tunable; valid range 3-5 per spec
MAX_REFINED_LEN = 200

# Default prompt — tune to taste. Was previously left literally empty, which
# caused jobs to silently report "done" with zero work performed.
_SYSTEM_PROMPT = (
    "You filter and shorten music facts. Drop generic or uninteresting items "
    "(birthdays, generic chart positions, common label info, well-known "
    "trivia). Keep the unusual, specific, or unexpectedly revealing facts. "
    "Shorten kept facts to one sharp sentence, preserving the concrete detail. "
    "Output strictly the JSON array described in the user message — no "
    "preamble, no explanation, no markdown fences."
)


def _build_user_prompt(*, facts: list[str], lang: str) -> str:
    facts_payload = [{"id": i + 1, "text": t} for i, t in enumerate(facts)]
    return (
        "Facts batch (JSON array, each item has id and text):\n"
        + json.dumps(facts_payload, ensure_ascii=False)
        + f"\n\nRespond in {lang} as a JSON array. Each item must include "
          "\"id\" (int) and \"keep\" (bool). For items with keep=true include "
          "\"refined_text\" (str, single sharp sentence)."
    )


def _parse_llm_response(raw: str) -> list[dict[str, Any]]:
    """Strict JSON-array validation. Raises ValueError on any malformed input."""
    try:
        arr = json.loads(raw)
    except Exception as e:
        raise ValueError(f"LLM response is not JSON: {e}")
    if not isinstance(arr, list):
        raise ValueError("LLM response is not a JSON array")
    out: list[dict[str, Any]] = []
    for item in arr:
        if not isinstance(item, dict) or "id" not in item or "keep" not in item:
            raise ValueError("each item must include id and keep")
        refined = item.get("refined_text", "")
        if item["keep"] and not refined:
            raise ValueError("keep=true requires refined_text")
        if refined and len(refined) > MAX_REFINED_LEN:
            refined = refined[:MAX_REFINED_LEN].rstrip() + "…"
        out.append({
            "id": int(item["id"]),
            "keep": bool(item["keep"]),
            "refined_text": refined,
        })
    return out


async def _process_one_scope(
    *, scope: str, scope_key: str, facts: list[str],
    collection_name: str, lang: str,
    llm_base_url: str | None = None, llm_model: str | None = None,
) -> tuple[int, int]:
    """Process one scope (song or artist) — batched LLM calls + persist.

    Returns (n_kept, n_batch_failures).
    """
    if len(facts) < 2:
        return (0, 0)

    if not _SYSTEM_PROMPT.strip():
        logger.warning(
            "[refined_facts] system prompt is empty — operator must set it. "
            "Skipping %s %s.", scope, scope_key,
        )
        return (0, 1)

    kept: list[str] = []
    n_failures = 0

    for start in range(0, len(facts), REFINED_FACTS_BATCH_SIZE):
        batch = facts[start : start + REFINED_FACTS_BATCH_SIZE]
        user = _build_user_prompt(facts=batch, lang=lang)
        try:
            raw = await ask_llm(
                user,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.3,
                base_url=llm_base_url,
                model=llm_model,
            )
            parsed = _parse_llm_response(raw or "")
            kept.extend(item["refined_text"] for item in parsed if item["keep"])
        except Exception as e:
            logger.warning(
                "[refined_facts] batch failed for %s %s: %s",
                scope, scope_key, e,
            )
            n_failures += 1

    MetadataDB.set_refined_facts(
        scope=scope, scope_key=scope_key,
        collection_name=collection_name, lang=lang,
        refined=kept,
    )
    return (len(kept), n_failures)


async def run(job, db_client, llm) -> None:
    """Iterate songs in the collection; for each, refine its song facts and
    (one pass per artist) its artist facts."""
    if not _SYSTEM_PROMPT.strip():
        # Fail fast and visibly instead of silently iterating with no work.
        raise RuntimeError(
            "refined_facts: _SYSTEM_PROMPT is empty — set it in "
            "app/services/ai_tasks/refined_facts.py before running this task."
        )

    qdrant = db_client.qdrant
    n_done = 0
    n_failed = 0
    n_skipped = 0
    offset = None
    seen_artist_slugs: set[str] = set()

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
            # Compute slugs from artist+title — see sonic_vibe.run for the
            # rationale. Qdrant payload does not carry precomputed slugs.
            artist_name = (p.get("artist") or "").strip()
            title_text = (p.get("title") or "").strip()
            song_slug = get_song_facts_key(artist_name, title_text) if (artist_name and title_text) else ""
            artist_slug = _slugify_artist(artist_name) if artist_name else ""

            did_work = False

            # Song facts.
            if song_slug:
                try:
                    song_facts = MetadataDB.get_song_facts(song_slug, job.collection_name)
                except Exception:
                    song_facts = []
                if song_facts:
                    _, fail = await _process_one_scope(
                        scope="song", scope_key=tid, facts=song_facts,
                        collection_name=job.collection_name, lang=job.lang,
                        llm_base_url=job.llm_base_url, llm_model=job.llm_model,
                    )
                    n_failed += fail
                    did_work = True

            # Artist facts — once per artist.
            if artist_slug and artist_slug not in seen_artist_slugs:
                seen_artist_slugs.add(artist_slug)
                try:
                    art_facts = MetadataDB.get_artist_facts(artist_slug, job.collection_name)
                except Exception:
                    art_facts = []
                if art_facts:
                    _, fail = await _process_one_scope(
                        scope="artist", scope_key=artist_slug, facts=art_facts,
                        collection_name=job.collection_name, lang=job.lang,
                        llm_base_url=job.llm_base_url, llm_model=job.llm_model,
                    )
                    n_failed += fail
                    did_work = True

            if did_work:
                n_done += 1
            else:
                # No facts (song or artist) — nothing to refine for this track.
                n_skipped += 1
            MetadataDB.update_ai_job(
                job_id=job.job_id,
                n_done=n_done, n_failed=n_failed, n_skipped=n_skipped,
            )

        if offset is None:
            break

    if n_done == 0 and n_skipped > 0 and n_failed == 0:
        logger.warning(
            "[refined_facts] all %d tracks skipped — no song_facts or "
            "artist_facts in collection %s. Run facts indexing first.",
            n_skipped, job.collection_name,
        )


# Register at import time
ai_indexing_service.register_task("refined_facts", run)
