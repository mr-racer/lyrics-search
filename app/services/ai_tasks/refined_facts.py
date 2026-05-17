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
_FACTS_REFINE_PROMPT = """
You are a music editor and an expert in curating rare trivia. You will receive an array of 5 facts about a musician or a song.

YOUR TASK:
1. Evaluate each fact. Ignore the trivial ones (standard chart positions, release dates, ordinary awards, boring biographical info).
2. Select only the most obscure, paradoxical, or unusual facts (weird studio habits, strange incidents, hidden technical or lyrical details).
3. Condense the selected facts as much as possible without losing the core meaning. Make them punchy and concise.
4. Return the result STRICTLY as a valid JSON.

JSON FORMAT:
{{
  "selected_facts": [
    {{
      "reasoning": "Brief explanation (1-2 words) of why the fact is interesting",
      "short_fact": "Condensed, punchy, and interesting fact"
    }}
  ]
}}

RULES:
- If none of the 5 facts seem interesting, return an empty array: {{"selected_facts": []}}
- Do not add any explanations or text before or after the JSON.
- Output the "short_fact" values in {lang}.

PROCEED WITH THIS FACTS:
{facts}
""".strip()

# def _build_user_prompt(*, facts: list[str], lang: str) -> str:
#     facts_payload = [{"id": i + 1, "text": t} for i, t in enumerate(facts)]
#     return (
#         "Facts batch (JSON array, each item has id and text):\n"
#         + json.dumps(facts_payload, ensure_ascii=False)
#         + f"\n\nRespond in {lang} as a JSON array. Each item must include "
#           "\"id\" (int) and \"keep\" (bool). For items with keep=true include "
#           "\"refined_text\" (str, single sharp sentence)."
#     )


def _parse_llm_response(raw: str) -> list[dict[str, Any]]:
    """Strict JSON-array validation. Raises ValueError on any malformed input."""
    try:
        arr = json.loads(raw)
    except Exception as e:
        raise ValueError(f"LLM response is not JSON: {e}")
    if not isinstance(arr, dict):
        raise ValueError("LLM response is not a JSON dict")
    out: list[dict[str, Any]] = []
    facts = arr.get('selected_facts', None)
    if facts is None:
        raise ValueError("corrupted facts structure (no 'selected_facts' in dict)")
    for item in facts:
        if not isinstance(item, dict):
            raise ValueError("fact is not a valid dict")
        refined = item.get("short_fact")
        # if refined and len(refined) > MAX_REFINED_LEN:
        #     refined = refined[:MAX_REFINED_LEN].rstrip() + "…"
        if refined:
            out.append({
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
    # if len(facts) < 2:
    #     return (0, 0)

    kept: list[str] = []
    n_failures = 0

    for start in range(0, len(facts), REFINED_FACTS_BATCH_SIZE):
        batch = facts[start : start + REFINED_FACTS_BATCH_SIZE]
        facts_to_prompt = str()
        for fact in batch:
            facts_to_prompt += f"- {fact}\n"
        user = _FACTS_REFINE_PROMPT.format(facts=facts_to_prompt, lang=lang)
        try:
            raw = await ask_llm(
                user,
                temperature=0.3,
                base_url=llm_base_url,
                model=llm_model,
            )
            parsed = _parse_llm_response(raw or "")
            kept.extend(item["refined_text"] for item in parsed)
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
            p = pt.payload or {}
            # Compute slugs from artist+title — see sonic_vibe.run for the
            # rationale. Qdrant payload does not carry precomputed slugs.
            artist_name = (p.get("artist") or "").strip()
            title_text = (p.get("title") or "").strip()
            song_slug = get_song_facts_key(artist_name, title_text) if (artist_name and title_text) else ""
            artist_slug = _slugify_artist(artist_name) if artist_name else ""

            did_work = False

            # Song facts — keyed by song_slug so search_service can merge
            # refined facts into TrackHit.song_facts using the same slug
            # that load_all_song_facts_for_collection() uses as dict key.
            if song_slug:
                try:
                    song_facts = MetadataDB.get_song_facts(song_slug, job.collection_name)
                except Exception:
                    song_facts = []
                if song_facts:
                    _, fail = await _process_one_scope(
                        scope="song", scope_key=song_slug, facts=song_facts,
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
