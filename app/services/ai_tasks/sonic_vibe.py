"""Sonic Vibe task — one short fact-based line per track via LLM.

Reads a track's curated song facts (up to MAX_FACTS) + sonic_tags + year;
asks the LLM for the single most interesting fact as one line, or SKIP when
there is no usable fact. Only runs for tracks that HAVE facts. Persists the
line in the sonic_vibes table (PK track_id+collection+lang); a SKIP — or a
track with no facts — leaves the slot empty, so no vibe is shown.

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
MAX_FACTS = 10  # how many curated facts to show the model per track

_SYSTEM_PROMPT = """
You write ONE short line shown under a track in a music player. Its job is to tell the listener one true, concrete thing about THIS track that a fan would find interesting — a fact, not a mood. If you don't have such a fact, you output nothing.

WHAT YOU CAN AND CANNOT KNOW:
You have NOT heard this track. You only have text: a few sonic descriptor tags, a couple of curated facts, and the year. Therefore:
- You MAY state what the FACTS say (and the era, if it adds to a fact).
- You MAY NOT describe how the vocals sound, what instruments are playing, the "feel", a scene, or what the song is about — UNLESS it is explicitly in the facts. Inventing these is the main failure. No "icy vocals", "Japanese drums", "neon", "nightclub", "political tension" unless a fact literally states it.
- If you mention a sonic descriptor tag at all, copy it EXACTLY as given — same instrument, same technique, same word. NEVER swap a tag for a different-but-related one (e.g. a tag saying "throat singing" must never become "horns", "brass" must never become "strings") and NEVER invent a tag that isn't in the given list. When in doubt, leave the tag out rather than guess at it.

YOUR ONLY TWO OPTIONS:

1. There IS a usable fact. Pick the SINGLE best one by this priority — higher beats lower, always take the highest available:

   A. CREATION STORY — the non-obvious path the track took to exist: made in one night / hours before a flight, sat unreleased for years, was meant for a different album, started as something else (e.g. AI-written lines later rewritten), an accident or constraint that shaped it. These are the most interesting — a fan would retell them. Prefer these above all.
   B. CONCRETE PRODUCTION FACT — who produced it, a notable guest, an unusual instrument or recording method (only if the fact states it). NEVER build the line on a sample or interpolation — the player already shows sample credits separately; if the only production fact is "this samples X", treat it as unusable and look for another fact (or SKIP).
   C. EXTERNAL RESULT / CONTEXT — chart milestone, award, its role on the album, real-world reaction or controversy.

   → Write ONE line from the single highest-priority fact you have. Don't cram two.

   Do NOT pick a fact that just restates what the song is ABOUT or quotes its lyrics — the listener is hearing that right now. If the only "facts" are lyric content, go to option 2.

2. There is NO usable fact of type A–C — facts are missing, generic, biographical fluff, or only describe the song's lyrical content. → Output exactly: SKIP
   Output the word SKIP and nothing else. A line built only on tags, era, or what the song is about is NOT good enough — SKIP it. An empty slot is better than a generic line.

When unsure whether a fact is interesting enough, prefer SKIP. Never reach for invented atmosphere to fill the line.

STYLE (when you do write a line):
- INVENT NOTHING. Your line must say EXACTLY what the chosen fact says — same claim, same specifics, no added color. You may only translate it into {lang_name} and shorten it; you may NOT add, merge, reinterpret, exaggerate, or "improve" details. If the fact says "recorded in a hotel room", the line says a hotel room — not "a cramped Tokyo hotel room at 4am".
- Plain and clear, like a knowledgeable friend pointing something out — NOT a magazine pull-quote, NOT poetry. No purple adjectives, no invented atmosphere, no clichés.
- One line, max ~120 characters. No emoji, no quotation marks.
- NAMES: you may name a producer, featured guest, or album — those are the interesting facts. Do NOT repeat the main artist's name or the track title; the UI already shows them right next to your line.
- Any name you DO write (producer, guest, album) must appear EXACTLY as given in the input — character for character. NEVER translate, transliterate, localize, or grammatically decline a name into {lang_name}.

EXAMPLE (style and selection only — do NOT reuse this content):
Facts given:
  - Recorded in one night in a hotel room, hours before the artist had to fly out
  - Produced by a well-known beatmaker
  - The hook is about feeling watched by everyone
Correct output: Записан за одну ночь в отеле — за несколько часов до вылета
Why: the creation story (A) outranks the producer fact (B); the lyric-content fact is ignored because the listener is already hearing it.

Respond ONLY in {lang_name}; do not mix languages. (SKIP stays as the literal word SKIP.)
""".strip()

_LANG_NAMES = {"ru": "Russian", "en": "English"}


def _build_user_prompt(
    *, tags: list[str], payload: dict, facts: list[str], lang: str,
) -> str:
    year = payload.get("year")
    era = f"{(year // 10) * 10}s" if isinstance(year, int) and year > 0 else None
    lines = ["FACTS (pick the single best, or SKIP):"]
    for f in facts[:MAX_FACTS]:
        lines.append(f"- {f}")
    lines.append("")
    lines.append(f"Sonic descriptor tags: {', '.join(tags) if tags else '(none)'}")
    lines.append(f"Era: {era or '(unknown)'}")
    lines.append("")
    lines.append("Your one line, or SKIP:")
    return "\n".join(lines)


def _validate(phrase: str) -> str:
    """Trim, drop wrapping quotes, enforce length cap."""
    phrase = (phrase or "").strip().strip('"').strip("'")
    while len(phrase) > 1 and phrase[-1] == phrase[-2] and phrase[-1] in ".!?":
        phrase = phrase[:-1]
    if len(phrase) > MAX_PHRASE_CHARS:
        phrase = phrase[:MAX_PHRASE_CHARS].rstrip() + "…"
    return phrase


def _is_skip(raw: str) -> bool:
    """True when the model declined to write a line (literal SKIP) or said nothing.

    Tolerates wrapping quotes/punctuation and a trailing explanation; a real
    fact line is extremely unlikely to begin with the word "skip".
    """
    s = (raw or "").strip().strip('"').strip("'").strip()
    if not s:
        return True
    first = s.split()[0].rstrip(".,!:;—-").upper()
    return first == "SKIP"


async def run(job, db_client, llm) -> None:
    """Iterate the collection's tracks, generate a fact-based line for each.

    Skip rule: sonic vibes are FACT-BASED — a track with no curated song facts
    is skipped (no LLM call); tags alone are not enough. When facts exist the
    model may still answer SKIP (no fact worth showing), which leaves the slot
    empty — no vibe persisted. Also skip if a vibe is already cached for this
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

            if not facts:
                # Sonic vibes are fact-based now — no facts, no LLM call (tags
                # alone are not enough). Count as skipped (NOT done) so the UI
                # can tell the job finished without real work on this track.
                n_skipped += 1
                MetadataDB.update_ai_job(job_id=job.job_id, n_skipped=n_skipped)
                continue

            user = _build_user_prompt(tags=tags, payload=p, facts=facts, lang=lang)

            try:
                raw = await ask_llm(
                    user,
                    system_prompt=system,
                    temperature=0.7,
                    base_url=job.llm_base_url,
                    model=job.llm_model,
                )
            except Exception as e:
                logger.warning("[sonic_vibe] LLM error on %s: %s", tid, e)
                n_failed += 1
                MetadataDB.update_ai_job(job_id=job.job_id, n_failed=n_failed)
                continue

            if _is_skip(raw):
                # Model judged no fact worth showing — leave the slot empty.
                # Still "processed" (the LLM gave a verdict), so count as done.
                n_done += 1
                MetadataDB.update_ai_job(job_id=job.job_id, n_done=n_done)
                continue

            phrase = _validate(raw or "")
            if phrase:
                MetadataDB.set_sonic_vibe(tid, job.collection_name, lang, phrase)
                n_done += 1
            else:
                n_failed += 1
            MetadataDB.update_ai_job(
                job_id=job.job_id, n_done=n_done, n_failed=n_failed,
            )

        if offset is None:
            break

    if n_done == 0 and n_skipped > 0 and n_failed == 0:
        # The whole collection lacks the inputs sonic_vibe needs — surface
        # this as a job-level note so the UI can show why nothing happened.
        logger.warning(
            "[sonic_vibe] all %d tracks skipped — no song_facts in collection "
            "%s. Sonic vibes are fact-based; run facts indexing first.",
            n_skipped, job.collection_name,
        )


# Register at import time
ai_indexing_service.register_task("sonic_vibe", run)
