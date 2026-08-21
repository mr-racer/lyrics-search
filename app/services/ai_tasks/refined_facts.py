"""Refined Facts task — batch-filter and shorten existing song/artist facts via LLM.

The system prompt is intentionally left empty — the operator fills it in
based on their preferred filtering criteria (skip uninteresting bio
chronology, keep unusual/specific facts, shorten everything to a sharp
sentence). The user message format and JSON response format are FIXED;
do not change them without updating _parse_llm_response.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any, Optional

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service
from app.services.llm_client import ask_llm
from app.services.song_facts_service import get_song_facts_key
# Split collaborations into per-participant canonical slugs, exactly as the
# indexing path and the artist_bio task do, so refined facts land under the
# same slug the artist page queries. (artist_slugs uses artist_facts_service's
# _slugify internally, so "Guns N' Roses" resolves to the same slug as in
# storage; it also alias-resolves and dedupes.)
from app.services.artist_split import artist_slugs

logger = logging.getLogger(__name__)

REFINED_FACTS_BATCH_SIZE = 5  # tunable; valid range 3-5 per spec
ANNOTATION_BATCH_SIZE = 6     # genius line-note classifier batches
MAX_REFINED_LEN = 200
MIN_FACT_CHARS = 30           # junk gate: shorter raw facts never reach the LLM
DEDUP_RATIO = 0.72            # difflib similarity above which two facts collapse

# Default prompt — tune to taste. Was previously left literally empty, which
# caused jobs to silently report "done" with zero work performed.
_FACTS_REFINE_PROMPT = """
You are a music editor and an expert in curating rare trivia. You will receive an array of 5 facts about a musician or a song.

YOUR TASK:
1. Evaluate each fact. Ignore the trivial ones (standard chart positions, release dates, ordinary awards, boring biographical info).
2. Select only the most obscure, paradoxical, or unusual facts (weird studio habits, strange incidents, hidden technical or lyrical details).
3. Lightly shorten each selected fact so it reads cleanly, but DO NOT lose key information — keep the names, places, numbers, and context that make it interesting. Aim for one natural, readable sentence, never a telegraphic fragment.
4. Return the result STRICTLY as a valid JSON.
{subject_clause}
JSON FORMAT:
{{
  "selected_facts": [
    {{
      "reasoning": "Brief explanation (1-2 words) of why the fact is interesting",
      "short_fact": "Trimmed but information-rich fact"
    }}
  ]
}}

RULES:
- If none of the 5 facts seem interesting, return an empty array: {{"selected_facts": []}}
- Do not add any explanations or text before or after the JSON.
- Do NOT wrap the JSON in markdown code blocks (no ```json or ``` markers).
- Output the "short_fact" values in {lang}.
- Any person or artist name in a fact (the subject, a producer, a featured guest, a sampled artist, a band member) must stay EXACTLY as given in the source fact — character for character. NEVER translate, transliterate, localize, or grammatically decline a name into {lang}.
- This applies to EVERY proper name: people, family members, brands, shows, places. A name spelled in Latin letters in the source must appear in your output in the SAME Latin letters — writing it in Cyrillic is an error.{artist_line}

PROCEED WITH THIS FACTS:
{facts}
""".strip()

# Injected into the prompt for SONG-scope batches only. The listener already
# sees which song is playing, so repeating its title in every fact is noise —
# tell the model to swap the title for a neutral "this song" reference.
_SONG_SUBJECT_CLAUSE = (
    "\nCONTEXT: All of these facts are about the song \"{title}\", which the "
    "listener is already looking at. Do NOT repeat the song's title in your "
    "output. Refer to the song in plain words woven naturally into the "
    "sentence — in English \"this song\" / \"it\", in Russian «эта песня», "
    "grammatically declined to fit the sentence (эта песня / этой песни / "
    "эту песню / в этой песне). NEVER put this reference in quotation marks "
    "and never treat it as a title — it is an ordinary noun phrase. Often "
    "the cleanest option is to drop the subject entirely (e.g. «Записана за "
    "одну ночь…», \"Was written for another album…\").\n"
)

# ── v2: genius line-note (annotation) classifier ─────────────────────────────
#
# 65% of raw song facts are per-line fan notes from Genius ("Lyrics string:
# <fragment>. Fact: <note>"). The generic refine prompt can't tell them from
# editorial facts, so meaning-retellings leak to the UI. This prompt classifies
# each note: KEEP only external facts/realia/creation stories, DROP retellings.
# Kept notes are rewritten in {lang} and MUST embed the lyric quote in «…».
_ANNOTATION_PROMPT = """
You are an editor for a music player. Below are fan-written notes on individual lyric lines of the song {artist} — {title} from Genius.com. Select only the notes that contain an INTERESTING EXTERNAL FACT, and rewrite each selected one in {lang}.

KEEP (keep=true) ONLY if the note contains at least one of:
- a reference to a specific person, work, film, book, brand, place, or a historical or cultural phenomenon;
- a verified fact about the song's creation or recording, or the artist's own words (interview, book, live show);
- wordplay or a hidden device with a precise mechanism (a homophone, a double meaning of a specific phrase).

DROP (keep=false):
- retellings of what the lines mean ("the narrator feels...", "the metaphor highlights...");
- generic musings about life, emotions, depression, relationships, partying;
- explanations of common words, slang, or well-known nicknames with no story behind them;
- for soundtrack songs: retellings of the film's plot (keep only facts about the song itself).

For every keep=true write "fact" in {lang}:
- ALWAYS include the lyric quote in «...» (may be shortened to 3-6 words);
- explain any realia so that a listener who knows no English and no context understands it;
- keep names of people, bands, works in LATIN exactly as in the source; never translate, transliterate or decline them — the artist's original spelling is "{artist}", copy it and every other Latin-spelled name character for character, never in Cyrillic;
- 1-2 sentences, keep every essential detail, invent NOTHING beyond the source;
- "confirmed": false if the note hedges (possibly, probably, could, may, might, PROPOSED SUGGESTION), else true.

Respond with STRICT JSON, no markdown, no text around it:
{{"items":[{{"id":"M1","keep":true,"confirmed":true,"fact":"..."}},{{"id":"M2","keep":false}}]}}

NOTES:
{items}
""".strip()

_ANNOTATION_RE = re.compile(r"^Lyrics string:\s*(.*?)\.?\s*Fact:\s*(.*)$", re.S)
_SECTION_QUOTE_RE = re.compile(r"^\[.*\]$")
# Latin capitalized tokens for the anti-hallucination entity check. Trailing
# punctuation is stripped separately — keeping '.' inside the class is needed
# for initialisms (D.M.C., N.W.A) but must not glue sentence periods on.
_LATIN_TOKEN_RE = re.compile(r"[A-Z][A-Za-z0-9'&.\-]{2,}")
_TOKEN_STRIP = ".,;:!?»«"


def _parse_annotation(fact: str) -> Optional[tuple[str, str]]:
    """Split a raw genius_annotation into (quote, note); None if malformed."""
    m = _ANNOTATION_RE.match(fact)
    if not m:
        return None
    quote = " ".join(m.group(1).split())
    note = " ".join(m.group(2).split())
    if _SECTION_QUOTE_RE.match(quote):
        # "[Chorus: …]" — a section marker, not a lyric line. The note may
        # still hold a real fact; it just can't carry a quote requirement.
        quote = ""
    return quote, note


def _junk_reason(text: str) -> Optional[str]:
    """Code-side junk gate (spec 1.1): empty/'?'/too-short never reach the LLM."""
    t = (text or "").strip()
    if not t or t in {"?", "??", "..."}:
        return "junk_empty"
    if len(t) < MIN_FACT_CHARS:
        return "junk_short"
    return None


def _latin_tokens(text: str) -> set[str]:
    return {
        tok.strip(_TOKEN_STRIP)
        for tok in _LATIN_TOKEN_RE.findall(text or "")
        if tok.strip(_TOKEN_STRIP)
    }


_LETTER_RUN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")
_CYR_RE = re.compile(r"[А-Яа-яЁё]")
_LAT_RE = re.compile(r"[A-Za-z]")


def _has_garbled_script(text: str) -> bool:
    """True when any single letter-run mixes Latin and Cyrillic («анаointment»).

    Hyphenated legit mixes ("Grammy-номинация") survive: the hyphen splits the
    runs, and each side is single-script.
    """
    for run in _LETTER_RUN_RE.findall(text or ""):
        if _CYR_RE.search(run) and _LAT_RE.search(run):
            return True
    return False


def _entities_ok(fact_text: str, source_text: str) -> bool:
    """Every Latin name in the LLM output must appear in the source batch.

    This is the code-side anti-hallucination gate: 0 invented names survived
    it in the 10-song production experiment. Comparison is case-insensitive
    substring — declensions don't apply to Latin names kept verbatim.
    """
    src = (source_text or "").lower()
    return all(tok.lower() in src for tok in _latin_tokens(fact_text))


def _dedupe_refined(items: list[dict]) -> list[dict]:
    """Collapse near-duplicates across the annotation/editorial streams.

    Two rules (spec 1.4): difflib ratio > DEDUP_RATIO keeps the longer text;
    a fact whose (names+numbers) entity set is a subset of another's — both
    non-empty — is absorbed by the richer fact. O(n²) is fine: n ≤ a few dozen.
    """
    def _ents(text: str) -> set[str]:
        return {t.lower() for t in _latin_tokens(text)} | set(re.findall(r"\d+", text))

    kept: list[dict] = []
    for cand in items:
        c_text = cand["text"].lower()
        c_ents = _ents(cand["text"])
        absorbed = False
        for i, existing in enumerate(kept):
            e_ents = _ents(existing["text"])
            similar = difflib.SequenceMatcher(
                None, c_text, existing["text"].lower(),
            ).ratio() > DEDUP_RATIO
            # Proper-subset containment only: equal sets are two different
            # facts about the same entities and must both survive.
            proper_subset = bool(c_ents and e_ents and (c_ents < e_ents or e_ents < c_ents))
            if not similar and not proper_subset:
                continue
            if proper_subset and not similar:
                winner = cand if e_ents < c_ents else existing
            else:
                winner = cand if len(cand["text"]) > len(existing["text"]) else existing
            kept[i] = winner
            absorbed = True
            break
        if not absorbed:
            kept.append(cand)
    return kept


def _parse_annotation_response(raw: str, valid_ids: set[str]) -> dict[str, dict]:
    """Parse the classifier JSON → {id: {keep, confirmed, fact}}. Raises ValueError."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError("annotation response is not JSON")
        data = json.loads(m.group(0))
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("annotation response missing 'items' list")
    out: dict[str, dict] = {}
    for it in data["items"]:
        if not isinstance(it, dict):
            continue
        iid = str(it.get("id", ""))
        if iid not in valid_ids:
            continue
        out[iid] = {
            "keep": bool(it.get("keep")),
            "confirmed": bool(it.get("confirmed", True)),
            "fact": (it.get("fact") or "").strip(),
        }
    return out


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


async def _refine_editorial_batches(
    *, facts: list[str], lang: str, subject_clause: str, subject_hint: str,
    scope: str, scope_key: str, artist_line: str = "",
    llm_base_url: str | None = None, llm_model: str | None = None,
) -> tuple[list[dict], int]:
    """Run editorial facts through the classic refine prompt, batch by batch.

    ``subject_hint`` (artist/title or slug words) extends the entity-check
    source: the model legitimately names the subject even when a single batch
    fact doesn't spell it out. A batch whose output fails the entity check is
    retried once; on the second failure only the clean facts survive.

    Returns (kept_items, n_batch_failures) — items are
    ``{"text", "confirmed": True, "src": "editorial"}``.
    """
    kept: list[dict] = []
    n_failures = 0

    for start in range(0, len(facts), REFINED_FACTS_BATCH_SIZE):
        batch = facts[start : start + REFINED_FACTS_BATCH_SIZE]
        facts_to_prompt = "".join(f"- {fact}\n" for fact in batch)
        user = _FACTS_REFINE_PROMPT.format(
            facts=facts_to_prompt, lang=lang, subject_clause=subject_clause,
            artist_line=artist_line,
        )
        src_text = " ".join(batch) + " " + subject_hint

        def _is_bad(t: str) -> bool:
            return not _entities_ok(t, src_text) or _has_garbled_script(t)

        try:
            for attempt in (0, 1):
                raw = await ask_llm(
                    user,
                    temperature=0.3,
                    base_url=llm_base_url,
                    model=llm_model,
                )
                parsed = _parse_llm_response(raw or "")
                texts = [item["refined_text"] for item in parsed]
                bad = [t for t in texts if _is_bad(t)]
                if not bad and attempt == 0:
                    break
                if attempt == 0:
                    continue  # retry the whole batch once
                if bad:
                    logger.warning(
                        "[refined_facts] %s %s: dropped %d fact(s) with names "
                        "absent from source or garbled script after retry",
                        scope, scope_key, len(bad),
                    )
                texts = [t for t in texts if not _is_bad(t)]
            kept.extend(
                {"text": t, "confirmed": True, "src": "editorial"} for t in texts
            )
        except Exception as e:
            logger.warning(
                "[refined_facts] batch failed for %s %s: %s",
                scope, scope_key, e,
            )
            n_failures += 1

    return kept, n_failures


async def _refine_annotations(
    *, annotations: list[tuple[str, str]], artist: str, title: str, lang: str,
    scope_key: str,
    llm_base_url: str | None = None, llm_model: str | None = None,
) -> tuple[list[dict], int]:
    """Classify genius line notes (KEEP/DROP) and rewrite the keepers.

    ``annotations`` is a list of pre-parsed ``(quote, note)`` pairs (already
    junk-filtered). Validation per kept item: Latin names must appear in that
    item's own quote+note (+artist/title), and when the item has a real lyric
    quote the rewritten fact must embed a «…» quote. One retry per batch,
    then offending items are dropped.

    Returns (kept_items, n_batch_failures) — items are
    ``{"text", "confirmed": bool, "src": "annotation"}``.
    """
    kept: list[dict] = []
    n_failures = 0

    for start in range(0, len(annotations), ANNOTATION_BATCH_SIZE):
        batch = annotations[start : start + ANNOTATION_BATCH_SIZE]
        ids = {f"M{i + 1}" for i in range(len(batch))}
        lines = [
            f'M{i + 1}. Line: "{q[:160]}" | Note: {n[:600]}'
            for i, (q, n) in enumerate(batch)
        ]
        user = _ANNOTATION_PROMPT.format(
            artist=artist or scope_key, title=title or "",
            lang=lang, items="\n".join(lines),
        )

        def _valid_items(parsed: dict[str, dict]) -> tuple[list[dict], int]:
            """Split parsed keeps into (accepted, n_rejected)."""
            good: list[dict] = []
            n_bad = 0
            for i, (quote, note) in enumerate(batch):
                it = parsed.get(f"M{i + 1}")
                if not it or not it["keep"] or not it["fact"]:
                    continue
                src = f"{quote} {note} {artist} {title}"
                quote_ok = ("«" in it["fact"]) if quote else True
                if (
                    quote_ok
                    and _entities_ok(it["fact"], src)
                    and not _has_garbled_script(it["fact"])
                ):
                    good.append({
                        "text": it["fact"],
                        "confirmed": it["confirmed"],
                        "src": "annotation",
                    })
                else:
                    n_bad += 1
            return good, n_bad

        try:
            raw = await ask_llm(
                user, temperature=0.2,
                base_url=llm_base_url, model=llm_model,
            )
            good, n_bad = _valid_items(_parse_annotation_response(raw or "", ids))
            if n_bad:
                # One retry of the whole batch, then keep the clean subset.
                raw = await ask_llm(
                    user, temperature=0.2,
                    base_url=llm_base_url, model=llm_model,
                )
                retry_good, retry_bad = _valid_items(
                    _parse_annotation_response(raw or "", ids),
                )
                if retry_bad < n_bad or len(retry_good) > len(good):
                    good, n_bad = retry_good, retry_bad
                if n_bad:
                    logger.warning(
                        "[refined_facts] %s: %d annotation fact(s) failed "
                        "quote/entity validation after retry — dropped",
                        scope_key, n_bad,
                    )
            kept.extend(good)
        except Exception as e:
            logger.warning(
                "[refined_facts] annotation batch failed for %s: %s",
                scope_key, e,
            )
            n_failures += 1

    return kept, n_failures


async def _process_one_scope(
    *, scope: str, scope_key: str, facts: list[str],
    collection_name: str, lang: str,
    subject_title: str | None = None,
    llm_base_url: str | None = None, llm_model: str | None = None,
) -> tuple[int, int]:
    """Process one editorial-only scope (artist, or legacy callers) + persist.

    ``subject_title`` is the song title for song-scope batches (used to tell
    the model to strip the redundant title from each fact). It is None for
    artist scope, where there is no single subject title.

    Returns (n_kept, n_batch_failures).
    """
    subject_clause = (
        _SONG_SUBJECT_CLAUSE.format(title=subject_title)
        if subject_title else ""
    )
    facts = [f for f in facts if _junk_reason(f) is None]

    kept, n_failures = await _refine_editorial_batches(
        facts=facts, lang=lang, subject_clause=subject_clause,
        subject_hint=(subject_title or "") + " " + scope_key.replace("-", " "),
        scope=scope, scope_key=scope_key,
        artist_line=(
            f'\n- The subject\'s original name is "{subject_title}" — use this exact spelling.'
            if subject_title else ""
        ),
        llm_base_url=llm_base_url, llm_model=llm_model,
    )
    kept = _dedupe_refined(kept)

    MetadataDB.set_refined_facts(
        scope=scope, scope_key=scope_key,
        collection_name=collection_name, lang=lang,
        refined=kept,
    )
    return (len(kept), n_failures)


async def _process_song_scope(
    *, scope_key: str, meta_facts: list[dict], artist: str, title: str,
    collection_name: str, lang: str,
    llm_base_url: str | None = None, llm_model: str | None = None,
) -> tuple[int, int]:
    """v2 song pipeline: split raw facts by category, refine both streams,
    dedupe across them and persist one combined refined list.

    ``meta_facts`` comes from ``get_song_facts_with_meta`` —
    ``[{"fact", "category"}, ...]``. Returns (n_kept, n_batch_failures).
    """
    editorial: list[str] = []
    annotations: list[tuple[str, str]] = []
    for mf in meta_facts:
        fact = mf.get("fact") or ""
        if mf.get("category") == "genius_annotation":
            parsed = _parse_annotation(fact)
            if parsed is None:
                continue
            quote, note = parsed
            if _junk_reason(note) is None:
                annotations.append((quote, note))
        else:
            if _junk_reason(fact) is None:
                editorial.append(fact)

    subject_clause = _SONG_SUBJECT_CLAUSE.format(title=title) if title else ""

    kept_editorial, fail_e = await _refine_editorial_batches(
        facts=editorial, lang=lang, subject_clause=subject_clause,
        subject_hint=f"{artist} {title}",
        scope="song", scope_key=scope_key,
        artist_line=(
            f'\n- The artist\'s original name is "{artist}" and the song is '
            f'"{title}" — use these exact Latin spellings for them and for '
            "any related names."
            if artist else ""
        ),
        llm_base_url=llm_base_url, llm_model=llm_model,
    )
    kept_annotations, fail_a = await _refine_annotations(
        annotations=annotations, artist=artist, title=title, lang=lang,
        scope_key=scope_key,
        llm_base_url=llm_base_url, llm_model=llm_model,
    )

    combined = _dedupe_refined(kept_editorial + kept_annotations)

    MetadataDB.set_refined_facts(
        scope="song", scope_key=scope_key,
        collection_name=collection_name, lang=lang,
        refined=combined,
    )
    return (len(combined), fail_e + fail_a)


async def run(job, db_client, llm) -> None:
    """Iterate songs in the collection; for each, refine its song facts and
    (one pass per artist) its artist facts.

    When ``job.new_track_ids`` is set (auto run after an append/upload) only
    those tracks' song facts — and the participant slugs of exactly those
    tracks — are processed; the rest were enriched by an earlier run.
    """

    qdrant = db_client.qdrant
    n_done = 0
    n_failed = 0
    n_skipped = 0
    seen_artist_slugs: set[str] = set()

    if job.new_track_ids:
        # Auto run: a small batch — one batched retrieve instead of a
        # whole-collection scroll (the scroll loop below is the legacy
        # manual path).
        points = qdrant.retrieve(
            collection_name=job.collection_name,
            ids=list(job.new_track_ids),
            with_payload=True,
            with_vectors=False,
        )
        batches = [points] if points else []
    else:
        batches = []
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
            batches.append(points)
            if offset is None:
                break

    for points in batches:
        for pt in points:
            p = pt.payload or {}
            # Compute slugs from artist+title — see sonic_vibe.run for the
            # rationale. Qdrant payload does not carry precomputed slugs.
            artist_name = (p.get("artist") or "").strip()
            title_text = (p.get("title") or "").strip()
            song_slug = get_song_facts_key(artist_name, title_text) if (artist_name and title_text) else ""
            # Per-participant canonical slugs. Prefer the slugs computed at index
            # time (payload); fall back to splitting the raw tag ourselves. A
            # collaboration "Calvin Harris, Dua Lipa" yields two slugs so each
            # artist's facts are refined under their OWN page slug.
            participant_slugs = (
                p.get("artist_slugs")
                or (artist_slugs(artist_name) if artist_name else [])
            )

            handled = False   # at least one scope was processed or cached

            # Song facts — keyed by song_slug so search_service can merge
            # refined facts into TrackHit.song_facts using the same slug
            # that load_all_song_facts_for_collection() uses as dict key.
            if song_slug:
                # Skip if refined facts already cached for this song+lang.
                # None means "never run"; [] means "AI ran, judged nothing interesting".
                existing = MetadataDB.get_refined_facts(
                    scope="song", scope_key=song_slug,
                    collection_name=job.collection_name, lang=job.lang,
                )
                if existing is None:
                    try:
                        song_facts = MetadataDB.get_song_facts_with_meta(
                            song_slug, job.collection_name,
                        )
                    except Exception:
                        song_facts = []
                    if song_facts:
                        _, fail = await _process_song_scope(
                            scope_key=song_slug, meta_facts=song_facts,
                            artist=artist_name, title=title_text,
                            collection_name=job.collection_name, lang=job.lang,
                            llm_base_url=job.llm_base_url, llm_model=job.llm_model,
                        )
                        n_failed += fail
                        n_done += len(song_facts)
                        handled = True
                else:
                    # Cached — count the raw facts that were already processed
                    try:
                        song_facts = MetadataDB.get_song_facts(song_slug, job.collection_name)
                    except Exception:
                        song_facts = []
                    n_done += len(song_facts)
                    handled = True

            # Artist facts — once per participant (collaborations split above).
            for artist_slug in participant_slugs:
                if not artist_slug or artist_slug in seen_artist_slugs:
                    continue
                seen_artist_slugs.add(artist_slug)
                # Skip if refined facts already cached for this artist+lang.
                existing = MetadataDB.get_refined_facts(
                    scope="artist", scope_key=artist_slug,
                    collection_name=job.collection_name, lang=job.lang,
                )
                if existing is None:
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
                        n_done += len(art_facts)
                        handled = True
                else:
                    # Cached — count the raw facts that were already processed
                    try:
                        art_facts = MetadataDB.get_artist_facts(artist_slug, job.collection_name)
                    except Exception:
                        art_facts = []
                    n_done += len(art_facts)
                    handled = True

            # Per-track accounting: n_done now tracks facts, not tracks.
            if not handled and (song_slug or participant_slugs):
                # Track has at least one slug but no scope had facts.
                n_skipped += 1
            # No slug at all — uncounted, doesn't affect progress.

            MetadataDB.update_ai_job(
                job_id=job.job_id,
                n_done=n_done, n_failed=n_failed, n_skipped=n_skipped,
            )

    if n_done == 0 and n_skipped > 0 and n_failed == 0:
        logger.warning(
            "[refined_facts] all %d tracks skipped — no song_facts or "
            "artist_facts in collection %s. Run facts indexing first.",
            n_skipped, job.collection_name,
        )


# Register at import time
ai_indexing_service.register_task("refined_facts", run)
