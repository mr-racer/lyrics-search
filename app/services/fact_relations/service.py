"""Pipeline entry point: raw facts -> GLiNER2 -> LLM classifier -> gates -> DB.

**Production uses the producer leg only.** Sampling links are owned by the
facts_v2 pipeline (``ai_tasks/refined_facts._store_links``), which labels a
fact before reading it instead of matching a lexicon, and cleans its output
through ``facts_v2.sample_links``. :func:`process_song_facts` therefore calls
:func:`collect_claims` with ``samples=False``. The sampling stages below still
exist for ``scripts/dry_run_relations.py``, which reports what this extractor
would have said; nothing writes them.

Four stages, in order of how much they cost:

1. **Lexical prefilter** (:mod:`gates`). A fact is read for samples only if it
   says in words that sound was taken. On the 2026-07-26 snapshot this alone
   removes the whole Adele/"Hello" family — *"the first line matches the
   melody of Lionel Richie's…"* — before any triage runs. It guards the sample
   leg only: producers are a separate feature and keep their old behaviour.
2. **GLiNER2 + triage** find candidate spans and split them into AS_IS
   (explicit wording) and LLM (needs a decision).
3. **The LLM classifies, it does not confirm.** It answers *what kind of link
   is this* from a closed list — sample, interpolation, cover, remix,
   lyrical_reference, inspiration, other. Asking a 12B model "should this be
   included?" was the old design and it drowned; asking "what is it?" is a
   question with a checkable answer.
4. **Code gates** (:func:`gates.apply_gates`) drop what is decidable without a
   model: self-references, same-album links, impossible release order,
   contradictory directions, duplicates, and anything past the per-song cap.

Claims accumulate across ALL of a song's facts and are gated once at the end —
the cap and the both-directions check are only meaningful over the whole set.

Both GLiNER2 and the LLM are optional in practice: a per-fact extractor
failure is logged and that fact is skipped; a failing/unavailable LLM degrades
to AS_IS-only for that fact. Nothing here raises out to the caller.
"""
import asyncio
import json
import logging

from .extractor import get_extractor
from .gates import fact_mentions_sampling
from .llm_re import build_llm_messages, mark_fact, merge_results, parse_llm_re
from .triage import triage_producer, triage_samples

logger = logging.getLogger(__name__)

_EMPTY = {"producers": [], "links": []}


def _clean(v):
    return v if v else None


def _as_is_dict(producer_hits, source_hits, usage_hits):
    """Triage's confident hits in the classifier's own shape.

    AS_IS means the fact used explicit sampling wording (``SOURCE_CUE`` /
    ``USAGE_CUE``), so the relation is already known — the LLM is not needed
    to label these, only to label the rest.
    """
    producers = [name for bucket, name in producer_hits if bucket == "AS_IS"]
    links = []
    for bucket, song, artist in source_hits:
        if bucket == "AS_IS":
            links.append({"song": _clean(song), "artist": _clean(artist),
                          "relation": "sample", "direction": "source"})
    for bucket, song, artist in usage_hits:
        if bucket == "AS_IS":
            links.append({"song": _clean(song), "artist": _clean(artist),
                          "relation": "sample", "direction": "usage"})
    return {"producers": producers, "links": links}


def _llm_candidates(producer_hits, source_hits, usage_hits):
    candidates = []
    for bucket, name in producer_hits:
        if bucket == "LLM" and name:
            candidates.append((name, "Person"))
    for bucket, song, artist in (*source_hits, *usage_hits):
        if bucket != "LLM":
            continue
        if song:
            candidates.append((song, "Song"))
        if artist:
            candidates.append((artist, "Artist"))
    return candidates


def _resolve_llm(fact, candidates, title, artist, ask_llm_fn):
    """Run the LLM leg for one fact's LLM-bucket candidates.

    Returns a parsed dict (classifier shape) or ``None`` -- never raises, so
    an unreachable/misbehaving LLM just means this fact's LLM bucket
    contributes nothing (AS_IS results still get written).
    """
    marked = mark_fact(fact, candidates)
    messages = build_llm_messages(title, artist, marked)
    try:
        raw = ask_llm_fn(messages)
    except Exception as e:
        logger.warning("[fact_relations] LLM call failed: %s", e)
        return None
    if raw is None:
        return None
    try:
        return parse_llm_re(raw)
    except Exception as e:
        logger.warning("[fact_relations] LLM reply parse failed: %s", e)
        return None


def collect_claims(facts, title, artist, ask_llm_fn, extractor=None, samples=True):
    """Every producer name and candidate link a song's facts yield, ungated.

    Split out from :func:`process_song_facts` so the dry-run script can see
    what the model produced *before* the gates ran — that is the "было ->
    стало" column of its report.

    ``samples=False`` skips the sampling leg entirely: no sample triage, and
    no Song/Artist candidates in the LLM call. That is what production passes
    now — sampling links are owned by the facts_v2 pipeline
    (``ai_tasks/refined_facts._store_links``), and running this leg too would
    both pay for a second opinion and overwrite the first one. The default
    stays True for ``scripts/dry_run_relations.py``, which reports on what
    this extractor WOULD say.
    """
    extractor = extractor or get_extractor()
    final = dict(_EMPTY)

    for fact in facts or []:
        try:
            gliner_out = extractor.extract(fact)
        except Exception as e:
            logger.warning("[fact_relations] extraction failed on a fact: %s", e)
            continue

        # The prefilter deliberately guards the SAMPLE leg only. Producers are
        # a separate feature with its own consumers (the credits drawer), and
        # silently changing what they see is not this change's business.
        producer_hits = triage_producer(fact, gliner_out, title)
        source_hits, usage_hits = (
            triage_samples(fact, gliner_out, title, artist)
            if samples and fact_mentions_sampling(fact) else ([], [])
        )

        as_is = _as_is_dict(producer_hits, source_hits, usage_hits)
        candidates = _llm_candidates(producer_hits, source_hits, usage_hits)

        llm_result = None
        if candidates:
            llm_result = _resolve_llm(fact, candidates, title, artist, ask_llm_fn)

        merged = merge_results(as_is, llm_result)
        for link in merged["links"]:
            link.setdefault("evidence", fact)
        final = merge_results(final, merged)

    return final


def process_song_facts(
    slug, facts, title, artist, db, ask_llm_fn, extractor=None, ctx=None,
):
    """Extract PRODUCER credits from ``facts`` and persist them for ``slug``.

    Sampling links used to come out of here as well, through the same GLiNER2
    triage plus a relation classifier. They no longer do. The facts_v2
    pipeline labels each fact before reading it and hands the sampling ones to
    a dedicated prompt, so running both meant two models answering the same
    question and the loser silently overwriting the winner: this function
    persisted with ``replace_sample_links``, which deletes a song's rows
    before inserting, and it ran AFTER ``refined_facts`` in the auto pipeline.
    Producers stay here — facts_v2 does not extract them.

    Parameters
    ----------
    slug        : song slug (``songs.slug``).
    facts       : raw fact strings for this song (English).
    title       : subject song title (for triage/LLM subject matching).
    artist      : subject artist name or slug (LLM normalizes dashes).
    db          : object exposing ``set_song_producers``.
    ask_llm_fn  : callable(messages) -> dict|str|None. Called once per fact
                  that has any LLM-bucket candidate. May raise or return None;
                  both degrade gracefully to AS_IS-only for that fact.
    extractor   : object exposing ``extract(fact) -> dict`` (GLiNER2 output
                  shape). Defaults to the lazy ``get_extractor()`` singleton;
                  tests inject a fake to avoid loading torch/gliner2.
    ctx         : :class:`gates.SongContext`, accepted for call compatibility.
                  Nothing here reads it any more — the gates it carried data
                  for guarded the sampling leg.

    Returns ``{"producers": [...]}`` — what was written to the DB.
    """
    claims = collect_claims(
        facts, title, artist, ask_llm_fn, extractor, samples=False,
    )
    producers = claims["producers"]
    # Producers only: writing samples_json from here would blank the sample
    # cache that facts_v2 fills for the same song.
    db.set_song_producers(slug, json.dumps(producers))
    return {"producers": producers}


async def process_song_facts_async(
    slug, facts, title, artist, db, extractor=None, ctx=None,
):
    """Async wrapper around :func:`process_song_facts` for event-loop callers.

    The pipeline is synchronous and CPU-bound (GLiNER2), so it runs in
    ``asyncio.to_thread``. Its LLM leg needs a *sync* ``ask_llm_fn``; we bridge
    ``llm_client.ask_llm`` back onto the running loop via
    ``run_coroutine_threadsafe`` rather than spinning up a second event loop --
    that keeps ``llm_client``'s process-global async HTTP client on the single
    loop it was created on.
    """
    from app.services.llm_client import ask_llm

    loop = asyncio.get_running_loop()

    def _ask(messages):
        system = next((m["content"] for m in messages if m.get("role") == "system"), None)
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        return asyncio.run_coroutine_threadsafe(
            ask_llm(user, system_prompt=system, parse_json=True, temperature=0), loop,
        ).result()

    return await asyncio.to_thread(
        process_song_facts, slug, facts, title, artist, db, _ask, extractor, ctx,
    )
