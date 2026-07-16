"""Pipeline entry point: raw facts -> GLiNER2 -> triage -> LLM -> DB write.

For each fact of a song: run the (lazy, injectable) GLiNER2 extractor, triage
its output (Task 2) into AS_IS / LLM buckets, resolve the LLM bucket with one
LLM call per fact (Task 3's mark/build/parse), merge AS_IS + LLM per fact, and
accumulate the merged result across all facts of the song. The final
producers/samples/sampled_by dict is written to ``songs.producers`` /
``songs.samples_json`` via ``db.set_song_relations`` and returned.

Both GLiNER2 and the LLM are optional in practice: a per-fact extractor
failure is logged and that fact is skipped (its AS_IS/LLM contribution is
just empty); a failing/unavailable LLM (``ask_llm_fn`` raises, or returns
None/unparsable output) degrades to AS_IS-only for that fact. Nothing here
raises out to the caller -- see the try/except hook in
``song_facts_service.py`` for the outermost safety net during indexing.
"""
import asyncio
import json
import logging

from .extractor import get_extractor
from .llm_re import build_llm_messages, mark_fact, merge_results, parse_llm_re
from .triage import triage_producer, triage_samples

logger = logging.getLogger(__name__)

_EMPTY = {"producers": [], "samples": [], "sampled_by": []}


def _clean(v):
    return v if v else None


def _as_is_dict(producer_hits, source_hits, usage_hits):
    producers = [name for bucket, name in producer_hits if bucket == "AS_IS"]
    samples = [{"song": _clean(song), "artist": _clean(artist)}
               for bucket, song, artist in source_hits if bucket == "AS_IS"]
    sampled_by = [{"song": _clean(song), "artist": _clean(artist)}
                  for bucket, song, artist in usage_hits if bucket == "AS_IS"]
    return {"producers": producers, "samples": samples, "sampled_by": sampled_by}


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

    Returns a parsed dict (Task 3 shape) or ``None`` -- never raises, so an
    unreachable/misbehaving LLM just means this fact's LLM bucket contributes
    nothing (AS_IS results still get written).
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


def process_song_facts(slug, facts, title, artist, db, ask_llm_fn, extractor=None):
    """Extract producers/samples from ``facts`` and persist them for ``slug``.

    Parameters
    ----------
    slug        : song slug (``songs.slug``).
    facts       : raw fact strings for this song (English).
    title       : subject song title (for triage/LLM subject matching).
    artist      : subject artist name or slug (LLM normalizes dashes).
    db          : object exposing ``set_song_relations(slug, producers_json,
                  samples_json)`` -- production passes the ``MetadataDB`` class.
    ask_llm_fn  : callable(messages) -> dict|str|None. Called once per fact
                  that has any LLM-bucket candidate. May raise or return None;
                  both degrade gracefully to AS_IS-only for that fact.
    extractor   : object exposing ``extract(fact) -> dict`` (GLiNER2 output
                  shape). Defaults to the lazy ``get_extractor()`` singleton;
                  tests inject a fake to avoid loading torch/gliner2.

    Returns the merged ``{"producers": [...], "samples": [...],
    "sampled_by": [...]}`` dict that was written to the DB.
    """
    extractor = extractor or get_extractor()
    final = dict(_EMPTY)

    for fact in facts or []:
        try:
            gliner_out = extractor.extract(fact)
        except Exception as e:
            logger.warning("[fact_relations] slug=%s extraction failed on a fact: %s", slug, e)
            continue

        producer_hits = triage_producer(fact, gliner_out, title)
        source_hits, usage_hits = triage_samples(fact, gliner_out, title, artist)

        as_is = _as_is_dict(producer_hits, source_hits, usage_hits)
        candidates = _llm_candidates(producer_hits, source_hits, usage_hits)

        llm_result = None
        if candidates:
            llm_result = _resolve_llm(fact, candidates, title, artist, ask_llm_fn)

        merged = merge_results(as_is, llm_result)
        final = merge_results(final, merged)

    db.set_song_relations(
        slug,
        json.dumps(final["producers"]),
        json.dumps({"samples": final["samples"], "sampled_by": final["sampled_by"]}),
    )
    return final


async def process_song_facts_async(slug, facts, title, artist, db, extractor=None):
    """Async wrapper around :func:`process_song_facts` for event-loop callers.

    The pipeline is synchronous and CPU-bound (GLiNER2), so it runs in
    ``asyncio.to_thread``. Its LLM leg needs a *sync* ``ask_llm_fn``; we bridge
    ``llm_client.ask_llm`` back onto the running loop via
    ``run_coroutine_threadsafe`` rather than spinning up a second event loop --
    that keeps ``llm_client``'s process-global async HTTP client on the single
    loop it was created on. Shared by the inline indexing hook
    (``song_facts_service``) and the ``fact_relations`` backfill ai_task.
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
        process_song_facts, slug, facts, title, artist, db, _ask, extractor,
    )
