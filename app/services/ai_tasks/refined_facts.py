"""Refined Facts task — classify every raw fact, then rewrite the keepers.

This module is the job runner: walk the collection, feed each entity to
``services/facts_v2``, persist per fact, report progress. The processing itself
lives in that package.

What changed from the batch-of-five version, and why it is worth the extra
calls (0.68 s/fact against 0.55, measured):

* **Facts are labelled, not merely selected.** The frontend can group and hide,
  the sample stage runs on the facts that mention a sample instead of on all
  58k, and "dropped" stops being invisible.
* **One fact per rewrite, with the prompt chosen by label.** An award keeps
  "won or merely nominated"; a video keeps its director; a sample is not
  rewritten at all, it becomes a row in ``sample_links``.
* **Results are stored per raw fact.** A run over the whole corpus takes about
  ten hours and will be interrupted; it now resumes instead of restarting.

Compatibility kept on purpose: ``sonic_vibe`` imports four helpers from here,
and the unit tests patch ``refined_facts.ask_llm``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service, text_quality as tq
from app.services.artist_split import artist_slugs, name_for_slug
from app.services.facts_v2 import pipeline as fv2
from app.services.facts_v2 import sample_links as sl
from app.services.facts_v2.verify_lane import (
    VerifyLane, clean_and_store, seed_collection,
)
from app.services.llm_client import ask_llm
from app.services.song_facts_service import get_song_facts_key

logger = logging.getLogger(__name__)

MAX_REFINED_LEN = 200          # kept for callers that still read it


# ── compatibility shims ──────────────────────────────────────────────────────
# sonic_vibe imports these four. They now delegate, so there is one definition
# of each rule rather than two that drift apart.

def _junk_reason(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t or t in {"?", "??", "..."}:
        return "junk_empty"
    if len(t) < fv2.MIN_FACT_CHARS:
        return "junk_short"
    return None


def _parse_annotation(fact: str):
    """Split a raw genius_annotation into (quote, note); None if malformed."""
    parsed = fv2.split_annotation(fact)
    if parsed is None:
        return None
    quote, note = parsed
    if re.match(r"^\[.*\]$", quote):
        quote = ""            # "[Chorus: …]" is a section marker, not a line
    return quote, note


def _entities_ok(fact_text: str, source_text: str) -> bool:
    return not tq.invented_names(fact_text, source_text)


def _has_garbled_script(text: str) -> bool:
    return tq.garbled_script(text)


# ── the runner ───────────────────────────────────────────────────────────────

def _asker(job):
    """`ask(prompt, temperature) -> str` bound to this job's model.

    Indirection on purpose: it is the seam the pipeline is tested through, and
    it keeps ``ask_llm`` patchable by name in this module, which three unit
    tests rely on.
    """
    async def ask(prompt: str, temperature: float = 0.3) -> str:
        return await ask_llm(prompt, temperature=temperature,
                             base_url=job.llm_base_url, model=job.llm_model)
    return ask


def _persist(job, scope: str, scope_key: str, origin_kind: str,
             artist_name: str = ""):
    """Callback that writes one finished fact, as soon as it is finished.

    A fact is always stored under the entity it was READ from. An earlier
    version relocated a fact to the other page when the model said it belonged
    there, and that is gone — see ``facts_v2.pipeline.route`` for the measured
    damage. ``artist_name`` stays in the signature because the roster of
    callers passes it and the sample-link writer alongside this one needs it.
    """
    def store(rec: dict) -> None:
        fact = rec["fact"]
        labels = [x for x in rec.get("labels", []) if not x.startswith("gate:")]
        if not labels:
            labels = rec.get("labels", [])       # keep the gate reason as the label
        text = rec.get("refined") or None

        MetadataDB.set_refined_fact_item(
            scope=scope, scope_key=scope_key, lang=job.lang,
            origin_kind=origin_kind, origin_id=int(fact["id"]),
            labels=labels, text=text,
            confirmed=not rec.get("unconfirmed", False),
            src=("annotation" if fact.get("category") == "genius_annotation"
                 else "editorial"),
            collection_name=job.collection_name,
        )
    return store


async def _do_entity(job, ask, *, scope: str, scope_key: str, entity: dict,
                     origin_kind: str, facts: list,
                     links_ctx: "LinkContext | None" = None) -> tuple:
    """Process one song or artist. Returns (n_facts, n_failed)."""
    ids = [int(f["id"]) for f in facts]
    done = MetadataDB.processed_origin_ids(origin_kind, job.lang, ids)
    todo = [f for f in facts if int(f["id"]) not in done]
    if not todo:
        return len(facts), 0

    artist_name = entity.get("artist") or entity.get("name") or ""
    recs = await fv2.process_entity(
        ask, entity, scope, todo, lang_name=_LANG_NAME.get(job.lang, "Russian"),
        lang_code=job.lang,
        on_result=_persist(job, scope, scope_key, origin_kind, artist_name))
    _store_links(job, scope, scope_key, entity, recs, links_ctx)
    return len(facts), sum(1 for r in recs if r.get("error"))


_LANG_NAME = {"ru": "Russian", "en": "English"}


class LinkContext:
    """Per-run state the sampling-link writer needs.

    The library index behind ``resolve`` is read once for the whole run. It was
    tempting to build it inside the writer, but the writer runs once per song:
    on a 6,000-track library that is 6,000 full reads of the song table.
    """

    def __init__(self, collection_name: str, lane=None):
        try:
            self.resolve, self.n_index = sl.library_resolver_from_db(collection_name)
        except Exception:                           # noqa: BLE001
            logger.warning("[refined_facts] library index unavailable — links "
                           "will not resolve to owned tracks", exc_info=True)
            self.resolve, self.n_index = None, 0
        self.lane = lane


def _store_links(job, scope: str, scope_key: str, entity: dict, recs: list,
                 ctx: "LinkContext | None" = None) -> None:
    """Clean this song's sampling links and write them.

    What the model returns is a raw pair of names. Before this ran through
    ``sample_links.clean`` the raw pair went straight into the table, and
    production showed exactly what that costs: an album stored as a song
    ("Glass Animals — Dreamland"), the same link under two spellings ("Billy
    Squier — The Big Beat" beside "Billy Squire — Big Beat"), and ``dst_slug``
    null on every row — which left the derived "sampled by" side, built from
    that column, empty for the whole library.

    ``clean`` does the free tiers only (shape, evidence against the fact text,
    dedupe, fuzzy merge, and resolution against the user's own files). The one
    tier that costs a network round trip is MusicBrainz, and it happens in
    :class:`facts_v2.verify_lane.VerifyLane` while this keeps extracting.

    Links are kept unless ``clean`` rejects them outright: an unchecked link is
    not a wrong one, and the lane prunes what MusicBrainz disowns.

    ``replace_sample_links`` is a delete-then-insert per source song, so this
    collects every link of the entity and writes once — calling it per link
    would leave only the last one.
    """
    if scope != "song":
        # An artist fact can carry a sampling label too, but sample_links is
        # keyed by SONG slug: rows written under an artist slug match nothing
        # the readers ask for and never reach the cache rebuild.
        return

    src_artist = entity.get("artist") or entity.get("name") or ""
    src_title = entity.get("title") or ""
    rows, seen = [], set()
    for rec in recs:
        fact_text = (rec.get("fact") or {}).get("fact") or ""
        for link in (rec.get("links") or []):
            artist = (link.get("artist") or "").strip()
            title = (link.get("title") or "").strip()
            if not artist or not title:
                continue
            direction = link.get("direction") or "source"
            key = (direction, sl.db_key(artist, title))
            if key in seen:
                continue                    # one fact can restate another's link
            seen.add(key)
            rows.append({
                "artist": artist, "title": title, "direction": direction,
                "relation": link.get("relation") or "sample",
                "src_slug": scope_key, "src_artist": src_artist,
                "src_title": src_title, "fact": fact_text,
            })
    if not rows:
        return

    try:
        clean_and_store(
            job.collection_name, scope_key, rows,
            resolve=getattr(ctx, "resolve", None),
            lane=getattr(ctx, "lane", None),
        )
    except Exception:                               # noqa: BLE001
        logger.warning("[refined_facts] sample link store failed for %s",
                       scope_key, exc_info=True)


async def run(job, db_client, llm) -> None:
    """Iterate songs in the collection; refine song facts and artist facts.

    When ``job.new_track_ids`` is set (the automatic run after an append or an
    upload) only those tracks — and the participant slugs of exactly those
    tracks — are processed. Everything else was enriched by an earlier run, and
    re-walking the whole library to add three songs is hours of work for
    nothing.
    """
    qdrant = db_client.qdrant
    ask = _asker(job)
    n_done = n_failed = n_skipped = 0
    seen_artist_slugs: set = set()

    # The verification lane starts with the extraction rather than after it:
    # MusicBrainz is paced at about a call a second and the network is idle
    # while the LLM works, so the checking is paid for out of time this run
    # spends anyway. Building the library index touches the whole song table —
    # off the event loop.
    lane = VerifyLane(job.collection_name)
    lane.start()
    links_ctx = await asyncio.to_thread(LinkContext, job.collection_name, lane)

    # Links already in the table go through the same cleaning and checking.
    # A resumed run returns before the writer for every fact it has already
    # processed, so without this the only links ever verified would be the
    # ones extracted in the very same run — and a library indexed before
    # verification existed could never be healed by re-running the task.
    seeded = await asyncio.to_thread(
        seed_collection, job.collection_name, resolve=links_ctx.resolve)
    for src_slug, links in seeded:
        lane.submit(src_slug, links)
    if seeded:
        logger.info("[refined_facts] %d song(s) with stored links queued for "
                    "verification", len(seeded))

    if job.new_track_ids is not None:
        # A small batch: one retrieve instead of a whole-collection scroll.
        points = qdrant.retrieve(
            collection_name=job.collection_name,
            ids=list(job.new_track_ids), with_payload=True, with_vectors=False,
        )
        batches = [points] if points else []
    else:
        def _scroll_sync() -> list:
            """Обойти коллекцию, не занимая event loop.

            Клиент Qdrant синхронный, а `run` живёт корутиной на главном цикле:
            обход библиотеки — это под сотню round-trip'ов подряд, и всё это
            время uvicorn не отвечает НИ НА ОДИН запрос. Снаружи выглядит как
            «сервис завис на обращениях к Qdrant» — так оно и есть.

            Заодно проекция payload: `with_payload=True` тянул на каждый трек
            весь текст песни, который здесь не нужен ни разу.
            """
            out: list = []
            offset = None
            while True:
                points, offset = qdrant.scroll(
                    collection_name=job.collection_name, limit=64, offset=offset,
                    with_payload=["artist", "title", "artist_slugs"],
                    with_vectors=False,
                )
                if not points:
                    break
                out.append(points)
                if offset is None:
                    break
            return out

        batches = await asyncio.to_thread(_scroll_sync)

    for points in batches:
        for pt in points:
            p = pt.payload or {}
            artist_name = (p.get("artist") or "").strip()
            title_text = (p.get("title") or "").strip()
            song_slug = (get_song_facts_key(artist_name, title_text)
                         if artist_name and title_text else "")
            participant_slugs = (p.get("artist_slugs")
                                 or (artist_slugs(artist_name) if artist_name else []))
            handled = False

            if song_slug:
                try:
                    facts = MetadataDB.get_song_facts_for_refine(
                        song_slug, job.collection_name)
                except Exception:                   # noqa: BLE001
                    facts = []
                if facts:
                    got, failed = await _do_entity(
                        job, ask, scope="song", scope_key=song_slug,
                        entity={"slug": song_slug, "artist": artist_name,
                                "title": title_text},
                        origin_kind="song_facts", facts=facts,
                        links_ctx=links_ctx)
                    n_done += got
                    n_failed += failed
                    handled = True

            for artist_slug in participant_slugs:
                if not artist_slug or artist_slug in seen_artist_slugs:
                    continue
                seen_artist_slugs.add(artist_slug)
                try:
                    facts = MetadataDB.get_artist_facts_for_refine(
                        artist_slug, job.collection_name)
                except Exception:                   # noqa: BLE001
                    facts = []
                if facts:
                    got, failed = await _do_entity(
                        job, ask, scope="artist", scope_key=artist_slug,
                        entity={"slug": artist_slug,
                                # The participant's own name, not the raw tag:
                                # "Dua Lipa x Angele" must research Angele on the
                                # angele page, not the headliner (see 4f7b97a).
                                "name": name_for_slug(artist_name, artist_slug)
                                or artist_name},
                        origin_kind="artist_facts", facts=facts)
                    n_done += got
                    n_failed += failed
                    handled = True

            if not handled and (song_slug or participant_slugs):
                n_skipped += 1
            MetadataDB.update_ai_job(job_id=job.job_id, n_done=n_done,
                                     n_failed=n_failed, n_skipped=n_skipped)

    # Whatever the lane has not checked yet is finished here. In practice the
    # queue drained long ago: it is fed one song at a time by a loop that
    # spends seconds per song on the LLM, and a check costs under a second.
    await lane.aclose()

    # The read cache the player uses is derived, and the second direction can
    # only be derived once every song has been written: "who sampled X" comes
    # from OTHER songs' rows. So it is rebuilt here, at the end, rather than
    # per song.
    try:
        written = await asyncio.to_thread(
            MetadataDB.rebuild_samples_cache, job.collection_name)
        logger.info("[refined_facts] sample cache rebuilt for %d songs", written)
    except Exception:                                   # noqa: BLE001
        logger.warning("[refined_facts] sample cache rebuild failed",
                       exc_info=True)

    if n_done == 0 and n_skipped > 0 and n_failed == 0:
        logger.warning(
            "[refined_facts] all %d tracks skipped — no song_facts or "
            "artist_facts in collection %s. Run facts indexing first.",
            n_skipped, job.collection_name)


# Register at import time
ai_indexing_service.register_task("refined_facts", run)
