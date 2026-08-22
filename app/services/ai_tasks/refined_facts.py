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

import logging
import re
from typing import Optional

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service, text_quality as tq
from app.services.artist_split import artist_slugs, name_for_slug
from app.services.facts_v2 import pipeline as fv2
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

    ``artist_name`` is bound HERE rather than read off the record: this fires
    DURING processing, so anything the caller means to attach afterwards is not
    there yet — a fact that should move to the artist page stayed on the song's
    because the name it needed to resolve a slug was still empty.
    """
    def store(rec: dict) -> None:
        fact = rec["fact"]
        labels = [x for x in rec.get("labels", []) if not x.startswith("gate:")]
        if not labels:
            labels = rec.get("labels", [])       # keep the gate reason as the label
        text = rec.get("refined") or None
        dest_scope, dest_key = scope, scope_key

        # A fact that moved scope is stored under the page it belongs to — and
        # under the DESTINATION's labels, because "about_artist" describes where
        # it came from and says nothing a reader could group by.
        if rec.get("moved_to") == "artist" and scope == "song":
            primary = (artist_slugs(artist_name) or [None])[0]
            if primary:
                dest_scope, dest_key = "artist", primary
                labels = rec.get("focus_labels") or labels
        elif rec.get("moved_to") == "song" and scope == "artist":
            # The destination song has to be resolvable; when it is not, the
            # fact stays where it is rather than landing on the wrong page.
            title = ((rec.get("move") or {}).get("title") or "").strip()
            if title and artist_name:
                dest_scope = "song"
                dest_key = get_song_facts_key(artist_name, title)
                labels = rec.get("focus_labels") or labels

        MetadataDB.set_refined_fact_item(
            scope=dest_scope, scope_key=dest_key, lang=job.lang,
            origin_kind=origin_kind, origin_id=int(fact["id"]),
            labels=labels, text=text,
            confirmed=not rec.get("unconfirmed", False),
            src=("annotation" if fact.get("category") == "genius_annotation"
                 else "editorial"),
            collection_name=job.collection_name,
        )
    return store


async def _do_entity(job, ask, *, scope: str, scope_key: str, entity: dict,
                     origin_kind: str, facts: list) -> tuple:
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
    _store_links(job, scope_key, entity, recs)
    return len(facts), sum(1 for r in recs if r.get("error"))


_LANG_NAME = {"ru": "Russian", "en": "English"}


def _store_links(job, scope_key: str, entity: dict, recs: list) -> None:
    """Write this song's sampling links. Verification is a separate pass.

    Deliberately not verified here: MusicBrainz answers about one request a
    second, so checking inline would make indexing wait on the network once per
    sampled track. ``scripts/verify_sample_links.py`` does that afterwards and
    caches its verdicts.

    ``replace_sample_links`` is a delete-then-insert per source song, so this
    collects every link of the entity and writes once — calling it per link
    would leave only the last one.
    """
    from app.services.fact_relations.gates import dst_key

    rows, seen = [], set()
    for rec in recs:
        for link in (rec.get("links") or []):
            artist = (link.get("artist") or "").strip()
            title = (link.get("title") or "").strip()
            if not artist or not title:
                continue
            direction = link.get("direction") or "source"
            key = dst_key(artist, title)
            if (direction, key) in seen:
                continue                    # one fact can restate another's link
            seen.add((direction, key))
            rows.append({
                "direction": direction, "dst_key": key, "dst_artist": artist,
                "dst_title": title, "dst_slug": None,
                "relation": link.get("relation") or "sample",
                "evidence": (rec["fact"].get("fact") or "")[:400],
                "confidence": None,
            })
    if not rows:
        return
    try:
        MetadataDB.replace_sample_links(job.collection_name, scope_key, rows)
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

    if job.new_track_ids is not None:
        # A small batch: one retrieve instead of a whole-collection scroll.
        points = qdrant.retrieve(
            collection_name=job.collection_name,
            ids=list(job.new_track_ids), with_payload=True, with_vectors=False,
        )
        batches = [points] if points else []
    else:
        batches = []
        offset = None
        while True:
            points, offset = qdrant.scroll(
                collection_name=job.collection_name, limit=64, offset=offset,
                with_payload=True, with_vectors=False,
            )
            if not points:
                break
            batches.append(points)
            if offset is None:
                break

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
                        origin_kind="song_facts", facts=facts)
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

    if n_done == 0 and n_skipped > 0 and n_failed == 0:
        logger.warning(
            "[refined_facts] all %d tracks skipped — no song_facts or "
            "artist_facts in collection %s. Run facts indexing first.",
            n_skipped, job.collection_name)


# Register at import time
ai_indexing_service.register_task("refined_facts", run)
