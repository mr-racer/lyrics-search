"""Lyric-gems task — mine rare, quotable facts from every track's lyrics.

Two passes over the collection:

1. A cheap payload-only scroll builds the matching context: the library
   artist index (slug → display name, plus LLM-fetched aliases, cached
   forever in ``artist_aliases``) and the song/album catalog for songref.
2. The per-track pass runs the shared GLiNER2 model (entities schema) over
   cleaned lyric chunks in ``asyncio.to_thread`` — the sync model call must
   never sit on the event loop — then canonicalizes spans via gazetteers and
   the library catalog. Leftover confident candidates go to the LLM verifier
   (≤2 web searches, verdict cached in ``gem_resolution_cache``).

Incremental: a processed track (even with zero findings) has rows in
``track_gems`` (marker kind='none') and is skipped on re-runs. Cyrillic-heavy
tracks are marked processed without extraction for now — the model (multi-v1)
handles Russian, but the gazetteers/catalog matching are English-built; the
Russian pass is a planned follow-up, not a model limitation.

The ``llm`` parameter of run() is framework compatibility; LLM calls go
through the resolver, which resolves connection config itself.
"""

from __future__ import annotations

import asyncio
import logging

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service
from app.services.artist_facts_service import _slugify
from app.services.lyric_gems import matching, preprocess
from app.services.lyric_gems.gazetteers import capsule_hits
from app.services.lyric_gems.resolver import fetch_artist_aliases, verify_candidate

logger = logging.getLogger(__name__)

MIN_LYRICS_CHARS = 200
CYR_SKIP_RATIO = 0.3
GLINER_THRESHOLD = 0.4  # raw pass; matching.MIN_SCORE re-filters at 0.6
MAX_LLM_VERIFICATIONS_PER_TRACK = 5
MAX_ALIAS_FETCHES_PER_RUN = 300

ENTITY_SCHEMA = {
    "celebrity": "famous real person: musician, actor, athlete, historical figure",
    "fictional_character": (
        "fictional character from movies, comics or TV, e.g. Tony Montana, Batman, James Bond"
    ),
    "movie_or_show": "movie, film or TV show title",
    "song_or_album": "title of a song or music album being referenced",
}


def _extract_spans(model, chunk: str) -> list[tuple[str, str, float]]:
    """Sync GLiNER call → [(label, text, score)]. Runs inside to_thread."""
    try:
        out = model.extract_entities(
            chunk, ENTITY_SCHEMA, threshold=GLINER_THRESHOLD, include_confidence=True,
        )
    except TypeError:
        out = model.extract_entities(chunk, ENTITY_SCHEMA, threshold=GLINER_THRESHOLD)
    ent_map = out.get("entities", out) if isinstance(out, dict) else {}
    spans: list[tuple[str, str, float]] = []
    for label, items in (ent_map or {}).items():
        for it in items or []:
            if isinstance(it, dict):
                text = (it.get("text") or "").strip()
                score = float(it.get("confidence", it.get("score", 0.0)) or 0.0)
            else:
                text, score = str(it).strip(), 1.0
            if text:
                spans.append((label, text, score))
    return spans


async def _scroll_tracks(
    qdrant, collection_name: str, only_track_ids: list[str] | None = None,
) -> list[dict]:
    """Full payload scroll (lyrics included) off the event loop.

    ``only_track_ids`` (auto run after an append/upload): a batched retrieve
    of exactly those tracks instead of a whole-collection scroll.
    """
    fields = ["lyrics", "artist", "artists", "title", "album", "year", "artist_slugs"]

    def _scroll_sync():
        tracks: list[dict] = []
        if only_track_ids is not None:
            points = qdrant.retrieve(
                collection_name=collection_name,
                ids=list(only_track_ids),
                with_payload=fields,
                with_vectors=False,
            )
            for pt in points:
                p = pt.payload or {}
                tracks.append({
                    "track_id": str(pt.id),
                    "lyrics": p.get("lyrics") or "",
                    "artist": p.get("artist") or "",
                    "artists": p.get("artists") or [],
                    "title": p.get("title") or "",
                    "album": p.get("album") or "",
                    "year": p.get("year"),
                    "artist_slugs": p.get("artist_slugs") or [],
                })
            return tracks

        offset = None
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=fields,
                with_vectors=False,
            )
            for pt in points:
                p = pt.payload or {}
                tracks.append({
                    "track_id": str(pt.id),
                    "lyrics": p.get("lyrics") or "",
                    "artist": p.get("artist") or "",
                    "artists": p.get("artists") or [],
                    "title": p.get("title") or "",
                    "album": p.get("album") or "",
                    "year": p.get("year"),
                    "artist_slugs": p.get("artist_slugs") or [],
                })
            offset = next_offset
            if offset is None:
                return tracks

    return await asyncio.to_thread(_scroll_sync)


def _collect_library_artists(tracks: list[dict]) -> tuple[dict[str, str], dict[str, int]]:
    """``({slug: display_name}, {slug: track_count})`` over the collection.

    The counts feed ``matching.RARE_ARTIST_TRACKS``: a one-word name belonging
    to an artist with almost no tracks here ("Kobe") is far more likely to be
    someone else entirely in another artist's lyrics, so it goes to the LLM
    verifier rather than straight into the gems.
    """
    artists: dict[str, str] = {}
    counts: dict[str, int] = {}
    for tr in tracks:
        names = list(tr["artists"]) or ([tr["artist"]] if tr["artist"] else [])
        slugs = list(tr["artist_slugs"])
        for i, name in enumerate(names):
            name = (name or "").strip()
            if not name:
                continue
            slug = slugs[i] if i < len(slugs) else _slugify(name)
            artists.setdefault(slug, name)
            counts[slug] = counts.get(slug, 0) + 1
    return artists, counts


def _own_slugs(track: dict, kin: dict[str, frozenset[str]]) -> frozenset[str]:
    """Every artist slug the track itself counts as.

    Namedrop identity is compared on slugs, not display names — an alias that
    resolves to the performer ("Marshall Mathers" on an Eminem track) has to
    lose here, and the alias cache is exactly what makes the display-name
    comparison miss it. ``kin`` widens the set across group/member lines.
    """
    slugs = {s for s in (track.get("artist_slugs") or []) if s}
    for name in [track.get("artist") or "", *(track.get("artists") or [])]:
        if name.strip():
            slugs.add(_slugify(name))
    return frozenset(slugs | {k for s in slugs for k in kin.get(s, ())})


async def _ensure_aliases(artists: dict[str, str], job) -> None:
    """One-time LLM alias fetch per library artist (marker-cached forever).

    A failed call (LLM down) writes NO marker, so the artist is retried on
    the next run. Budgeted per run to bound a first-run on a huge library."""
    fetched = 0
    for slug, display in artists.items():
        if fetched >= MAX_ALIAS_FETCHES_PER_RUN:
            logger.info("[lyric_gems] alias budget (%d) reached", MAX_ALIAS_FETCHES_PER_RUN)
            return
        if MetadataDB.has_artist_aliases(slug):
            continue
        found = await fetch_artist_aliases(
            display, base_url=job.llm_base_url, model_name=job.llm_model,
        )
        if found is None:
            logger.info("[lyric_gems] alias fetch failed for %r — skipping alias stage", display)
            return  # LLM unreachable: don't burn the rest of the budget
        MetadataDB.set_artist_aliases(slug, found["aliases"])
        if found["members"]:
            MetadataDB.set_artist_aliases(
                slug, found["members"], source=MetadataDB.MEMBER_ALIAS_SOURCE,
            )
        fetched += 1


def _kin_slugs(
    artist_index: dict[str, tuple[str, str]], members_map: dict[str, str],
) -> dict[str, frozenset[str]]:
    """{slug: slugs of the same act} — a group and its members, both ways.

    Bad Meets Evil IS Eminem and Royce, so neither side name-dropping the
    other is a discovery. Only members the library also knows as artists in
    their own right can appear here; the rest have no slug to link to.
    """
    kin: dict[str, set[str]] = {}
    for member_name, group_slug in members_map.items():
        hit = artist_index.get(matching.norm_name(member_name))
        if hit is None or hit[0] == group_slug:
            continue
        kin.setdefault(group_slug, set()).add(hit[0])
        kin.setdefault(hit[0], set()).add(group_slug)
    return {k: frozenset(v) for k, v in kin.items()}


async def _verify_with_cache(*, kind, candidate, quote, track, target=None, context_key="", job) -> dict:
    """LLM verdict with the forever-cache in front."""
    cached = MetadataDB.get_gem_resolution(kind, candidate, context_key)
    if cached is not None:
        return cached
    verdict = await verify_candidate(
        kind=kind,
        candidate=candidate,
        quote=quote,
        track_artist=track["artist"],
        track_title=track["title"],
        year=track["year"] if isinstance(track["year"], int) else None,
        target=target,
        base_url=job.llm_base_url,
        model_name=job.llm_model,
    )
    # Cache everything except transport errors — those deserve a retry later.
    if not str(verdict.get("why", "")).startswith("agent error"):
        MetadataDB.set_gem_resolution(kind, candidate, verdict, context_key)
    return verdict


async def run(job, db_client, llm) -> None:
    """Process every track of the collection; persist gems + marker rows.

    When ``job.new_track_ids`` is set (auto run after an append/upload) only
    those tracks are processed; the matching context (artist index, song
    catalog) is still built from the batch itself — cheap and sufficient for
    name-drops within the new songs.
    """
    qdrant = db_client.qdrant
    only_ids = (list(job.new_track_ids)
                if job.new_track_ids is not None else None)
    tracks = await _scroll_tracks(qdrant, job.collection_name, only_track_ids=only_ids)

    artists, artist_counts = _collect_library_artists(tracks)
    await _ensure_aliases(artists, job)
    artist_index = matching.build_artist_index(artists, MetadataDB.get_artist_aliases_map())
    song_catalog = matching.build_song_catalog(tracks)
    rare_slugs = frozenset(
        slug for slug, n in artist_counts.items() if n < matching.RARE_ARTIST_TRACKS
    )
    kin = _kin_slugs(artist_index, MetadataDB.get_artist_members_map())
    logger.info(
        "[lyric_gems] %s: %d tracks, %d artists in index, %d catalog titles",
        job.collection_name, len(tracks), len(artist_index), len(song_catalog),
    )

    model = None  # lazy — a fully-cached re-run must not load GLiNER at all
    n_done = n_failed = n_skipped = 0

    for track in tracks:
        tid = track["track_id"]
        if MetadataDB.has_track_gems(tid, job.collection_name):
            n_done += 1
            MetadataDB.update_ai_job(job_id=job.job_id, n_done=n_done)
            continue

        lyrics = track["lyrics"].strip()
        if len(lyrics) < MIN_LYRICS_CHARS or preprocess.cyr_ratio(lyrics) > CYR_SKIP_RATIO:
            MetadataDB.set_track_gems(tid, job.collection_name, [])
            n_skipped += 1
            MetadataDB.update_ai_job(job_id=job.job_id, n_skipped=n_skipped)
            continue

        try:
            if model is None:
                from app.services.fact_relations.extractor import get_model

                model = await asyncio.to_thread(get_model)

            cleaned = preprocess.clean_lyrics(lyrics)
            spans: list[tuple[str, str, float]] = []
            for chunk in preprocess.chunk_text(cleaned):
                spans.extend(await asyncio.to_thread(_extract_spans, model, chunk))

            gems: list[dict] = []
            seen: set[tuple[str, str]] = set()
            # One lyric line yields one pop-culture gem: "Batman brought his
            # own Robin" is a Batman reference, not two separate finds.
            pop_quotes: dict[str, dict] = {}
            llm_budget = MAX_LLM_VERIFICATIONS_PER_TRACK
            own_keys = matching.own_name_keys(track["artist"], track["artists"], track["title"])
            own_slugs = _own_slugs(track, kin)

            def _add(gem: dict) -> None:
                key = (gem["kind"], gem["canonical"])
                if key in seen:
                    return
                if not gem.get("quote"):
                    gem["quote"] = preprocess.find_quote(gem.get("surface") or gem["canonical"], lyrics)
                gem.pop("surface", None)
                if not gem["quote"]:
                    return
                if gem["kind"] == "popculture":
                    rival = pop_quotes.get(gem["quote"])
                    if rival is not None:
                        if (gem.get("score") or 0) <= (rival.get("score") or 0):
                            return
                        seen.discard((rival["kind"], rival["canonical"]))
                        gems.remove(rival)
                    pop_quotes[gem["quote"]] = gem
                seen.add(key)
                gems.append(gem)

            for gem in capsule_hits(cleaned, lyrics, track["title"]):
                _add(gem)

            for label, text, score in spans:
                if not preprocess.appears_capitalized(text, lyrics):
                    continue
                if label == "celebrity":
                    gem, needs_llm = matching.match_namedrop(
                        text, score, artist_index, own_keys, track["title"],
                        own_slugs=own_slugs, rare_slugs=rare_slugs,
                    )
                    if gem is None:
                        continue
                    gem["surface"] = text
                    if needs_llm:
                        if llm_budget <= 0:
                            continue
                        quote = preprocess.find_quote(text, lyrics)
                        if not quote:
                            continue
                        llm_budget -= 1
                        verdict = await _verify_with_cache(
                            kind="namedrop", candidate=text, quote=quote,
                            track=track, target=gem["canonical"],
                            context_key=f"{track['artist']}|{track['title']}", job=job,
                        )
                        if verdict.get("verdict") != "yes":
                            continue
                    _add(gem)
                elif label == "song_or_album":
                    # The quote is a gate input here, not just decoration: a
                    # title only counts when the line presents it as a work.
                    quote = preprocess.find_quote(text, lyrics)
                    if not quote:
                        continue
                    gem = matching.match_songref(
                        text, score, song_catalog, track["title"], track["album"],
                        quote=quote,
                    )
                    if gem is not None:
                        gem["surface"] = text
                        _add(gem)
                elif label in ("fictional_character", "movie_or_show"):
                    gem, needs_llm = matching.match_popculture(text, score)
                    if gem is not None:
                        gem["surface"] = text
                        _add(gem)
                    elif needs_llm and llm_budget > 0:
                        quote = preprocess.find_quote(text, lyrics)
                        if not quote:
                            continue
                        llm_budget -= 1
                        verdict = await _verify_with_cache(
                            kind="popculture", candidate=text, quote=quote,
                            track=track, job=job,
                        )
                        if verdict.get("verdict") == "yes":
                            _add({
                                "kind": "popculture",
                                "canonical": verdict["canonical"],
                                "display": verdict["canonical"],
                                "quote": quote,
                                "detail": {"source": "llm"},
                                "score": score,
                            })

            MetadataDB.set_track_gems(tid, job.collection_name, gems)
            n_done += 1
            MetadataDB.update_ai_job(job_id=job.job_id, n_done=n_done)
        except Exception:
            logger.exception("[lyric_gems] failed on track %s", tid)
            n_failed += 1
            MetadataDB.update_ai_job(job_id=job.job_id, n_failed=n_failed)

    logger.info(
        "[lyric_gems] %s finished: %d done, %d skipped, %d failed",
        job.collection_name, n_done, n_skipped, n_failed,
    )


# Register at import time
ai_indexing_service.register_task("lyric_gems", run)
