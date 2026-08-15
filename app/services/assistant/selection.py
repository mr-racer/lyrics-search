"""Claims → a playlist: weighting, filtering, triage, curation.

The stages between "pages said these songs exist" and "here is the playlist".

The division of labour is the same as everywhere else in this package. Code
decides what exists, how much each source is worth and what the era rules out.
The model gets exactly one judgement — which of the surviving tracks the page was
actually offering as an answer — and can express it only as a list of ids.

The per-track "why this fits" line goes through ``reason_gate.clean_reason``, the
same gate the old playlist path used: the prompt already forbids filler and a 12b
model obeys that unevenly.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.services.assistant.llm import as_str
from app.services.assistant.prompts import CURATE_SYSTEM, TRIAGE_SYSTEM
from app.services.assistant.reason_gate import clean_reason
from app.services.library_catalog import filter_by_era

logger = logging.getLogger(__name__)


def merge_claims(resolved: list, *, source_weights: dict) -> list:
    """One row per track, carrying the weight of every source that found it.

    Finding the same song on Wikipedia and again on a listicle is corroboration,
    and corroboration is the whole reason the curated sources count double: two
    independent pages naming a track is stronger evidence than one page naming it
    twice.
    """
    merged: dict = {}
    for track in resolved:
        key = track.track_id or f"{track.artist}|{track.title}".lower()
        weight = source_weights.get(
            track.sources[0] if track.sources else "web", 1.0)
        existing = merged.get(key)
        if existing is None:
            track.weight = weight
            merged[key] = track
            continue
        existing.weight += weight
        for source in track.sources:
            if source not in existing.sources:
                existing.sources.append(source)
        # Keep whichever provenance is more specific — a bare page beats nothing,
        # a section beats a bare page.
        if not existing.section and track.section:
            existing.section = track.section
            existing.page_title = track.page_title
            existing.context = track.context
    return list(merged.values())


def rank_tracks(tracks: list) -> list:
    """Corroborated first, exact matches before fuzzy ones, then by title."""
    return sorted(tracks,
                  key=lambda t: (-t.weight, t.match != "exact", t.title.lower()))


def select_tracks(catalog, claims: list, *, era: Optional[tuple] = None,
                  source_weights: Optional[dict] = None) -> tuple:
    """The whole deterministic half: resolve, merge, filter, rank.

    Returns ``(tracks, missing)``. Nothing here consults a model, so the result is
    the same on every run over the same pages — which is what makes it possible to
    tell a retrieval change from a model mood.
    """
    if catalog is None:
        return [], list(claims)
    source_weights = source_weights or {}

    resolved, missing = catalog.resolve_tracks(claims)
    merged = merge_claims(resolved, source_weights=source_weights)

    before = len(merged)
    # Applied BEFORE anything reaches a model: an out-of-era track in the triage
    # context is context spent on a track that will be dropped anyway.
    kept = filter_by_era(merged, era)
    if before != len(kept):
        logger.info("[selection] era %s dropped %d/%d", era, before - len(kept),
                    before)
    return rank_tracks(kept), missing


_CELL_SPLIT = re.compile(r"\s*[|·]\s*")


def _row_extra(track) -> str:
    """The parts of the source row that are not already on the track's line.

    A table row is stored whole, artist and title included, which is right for a
    claim and pure repetition here. On a 30-track GTA playlist 26 rows restated
    the artist and 23 the title — the model paid for them twice and learned
    nothing. What survives the cut is the part that carries the fact: the album,
    and for a soundtrack the album IS the film.
    """
    known = [f for f in (_fold(track.artist), _fold(track.title)) if f]
    keep = []
    for part in _CELL_SPLIT.split(track.context or ""):
        folded = _fold(part)
        # Containment either way, not equality: the row spells the same act
        # differently often enough to matter — "Black Eyed Peas" beside "The
        # Black Eyed Peas", "Gorillaz ft. De La Soul" beside "Gorillaz".
        if folded and not any(folded in k or k in folded for k in known):
            keep.append(part.strip())
    return " · ".join(keep)[:120]


def _fold(text: str) -> str:
    return " ".join((text or "").casefold().split())


def _listing(keys: dict) -> str:
    """The tracks, grouped by where they were found.

    Both model calls in this file need the provenance, and for the same reason: a
    model shown nothing but "Artist — Title" knows only what it already knew, so
    when asked for anything specific about a track it writes what it can guess.
    That is where "идеальный драйв для сцен погони" came from on a request for
    film soundtracks — the film was in the material all along, in the album Apple
    ships with the row, and this call never saw it.

    Grouped rather than repeated per track because the flat form was four times
    the size of the bare listing and three quarters of that was one page title
    printed thirty times. Grouping is safe here specifically: neither caller reads
    sequence off this text — triage walks its result in rank order and curation
    re-sorts by weight — so the only thing the order changes is which tracks the
    model sees side by side, and side by side with its own station is the right
    neighbourhood anyway.
    """
    groups: dict = {}
    for key, track in keys.items():
        where = " · ".join(p for p in (track.page_title, track.section) if p)
        line = f"{key}. {track.artist} — {track.title}"
        if track.year:
            line += f" ({track.year})"
        extra = _row_extra(track)
        if extra:
            line += f" — {extra}"
        groups.setdefault(where, []).append(line)

    out: list = []
    for where, lines in groups.items():
        if where:
            out.append(f"found on: {where}")
            out.extend("  " + line for line in lines)
        else:
            out.extend(lines)
    return "\n".join(out)


async def triage_tracks(llm, message: str, tracks: list, *, config,
                        sink=None) -> list:
    """Let the model drop what the page had for another reason.

    Nothing before this can remove a track once it matched the library, and
    matching only proves the library HAS it — not that the page was offering it as
    an answer. A discography's "Other appearances" table, a listicle's sidebar and
    a chart-position grid all resolve just as cleanly as the tracklist.

    The model sees where each track came from and answers with ids only. An id it
    was not offered is discarded; an empty answer is read as a model failure
    rather than a verdict, because a 12b losing the format is far likelier than a
    page containing nothing relevant, and the cost of believing it is an empty
    playlist.
    """
    if not config.llm_triage or len(tracks) < config.triage_min_candidates:
        return tracks

    keys = {f"T{i + 1}": t for i, t in enumerate(tracks)}

    if sink is not None:
        sink.put("triage", candidates=len(keys))
    raw = await llm.ask_json([
        {"role": "system", "content": TRIAGE_SYSTEM},
        {"role": "user",
         "content": f"Request: {message}\n\nCandidates:\n{_listing(keys)}"},
    ], required=("keep",))
    if raw is None:
        logger.info("[selection] triage unavailable — keeping all %d", len(tracks))
        return tracks

    ids = {as_str(k, 8).upper() for k in (raw.get("keep") or [])
           if isinstance(k, (str, int))}
    # Walked in RANK order, not in the order the model listed its ids. Its own
    # prompt says it cannot reorder anything, and it has no business doing so: the
    # caller cuts this list to the target count straight afterwards, so the
    # sequence the model happened to type would decide which tracks survive. That
    # is how a three-source track ended up below a single-listicle one.
    kept = [track for key, track in keys.items() if key in ids]
    if not kept:
        logger.info("[selection] triage kept nothing — ignoring it")
        return tracks

    why = as_str(raw.get("dropped_because"), 160)
    if sink is not None:
        sink.put("triage_done", kept=len(kept), dropped=len(tracks) - len(kept),
                 why=why)
    logger.info("[selection] triage dropped %d/%d: %s", len(tracks) - len(kept),
                len(tracks), why)
    return kept


async def curate_tracks(llm, message: str, tracks: list, *, config,
                        sink=None) -> tuple:
    """Order and describe. The model cannot add, remove or rename a track."""
    fallback_title = message.strip()[:80]
    if not tracks:
        return fallback_title, "", tracks

    lang = "Russian" if (config.lang or "").lower().startswith("ru") else "English"
    keys = {f"T{i + 1}": t for i, t in enumerate(tracks)}

    raw = await llm.ask_json([
        {"role": "system", "content": CURATE_SYSTEM.format(lang=lang)},
        {"role": "user",
         "content": f"Request: {message}\n\nTracks:\n{_listing(keys)}"},
    ], required=("order",))
    if raw is None:
        return fallback_title, "", tracks

    ordered: list = []
    seen: set = set()
    for item in (raw.get("order") or []):
        if not isinstance(item, dict):
            continue
        key = as_str(item.get("id"), 8).upper()
        track = keys.get(key)
        if track is None or key in seen:
            continue
        seen.add(key)
        track.reason = clean_reason(item.get("reason"), title=track.title,
                                    artist=track.artist)
        ordered.append(track)
    # Anything the model forgot keeps its place at the end. Losing a track to a
    # formatting slip is not an acceptable failure mode.
    ordered += [t for k, t in keys.items() if k not in seen]

    if getattr(config, "curate_respects_weight", True):
        # Corroboration outranks sequencing, and the sort is STABLE: tracks of
        # equal weight keep the flow the model built for them. Only the tiers
        # move — a track three pages named goes above one a single listicle did,
        # which for a request like "хиты X" is the whole question.
        ordered.sort(key=lambda t: -t.weight)

    if sink is not None:
        sink.put("curated", tracks=len(ordered),
                 with_reason=sum(1 for t in ordered if t.reason))
    return (as_str(raw.get("title"), 120) or fallback_title,
            as_str(raw.get("comment"), 400), ordered)
