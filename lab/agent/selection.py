"""Claims → a playlist: weighting, filtering, triage, curation.

These are the stages between "pages said these songs exist" and "here is the
playlist". They live here as functions rather than inside ``PlaylistBranch``
for one reason: a notebook driving the pipeline a stage at a time has to run
the SAME selection the agent runs. Reimplementing the weighting or the era
filter in a cell is how a lab result stops describing the thing it is meant to
be measuring.

The division of labour is the same as everywhere else in this package. Code
decides what exists, how much each source is worth and what the era rules out.
The model gets exactly one judgement — which of the surviving tracks the page
was actually offering as an answer — and can express it only as a list of ids.
"""

from __future__ import annotations

import logging
from typing import Optional

from lab.agent.catalog import filter_by_era
from lab.agent.llm import as_str
from lab.agent.models import ResolvedTrack, TrackRef
from lab.agent.prompts import CURATE_SYSTEM, TRIAGE_SYSTEM
from lab.agent.reasons import clean_reason

logger = logging.getLogger(__name__)


def merge_claims(resolved: list[ResolvedTrack], *,
                 source_weights: dict) -> list[ResolvedTrack]:
    """One row per track, carrying the weight of every source that found it.

    Finding the same song on Wikipedia and again on a listicle is
    corroboration, and corroboration is the whole reason the curated sources
    count double: two independent pages naming a track is stronger evidence
    than one page naming it twice.
    """
    merged: dict[str, ResolvedTrack] = {}
    for track in resolved:
        key = track.track_id or f"{track.artist}|{track.title}".lower()
        weight = source_weights.get(track.sources[0] if track.sources else "web", 1.0)
        existing = merged.get(key)
        if existing is None:
            track.weight = weight
            merged[key] = track
            continue
        existing.weight += weight
        for source in track.sources:
            if source not in existing.sources:
                existing.sources.append(source)
        # Keep whichever provenance is more specific — a bare page beats
        # nothing, a section beats a bare page.
        if not existing.section and track.section:
            existing.section = track.section
            existing.page_title = track.page_title
            existing.context = track.context
    return list(merged.values())


def rank_tracks(tracks: list[ResolvedTrack]) -> list[ResolvedTrack]:
    """Corroborated first, exact matches before fuzzy ones, then by title."""
    return sorted(tracks,
                  key=lambda t: (-t.weight, t.match != "exact", t.title.lower()))


def select_tracks(catalog, claims: list[TrackRef], *,
                  era: Optional[tuple[int, int]] = None,
                  source_weights: Optional[dict] = None,
                  ) -> tuple[list[ResolvedTrack], list[TrackRef]]:
    """The whole deterministic half: resolve, merge, filter, rank.

    Returns ``(tracks, missing)``. Nothing here consults a model, so the result
    is the same on every run over the same pages — which is what makes it
    possible to tell a retrieval change from a model mood.
    """
    if catalog is None:
        return [], list(claims)
    source_weights = source_weights or {}

    resolved, missing = catalog.resolve_tracks(claims)
    merged = merge_claims(resolved, source_weights=source_weights)

    before = len(merged)
    # Applied BEFORE anything reaches a model: an out-of-era track in the
    # triage context is context spent on a track that will be dropped anyway.
    kept = filter_by_era(merged, era)
    if before != len(kept):
        logger.info("[selection] era %s dropped %d/%d", era, before - len(kept),
                    before)
    return rank_tracks(kept), missing


async def triage_tracks(llm, message: str, tracks: list[ResolvedTrack], *,
                        config, sink=None) -> list[ResolvedTrack]:
    """Let the model drop what the page had for another reason.

    Nothing before this can remove a track once it matched the library, and
    matching only proves the library HAS it — not that the page was offering
    it as an answer. A discography's "Other appearances" table, a listicle's
    sidebar and a chart-position grid all resolve just as cleanly as the
    tracklist.

    The model sees where each track came from and answers with ids only. An id
    it was not offered is discarded; an empty answer is read as a model failure
    rather than a verdict, because a 12b losing the format is far likelier than
    a page containing nothing relevant, and the cost of believing it is an
    empty playlist.
    """
    if not config.llm_triage or len(tracks) < config.triage_min_candidates:
        return tracks

    keys = {f"T{i + 1}": t for i, t in enumerate(tracks)}
    listing = []
    for key, track in keys.items():
        where = " · ".join(p for p in (track.page_title, track.section) if p)
        line = f"{key}. {track.artist} — {track.title}"
        if track.year:
            line += f" ({track.year})"
        if where:
            line += f"\n     found under: {where}"
        if track.context:
            line += f"\n     row: {track.context[:160]}"
        listing.append(line)

    if sink is not None:
        sink.put("triage", candidates=len(keys))
    raw = await llm.ask_json([
        {"role": "system", "content": TRIAGE_SYSTEM},
        {"role": "user",
         "content": f"Request: {message}\n\nCandidates:\n" + "\n".join(listing)},
    ], required=("keep",))
    if raw is None:
        logger.info("[selection] triage unavailable — keeping all %d", len(tracks))
        return tracks

    ids = [as_str(k, 8).upper() for k in (raw.get("keep") or [])
           if isinstance(k, (str, int))]
    kept = [keys[k] for k in dict.fromkeys(ids) if k in keys]
    if not kept:
        logger.info("[selection] triage kept nothing — ignoring it")
        return tracks

    why = as_str(raw.get("dropped_because"), 160)
    if sink is not None:
        sink.put("triage_done", kept=len(kept), dropped=len(tracks) - len(kept),
                 why=why)
    logger.info("[selection] triage dropped %d/%d: %s",
                len(tracks) - len(kept), len(tracks), why)
    return kept


async def curate_tracks(llm, message: str, tracks: list[ResolvedTrack], *,
                        config, sink=None) -> tuple[str, str, list[ResolvedTrack]]:
    """Order and describe. The model cannot add, remove or rename a track."""
    fallback_title = message.strip()[:80]
    if not tracks:
        return fallback_title, "", tracks

    lang = "Russian" if (config.lang or "").lower().startswith("ru") else "English"
    keys = {f"T{i + 1}": t for i, t in enumerate(tracks)}
    listing = "\n".join(
        f"{k}. {t.artist} — {t.title}" + (f" ({t.year})" if t.year else "")
        for k, t in keys.items())

    raw = await llm.ask_json([
        {"role": "system", "content": CURATE_SYSTEM.format(lang=lang)},
        {"role": "user", "content": f"Request: {message}\n\nTracks:\n{listing}"},
    ], required=("order",))
    if raw is None:
        return fallback_title, "", tracks

    ordered: list[ResolvedTrack] = []
    seen: set[str] = set()
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

    if sink is not None:
        sink.put("curated", tracks=len(ordered),
                 with_reason=sum(1 for t in ordered if t.reason))
    return (as_str(raw.get("title"), 120) or fallback_title,
            as_str(raw.get("comment"), 400), ordered)
