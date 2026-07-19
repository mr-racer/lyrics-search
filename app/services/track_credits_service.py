"""Track-credit logic: producer credits and record labels.

Backs the player's producers drawer and the album gatefold's label block.
Everything here reads only the SQLite ``track_metadata`` mirror — no Qdrant —
so credits keep working while Qdrant is down.

Producer names come from the free-text ``producer`` tag ("Dr. Dre, Mel-Man" /
"A; B" / "A & B"); ``split_credit_names`` is the single source of truth for
splitting it (the player keeps a client-side copy only as a display fallback
while the resolve request is in flight). Label strings may carry several
labels ("GOOD Music, Def Jam") and are split by ``split_labels``.

Label names are matched with ``label_key`` (lowercase + collapsed whitespace).
Deliberately NOT ``artist_facts_service._slugify`` — labels are not artist
names and must not share the artist slug space.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

from ..domain.models import ProducerCredit, ProducerTrack
from ..resources.metadata_db import MetadataDB, canonical_track_id
from .artist_split import artist_refs_for_track, display_title_for_track, artist_slugs

logger = logging.getLogger(__name__)

# Same separators the player UI used client-side: , ; / & feat.
# NB: ``feat\b\.?`` (boundary BEFORE the optional dot) — ``feat\.?\b`` never
# matches the dot ("." and " " are both non-word, so no boundary sits between
# them) and left a ". Name" remainder.
_CREDIT_SPLIT_RE = re.compile(r"\s*(?:[,;/]|&|\bfeat\b\.?)\s*", re.IGNORECASE)
# Labels only split on , and ; — "/" and "&" appear inside real label names
# ("Kobalt Music Group", "A&M Records").
_LABEL_SPLIT_RE = re.compile(r"\s*[,;]\s*")

# The player shows at most 3 producers; counting honors the same cap so the
# drawer's "· N" badges agree with what a visitor sees on the other tracks.
MAX_CREDIT_NAMES = 3


def split_credit_names(raw: Optional[str], cap: int = MAX_CREDIT_NAMES) -> list[str]:
    """Split a raw producer tag into up to ``cap`` clean names."""
    parts = [p.strip() for p in _CREDIT_SPLIT_RE.split(str(raw or "")) if p.strip()]
    return parts[:cap]


def _name_key(name: str) -> str:
    """Case/whitespace-insensitive identity for a producer name."""
    return " ".join(str(name or "").lower().split())


def split_labels(raw: Optional[str]) -> list[str]:
    """Split a raw label tag into clean label names (no cap)."""
    return [p.strip() for p in _LABEL_SPLIT_RE.split(str(raw or "")) if p.strip()]


def label_key(name: str) -> str:
    """Case/whitespace-insensitive identity for a label name."""
    return " ".join(str(name or "").lower().split())


def aggregate_labels(raw_values: Iterable[Optional[str]]) -> list[str]:
    """Collapse per-track label strings into a deduped album-level list.

    Keeps the first-seen casing of every label, ordered by frequency
    (ties broken by first appearance).
    """
    counts: dict[str, int] = {}
    display: dict[str, str] = {}
    order: dict[str, int] = {}
    for raw in raw_values:
        for name in split_labels(raw):
            k = label_key(name)
            if k not in display:
                display[k] = name
                order[k] = len(order)
            counts[k] = counts.get(k, 0) + 1
    return [display[k] for k in sorted(counts, key=lambda k: (-counts[k], order[k]))]


def _library_artist_slug(collection_name: str, name: str) -> Optional[str]:
    """Canonical artist slug for ``name`` IF the library has them as an artist.

    Slugs route through ``artist_slugs`` (→ ``artist_facts_service._slugify``,
    the same path the artist page resolves), then must exist in
    ``track_artist_slugs`` for this collection — whole-slug equality only.
    """
    slugs = artist_slugs(name)
    slug = slugs[0] if slugs else ""
    if slug and MetadataDB.has_artist_slug(collection_name, slug):
        return slug
    return None


def resolve_producers(collection_name: str, track_id: str) -> list[ProducerCredit]:
    """Resolve a track's producer names against the user's library.

    For every name in the track's producer tag: is the producer also a library
    artist (→ ``artist_slug``), and how many OTHER tracks in the collection
    credit them as producer (→ ``produced_count``). Degrades to ``[]`` on any
    error — the player's drawer falls back to plain client-split pills.
    """
    try:
        raw = MetadataDB.get_track_producer(collection_name, track_id)
        names = split_credit_names(raw)
        if not names:
            return []

        wanted = {_name_key(n) for n in names}
        counts: dict[str, int] = {}
        current_id = canonical_track_id(track_id)
        for row in MetadataDB.get_produced_tracks(collection_name):
            if row["track_id"] == current_id:
                continue
            # Per-row set: a name repeated inside one tag counts once.
            for key in {_name_key(n) for n in split_credit_names(row["producer"])}:
                if key in wanted:
                    counts[key] = counts.get(key, 0) + 1

        return [
            ProducerCredit(
                name=name,
                artist_slug=_library_artist_slug(collection_name, name),
                produced_count=counts.get(_name_key(name), 0),
            )
            for name in names
        ]
    except Exception:
        logger.exception(
            "[credits] resolve_producers failed (collection=%s, track=%s)",
            collection_name, track_id,
        )
        return []


def tracks_produced_by(collection_name: str, name: str) -> list[ProducerTrack]:
    """All tracks in the collection whose producer tag credits ``name``.

    Sorted by artist, then title. Degrades to ``[]`` on any error.
    """
    key = _name_key(name)
    if not key:
        return []
    try:
        out: list[ProducerTrack] = []
        for row in MetadataDB.get_produced_tracks(collection_name):
            keys = {_name_key(n) for n in split_credit_names(row["producer"])}
            if key not in keys:
                continue
            out.append(ProducerTrack(
                track_id=row["track_id"],
                title=row["title"] or "—",
                title_display=display_title_for_track(row),
                artist=row["artist"] or "—",
                album=row["album"],
                year=row["year"],
                duration=row["duration"],
                cover_art_path=row["cover_art_path"],
                artist_refs=artist_refs_for_track(row),
            ))
        out.sort(key=lambda t: (t.artist.lower(), t.title.lower()))
        return out
    except Exception:
        logger.exception(
            "[credits] tracks_produced_by failed (collection=%s, name=%s)",
            collection_name, name,
        )
        return []
