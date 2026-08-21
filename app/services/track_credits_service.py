"""Track-credit logic: producer credits and record labels.

Backs the player's producers drawer and the album gatefold's label block.
Everything here reads only the SQLite ``track_metadata`` mirror — no Qdrant —
so credits keep working while Qdrant is down.

Producer names come from the free-text ``producer`` tag ("Dr. Dre, Mel-Man" /
"A; B" / "A & B"); ``split_credit_names`` is the single source of truth for
splitting it (the player keeps a client-side copy only as a display fallback
while the resolve request is in flight). Label strings may carry several
labels ("GOOD Music, Def Jam" / "Columbia Records & ARXOXO LLC") and are
split by ``split_labels``.

Label names are matched with ``label_key`` (lowercase + collapsed whitespace).
Deliberately NOT ``artist_facts_service._slugify`` — labels are not artist
names and must not share the artist slug space.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Iterable, Optional

from ..domain.models import ProducerCredit, ProducerTrack
from ..resources.metadata_db import MetadataDB, canonical_track_id
from .artist_split import artist_refs_for_track, display_title_for_track, artist_slugs
# Same normalization rules the playlist resolver matches web tracklists with:
# fold + feat-stripped title keys, substring-tolerant artist equality. Shared
# on purpose — a second local copy would silently drift (cf. the two _slugify).
from .playlist_agent.resolver import _artist_matches, _title_key
# The song-facts slug is the key the GLiNER2+Genius extraction is stored under
# (songs.producers / producers_genius) — needed to merge those credits into
# the producer view below.
from .song_facts_service import get_song_facts_key

logger = logging.getLogger(__name__)

# Same separators the player UI used client-side: , ; / & feat.
# NB: ``feat\b\.?`` (boundary BEFORE the optional dot) — ``feat\.?\b`` never
# matches the dot ("." and " " are both non-word, so no boundary sits between
# them) and left a ". Name" remainder.
_CREDIT_SPLIT_RE = re.compile(r"\s*(?:[,;/]|&|\bfeat\b\.?)\s*", re.IGNORECASE)
# Labels split on , ; and a whitespace-flanked & — Genius renders its list of
# release entities as "A, B & C", so " & " glues together things like
# "Columbia Records & ARXOXO LLC" (major label + the artist's own LLC).
# "&" WITHOUT spaces stays inside real label names ("A&M Records"); "/" is
# never a separator here.
_LABEL_SPLIT_RE = re.compile(r"\s*[,;]\s*|\s+&\s+")

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


def _artist_thumb(collection_name: str, slug: Optional[str]) -> Optional[str]:
    """Real photo (thumb) for a library artist, or None.

    Deliberately NOT the cutout: the producer popup shows a portrait card,
    and a transparent floating-head PNG reads wrong there. No thumb → the
    player falls back to a monogram avatar.
    """
    if not slug:
        return None
    try:
        ad = MetadataDB.get_artist_audiodb(slug, collection_name) or {}
    except Exception:
        return None
    return ad.get("thumb_path") or None


# ── Effective producer view (tag column ∪ songs extraction) ──────────────────
# The player's pills show producers from the read-time overlay
# (``apply_song_relations``): the tag column when the import filled it, else
# the Genius+GLiNER merge in ``songs``. Resolve/counts/track-lists must see
# the SAME view — reading only the tag column left every extraction-only
# track (≈1/4 of a real library: pills present, resolve empty) with dead
# muted pills. Cached briefly per collection: resolve is prefetched on every
# track change and the merge walks the whole library.
_CREDITED_TTL_SEC = 60.0
_credited_cache: dict[str, tuple[float, list[dict]]] = {}


def clear_credited_cache() -> None:
    """Drop the effective-producer cache (tests / after bulk writes)."""
    _credited_cache.clear()


def _credited_rows(collection_name: str) -> list[dict]:
    """Slim rows of every track that carries an EFFECTIVE producer credit.

    Each returned row's ``producer`` holds the effective string: the
    ``track_metadata.producer`` tag when present, else the merged extraction
    from ``songs`` (same merge order as the player overlay — Genius first).
    """
    now = time.monotonic()
    hit = _credited_cache.get(collection_name)
    if hit and now - hit[0] < _CREDITED_TTL_SEC:
        return hit[1]

    rows = MetadataDB.get_tracks_slim(collection_name)
    keyed: dict[str, list[dict]] = {}
    for r in rows:
        if (r.get("producer") or "").strip():
            continue
        if not (r.get("artist") and r.get("title")):
            continue
        keyed.setdefault(get_song_facts_key(r["artist"], r["title"]), []).append(r)
    if keyed:
        try:
            rels = MetadataDB.get_song_relations_bulk(list(keyed.keys()))
        except Exception:
            logger.warning("[credits] relations merge failed (tag-only view)", exc_info=True)
            rels = {}
        for slug, rel in rels.items():
            prod = (rel or {}).get("producer")
            if not prod:
                continue
            for r in keyed.get(slug, []):
                r["producer"] = prod

    credited = [r for r in rows if (r.get("producer") or "").strip()]
    _credited_cache[collection_name] = (now, credited)
    return credited


def producer_index(collection_name: str) -> dict[str, dict]:
    """``{producer_key: {"name": display, "tracks": [track_id, ...]}}``.

    One entry per EFFECTIVE producer across the collection, keyed by the
    normalised name so "Kanye West" and "kanye west" are one person. Built on
    top of the same cached ``_credited_rows`` view the player overlay uses, so
    asking for it repeatedly costs a dict build, not a scan.

    The display name is whichever spelling was seen first — the credits data
    has no canonical form, and picking one arbitrarily beats showing a key.
    """
    index: dict[str, dict] = {}
    for row in _credited_rows(collection_name):
        track_id = row.get("track_id")
        if not track_id:
            continue
        for name in split_credit_names(row.get("producer")):
            key = _name_key(name)
            if not key:
                continue
            entry = index.setdefault(key, {"name": name, "tracks": []})
            entry["tracks"].append(track_id)
    return index


def resolve_producers(collection_name: str, track_id: str) -> list[ProducerCredit]:
    """Resolve a track's producer names against the user's library.

    For every name in the track's effective producer credit (tag column or
    the ``songs`` extraction — the same view the player's pills render): is
    the producer also a library artist (→ ``artist_slug`` + their photo in
    ``image``), and how many OTHER tracks in the collection credit them as
    producer (→ ``produced_count``). Degrades to ``[]`` on any error — the
    player's drawer falls back to plain client-split pills.
    """
    try:
        rows = _credited_rows(collection_name)
        current_id = canonical_track_id(track_id)
        raw = next(
            (r["producer"] for r in rows if r["track_id"] == current_id), None,
        )
        names = split_credit_names(raw)
        if not names:
            return []

        wanted = {_name_key(n) for n in names}
        counts: dict[str, int] = {}
        for row in rows:
            if row["track_id"] == current_id:
                continue
            # Per-row set: a name repeated inside one credit counts once.
            for key in {_name_key(n) for n in split_credit_names(row["producer"])}:
                if key in wanted:
                    counts[key] = counts.get(key, 0) + 1

        out: list[ProducerCredit] = []
        for name in names:
            slug = _library_artist_slug(collection_name, name)
            out.append(ProducerCredit(
                name=name,
                artist_slug=slug,
                image=_artist_thumb(collection_name, slug),
                produced_count=counts.get(_name_key(name), 0),
            ))
        return out
    except Exception:
        logger.exception(
            "[credits] resolve_producers failed (collection=%s, track=%s)",
            collection_name, track_id,
        )
        return []


def _producer_track(row: dict) -> ProducerTrack:
    """A slim ``track_metadata`` row → the playable shape the popovers render."""
    return ProducerTrack(
        track_id=row["track_id"],
        title=row["title"] or "—",
        title_display=display_title_for_track(row),
        artist=row["artist"] or "—",
        album=row.get("album"),
        year=row.get("year"),
        duration=row.get("duration"),
        cover_art_path=row.get("cover_art_path"),
        artist_refs=artist_refs_for_track(row),
    )


def tracks_produced_by(collection_name: str, name: str) -> list[ProducerTrack]:
    """All tracks whose effective producer credit names ``name``.

    Reads the same tag∪extraction view as ``resolve_producers``, so the
    popover list agrees with the pill counts. Sorted by artist, then title.
    Degrades to ``[]`` on any error.
    """
    key = _name_key(name)
    if not key:
        return []
    try:
        out: list[ProducerTrack] = []
        for row in _credited_rows(collection_name):
            keys = {_name_key(n) for n in split_credit_names(row["producer"])}
            if key not in keys:
                continue
            out.append(_producer_track(row))
        out.sort(key=lambda t: (t.artist.lower(), t.title.lower()))
        return out
    except Exception:
        logger.exception(
            "[credits] tracks_produced_by failed (collection=%s, name=%s)",
            collection_name, name,
        )
        return []


# "Artist — Title" (em/en dash with spaces) — the exact format the relations
# pipeline writes (``_fmt_sample_entry``) and the player's SampleRow splits.
_SAMPLE_REF_RE = re.compile(r"\s+[—–]\s+")


def resolve_sample_refs(
    collection_name: str, items: list[str],
) -> list[Optional[ProducerTrack]]:
    """Match sample-reference strings against the user's library.

    Exact-only matching (feat-stripped, fold-normalized title + substring-
    tolerant artist): a wrong «you have this» claim on a playable row is worse
    than a miss, so there is no fuzzy stage. A string without the dash is
    matched as a bare title and accepted only when the library has exactly one
    candidate. Returns a list aligned with ``items``; unmatched → None.
    """
    if not items:
        return []
    try:
        by_title: dict[str, list[dict]] = {}
        for row in MetadataDB.get_tracks_slim(collection_name):
            tkey = _title_key(row.get("title"))
            if tkey:
                by_title.setdefault(tkey, []).append(row)

        out: list[Optional[ProducerTrack]] = []
        for raw in items:
            text = str(raw or "").strip()
            parts = _SAMPLE_REF_RE.split(text, maxsplit=1)
            q_artist, q_title = (
                (parts[0].strip(), parts[1].strip()) if len(parts) == 2
                else (None, text)
            )
            picked = None
            candidates = by_title.get(_title_key(q_title), []) if q_title else []
            if q_artist:
                picked = next(
                    (c for c in candidates if _artist_matches(q_artist, c.get("artist"))),
                    None,
                )
            elif len(candidates) == 1:
                # No artist to disambiguate with — only a unique hit is safe.
                picked = candidates[0]
            out.append(_producer_track(picked) if picked else None)
        return out
    except Exception:
        logger.exception(
            "[credits] resolve_sample_refs failed (collection=%s, n=%d)",
            collection_name, len(items),
        )
        return [None] * len(items)
