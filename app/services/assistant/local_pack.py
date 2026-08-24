"""The grounding pack the library can build on its own, before anything is read.

The assistant used to search the web first and assemble its pack afterwards, so
a question it could already answer — "what does this fact mean", "what samples
are in this track" — cost a full round of SearXNG and page fetches. Worse, the
structural material never entered the pack at all: the samples card was BUILT
from ``sample_links`` and then answered from the open web, with its own verified
list nowhere in the prompt.

This module is the other half. It reads SQLite and nothing else — no Qdrant, no
network — and hands back a numbered pack:

1. **Structure first.** Sample and interpolation links (both directions, with
   the sentence each was extracted from), producer, label, lyric gems, the
   release line. These are facts ABOUT the subject rather than candidates FOR
   it, so they carry no probability and are never ranked away: a cross-encoder
   score would be answering a question nobody asked.
2. **Then the subject's facts, ranked** against the question.
3. **Then the facts of the songs it is structurally tied to.** The explanation
   of a sample routinely lives in the OTHER song's facts — and that song does
   not have to be guessed at, because ``sample_links`` stores its slug. A
   structural lookup returns exactly that record where a similarity search
   returns something merely like it. This is why
   ``app/services/facts_retrieval.py`` (dense over ``facts_acct_{id}``) is not
   used here: it is a Qdrant round-trip for a worse answer.

Both sample storages are read, always. The normalized ``sample_links`` table is
the newer one; the production library predates it and keeps every link in the
``songs.samples_json`` cache, so a reader that consults one finds nothing in
half the installs. Duplicates across the two collapse on the destination key.

Nothing here raises. A reader that fails costs the pack some items and is logged;
it never costs the turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.services.assistant.contracts import Evidence, Fact, Subject

logger = logging.getLogger(__name__)

# Link kinds that state something unambiguous, matching the fence
# ``fact_relations.gates.ACCEPTED_RELATIONS`` already applies at extraction.
_RELATION_RU = {"sample": "сэмплирует", "interpolation": "переигрывает мотив",
                "cover": "кавер на"}
_RELATION_EN = {"sample": "samples", "interpolation": "interpolates",
                "cover": "covers"}

# How many gems and how many neighbour songs may spend a slot. Both are guards
# against one lopsided track flooding a pack the model reads in one window: a
# track with thirty gems is a track whose lyrics are one long namedrop.
MAX_GEMS = 5
MAX_NEIGHBOURS = 6


@dataclass(slots=True)
class LocalPack:
    """What the library alone had to say. ``items`` are numbered 1..k."""

    items: list = field(default_factory=list)
    # The normalized sample links behind the pack, deduplicated across both
    # storages. The caller resolves these into playable tracks; the pack itself
    # only talks about them.
    links: list = field(default_factory=list)

    @property
    def ranked(self) -> list:
        """Items a cross-encoder actually scored — the only ones a veto may read."""
        return [i for i in self.items if i.ce_prob is not None]


def build(collection_name: str, subject: Optional[Subject], query: str, *,
          lang: str = "en", db=None, ranker: Optional[Callable] = None) -> LocalPack:
    """The library's own answer material for ``subject``, numbered and ordered.

    ``db`` and ``ranker`` are injectable so this is testable without SQLite or a
    GPU; production passes neither.
    """
    if subject is None or not subject.resolved:
        # Not a failure. Loading the wrong subject's material is worse than
        # loading none — it is invisible, and it makes the answer confidently
        # about someone else.
        return LocalPack()

    db = db if db is not None else _default_db()
    ru = (lang or "").lower().startswith("ru")

    links = _links(db, collection_name, subject.song_slug) if subject.song_slug else []
    structural = _structural_items(db, collection_name, subject, links, ru)
    facts = _ranked_facts(db, collection_name, subject, links, query, ranker)

    items: list = []
    for text, source in structural:
        items.append(Evidence(n=len(items) + 1, text=text, kind="fact",
                              source=source))
    for fact in facts:
        items.append(Evidence(n=len(items) + 1, text=fact.text, kind="fact",
                              source=fact.source or "facts",
                              ce_prob=fact.ce_prob))

    logger.info("[local_pack] %s: %d structural + %d ranked facts, %d links",
                subject.song_slug or subject.artist_slug, len(structural),
                len(facts), len(links))
    return LocalPack(items=items, links=links)


# ── sample links ─────────────────────────────────────────────────────────────


def _links(db, collection_name: str, song_slug: str) -> list:
    """Both directions, from both storages, deduplicated on the other side."""
    out: list = []
    seen: set = set()

    def take(direction: str, entries) -> None:
        for entry in entries or ():
            if not isinstance(entry, dict):
                continue
            song = (entry.get("song") or "").strip()
            artist = (entry.get("artist") or "").strip()
            slug = (entry.get("slug") or "").strip() or _song_key(artist, song)
            key = (direction, slug or f"{artist.lower()}|{song.lower()}")
            if not song or key in seen:
                continue
            seen.add(key)
            out.append({
                "direction": direction, "song": song, "artist": artist,
                "slug": slug or None,
                "relation": (entry.get("relation") or "sample").strip().lower(),
                "evidence": (entry.get("evidence") or "").strip() or None,
            })

    for reader, label in ((_read_sample_links, "sample_links"),
                          (_read_samples_json, "samples_json")):
        rel = reader(db, collection_name, song_slug)
        if rel is None:
            logger.info("[local_pack] %s unavailable for %s", label, song_slug)
            continue
        take("samples", rel.get("samples"))
        take("sampled_by", rel.get("sampled_by"))
    return out


def _read_sample_links(db, collection_name: str, slug: str) -> Optional[dict]:
    try:
        return db.get_sample_links(collection_name, slug) or {}
    except Exception:  # noqa: BLE001
        logger.warning("[local_pack] sample_links read failed", exc_info=True)
        return None


def _read_samples_json(db, collection_name: str, slug: str) -> Optional[dict]:
    try:
        return (db.get_song_relations_raw([slug]) or {}).get(slug) or {}
    except Exception:  # noqa: BLE001
        logger.warning("[local_pack] samples_json read failed", exc_info=True)
        return None


def _song_key(artist: str, song: str) -> str:
    if not (artist and song):
        return ""
    try:
        from app.services.song_facts_service import get_song_facts_key

        return get_song_facts_key(artist, song) or ""
    except Exception:  # noqa: BLE001
        return ""


# ── structural items ─────────────────────────────────────────────────────────


def _structural_items(db, collection_name: str, subject: Subject, links: list,
                      ru: bool) -> list:
    """``[(text, source), …]`` — verified statements, in reading order."""
    items: list = []
    verbs = _RELATION_RU if ru else _RELATION_EN

    for link in links:
        verb = verbs.get(link["relation"], verbs["sample"])
        who = f" — {link['artist']}" if link["artist"] else ""
        if link["direction"] == "samples":
            text = (f"«{subject.song_title or 'этот трек'}» {verb} "
                    f"«{link['song']}»{who}" if ru else
                    f"“{subject.song_title or 'this track'}” {verb} "
                    f"“{link['song']}”{who}")
        else:
            text = (f"«{link['song']}»{who} {verb} "
                    f"«{subject.song_title or 'этот трек'}»" if ru else
                    f"“{link['song']}”{who} {verb} "
                    f"“{subject.song_title or 'this track'}”")
        if link["evidence"]:
            # The sentence the link was extracted from. This is the difference
            # between a list and a story, and it is already in the database.
            text = f"{text}. {link['evidence']}"
        items.append((text, "credits"))

    if subject.song_slug:
        relations = _relations(db, subject.song_slug)
        producer = (relations.get("producer") or "").strip()
        label = (relations.get("label") or "").strip()
        if producer:
            items.append((f"Продюсер: {producer}" if ru
                          else f"Produced by {producer}", "credits"))
        if label:
            items.append((f"Лейбл: {label}" if ru else f"Label: {label}",
                          "credits"))

    if subject.track_id:
        for gem in _gems(db, collection_name, subject.track_id)[:MAX_GEMS]:
            display = (gem.get("display") or gem.get("canonical") or "").strip()
            if not display:
                continue
            text = (f"В тексте упоминается {display}" if ru
                    else f"The lyrics reference {display}")
            quote = (gem.get("quote") or "").strip()
            if quote:
                text = f"{text} — «{quote}»"
            items.append((text, "gems"))

    catalog = _catalog_line(db, collection_name, subject, ru)
    if catalog:
        items.append((catalog, "catalog"))
    return items


def _relations(db, slug: str) -> dict:
    try:
        return (db.get_song_relations_bulk([slug]) or {}).get(slug) or {}
    except Exception:  # noqa: BLE001
        logger.warning("[local_pack] credits read failed", exc_info=True)
        return {}


def _gems(db, collection_name: str, track_id: str) -> list:
    try:
        return db.get_track_gems(track_id, collection_name) or []
    except Exception:  # noqa: BLE001
        logger.warning("[local_pack] gems read failed", exc_info=True)
        return []


def _catalog_line(db, collection_name: str, subject: Subject, ru: bool) -> str:
    """Album / year / genre — the things facts rarely repeat and users ask for."""
    lookup = getattr(db, "get_track", None)
    if lookup is None or not subject.track_id:
        return ""
    try:
        row = lookup(collection_name, subject.track_id) or {}
    except Exception:  # noqa: BLE001
        logger.warning("[local_pack] catalog read failed", exc_info=True)
        return ""
    bits = []
    if row.get("album"):
        bits.append(f"альбом {row['album']}" if ru else f"album {row['album']}")
    if row.get("year"):
        bits.append(f"вышел в {row['year']}" if ru else f"released {row['year']}")
    if row.get("genre"):
        bits.append(f"жанр {row['genre']}" if ru else f"genre {row['genre']}")
    if not bits:
        return ""
    head = " — ".join(p for p in (subject.song_title, subject.artist_name) if p)
    return f"{head}: " + ", ".join(bits)


# ── facts ────────────────────────────────────────────────────────────────────


def _ranked_facts(db, collection_name: str, subject: Subject, links: list,
                  query: str, ranker: Optional[Callable]) -> list:
    pool = _fact_pool(db, collection_name, subject, links)
    if not pool:
        return []
    rank = ranker if ranker is not None else _default_ranker(collection_name)
    try:
        return list(rank(pool, query) or [])
    except Exception:  # noqa: BLE001 — an unavailable model degrades the order,
        logger.warning("[local_pack] ranking failed — keeping the pool order",
                       exc_info=True)              # never the call
        return pool


def _fact_pool(db, collection_name: str, subject: Subject, links: list) -> list:
    """The subject's facts plus the facts of what it is structurally tied to."""
    pool: list = []
    seen: set = set()

    def take(kind: str, slug: str) -> None:
        if not slug:
            return
        rows = _facts_for(db, kind, slug, collection_name)
        for row in rows:
            text = (row.get("fact") or "").strip()
            key = " ".join(text.lower().split())
            if not text or key in seen:
                continue
            seen.add(key)
            pool.append(Fact(row_id=len(pool), kind=kind, slug=slug, text=text,
                             source=row.get("source") or "",
                             category=row.get("category") or ""))

    take("song", subject.song_slug)
    take("artist", subject.artist_slug)

    for link in links[:MAX_NEIGHBOURS]:
        take("song", link.get("slug") or "")
        take("artist", _artist_slug(link.get("artist")))
    return pool


def _facts_for(db, kind: str, slug: str, collection_name: str) -> list:
    reader = (db.get_song_facts_rich if kind == "song"
              else db.get_artist_facts_rich)
    try:
        return reader(slug, collection_name) or []
    except Exception:  # noqa: BLE001
        logger.warning("[local_pack] %s facts read failed for %s", kind, slug,
                       exc_info=True)
        return []


def _artist_slug(name: Optional[str]) -> str:
    """The artist slug for a name, via the ONE slugifier that matches.

    The repo has more than one ``_slugify`` and only ``artist_facts_service``'s
    produces slugs the artist tables are keyed by. Using the other one here
    would silently match nothing, which reads exactly like "this artist has no
    facts".
    """
    if not name:
        return ""
    try:
        from app.services.artist_facts_service import _slugify

        return _slugify(name) or ""
    except Exception:  # noqa: BLE001
        return ""


# ── production wiring ────────────────────────────────────────────────────────


def resolve_links(collection_name: str, links: list, *,
                  exclude_track_id: Optional[str] = None) -> list:
    """The in-library counterparts of ``links``, as playable track rows.

    SQLite only, and by title + artist rather than by slug: the slug is the
    facts key, and a library track is found by what its tags say. A link whose
    other side the listener does not own resolves to nothing, which is the
    common case and not a problem — the answer still talks about it.
    """
    if not links:
        return []
    try:
        from app.resources import track_store
    except Exception:  # noqa: BLE001 — the mirror has not landed on this branch
        logger.info("[local_pack] track_store unavailable — no related tracks",
                    exc_info=True)
        return []

    out: list = []
    seen: set = set()
    for link in links:
        song, artist = link.get("song"), link.get("artist")
        if not song:
            continue
        try:
            rows = track_store.find_track_by_title_artist(
                collection_name, song, artist or None, limit=1, strict=False)
        except Exception:  # noqa: BLE001
            logger.warning("[local_pack] track lookup failed for %r", song,
                           exc_info=True)
            continue
        for row in rows or []:
            track_id = str(row.get("track_id") or "")
            if not track_id or track_id in seen or track_id == exclude_track_id:
                continue
            seen.add(track_id)
            out.append(row)
    logger.info("[local_pack] %d of %d links are in the library", len(out),
                len(links))
    return out


def _default_db():
    """MetadataDB plus the track reader, as one object with the readers we call."""
    from app.resources.metadata_db import MetadataDB

    MetadataDB.init()
    return _ProductionDB(MetadataDB)


class _ProductionDB:
    """Thin adapter: everything is MetadataDB except the catalog line."""

    def __init__(self, metadata_db):
        self._db = metadata_db

    def __getattr__(self, name):
        return getattr(self._db, name)

    def get_track(self, collection_name: str, track_id: str):
        """The track row, from SQLite or not at all.

        No client is passed and ``strict=False`` on purpose: with a client
        ``track_store`` re-enables its bounded ``client.retrieve`` fallback,
        which is the Qdrant round-trip this pack exists to avoid, and with
        ``strict=True`` a library whose mirror has not been built yet raises
        instead of simply having no catalog line.
        """
        try:
            from app.resources import track_store

            return track_store.get_track(collection_name, track_id,
                                         strict=False)
        except Exception:  # noqa: BLE001 — the mirror may not have landed yet
            logger.info("[local_pack] track_store unavailable — no catalog line",
                        exc_info=True)
            return None


def _default_ranker(collection_name: str) -> Callable:
    """Rank a pool of facts with the same retriever the agent already uses."""
    def rank(facts: list, query: str) -> list:
        from app.services.assistant.config import AgentConfig
        from app.services.assistant.facts_source import FactsRetriever
        from app.services.retrieval import DEFAULT_HUB

        retriever = FactsRetriever(None, hub=DEFAULT_HUB, config=AgentConfig())
        return retriever.rank(facts, query)

    return rank
