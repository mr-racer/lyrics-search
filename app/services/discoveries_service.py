"""Discovery cards for the assistant page — the "СВЯЗИ В БИБЛИОТЕКЕ" rail.

Every card is a hook that already knows what it will ask the assistant: tapping
one sends a turn, it does not navigate. Three kinds, all built from data the
code can verify against the DB:

1. ``relation`` — a sample / interpolation / cover where BOTH sides are songs
   of this library (two covers and an arrow).
2. ``producer`` — a producer credited on N ≥ 3 tracks here.
3. ``artist`` — an artist whose biography is already generated, so the facts
   branch has something to answer with.

What is deliberately NOT here: lyric gems (mined findings turned out weak —
noise beats signal), ``lyrical_reference`` / ``inspiration`` / ``other`` links
(too uneven to state as fact), and raw fact text (``refined_facts`` has no
categories, so "interesting" cannot be filtered; a fact is a reason to ask, not
something to print). The rail shows no model-written prose at all.

Prompts are built here, in both languages — same rule as ``humanize``: the SPA
animates strings, it never composes them.
"""

from __future__ import annotations

import logging
import random
import threading
import time

from app.resources.metadata_db import MetadataDB
from app.services.assistant.humanize import is_ru, plural_ru
from app.services.song_facts_service import get_song_facts_key
from app.services.track_credits_service import split_credit_names

logger = logging.getLogger(__name__)

CACHE_TTL = 600.0
MAX_CARDS = 12
# A producer has to recur before "12 треков у тебя" is a finding rather than a
# credit — two tracks is a coincidence, three is a pattern worth a playlist.
MIN_PRODUCER_TRACKS = 3
# The three link kinds that state something unambiguous. The rest of the closed
# list fact_relations extracts ("lyrical_reference", "inspiration", "other") is
# exactly where its quality wobbles, so it never reaches the rail.
ALLOWED_RELATIONS = ("sample", "interpolation", "cover")

_RELATION_RU = {"sample": "сэмплирует", "interpolation": "переигрывает мотив",
                "cover": "кавер на"}
_RELATION_EN = {"sample": "samples", "interpolation": "interpolates",
                "cover": "covers"}

_CACHE: dict = {}
_LOCK = threading.Lock()


# ── library index ────────────────────────────────────────────────────────────


def _index(points: list) -> dict:
    """``{song_slug: track}`` plus per-artist track lists, from light payloads.

    The key is ``get_song_facts_key`` — the same slug ``songs.slug`` and the
    sample links use. Importing that one function (rather than re-deriving a
    slug here) is what keeps the join honest; there are three different
    ``_slugify`` in this project and they are not interchangeable.
    """
    by_slug: dict = {}
    by_artist: dict = {}
    for track_id, payload in points:
        payload = payload or {}
        title = (payload.get("title") or "").strip()
        artist = (payload.get("artist") or "").strip()
        if not title or not artist:
            continue
        track = {
            "track_id": track_id,
            "title": title,
            "artist": artist,
            "cover_art_path": payload.get("cover_art_path"),
        }
        by_slug.setdefault(get_song_facts_key(artist, title), track)
        slug = payload.get("primary_artist_slug") or ""
        if slug:
            by_artist.setdefault(slug, []).append(track)
    return {"by_slug": by_slug, "by_artist": by_artist}


# ── card builders ────────────────────────────────────────────────────────────


def _relation_cards(collection_name: str, idx: dict, ru: bool) -> list[dict]:
    try:
        links = MetadataDB.get_in_library_sample_links(collection_name)
    except Exception:
        logger.exception("[discoveries] sample links unavailable")
        return []

    by_slug = idx["by_slug"]
    seen: set = set()
    cards: list[dict] = []
    for link in links:
        relation = (link.get("relation") or "").strip().lower()
        if relation not in ALLOWED_RELATIONS:
            continue
        src = by_slug.get(link["src_slug"])
        dst = by_slug.get(link["dst_slug"])
        if not src or not dst or src["track_id"] == dst["track_id"]:
            continue
        # One finding per pair regardless of which side the extraction stored.
        pair = frozenset((src["track_id"], dst["track_id"]))
        if pair in seen:
            continue
        seen.add(pair)
        verb = (_RELATION_RU if ru else _RELATION_EN)[relation]
        cards.append({
            "kind": "relation",
            "intent": "facts",
            "prompt": (f"расскажи про «{src['title']}» {src['artist']}" if ru
                       else f"tell me about “{src['title']}” by {src['artist']}"),
            "headline": f"{src['title']} {verb} {dst['title']}",
            "subline": (f"{src['artist']} · {dst['artist']} — оба у тебя есть" if ru
                        else f"{src['artist']} · {dst['artist']} — you have both"),
            "badge": verb,
            "items": [src, dst],
            "track_id": src["track_id"],
        })
    return cards


def _producer_cards(collection_name: str, idx: dict, ru: bool) -> list[dict]:
    by_slug = idx["by_slug"]
    if not by_slug:
        return []
    try:
        relations = MetadataDB.get_song_relations_bulk(list(by_slug.keys()))
    except Exception:
        logger.exception("[discoveries] producer credits unavailable")
        return []

    counts: dict = {}
    display: dict = {}
    tracks: dict = {}
    for slug, rel in relations.items():
        track = by_slug.get(slug)
        if not track:
            continue
        for name in split_credit_names(rel.get("producer")):
            key = " ".join(name.lower().split())
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
            display.setdefault(key, name)
            tracks.setdefault(key, []).append(track)

    cards = []
    for key, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        if count < MIN_PRODUCER_TRACKS:
            continue
        name = display[key]
        word = plural_ru(count, "трек", "трека", "треков") if ru else (
            "track" if count == 1 else "tracks")
        cards.append({
            "kind": "producer",
            "intent": "playlist",
            "prompt": (f"собери треки, которые спродюсировал {name}" if ru
                       else f"build a playlist of tracks produced by {name}"),
            "headline": name,
            "subline": (f"{count} {word} у тебя" if ru else f"{count} {word} in your library"),
            "badge": "продюсер" if ru else "producer",
            "items": tracks[key][:3],
            "count": count,
        })
    return cards


def _artist_cards(collection_name: str, idx: dict, ru: bool, lang: str) -> list[dict]:
    try:
        slugs = MetadataDB.get_artist_slugs_with_bio(collection_name, lang)
    except Exception:
        logger.exception("[discoveries] artist bios unavailable")
        return []

    by_artist = idx["by_artist"]
    cards = []
    # Most-represented artists first: a listener recognises them, and a fuller
    # catalogue means a fuller grounding pack behind the answer.
    for slug in sorted(slugs, key=lambda s: -len(by_artist.get(s, []))):
        artist_tracks = by_artist.get(slug) or []
        if not artist_tracks:
            continue
        name = artist_tracks[0]["artist"]
        count = len(artist_tracks)
        word = plural_ru(count, "трек", "трека", "треков") if ru else (
            "track" if count == 1 else "tracks")
        cards.append({
            "kind": "artist",
            "intent": "facts",
            "prompt": f"расскажи про {name}" if ru else f"tell me about {name}",
            "headline": name,
            "subline": (f"{count} {word} в библиотеке" if ru
                        else f"{count} {word} in your library"),
            "badge": "артист" if ru else "artist",
            "items": artist_tracks[:1],
            "artist_slug": slug,
            "count": count,
        })
    return cards


# ── assembly ─────────────────────────────────────────────────────────────────


def _interleave(groups: list[list[dict]], limit: int) -> list[dict]:
    """Round-robin so the rail never opens with six cards of one kind."""
    out: list[dict] = []
    i = 0
    while len(out) < limit and any(len(g) > i for g in groups):
        for group in groups:
            if i < len(group):
                out.append(group[i])
                if len(out) >= limit:
                    break
        i += 1
    return out


def build_discoveries(qdrant, collection_name: str, *, lang: str = "en",
                      limit: int = MAX_CARDS) -> list[dict]:
    """Assemble the rail. Returns ``[]`` when there is nothing to show — the
    SPA then renders no rail at all rather than an empty promise."""
    from app.resources.qdrant_utils import light_points

    ru = is_ru(lang)
    lang_key = "ru" if ru else "en"
    now = time.monotonic()
    cache_key = (collection_name, lang_key)
    with _LOCK:
        entry = _CACHE.get(cache_key)
        if entry and now - entry[0] < CACHE_TTL:
            return entry[1][:limit]

    try:
        points = light_points(qdrant, collection_name)
    except Exception:
        logger.exception("[discoveries] library unreadable")
        return []
    idx = _index(points)

    groups = [
        _relation_cards(collection_name, idx, ru),
        _producer_cards(collection_name, idx, ru),
        _artist_cards(collection_name, idx, ru, lang_key),
    ]
    # Shuffled per day, not per request: a rail that reshuffles on every visit
    # reads as broken, one frozen forever stops being a reason to come back.
    seed = random.Random(f"{collection_name}:{time.strftime('%Y-%m-%d')}")
    for group in groups:
        seed.shuffle(group)

    cards = _interleave(groups, MAX_CARDS)
    with _LOCK:
        _CACHE[cache_key] = (now, cards)
    return cards[:limit]


def invalidate(collection_name: str | None = None) -> None:
    with _LOCK:
        if collection_name is None:
            _CACHE.clear()
        else:
            for key in [k for k in _CACHE if k[0] == collection_name]:
                _CACHE.pop(key, None)
