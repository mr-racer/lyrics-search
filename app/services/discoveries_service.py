"""Discovery cards for the assistant page — the "СВЯЗИ В БИБЛИОТЕКЕ" rail.

Every card is a hook that already knows what it will ask the assistant: tapping
one sends a turn, it does not navigate. Four kinds, all built from data the
code can verify against the DB:

0. ``samples`` — a track built from ≥ 2 sampled records («какие сэмплы в X?»).
   The other side does not have to be a library song: the count is a property
   of the track, and the assistant's answer resolves the details anyway.
1. ``relation`` — a sample / interpolation / cover where BOTH sides are songs
   of this library (two covers and an arrow). This one carries a ``fact``: the
   statement itself, echoed back as ``AssistantRequest.focus_fact`` so the
   assistant explains the link rather than reciting the track's whole dossier.
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
from app.services.text_normalize import fold
from app.services.track_credits_service import split_credit_names

logger = logging.getLogger(__name__)

CACHE_TTL = 600.0
MAX_CARDS = 16
# A producer has to recur before "12 треков у тебя" is a finding rather than a
# credit — two tracks is a coincidence, three is a pattern worth a playlist.
MIN_PRODUCER_TRACKS = 3
# Same bar for "tell me about X": one track in the library is not an artist the
# listener would recognise as theirs.
MIN_ARTIST_TRACKS = 3
# "Какие сэмплы в X?" is only worth asking about a track that is actually built
# from other records — one link answers itself, two start a story.
MIN_SAMPLE_LINKS = 2
# Cards are shuffled per day for variety, but only inside the strongest N of
# each kind — shuffling the whole list is what put "Breakbot · 1 track" ahead
# of "Limp Bizkit · 42 tracks" on the first prod run.
TOP_POOL = 10
# The link kinds that state something unambiguous. Note that
# ``fact_relations.gates.ACCEPTED_RELATIONS`` already narrows the extraction to
# sample/interpolation *before* anything is stored, so this is a second fence
# for the normalized table, not the only one.
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


def _relation_pairs(collection_name: str, by_slug: dict) -> list[tuple]:
    """``(src_slug, dst_slug, relation)`` for links whose both sides are here.

    Two sources, because deployments differ: the normalized ``sample_links``
    table (which already carries a resolved ``dst_slug``) and the older
    ``songs.samples_json`` read cache, which is all a library indexed before
    that table existed has — 1500 songs' worth on the production instance.
    The cache stores no relation kind, but it does not need to: the extraction
    gate writes ONLY sample/interpolation links into it.
    """
    pairs: list[tuple] = []
    try:
        for link in MetadataDB.get_in_library_sample_links(collection_name):
            relation = (link.get("relation") or "").strip().lower()
            if relation in ALLOWED_RELATIONS:
                pairs.append((link["src_slug"], link["dst_slug"], relation))
    except Exception:
        logger.exception("[discoveries] sample_links unavailable")

    try:
        cached = MetadataDB.get_song_relations_raw(list(by_slug.keys()))
    except Exception:
        logger.exception("[discoveries] samples_json unavailable")
        cached = {}
    for slug, rel in cached.items():
        for entry in rel.get("samples") or []:
            song, artist = entry.get("song"), entry.get("artist")
            # No artist means the fact named a title only — nothing to resolve
            # against, and guessing which "A hawk chases a dove" it is would be
            # exactly the invention this rail exists to avoid.
            if not song or not artist:
                continue
            pairs.append((slug, get_song_facts_key(artist, song), "sample"))
    return pairs


def _relation_cards(collection_name: str, idx: dict, ru: bool) -> list[dict]:
    by_slug = idx["by_slug"]
    seen: set = set()
    cards: list[dict] = []
    for src_slug, dst_slug, relation in _relation_pairs(collection_name, by_slug):
        src = by_slug.get(src_slug)
        dst = by_slug.get(dst_slug)
        if not src or not dst or src["track_id"] == dst["track_id"]:
            continue
        # One finding per pair regardless of which side the extraction stored.
        pair = frozenset((src["track_id"], dst["track_id"]))
        if pair in seen:
            continue
        seen.add(pair)
        verb = (_RELATION_RU if ru else _RELATION_EN)[relation]
        # The card states a fact, so tapping it must ask about THAT fact. The
        # old prompt («расскажи про «X»») asked about the track instead, and the
        # answer came back as a list of everything known about it with the
        # sample link nowhere in it — the finding the card was built on was the
        # one thing the answer left out.
        fact = (f"«{src['title']}» ({src['artist']}) {verb} «{dst['title']}» ({dst['artist']})"
                if ru else
                f"“{src['title']}” by {src['artist']} {verb} “{dst['title']}” by {dst['artist']}")
        cards.append({
            "kind": "relation",
            "intent": "general",
            "prompt": (f"объясни: {fact}" if ru else f"explain this: {fact}"),
            "fact": fact,
            "headline": f"{src['title']} {verb} {dst['title']}",
            "subline": (f"{src['artist']} · {dst['artist']} — оба у тебя есть" if ru
                        else f"{src['artist']} · {dst['artist']} — you have both"),
            "badge": verb,
            "items": [src, dst],
            "track_id": src["track_id"],
        })
    return cards


def _sample_cards(collection_name: str, idx: dict, ru: bool) -> list[dict]:
    """Tracks built from many samples — «какие сэмплы использованы в X?».

    Counts merge both storages (the normalized ``sample_links`` table and the
    older ``samples_json`` read cache) and DON'T require the other side to be a
    library song: how many records a track is built from is a property of the
    track itself. Unresolved entries still count; duplicates across the two
    storages collapse on the destination key.
    """
    by_slug = idx["by_slug"]
    if not by_slug:
        return []
    used: dict[str, set] = {}

    def _dst_key(artist: str | None, song: str | None, slug: str | None) -> str | None:
        if slug:
            return slug
        if song and artist:
            return get_song_facts_key(artist, song)
        if song:
            return f"?:{song.strip().lower()}"
        return None

    try:
        for link in MetadataDB.get_outgoing_sample_links(collection_name):
            relation = (link.get("relation") or "").strip().lower()
            if relation not in ("sample", "interpolation"):
                continue
            key = _dst_key(link.get("dst_artist"), link.get("dst_title"), link.get("dst_slug"))
            if key and link["src_slug"] in by_slug:
                used.setdefault(link["src_slug"], set()).add(key)
    except Exception:
        logger.exception("[discoveries] sample_links unavailable")

    try:
        cached = MetadataDB.get_song_relations_raw(list(by_slug.keys()))
    except Exception:
        logger.exception("[discoveries] samples_json unavailable")
        cached = {}
    for slug, rel in cached.items():
        for entry in rel.get("samples") or []:
            key = _dst_key(entry.get("artist"), entry.get("song"), entry.get("slug"))
            if key:
                used.setdefault(slug, set()).add(key)

    cards = []
    for slug, dsts in sorted(used.items(), key=lambda kv: -len(kv[1])):
        count = len(dsts)
        if count < MIN_SAMPLE_LINKS:
            continue
        track = by_slug.get(slug)
        if not track:
            continue
        word = plural_ru(count, "сэмпл", "сэмпла", "сэмплов") if ru else (
            "sample" if count == 1 else "samples")
        cards.append({
            "kind": "samples",
            "intent": "general",
            "prompt": (f"какие сэмплы использованы в «{track['title']}»?" if ru
                       else f"what samples are used in “{track['title']}”?"),
            "headline": track["title"],
            "subline": track["artist"],
            "badge": f"{count} {word}",
            "items": [track],
            "track_id": track["track_id"],
            "count": count,
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
        performer = fold(track.get("artist") or "")
        for name in split_credit_names(rel.get("producer")):
            key = " ".join(name.lower().split())
            if not key:
                continue
            # A performer credited on their own track is not a finding: the
            # prod run offered "Lorde produced 21 of your tracks", all of them
            # Lorde's. Self-credits are dropped, other people's are counted.
            folded = fold(name)
            if folded and performer and (folded == performer or folded in performer):
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
        if len(artist_tracks) < MIN_ARTIST_TRACKS:
            continue
        name = artist_tracks[0]["artist"]
        count = len(artist_tracks)
        word = plural_ru(count, "трек", "трека", "треков") if ru else (
            "track" if count == 1 else "tracks")
        cards.append({
            "kind": "artist",
            "intent": "general",
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
        _sample_cards(collection_name, idx, ru),
        _relation_cards(collection_name, idx, ru),
        _producer_cards(collection_name, idx, ru),
        _artist_cards(collection_name, idx, ru, lang_key),
    ]
    # Shuffled per day, not per request: a rail that reshuffles on every visit
    # reads as broken, one frozen forever stops being a reason to come back.
    # Only the strongest TOP_POOL of each kind take part — the groups arrive
    # ranked, and shuffling all of them throws that ranking away.
    seed = random.Random(f"{collection_name}:{time.strftime('%Y-%m-%d')}")
    for i, group in enumerate(groups):
        pool = group[:TOP_POOL]
        seed.shuffle(pool)
        groups[i] = pool

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
