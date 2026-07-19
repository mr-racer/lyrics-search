"""Metadata routes — CRUD for artist/song facts stored in SQLite.

The collection is derived from the JWT user as ``acct_<user.id>``; clients no
longer pass a collection (Phase D-hard removed the parameter).
"""

from __future__ import annotations

import random as _random
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app.api.dependencies import get_current_user
from app.api.helpers import derive_collection_for_user
from app.domain.models import TrackMetadata, User
from app.resources.metadata_db import MetadataDB
from app.services._payload_coerce import coerce_float, coerce_year
from app.services.artist_facts_service import _slugify as _slugify_artist
from app.services.artist_split import (
    split_artists,
    artist_slugs as _artist_slugs,
    artist_refs_for_track,
    display_title_for_track,
)
from app.services.song_facts_service import get_song_facts_key, apply_song_relations

router = APIRouter(tags=["Metadata"])


def _track_from_qdrant_payload(point_id: str, pl: dict) -> TrackMetadata:
    """Build a COMPLETE TrackMetadata from a Qdrant payload for the player.

    Crucially includes the per-participant artist fields (``artists``,
    ``primary_artist_slug``, ``artist_refs``). The player merges this enrichment
    over the slim list-source shape, so omitting ``artist_refs`` here would wipe
    the refs a collaboration needs — the frontend would then slugify the whole
    raw tag ("Eminem, Nate Dogg" -> "eminem,-nate-dogg") and 404 the 2nd artist.
    Mirrors ``routes/artists._track_from_payload``.
    """
    raw = pl.get("artist") or ""
    names = pl.get("artists") or split_artists(raw)
    slugs = pl.get("artist_slugs") or _artist_slugs(raw)
    primary = pl.get("primary_artist_slug") or (slugs[0] if slugs else None)
    return TrackMetadata(
        track_id=point_id,
        title=pl.get("title") or "",
        title_display=display_title_for_track(pl),
        artist=raw,
        artists=names or None,
        primary_artist_slug=primary,
        artist_refs=artist_refs_for_track(pl),
        album=pl.get("album"),
        year=coerce_year(pl.get("year")),
        genre=pl.get("genre"),
        duration_sec=coerce_float(pl.get("duration")) or 0.0,
        file_path=pl.get("file_path") or "",
        lyrics=pl.get("lyrics"),
        cover_art_path=pl.get("cover_art_path"),
        producer=pl.get("producer"),
        label=pl.get("label"),
        samples=pl.get("samples"),
        sampled_by=pl.get("sampled_by"),
        bitrate_kbps=pl.get("bitrate_kbps"),
    )


# ── Request/Response models ──────────────────────────────────────────────────

class FactIn(BaseModel):
    """Payload for adding a fact."""
    fact: str


class FactOut(BaseModel):
    """Single fact returned by the API."""
    fact: str
    source: Optional[str] = None


class RandomFact(BaseModel):
    """Random fact with attribution context.

    ``image`` is a relative ``/covers/...`` URL: the track's album art for
    song facts (artist photo as fallback), the cached AudioDB photo for
    artist facts. None when neither exists — the frontend shows a placeholder.
    """
    fact: str
    context: str
    type: Literal["artist", "song"]
    artist: str = ""
    title: Optional[str] = None
    image: Optional[str] = None


# ── Artist facts ─────────────────────────────────────────────────────────────

@router.get("/metadata/artists/{slug}/facts")
def get_artist_facts(
    slug: str,
    current_user: User = Depends(get_current_user),
) -> List[str]:
    """Return all cached facts for an artist in a collection."""
    derived = derive_collection_for_user(current_user)
    return MetadataDB.get_artist_facts(slug, derived)


@router.post("/metadata/artists/{slug}/facts", status_code=201)
def add_artist_fact(
    slug: str,
    body: FactIn,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Add a new fact for an artist."""
    derived = derive_collection_for_user(current_user)
    MetadataDB.add_artist_fact(
        slug=slug,
        collection_name=derived,
        fact_text=body.fact,
        source="manual",
    )
    return {"ok": True}


# ── Song facts ───────────────────────────────────────────────────────────────

@router.get("/metadata/songs/{slug}/facts")
def get_song_facts(
    slug: str,
    current_user: User = Depends(get_current_user),
) -> List[str]:
    """Return all cached facts for a song in a collection."""
    derived = derive_collection_for_user(current_user)
    return MetadataDB.get_song_facts(slug, derived)


@router.post("/metadata/songs/{slug}/facts", status_code=201)
def add_song_fact(
    slug: str,
    body: FactIn,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Add a new fact for a song."""
    derived = derive_collection_for_user(current_user)
    MetadataDB.add_song_fact(
        slug=slug,
        collection_name=derived,
        fact_text=body.fact,
        source="manual",
    )
    return {"ok": True}


# ── Random facts (landing page) ──────────────────────────────────────────────

def _prettify_name(name: str | None) -> str | None:
    """Display-name guard: slug-shaped strings → Title Case.

    ``upsert_artist_audiodb`` inserts ``name=slug`` as a fallback when the
    facts pipeline hasn't named the artist yet, so attribution rows can carry
    "the-weeknd" instead of "The Weeknd". Only strings that LOOK like slugs
    (all-lowercase latin/digits with hyphens) are rewritten — anything with
    uppercase, spaces or non-latin letters passes through untouched.
    """
    if not name:
        return name
    import re
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) and ("-" in name or len(name) > 2):
        return name.replace("-", " ").title()
    return name


def _artist_photo(slug: str | None, collection_name: str) -> str | None:
    """Cached AudioDB photo (thumb/cutout) for an artist — a cheap SQLite read."""
    if not slug:
        return None
    try:
        ad = MetadataDB.get_artist_audiodb(slug, collection_name) or {}
    except Exception:
        return None
    return ad.get("thumb_path") or ad.get("cutout_path")


def _artist_any_cover(request: Request, collection_name: str, artist: str) -> str | None:
    """Fallback thumb for artist facts without an AudioDB photo: the cover of
    any track by this artist (one filtered scroll)."""
    db_client = getattr(request.app.state, "db_client", None) if request else None
    if db_client is None or not artist:
        return None
    from qdrant_client import models as qmodels
    flt = qmodels.Filter(
        should=[
            qmodels.FieldCondition(key="artist", match=qmodels.MatchValue(value=artist)),
            qmodels.FieldCondition(key="artists", match=qmodels.MatchValue(value=artist)),
        ],
    )
    try:
        points, _ = db_client.qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=flt,
            limit=24,
            with_payload=["cover_art_path"],
            with_vectors=False,
        )
    except Exception:
        return None
    for p in points or []:
        cover = (p.payload or {}).get("cover_art_path")
        if cover:
            return cover
    return None


def _song_cover(request: Request, collection_name: str, artist: str, title: str) -> str | None:
    """cover_art_path of the track matching (artist, title) — one filtered scroll.

    The SQLite fact row stores the PRIMARY artist's name, while the Qdrant
    ``artist`` payload keeps the raw tag ("Eminem, Nate Dogg") — so the match
    also accepts the per-participant ``artists`` list (element-wise match).
    Exact title match first; on a miss (SQLite titles can be normalized
    differently than the tag payload) fall back to a case-insensitive
    comparison over the artist's tracks.
    """
    db_client = getattr(request.app.state, "db_client", None) if request else None
    if db_client is None or not artist or not title:
        return None
    from qdrant_client import models as qmodels
    flt = qmodels.Filter(
        must=[qmodels.FieldCondition(key="title", match=qmodels.MatchValue(value=title))],
        should=[
            qmodels.FieldCondition(key="artist", match=qmodels.MatchValue(value=artist)),
            qmodels.FieldCondition(key="artists", match=qmodels.MatchValue(value=artist)),
        ],
    )
    try:
        points, _ = db_client.qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=flt,
            limit=1,
            with_payload=["cover_art_path"],
            with_vectors=False,
        )
    except Exception:
        return None
    if points:
        return (points[0].payload or {}).get("cover_art_path")
    # Fallback: exact-title miss — compare case-insensitively across the
    # artist's tracks (covers "Back In Black" vs "Back in Black" mismatches).
    flt2 = qmodels.Filter(
        should=[
            qmodels.FieldCondition(key="artist", match=qmodels.MatchValue(value=artist)),
            qmodels.FieldCondition(key="artists", match=qmodels.MatchValue(value=artist)),
        ],
    )
    try:
        points, _ = db_client.qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=flt2,
            limit=64,
            with_payload=["cover_art_path", "title"],
            with_vectors=False,
        )
    except Exception:
        return None
    want = title.casefold().strip()
    for p in points or []:
        payload = p.payload or {}
        if (payload.get("title") or "").casefold().strip() == want and payload.get("cover_art_path"):
            return payload.get("cover_art_path")
    return None


@router.get("/metadata/random-facts")
def get_random_facts(
    request: Request,
    current_user: User = Depends(get_current_user),
    limit: int = Query(5, ge=1, le=20, description="Number of random facts"),
    lang: str = Query("en", description="Preferred language for refined facts"),
) -> List[RandomFact]:
    """Return random facts from the collection's fact pool.

    Refined (AI-shortened) facts in the requested language are preferred;
    raw English facts only top up the remainder. Each fact carries an
    attribution image (album cover / artist photo) for the home strip.
    """
    derived = derive_collection_for_user(current_user)

    # Lyric gems join the rotation (up to 2 slots) as regular song-type facts —
    # same card shape, so the home strip needs no new rendering path.
    gems_out: List[RandomFact] = []
    db_client = getattr(request.app.state, "db_client", None)
    if db_client is not None:
        try:
            gem_rows = MetadataDB.get_random_gems(derived, limit=6)
        except Exception:
            gem_rows = []
        n_gems = min(2, max(0, limit - 1))
        for g in gem_rows:
            if len(gems_out) >= n_gems:
                break
            try:
                pts = db_client.qdrant.retrieve(
                    collection_name=derived, ids=[g["track_id"]],
                    with_payload=["artist", "title"], with_vectors=False,
                )
            except Exception:
                break
            if not pts:
                continue
            pl = pts[0].payload or {}
            g_artist = (pl.get("artist") or "").strip()
            g_title = (pl.get("title") or "").strip()
            if not g_artist or not g_title:
                continue
            g_image = (
                _song_cover(request, derived, g_artist, g_title)
                or _artist_photo(_slugify_artist(g_artist), derived)
                or _artist_any_cover(request, derived, g_artist)
            )
            gems_out.append(RandomFact(
                fact=_gem_fact_text(g, lang),
                context=f"{g_artist} — {g_title}",
                type="song", artist=g_artist, title=g_title, image=g_image,
            ))

    raw = MetadataDB.get_random_facts(
        collection_name=derived, limit=limit - len(gems_out), lang=lang,
    )
    out: List[RandomFact] = list(gems_out)
    for r in raw:
        artist_slug = r.pop("artist_slug", None)
        image = None
        if r.get("type") == "song":
            image = _song_cover(request, derived, r.get("artist") or "", r.get("title") or "")
        if not image:
            image = _artist_photo(artist_slug, derived)
        if not image:
            image = _artist_any_cover(request, derived, r.get("artist") or "")
        # Display-name guard: slug-shaped attribution ("the-weeknd") → Title Case.
        r["artist"] = _prettify_name(r.get("artist"))
        r["title"] = _prettify_name(r.get("title"))
        out.append(RandomFact(**{k: v for k, v in r.items() if v is not None} | {"image": image}))
    _random.shuffle(out)  # gems were prepended — don't always lead the strip
    return out


# ── Lyric gems (rare facts mined from lyrics) ────────────────────────────────

class TrackGem(BaseModel):
    """One lyric gem for the track drawer chips."""
    kind: Literal["capsule", "namedrop", "popculture", "songref"]
    canonical: str
    display: str
    quote: str
    detail: Optional[dict] = None


def _localize_gem(g: dict, lang: str) -> dict:
    """Pick the display language: gazetteers carry a ``ru`` alongside the
    original-name canonical; everything else shows as-is."""
    if lang == "ru":
        ru = (g.get("detail") or {}).get("ru")
        if ru:
            g = {**g, "display": ru}
    return g


_GEM_FACT_TEMPLATES = {
    "ru": {
        "capsule": "Привет из другой эпохи — {display}: «{quote}»",
        "namedrop": "В тексте зовут {display}: «{quote}»",
        "popculture": "Отсылка: {display} — «{quote}»",
        "songref": "Упоминает {label} «{display}»: «{quote}»",
    },
    "en": {
        "capsule": "A relic of another era — {display}: “{quote}”",
        "namedrop": "Name-drops {display}: “{quote}”",
        "popculture": "Pop-culture nod — {display}: “{quote}”",
        "songref": "Mentions the {label} “{display}”: “{quote}”",
    },
}
_SONGREF_LABELS = {
    "ru": {"album": "альбом", "track": "трек"},
    "en": {"album": "album", "track": "track"},
}


def _gem_fact_text(g: dict, lang: str) -> str:
    lng = lang if lang in _GEM_FACT_TEMPLATES else "en"
    g = _localize_gem(g, lng)
    label = ""
    if g["kind"] == "songref":
        ref_kind = (g.get("detail") or {}).get("ref_kind", "track")
        label = _SONGREF_LABELS[lng].get(ref_kind, ref_kind)
    return _GEM_FACT_TEMPLATES[lng][g["kind"]].format(
        display=g["display"], quote=g["quote"], label=label,
    )


@router.get("/metadata/tracks/{track_id}/gems")
def get_track_gems_endpoint(
    track_id: str,
    current_user: User = Depends(get_current_user),
    lang: str = Query("en"),
) -> List[TrackGem]:
    """Lyric gems for one track (empty list when none / not yet mined)."""
    derived = derive_collection_for_user(current_user)
    gems = MetadataDB.get_track_gems(track_id, derived)
    return [TrackGem(**_localize_gem(g, lang)) for g in gems]


# ── Track-level facts (landing player) ───────────────────────────────────────

class TrackFacts(BaseModel):
    """Merged facts for a single track."""
    artist_name: str
    title: str
    song_facts: List[str]
    artist_facts: List[str]


@router.get("/metadata/tracks/{track_id}/facts")
def get_track_facts(
    track_id: str,
    current_user: User = Depends(get_current_user),
    lang: str = Query("en"),
    request: Request = None,
) -> TrackFacts:
    """Return merged artist + song facts for a single track.

    Refined facts (from AI Indexing) take precedence per scope when present.
    An EXPLICIT empty refined list (AI ran but kept nothing) returns [],
    not a fallback to originals.
    """
    derived = derive_collection_for_user(current_user)
    empty = TrackFacts(artist_name="", title="", song_facts=[], artist_facts=[])
    if request is None:
        return empty
    db_client = request.app.state.db_client
    if db_client is None:
        return empty

    try:
        points = db_client.qdrant.retrieve(
            collection_name=derived,
            ids=[track_id],
            with_payload=["title", "artist"],
            with_vectors=False,
        )
    except Exception:
        return empty

    if not points:
        return empty

    payload = points[0].payload or {}
    artist = (payload.get("artist") or "").strip()
    title = (payload.get("title") or "").strip()
    if not artist or not title:
        return TrackFacts(artist_name=artist, title=title, song_facts=[], artist_facts=[])

    artist_slug = _slugify_artist(artist)
    song_key = get_song_facts_key(artist, title)

    # Prefer refined; fall back to originals if no refined row for that scope.
    # `is not None` is critical — an explicit [] from refined must short-circuit.
    # Refined facts are keyed by song_slug (not track_id) for consistency
    # with search_service caching.
    refined_song = MetadataDB.get_refined_facts(
        scope="song", scope_key=song_key, collection_name=derived, lang=lang,
    )
    refined_artist = MetadataDB.get_refined_facts(
        scope="artist", scope_key=artist_slug, collection_name=derived, lang=lang,
    )

    song_facts = (
        refined_song if refined_song is not None
        else MetadataDB.get_song_facts(song_key, derived)
    )
    artist_facts = (
        refined_artist if refined_artist is not None
        else MetadataDB.get_artist_facts(artist_slug, derived)
    )

    return TrackFacts(
        artist_name=artist, title=title,
        song_facts=song_facts, artist_facts=artist_facts,
    )


# ── Full track metadata ──────────────────────────────────────────────────────


@router.get("/metadata/tracks", response_model=List[TrackMetadata])
def get_tracks_metadata_batch(
    ids: str = Query(..., description="Comma-separated track_ids"),
    current_user: User = Depends(get_current_user),
    request: Request = None,
) -> List[TrackMetadata]:
    """Batch-resolve full TrackMetadata for many track_ids in one Qdrant retrieve.

    Used by the player to enrich the whole queue at once when a list-source
    (Playlists / Liked / Recently / Album) ships slim track shapes. Returning
    everything single-call avoids fan-out of N /metadata/tracks/{id} requests.
    Missing ids are silently dropped — caller can detect by comparing lengths.
    """
    derived = derive_collection_for_user(current_user)
    if request is None or request.app.state.db_client is None:
        raise HTTPException(status_code=503, detail="Qdrant unavailable")
    track_ids = [s for s in (ids.split(",") if ids else []) if s]
    if not track_ids:
        return []
    qdrant = request.app.state.db_client.qdrant
    try:
        points = qdrant.retrieve(
            collection_name=derived, ids=track_ids,
            with_payload=True, with_vectors=False,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Qdrant retrieve failed")
    out: List[TrackMetadata] = []
    for p in points:
        out.append(_track_from_qdrant_payload(str(p.id), p.payload or {}))
    apply_song_relations(out)
    return out


@router.get("/metadata/tracks/{track_id}", response_model=TrackMetadata)
def get_track_metadata(
    track_id: str,
    current_user: User = Depends(get_current_user),
    request: Request = None,
) -> TrackMetadata:
    """Return the full TrackMetadata for one track from Qdrant payload.

    Used by the player when the user opens a song from Library / Recently /
    Playlists — those endpoints return slim shapes (LikedSongTrack etc.) that
    lack `lyrics`, `producer`, `samples`, `file_path` etc. This endpoint
    backfills the missing fields so the player UI (lyrics back-face, info
    pills) is fully populated regardless of origin.
    """
    derived = derive_collection_for_user(current_user)
    if request is None or request.app.state.db_client is None:
        raise HTTPException(status_code=503, detail="Qdrant unavailable")
    qdrant = request.app.state.db_client.qdrant
    try:
        points = qdrant.retrieve(
            collection_name=derived, ids=[track_id],
            with_payload=True, with_vectors=False,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Qdrant retrieve failed")
    if not points:
        raise HTTPException(status_code=404, detail="track not found")
    track = _track_from_qdrant_payload(track_id, points[0].payload or {})
    apply_song_relations([track])
    return track


# ── Sonic Vibe (Plan 3 Task 14) ──────────────────────────────────────────────

class SonicVibeOut(BaseModel):
    track_id: str
    lang: str
    phrase: Optional[str] = None
    generated_at: Optional[str] = None


@router.get("/metadata/tracks/{track_id}/sonic-vibe", response_model=SonicVibeOut)
def get_sonic_vibe_endpoint(
    track_id: str,
    current_user: User = Depends(get_current_user),
    lang: str = Query("en"),
) -> SonicVibeOut:
    """Return the cached Sonic Vibe phrase. Does NOT lazy-generate — population
    happens through the AI Indexing batch job."""
    derived = derive_collection_for_user(current_user)
    cached = MetadataDB.get_sonic_vibe(track_id, derived, lang)
    if cached:
        return SonicVibeOut(track_id=track_id, lang=lang, **cached)
    return SonicVibeOut(track_id=track_id, lang=lang)
