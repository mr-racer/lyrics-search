"""Metadata routes — CRUD for artist/song facts stored in SQLite.

All endpoints require a ``collection`` query parameter so that facts are
scoped to the currently selected Qdrant collection.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from ...resources.metadata_db import MetadataDB
from ...services.song_facts_service import _slugify, get_song_facts_key

router = APIRouter(tags=["Metadata"])


# ── Request/Response models ──────────────────────────────────────────────────

class FactIn(BaseModel):
    """Payload for adding a fact."""
    fact: str
    collection: str


class FactOut(BaseModel):
    """Single fact returned by the API."""
    fact: str
    source: Optional[str] = None


class RandomFact(BaseModel):
    """Random fact with attribution context."""
    fact: str
    context: str
    type: Literal["artist", "song"]


# ── Artist facts ─────────────────────────────────────────────────────────────

@router.get("/metadata/artists/{slug}/facts")
def get_artist_facts(
    slug: str,
    collection: str = Query(..., description="Qdrant collection name"),
) -> List[str]:
    """Return all cached facts for an artist in a collection."""
    return MetadataDB.get_artist_facts(slug, collection)


@router.post("/metadata/artists/{slug}/facts", status_code=201)
def add_artist_fact(
    slug: str,
    body: FactIn,
) -> dict:
    """Add a new fact for an artist."""
    MetadataDB.add_artist_fact(
        slug=slug,
        collection_name=body.collection,
        fact_text=body.fact,
        source="manual",
    )
    return {"ok": True}


# ── Song facts ───────────────────────────────────────────────────────────────

@router.get("/metadata/songs/{slug}/facts")
def get_song_facts(
    slug: str,
    collection: str = Query(..., description="Qdrant collection name"),
) -> List[str]:
    """Return all cached facts for a song in a collection."""
    return MetadataDB.get_song_facts(slug, collection)


@router.post("/metadata/songs/{slug}/facts", status_code=201)
def add_song_fact(
    slug: str,
    body: FactIn,
) -> dict:
    """Add a new fact for a song."""
    MetadataDB.add_song_fact(
        slug=slug,
        collection_name=body.collection,
        fact_text=body.fact,
        source="manual",
    )
    return {"ok": True}


# ── Random facts (landing page) ──────────────────────────────────────────────

@router.get("/metadata/random-facts")
def get_random_facts(
    collection: str = Query(..., description="Qdrant collection name"),
    limit: int = Query(5, ge=1, le=20, description="Number of random facts"),
) -> List[RandomFact]:
    """Return random facts from the collection's fact pool."""
    raw = MetadataDB.get_random_facts(collection_name=collection, limit=limit)
    return [RandomFact(**r) for r in raw]


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
    collection: str = Query(..., description="Qdrant collection name"),
    lang: str = Query("en"),
    request: Request = None,
) -> TrackFacts:
    """Return merged artist + song facts for a single track.

    Refined facts (from AI Indexing) take precedence per scope when present.
    An EXPLICIT empty refined list (AI ran but kept nothing) returns [],
    not a fallback to originals.
    """
    empty = TrackFacts(artist_name="", title="", song_facts=[], artist_facts=[])
    if request is None:
        return empty
    db_client = request.app.state.db_client
    if db_client is None:
        return empty

    try:
        points = db_client.qdrant.retrieve(
            collection_name=collection,
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

    artist_slug = _slugify(artist)
    song_key = get_song_facts_key(artist, title)

    # Prefer refined; fall back to originals if no refined row for that scope.
    # `is not None` is critical — an explicit [] from refined must short-circuit.
    refined_song = MetadataDB.get_refined_facts(
        scope="song", scope_key=track_id, collection_name=collection, lang=lang,
    )
    refined_artist = MetadataDB.get_refined_facts(
        scope="artist", scope_key=artist_slug, collection_name=collection, lang=lang,
    )

    song_facts = (
        refined_song if refined_song is not None
        else MetadataDB.get_song_facts(song_key, collection)
    )
    artist_facts = (
        refined_artist if refined_artist is not None
        else MetadataDB.get_artist_facts(artist_slug, collection)
    )

    return TrackFacts(
        artist_name=artist, title=title,
        song_facts=song_facts, artist_facts=artist_facts,
    )


# ── Sonic Vibe (Plan 3 Task 14) ──────────────────────────────────────────────

class SonicVibeOut(BaseModel):
    track_id: str
    lang: str
    phrase: Optional[str] = None
    generated_at: Optional[str] = None


@router.get("/metadata/tracks/{track_id}/sonic-vibe", response_model=SonicVibeOut)
def get_sonic_vibe_endpoint(
    track_id: str,
    collection: str = Query(...),
    lang: str = Query("en"),
) -> SonicVibeOut:
    """Return the cached Sonic Vibe phrase. Does NOT lazy-generate — population
    happens through the AI Indexing batch job."""
    cached = MetadataDB.get_sonic_vibe(track_id, collection, lang)
    if cached:
        return SonicVibeOut(track_id=track_id, lang=lang, **cached)
    return SonicVibeOut(track_id=track_id, lang=lang)
