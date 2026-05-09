"""Metadata routes — CRUD for artist/song facts stored in SQLite.

All endpoints require a ``collection`` query parameter so that facts are
scoped to the currently selected Qdrant collection.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ...resources.metadata_db import MetadataDB

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
