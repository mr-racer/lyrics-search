"""Reading the library once per request, in the shape modes expect.

Kept as its own seam so the round logic can be tested against a plain list of
dicts instead of a faked Qdrant scroll plus a faked SQLite mirror.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §4.
"""
from __future__ import annotations

from typing import Dict, List

from app.resources.qdrant_utils import light_points


def load_library(qdrant_client, collection_name: str) -> List[Dict]:
    """Light payloads for the whole collection, each carrying its ``track_id``.

    ``light_points`` already strips lyrics and clap_chunks and memoises per
    collection, so calling this on every round is cheap after the first.
    """
    tracks: List[Dict] = []
    for point_id, payload in light_points(qdrant_client, collection_name):
        track = dict(payload or {})
        track["track_id"] = str(point_id)
        tracks.append(track)
    return tracks
