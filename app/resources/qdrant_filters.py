"""Qdrant filter construction for hybrid search queries.

Extracted from legacy search_engine/utils.py during Refactor 2.
"""

from __future__ import annotations

from qdrant_client import models


def build_filter(
    artist: str | None = None,
    album: str | None = None,
    title: str | None = None,
    genre: str | list[str] | None = None,
    year: int | None = None,
    year_ranges: list[str] | None = None,
    sonic_tags: list[str] | None = None,
) -> models.Filter | None:
    conditions = []

    if artist:
        conditions.append(models.FieldCondition(key="artist", match=models.MatchValue(value=artist)))
    if album:
        conditions.append(models.FieldCondition(key="album", match=models.MatchValue(value=album)))
    if title:
        conditions.append(models.FieldCondition(key="title", match=models.MatchValue(value=title)))

    if genre:
        conditions.append(
            models.FieldCondition(
                key="genre",
                match=models.MatchAny(any=genre) if isinstance(genre, list) else models.MatchValue(value=genre),
            )
        )

    if year:
        conditions.append(models.FieldCondition(key="year", match=models.MatchValue(value=year)))

    if year_ranges:
        conditions.append(models.FieldCondition(
            key="year_range", match=models.MatchAny(any=list(year_ranges)),
        ))

    if sonic_tags:
        for tag in sonic_tags:
            conditions.append(models.FieldCondition(
                key="sonic_tags", match=models.MatchValue(value=tag),
            ))

    return models.Filter(must=conditions) if conditions else None
