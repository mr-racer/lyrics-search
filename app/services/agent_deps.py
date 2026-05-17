"""Dependency injection for PydanticAI agents.

SearchDeps encapsulates everything agents need to interact with the
music library: search, filter resolution, and LLM configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.models import SearchFilters, TrackHit
from app.services.search_service import SearchService


@dataclass
class SearchDeps:
    """Зависимости, передаваемые в агенты через deps."""

    service: SearchService
    collection_name: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None

    # Кэш для resolve_filters — избегаем повторных scroll-запросов
    _filter_cache: dict[str, list[str]] = field(default_factory=dict)

    # ── Search ──────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        mode: str = "text",
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[TrackHit]:
        """Execute a search against the music library."""
        return await self.service.search(
            query=query,
            mode=mode,  # type: ignore[arg-type]
            limit=limit,
            filters=filters,
            collection_name=self.collection_name,
        )

    # ── Filter Resolution ──────────────────────────────────────────────

    async def resolve_filter_values(
        self,
        filter_key: str,
        raw_value: str,
    ) -> list[str]:
        """Return candidate values for a filter field from the Qdrant collection.

        Scrolls the collection once per field and caches the result so that
        subsequent lookups for the same filter_key are O(1).
        Returns an empty list if the collection is inaccessible.
        """
        cache_key = filter_key.lower()
        if cache_key in self._filter_cache:
            return self._filter_cache[cache_key]

        lyrics_db = self.service.lyrics_db
        collection = self.collection_name or lyrics_db.collection_name

        seen: set[str] = set()
        try:
            offset = None
            for _ in range(20):  # safety cap: max 20 pages × 128 = 2560 entries
                points, offset = lyrics_db.qdrant_client.scroll(
                    collection_name=collection,
                    limit=128,
                    offset=offset,
                    with_payload=[filter_key],
                    with_vectors=False,
                )
                for pt in points:
                    val = (pt.payload or {}).get(filter_key)
                    if val:
                        seen.add(str(val))
                if offset is None:
                    break
        except Exception:
            pass  # collection unavailable — return empty

        candidates = sorted(seen, key=str.lower)
        self._filter_cache[cache_key] = candidates
        return candidates

    async def resolve_filters(
        self,
        filter_lookup: dict[str, str] | None,
        resolved_filters: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Resolve raw filter terms to actual values present in the DB.

        For each key in `filter_lookup`, scroll the collection for candidate
        values and pick the best fuzzy match (threshold ≥ 60 via difflib).
        Already-resolved values in `resolved_filters` are passed through.
        """
        from difflib import SequenceMatcher

        result: dict[str, str] = dict(resolved_filters or {})

        if not filter_lookup:
            return result

        for key, raw in filter_lookup.items():
            if not raw:
                continue
            candidates = await self.resolve_filter_values(key, raw)
            if not candidates:
                continue

            best_match: str | None = None
            best_score = 0.0
            raw_lower = raw.lower()
            for candidate in candidates:
                score = SequenceMatcher(None, raw_lower, candidate.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = candidate

            if best_match and best_score >= 0.6:
                result[key] = best_match

        return result
