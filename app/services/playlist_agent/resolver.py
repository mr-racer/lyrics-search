"""Resolve LLM-proposed song titles against the library catalog.

Per item, two-stage matching:
  1. **exact** — normalized (``text_normalize.fold``) title equality, with an
     optional artist constraint;
  2. **fuzzy** — top-1 of ``catalog_search_service.search_catalog_tracks``
     (token + transliteration + IDF scoring) when exact misses.

Every result carries its ``match`` mode ("exact" | "fuzzy" | "none") so the
playlist agent can tell the LLM whether a title matched cleanly, only
approximately, or not at all — the LLM decides whether to trust a fuzzy hit.

``resolve_songs`` is decoupled from the real catalog via a small duck-typed
protocol so it can be unit-tested with a fake; ``CatalogAdapter`` binds it to
the real ``catalog_search_service`` for one (qdrant, collection).
"""
from app.services.text_normalize import fold


def _norm(s):
    return fold(s or "")


def _artist_matches(query_artist, song_artist):
    na, ns = _norm(query_artist), _norm(song_artist)
    return bool(na) and bool(ns) and (na in ns or ns in na)


def resolve_songs(items, catalog, artist_filter=None):
    """Resolve ``items`` against ``catalog``.

    Parameters
    ----------
    items         : list of ``{"title": str, "artist": str | None}``.
    catalog       : object exposing
                    ``iter_songs() -> iterable[{"track_id","title","artist"}]``
                    and ``search_tracks_fuzzy(query: str, limit: int) -> list``
                    of track dicts (``track_id``/``title``/``artist``).
    artist_filter : optional fallback artist applied to items whose own
                    ``artist`` is missing (e.g. an "artist hits" playlist).

    Returns a list aligned with ``items``; each element is
    ``{"query_title", "match", "track_id", "title", "artist"}`` where ``match``
    is "exact", "fuzzy", or "none".
    """
    by_title = {}
    for s in catalog.iter_songs():
        by_title.setdefault(_norm(s.get("title")), []).append(s)

    results = []
    for item in items:
        qtitle = (item.get("title") or "").strip()
        qartist = item.get("artist") or artist_filter
        picked = None
        mode = "none"

        # exact — normalized title equality (+ optional artist constraint)
        candidates = by_title.get(_norm(qtitle), [])
        if candidates:
            if qartist:
                picked = next(
                    (c for c in candidates if _artist_matches(qartist, c.get("artist"))),
                    None,
                )
            else:
                picked = candidates[0]
            if picked is not None:
                mode = "exact"

        # fuzzy fallback
        if picked is None and qtitle:
            query = f"{qtitle} {qartist}".strip() if qartist else qtitle
            hits = catalog.search_tracks_fuzzy(query, limit=3)
            if hits:
                picked = hits[0]
                mode = "fuzzy"

        if picked is None:
            results.append({"query_title": qtitle, "match": "none",
                            "track_id": None, "title": None, "artist": None})
        else:
            results.append({
                "query_title": qtitle, "match": mode,
                "track_id": picked.get("track_id"),
                "title": picked.get("title"),
                "artist": picked.get("artist"),
            })
    return results


class CatalogAdapter:
    """Bind ``resolve_songs``'s duck-typed protocol to the real
    ``catalog_search_service`` for one ``(qdrant, collection_name)``."""

    def __init__(self, qdrant, collection_name):
        self._qdrant = qdrant
        self._collection = collection_name

    def iter_songs(self):
        from app.services import catalog_search_service as css

        index = css._get_index(self._qdrant, self._collection)
        if index is None:
            return []
        return [
            {"track_id": d.meta.get("track_id"),
             "title": d.meta.get("title"),
             "artist": d.meta.get("artist", "")}
            for d in index.songs.docs
        ]

    def search_tracks_fuzzy(self, query, limit=3):
        from app.services import catalog_search_service as css

        return css.search_catalog_tracks(self._qdrant, self._collection, query, limit=limit)
