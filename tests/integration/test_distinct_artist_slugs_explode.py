"""Distinct-artist enumeration explodes collaborations into individuals."""

from unittest.mock import MagicMock

from app.services.library_service import LibraryService


def _pt(artist, slugs, names):
    pt = MagicMock()
    pt.payload = {"artist": artist, "artist_slugs": slugs, "artists": names}
    return pt


def test_collab_string_replaced_by_individuals():
    pts = [
        _pt("Dua Lipa, Angele", ["dua-lipa", "angele"], ["Dua Lipa", "Angele"]),
        _pt("Dua Lipa", ["dua-lipa"], ["Dua Lipa"]),
    ]
    qdrant = MagicMock()
    qdrant.scroll.return_value = (pts, None)
    result = LibraryService.list_distinct_artist_slugs(
        qdrant_client=qdrant, collection_name="c",
    )
    slugs = {s for s, _ in result}
    assert "dua-lipa" in slugs
    assert "angele" in slugs
    assert "dua-lipa-angele" not in slugs


def test_fallback_when_payload_lacks_slugs():
    # Un-backfilled point: only raw `artist`. Must still explode via splitter.
    pt = MagicMock()
    pt.payload = {"artist": "Calvin Harris & Dua Lipa"}
    qdrant = MagicMock()
    qdrant.scroll.return_value = ([pt], None)
    result = LibraryService.list_distinct_artist_slugs(
        qdrant_client=qdrant, collection_name="c",
    )
    slugs = {s for s, _ in result}
    assert "calvin-harris" in slugs
    assert "dua-lipa" in slugs


def test_fallback_alias_display_name_is_artist_name_not_slug():
    pt = MagicMock()
    pt.payload = {"artist": "Ye"}
    qdrant = MagicMock()
    qdrant.scroll.return_value = ([pt], None)
    result = LibraryService.list_distinct_artist_slugs(
        qdrant_client=qdrant, collection_name="c",
    )
    slug_to_name = dict(result)
    assert "kanye-west" in slug_to_name
    assert slug_to_name["kanye-west"] == "Ye"
