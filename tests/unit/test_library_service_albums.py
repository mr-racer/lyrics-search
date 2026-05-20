"""Unit tests for LibraryService.get_albums — pure aggregation over a fake Qdrant scroll."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services.library_service import LibraryService


def _mk_point(payload, point_id="tid-abc"):
    return SimpleNamespace(payload=payload, id=point_id)


def _stub_qdrant(points):
    """Build a MagicMock qdrant client whose .scroll() yields all points in one batch."""
    qdrant = MagicMock()
    qdrant.scroll.return_value = (points, None)
    qdrant.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="test_col")]
    )
    qdrant.get_collection.return_value = SimpleNamespace(points_count=len(points))
    return qdrant


def test_get_albums_groups_by_title_majority_vote_primary():
    points = [
        _mk_point({"album": "In Rainbows", "artist": "Radiohead",
                   "title": "Bodysnatchers", "year": 2007, "duration": 240,
                   "genre": "art rock", "cover_art_path": "/c/r1.jpg"}, point_id="r1"),
        _mk_point({"album": "In Rainbows", "artist": "Radiohead",
                   "title": "Nude", "year": 2007, "duration": 255,
                   "genre": "art rock", "cover_art_path": "/c/r1.jpg"}, point_id="r2"),
        _mk_point({"album": "In Rainbows", "artist": "Frank Ocean",
                   "title": "Wisemen feat", "year": 2007, "duration": 200,
                   "genre": "rnb", "cover_art_path": "/c/r1.jpg"}, point_id="r3"),
    ]
    qdrant = _stub_qdrant(points)
    res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col")

    assert res.qdrant_available is True
    assert len(res.albums) == 1
    a = res.albums[0]
    assert a.album_title == "In Rainbows"
    assert a.primary_artist == "Radiohead"
    assert [f.name for f in a.feat_artists] == ["Frank Ocean"]
    assert a.track_count == 3
    assert a.year == 2007
    assert a.year_range is None
    assert "art rock" in a.top_genres
    assert {t.track_id for t in a.tracks} == {"r1", "r2", "r3"}


def test_get_albums_emits_year_range_when_tracks_span_years():
    points = [
        _mk_point({"album": "Live 2003", "artist": "Sigur Rós", "title": "x",
                   "year": 2002, "duration": 300}),
        _mk_point({"album": "Live 2003", "artist": "Sigur Rós", "title": "y",
                   "year": 2003, "duration": 300}),
    ]
    qdrant = _stub_qdrant(points)
    res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col")
    a = res.albums[0]
    assert a.year is None
    assert a.year_range == "2002—2003"


def test_get_albums_skips_tracks_without_album_field():
    points = [
        _mk_point({"album": None, "artist": "A", "title": "loose", "duration": 100}),
        _mk_point({"album": "", "artist": "B", "title": "empty", "duration": 100}),
        _mk_point({"album": "Real", "artist": "C", "title": "ok", "duration": 100}),
    ]
    qdrant = _stub_qdrant(points)
    res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col")
    assert [a.album_title for a in res.albums] == ["Real"]


def test_get_albums_returns_empty_when_qdrant_unavailable():
    qdrant = MagicMock()
    qdrant.get_collections.side_effect = Exception("connect refused")
    res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col")
    assert res.qdrant_available is False
    assert res.albums == []


def test_get_albums_sort_alphabetical_default():
    points = [
        _mk_point({"album": "Banana", "artist": "X", "title": "t1", "duration": 100}),
        _mk_point({"album": "Apple", "artist": "X", "title": "t2", "duration": 100}),
        _mk_point({"album": "Cherry", "artist": "X", "title": "t3", "duration": 100}),
    ]
    qdrant = _stub_qdrant(points)
    res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col")
    assert [a.album_title for a in res.albums] == ["Apple", "Banana", "Cherry"]


def test_get_albums_year_sort_uses_year_range_when_year_none():
    """Albums with year_range only (multi-year) should still sort by their min year."""
    points = [
        _mk_point({"album": "Modern Album", "artist": "X", "title": "t", "year": 2020, "duration": 200}, point_id="m1"),
        _mk_point({"album": "Live 2003", "artist": "Y", "title": "t1", "year": 2002, "duration": 200}, point_id="l1"),
        _mk_point({"album": "Live 2003", "artist": "Y", "title": "t2", "year": 2003, "duration": 200}, point_id="l2"),
    ]
    qdrant = _stub_qdrant(points)
    res = LibraryService.get_albums(qdrant_client=qdrant, collection_name="test_col", sort="year_asc")
    # "Live 2003" has year=None, year_range="2002—2003" — should sort BEFORE "Modern Album" (2020)
    assert [a.album_title for a in res.albums] == ["Live 2003", "Modern Album"]
