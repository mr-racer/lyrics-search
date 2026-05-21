"""Smoke tests for Plan 19 Pydantic models."""
import pytest
from pydantic import ValidationError
from app.domain.models import (
    PlaylistTrack, PlaylistSummary, PlaylistDetail,
    PlaylistCreate, PlaylistUpdate, PlaylistTrackAdd, PlaylistReorderRequest,
    PlaylistsResponse,
)


def test_playlist_summary_minimal():
    s = PlaylistSummary(
        id=1, name="Mix", description=None,
        track_count=0, cover_track_ids=[], cover_art_paths=[],
        created_at="2026-05-21T10:00:00", updated_at="2026-05-21T10:00:00",
    )
    assert s.contains_track is None
    assert s.cover_track_ids == []


def test_playlist_summary_with_contains_track():
    s = PlaylistSummary(
        id=1, name="Mix", description=None,
        track_count=2, cover_track_ids=["t1", "t2"], cover_art_paths=["/c1", "/c2"],
        created_at="x", updated_at="y", contains_track=True,
    )
    assert s.contains_track is True


def test_playlist_track_required_fields():
    t = PlaylistTrack(track_id="x", position=1, added_at="z", title="A", artist="B")
    assert t.album is None and t.duration is None


def test_playlist_detail_with_missing_track_ids():
    d = PlaylistDetail(
        id=1, name="M", description=None, collection_name="c",
        tracks=[], missing_track_ids=["orphan1"],
        created_at="x", updated_at="y",
    )
    assert d.missing_track_ids == ["orphan1"]


def test_playlist_create_requires_name():
    with pytest.raises(ValidationError):
        PlaylistCreate(collection_name="c", name="", description=None)


def test_playlist_create_strips_name_whitespace():
    p = PlaylistCreate(collection_name="c", name="  Mix  ", description=None)
    assert p.name == "Mix"


def test_playlist_update_allows_both_none():
    PlaylistUpdate(name=None, description=None)


def test_playlist_reorder_request_accepts_list_of_strings():
    r = PlaylistReorderRequest(track_ids=["a", "b", "c"])
    assert r.track_ids == ["a", "b", "c"]


def test_playlists_response_wraps_list():
    r = PlaylistsResponse(playlists=[], collection_name="c")
    assert r.collection_name == "c"
