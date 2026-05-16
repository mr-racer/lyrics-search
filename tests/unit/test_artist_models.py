"""Unit tests for ArtistAggregate / ArtistAlbum pydantic models."""
from app.domain.models import (
    ArtistAggregate, ArtistAlbum, TrackMetadata,
)


def _mk_track(track_id: str, title: str, year: int, album: str = "X") -> TrackMetadata:
    return TrackMetadata(
        track_id=track_id, title=title, artist="Dua Lipa",
        album=album, year=year, duration_sec=180.0,
        file_path=f"/music/{track_id}.flac",
    )


def test_artist_album_minimal():
    album = ArtistAlbum(title="Future Nostalgia", year=2020, tracks=[])
    assert album.title == "Future Nostalgia"
    assert album.cover_art_path is None  # optional


def test_artist_aggregate_with_bio():
    agg = ArtistAggregate(
        slug="dua-lipa", name="Dua Lipa", genre="pop",
        track_count=3, album_count=1,
        decade_range="2020s", bio="From London…",
        facts=["a", "b"],
        albums=[ArtistAlbum(title="Future Nostalgia", year=2020,
                             cover_art_path="/covers/x.jpg",
                             tracks=[_mk_track("t1", "Physical", 2020)])],
    )
    assert agg.bio == "From London…"
    assert agg.albums[0].tracks[0].title == "Physical"


def test_artist_aggregate_no_bio():
    """bio=None must round-trip via .model_dump (frontend gates Bio tab on null)."""
    agg = ArtistAggregate(
        slug="x", name="X", track_count=0, album_count=0,
        facts=[], albums=[],
    )
    assert agg.bio is None
    assert agg.model_dump()["bio"] is None
