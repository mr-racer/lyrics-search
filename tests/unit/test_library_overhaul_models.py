"""Tests that the Library Overhaul response models parse and serialise cleanly."""
import pytest
from pydantic import ValidationError

from app.domain.models import (
    AlbumSummary, LibraryAlbumsResponse,
    LikedSongTrack, LikedSongsResponse,
    RecentTrack, RecentTracksResponse,
    ListeningStatsResponse, TopTrackBrief, TopArtistBrief, PeakHour,
)


def test_album_summary_minimal():
    a = AlbumSummary(
        album_title="In Rainbows",
        primary_artist="Radiohead",
        primary_artist_slug="radiohead",
        feat_artists=[],
        year=2007,
        cover_art_path=None,
        track_count=1,
        duration_seconds=237,
        top_genres=["art rock"],
        tracks=[],
    )
    assert a.year_range is None
    assert a.feat_artists == []


def test_album_summary_with_year_range():
    a = AlbumSummary(
        album_title="Live 2003",
        primary_artist="Sigur Rós",
        primary_artist_slug="sigur-ros",
        feat_artists=[],
        year=None,
        year_range="2002—2003",
        cover_art_path=None,
        track_count=3,
        duration_seconds=900,
        top_genres=["post-rock"],
        tracks=[],
    )
    assert a.year is None
    assert a.year_range == "2002—2003"


def test_listening_stats_empty():
    r = ListeningStatsResponse(
        total_seconds_listened=0,
        since=None,
        top_track=None,
        top_artist=None,
        peak_hour=None,
    )
    assert r.total_seconds_listened == 0
