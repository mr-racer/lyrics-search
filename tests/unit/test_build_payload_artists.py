"""_build_payload_for_upsert must emit canonical artist fields."""

from app.services.indexing_service import _build_payload_for_upsert


def test_payload_has_split_artist_fields():
    info = {
        "lyrics": "some words here long enough",
        "title": "Heartless",
        "artist": "Kanye West, Sia",
        "album": "808s",
    }
    p = _build_payload_for_upsert(info, slug=None)
    assert p["artist"] == "Kanye West, Sia"            # raw preserved for display
    assert p["artists"] == ["Kanye West", "Sia"]
    assert p["artist_slugs"] == ["kanye-west", "sia"]
    assert p["primary_artist_slug"] == "kanye-west"


def test_payload_solo_artist():
    info = {"lyrics": "x", "title": "T", "artist": "Radiohead", "album": "A"}
    p = _build_payload_for_upsert(info, slug=None)
    assert p["artists"] == ["Radiohead"]
    assert p["artist_slugs"] == ["radiohead"]
    assert p["primary_artist_slug"] == "radiohead"


def test_payload_missing_artist():
    info = {"lyrics": "x", "title": "T", "artist": "", "album": "A"}
    p = _build_payload_for_upsert(info, slug=None)
    assert p["artists"] == []
    assert p["artist_slugs"] == []
    assert p["primary_artist_slug"] is None


def test_payload_carries_track_and_disc_number():
    info = {
        "lyrics": "x", "title": "T", "artist": "A", "album": "Alb",
        "track_number": 5, "disc_number": 2,
    }
    p = _build_payload_for_upsert(info, slug=None)
    assert p["track_number"] == 5
    assert p["disc_number"] == 2


def test_payload_track_number_defaults_to_none():
    info = {"lyrics": "x", "title": "T", "artist": "A", "album": "Alb"}
    p = _build_payload_for_upsert(info, slug=None)
    assert p["track_number"] is None
    assert p["disc_number"] is None
