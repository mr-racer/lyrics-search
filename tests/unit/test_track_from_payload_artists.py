"""_track_from_payload surfaces canonical artist fields, with fallback."""

from app.api.routes.artists import _track_from_payload


def test_uses_payload_fields_when_present():
    p = {
        "title": "Heartless", "artist": "Kanye West, Sia",
        "artists": ["Kanye West", "Sia"],
        "artist_slugs": ["kanye-west", "sia"],
        "primary_artist_slug": "kanye-west",
        "duration": 200,
    }
    t = _track_from_payload("id1", p)
    assert t.artists == ["Kanye West", "Sia"]
    assert t.primary_artist_slug == "kanye-west"


def test_falls_back_to_split_when_fields_absent():
    p = {"title": "T", "artist": "Kanye West feat. Sia", "duration": 200}
    t = _track_from_payload("id1", p)
    assert t.artists == ["Kanye West", "Sia"]
    assert t.primary_artist_slug == "kanye-west"


def test_empty_artist_yields_none():
    p = {"title": "T", "artist": "", "artists": [], "artist_slugs": [], "duration": 0}
    t = _track_from_payload("id1", p)
    assert t.artists is None
    assert t.primary_artist_slug is None


def test_primary_derived_from_slugs_when_primary_absent():
    p = {"title": "T", "artist": "Kanye West, Sia",
         "artists": ["Kanye West", "Sia"], "artist_slugs": ["kanye-west", "sia"],
         "duration": 200}
    t = _track_from_payload("id1", p)
    assert t.primary_artist_slug == "kanye-west"
