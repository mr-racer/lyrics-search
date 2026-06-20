"""Unit tests for the catalog search engine (pure: fabricated points + history,
no Qdrant). Covers entity mode, tracks mode, and the music-player boosts."""
import pytest

pytestmark = pytest.mark.unit

from app.services.catalog_search_service import build_index, search_entities, search_tracks


def _slug(name: str) -> str:
    return name.lower().replace(" ", "-")


def _pt(tid, title, artist, album, **kw):
    return (tid, {
        "title": title,
        "artist": artist,
        "album": album,
        "artists": kw.get("artists", [artist]),
        "artist_slugs": kw.get("artist_slugs", [_slug(artist)]),
        "primary_artist_slug": kw.get("primary_artist_slug", _slug(artist)),
        "cover_art_path": kw.get("cover", f"/cov/{tid}.jpg"),
        "year": kw.get("year"),
        "genre": kw.get("genre"),
        "duration": kw.get("duration", 200.0),
        "file_path": kw.get("file_path", f"/m/{tid}.mp3"),
    })


LIB = [
    _pt("t1", "Bohemian Rhapsody", "Queen", "A Night at the Opera"),
    _pt("t2", "Love of My Life", "Queen", "A Night at the Opera"),
    _pt("t3", "Time", "Pink Floyd", "The Dark Side of the Moon"),
    _pt("t4", "Money", "Pink Floyd", "The Dark Side of the Moon"),
    _pt("t5", "Группа крови", "Кино", "Группа крови"),
]


def test_album_name_query_returns_album_entity_not_its_songs():
    idx = build_index(LIB)
    hits = search_entities(idx, "dark side of the moon")
    assert hits, "expected at least one hit"
    assert hits[0]["type"] == "album"
    assert "dark side of the moon" in hits[0]["album"].lower()
    # The album's songs ("Time", "Money") must not appear (titles don't match).
    titles = {h.get("title") for h in hits if h["type"] == "song"}
    assert "Time" not in titles and "Money" not in titles


def test_artist_name_query_returns_artist_entity():
    idx = build_index(LIB)
    hits = search_entities(idx, "pink floyd")
    assert hits[0]["type"] == "artist"
    assert hits[0]["artist"].lower() == "pink floyd"


def test_song_title_query_returns_song():
    idx = build_index(LIB)
    hits = search_entities(idx, "bohemian rhapsody")
    assert hits[0]["type"] == "song"
    assert hits[0]["title"] == "Bohemian Rhapsody"
    assert hits[0]["track_id"] == "t1"


def test_prefix_as_you_type():
    idx = build_index(LIB)
    hits = search_entities(idx, "bohem")
    assert hits[0]["type"] == "song" and hits[0]["title"] == "Bohemian Rhapsody"


def test_transliteration_latin_query_matches_cyrillic_artist():
    idx = build_index(LIB)
    hits = search_entities(idx, "kino")
    assert any(h["type"] == "artist" and h["artist"] == "Кино" for h in hits)


def test_history_boost_orders_equal_text_matches():
    pts = [
        _pt("h1", "Hello", "Adele", "25"),
        _pt("h2", "Hello", "Lionel Richie", "Can't Slow Down"),
    ]
    idx = build_index(pts)
    history = {"play_counts": {"h1": 50}, "reactions": {}}
    hits = [h for h in search_entities(idx, "hello", history=history) if h["type"] == "song"]
    assert hits[0]["track_id"] == "h1"  # more-played wins the tie


def test_history_boost_cannot_override_exact_name_match():
    pts = [
        _pt("m1", "Money", "Pink Floyd", "DSOTM"),
        _pt("m2", "Money Money Money", "ABBA", "Arrival"),
    ]
    idx = build_index(pts)
    history = {"play_counts": {"m2": 9999}, "reactions": {}}
    hits = [h for h in search_entities(idx, "money", history=history) if h["type"] == "song"]
    assert hits[0]["track_id"] == "m1"  # exact title beats a heavily-played partial


def test_tracks_mode_artist_query_explodes_into_tracks():
    idx = build_index(LIB)
    hits = search_tracks(idx, "pink floyd")
    ids = {h["track_id"] for h in hits}
    assert {"t3", "t4"} <= ids
    # tracks mode returns plain tracks with playable fields
    assert all(h.get("track_id") and h.get("file_path") for h in hits)


def test_tracks_mode_album_query_explodes_into_tracks():
    idx = build_index(LIB)
    hits = search_tracks(idx, "dark side of the moon")
    ids = {h["track_id"] for h in hits}
    assert {"t3", "t4"} <= ids


def test_tracks_mode_dedups_by_track_id():
    idx = build_index(LIB)
    hits = search_tracks(idx, "queen")
    ids = [h["track_id"] for h in hits]
    assert sorted(ids) == sorted(set(ids))
    assert {"t1", "t2"} <= set(ids)
