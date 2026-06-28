"""Consolidated artist tests: splitter, payload build, payload->track surfacing."""

from app.services.artist_split import (
    split_artists, artist_slugs, primary_artist, name_for_slug,
)
from app.services.indexing_service import _build_payload_for_upsert
from app.api.routes.artists import _track_from_payload


# --- from test_artist_split.py (already class-based, kept as-is) ---

class TestSplitArtists:
    def test_solo_artist_unchanged(self):
        assert split_artists("Kanye West") == ["Kanye West"]

    def test_comma_separator(self):
        assert split_artists("Kanye West, Sia") == ["Kanye West", "Sia"]

    def test_feat_dot(self):
        assert split_artists("Kanye West feat. Rihanna") == ["Kanye West", "Rihanna"]

    def test_ft_and_featuring(self):
        assert split_artists("Drake ft Future") == ["Drake", "Future"]
        assert split_artists("Drake featuring Future") == ["Drake", "Future"]

    def test_ampersand_and_plus_and_x(self):
        assert split_artists("Calvin Harris & Dua Lipa") == ["Calvin Harris", "Dua Lipa"]
        assert split_artists("MGK x Travis") == ["MGK", "Travis"]

    def test_primary_is_first(self):
        assert primary_artist("Kanye West, Sia") == "Kanye West"

    def test_dedupe_preserves_order(self):
        assert split_artists("Sia, Sia") == ["Sia"]

    def test_empty_and_none(self):
        assert split_artists("") == []
        assert split_artists("   ") == []

    def test_vs_separator(self):
        assert split_artists("Jay-Z vs Nas") == ["Jay-Z", "Nas"]

    def test_with_separator(self):
        assert split_artists("Santana with Rob Thomas") == ["Santana", "Rob Thomas"]


class TestKnownGroups:
    def test_earth_wind_and_fire_not_split(self):
        assert split_artists("Earth, Wind & Fire") == ["Earth, Wind & Fire"]

    def test_florence_plus_machine_not_split(self):
        assert split_artists("Florence + the Machine") == ["Florence + the Machine"]

    def test_acdc_not_split(self):
        assert split_artists("AC/DC") == ["AC/DC"]

    def test_charli_xcx_not_split_by_x(self):
        # No space-bounded 'x' token, so the trailing x in XCX must not split.
        assert split_artists("Charli XCX") == ["Charli XCX"]


class TestSlugsAndFalseInclusion:
    def test_artist_slugs(self):
        assert artist_slugs("Kanye West, Sia") == ["kanye-west", "sia"]

    def test_alias_resolves_to_canonical_slug(self):
        assert artist_slugs("Ye") == ["kanye-west"]

    def test_ye_not_found_inside_kanye(self):
        # The whole point: 'ye' is a distinct slug, never a substring match.
        assert "ye" not in artist_slugs("Kanye West")
        assert "ye" not in artist_slugs("Yeah Yeah Yeahs")


class TestNameForSlug:
    def test_picks_primary_participant(self):
        assert name_for_slug("Dua Lipa x Angele", "dua-lipa") == "Dua Lipa"

    def test_picks_featured_participant(self):
        # The whole point of the fix: the feat's page gets the feat's name.
        assert name_for_slug("Dua Lipa x Angele", "angele") == "Angele"

    def test_solo(self):
        assert name_for_slug("Dua Lipa", "dua-lipa") == "Dua Lipa"

    def test_no_match_returns_none(self):
        assert name_for_slug("Dua Lipa x Angele", "taylor-swift") is None

    def test_alias_resolves_to_canonical_slug(self):
        # "Ye" maps to kanye-west by alias; the tagged name "Ye" is returned.
        assert name_for_slug("Ye, Sia", "kanye-west") == "Ye"

    def test_none_input(self):
        assert name_for_slug(None, "x") is None


# --- from test_build_payload_artists.py ---

class TestBuildPayloadArtists:
    def test_payload_has_split_artist_fields(self):
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

    def test_payload_solo_artist(self):
        info = {"lyrics": "x", "title": "T", "artist": "Radiohead", "album": "A"}
        p = _build_payload_for_upsert(info, slug=None)
        assert p["artists"] == ["Radiohead"]
        assert p["artist_slugs"] == ["radiohead"]
        assert p["primary_artist_slug"] == "radiohead"

    def test_payload_missing_artist(self):
        info = {"lyrics": "x", "title": "T", "artist": "", "album": "A"}
        p = _build_payload_for_upsert(info, slug=None)
        assert p["artists"] == []
        assert p["artist_slugs"] == []
        assert p["primary_artist_slug"] is None

    def test_payload_carries_track_and_disc_number(self):
        info = {
            "lyrics": "x", "title": "T", "artist": "A", "album": "Alb",
            "track_number": 5, "disc_number": 2,
        }
        p = _build_payload_for_upsert(info, slug=None)
        assert p["track_number"] == 5
        assert p["disc_number"] == 2

    def test_payload_track_number_defaults_to_none(self):
        info = {"lyrics": "x", "title": "T", "artist": "A", "album": "Alb"}
        p = _build_payload_for_upsert(info, slug=None)
        assert p["track_number"] is None
        assert p["disc_number"] is None


# --- from test_track_from_payload_artists.py ---

class TestTrackFromPayloadArtists:
    def test_uses_payload_fields_when_present(self):
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

    def test_falls_back_to_split_when_fields_absent(self):
        p = {"title": "T", "artist": "Kanye West feat. Sia", "duration": 200}
        t = _track_from_payload("id1", p)
        assert t.artists == ["Kanye West", "Sia"]
        assert t.primary_artist_slug == "kanye-west"

    def test_empty_artist_yields_none(self):
        p = {"title": "T", "artist": "", "artists": [], "artist_slugs": [], "duration": 0}
        t = _track_from_payload("id1", p)
        assert t.artists is None
        assert t.primary_artist_slug is None

    def test_primary_derived_from_slugs_when_primary_absent(self):
        p = {"title": "T", "artist": "Kanye West, Sia",
             "artists": ["Kanye West", "Sia"], "artist_slugs": ["kanye-west", "sia"],
             "duration": 200}
        t = _track_from_payload("id1", p)
        assert t.primary_artist_slug == "kanye-west"
