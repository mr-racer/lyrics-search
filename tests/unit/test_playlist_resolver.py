"""Resolver: exact (normalized title) first, fuzzy fallback, else none."""
import pytest

from app.services.playlist_agent.resolver import resolve_songs


class FakeCatalog:
    def iter_songs(self):
        return [{"track_id": "1", "title": "Stronger", "artist": "Kanye West"}]

    def search_tracks_fuzzy(self, q, limit=3):
        return [{"track_id": "1", "title": "Stronger", "artist": "Kanye West", "score": 5.0}]


class EmptyCatalog:
    def iter_songs(self):
        return []

    def search_tracks_fuzzy(self, q, limit=3):
        return []


@pytest.mark.unit
def test_exact_match_first():
    res = resolve_songs([{"title": "stronger", "artist": None}], FakeCatalog())
    assert res[0]["match"] == "exact" and res[0]["track_id"] == "1"


@pytest.mark.unit
def test_fuzzy_fallback():
    res = resolve_songs([{"title": "strongr", "artist": None}], FakeCatalog())
    assert res[0]["match"] == "fuzzy" and res[0]["track_id"] == "1"


@pytest.mark.unit
def test_none_when_nothing():
    assert resolve_songs([{"title": "x", "artist": None}], EmptyCatalog())[0]["match"] == "none"


@pytest.mark.unit
def test_exact_respects_artist_filter():
    class TwoArtists:
        def iter_songs(self):
            return [{"track_id": "a", "title": "Forever", "artist": "Drake"},
                    {"track_id": "b", "title": "Forever", "artist": "Chris Brown"}]

        def search_tracks_fuzzy(self, q, limit=3):
            return []

    res = resolve_songs([{"title": "Forever", "artist": "Chris Brown"}], TwoArtists())
    assert res[0]["match"] == "exact" and res[0]["track_id"] == "b"


@pytest.mark.unit
def test_exact_with_wrong_artist_falls_to_fuzzy():
    res = resolve_songs([{"title": "Stronger", "artist": "Nonexistent Person"}], FakeCatalog())
    # exact title exists but artist doesn't match -> fuzzy (which here returns the track)
    assert res[0]["match"] == "fuzzy"


@pytest.mark.unit
def test_result_shape_and_alignment():
    res = resolve_songs(
        [{"title": "Stronger", "artist": None}, {"title": "Ghost", "artist": None}],
        FakeCatalog(),
    )
    assert len(res) == 2
    assert set(res[0]) == {"query_title", "match", "track_id", "title", "artist"}
    assert res[1]["match"] == "fuzzy"  # 'Ghost' misses exact, fuzzy fake returns a hit
