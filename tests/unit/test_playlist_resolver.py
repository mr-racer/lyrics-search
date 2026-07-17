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
def test_exact_with_wrong_artist_rejected_by_fuzzy_gate():
    # exact title exists but artist doesn't match; the fuzzy candidate is the
    # same track — strict gate must refuse it (right title, wrong artist).
    res = resolve_songs([{"title": "Stronger", "artist": "Nonexistent Person"}], FakeCatalog())
    assert res[0]["match"] == "none"


@pytest.mark.unit
def test_result_shape_and_alignment():
    res = resolve_songs(
        [{"title": "Stronger", "artist": None}, {"title": "Ghost", "artist": None}],
        FakeCatalog(),
    )
    assert len(res) == 2
    assert set(res[0]) == {"query_title", "match", "track_id", "title", "artist"}
    # 'Ghost' misses exact; the fuzzy hit ('Stronger') shares nothing with the
    # query title, so the similarity gate turns it into "none".
    assert res[1]["match"] == "none"


@pytest.mark.unit
def test_fuzzy_rejects_dissimilar_title():
    class NoisyCatalog:
        def iter_songs(self):
            return []

        def search_tracks_fuzzy(self, q, limit=3):
            # BM25F top-1 that shares only a stop-word-ish token with the query
            return [{"track_id": "9", "title": "Bohemian Rhapsody", "artist": "Queen"}]

    res = resolve_songs([{"title": "Sunflower", "artist": None}], NoisyCatalog())
    assert res[0]["match"] == "none"


@pytest.mark.unit
def test_fuzzy_accepts_transliterated_title():
    res = resolve_songs([{"title": "Стронгер", "artist": None}], FakeCatalog())
    assert res[0]["match"] == "fuzzy" and res[0]["track_id"] == "1"


@pytest.mark.unit
def test_fuzzy_artist_passes_via_similarity():
    # artist given in another script: substring check fails, but the
    # transliteration-aware similarity accepts it.
    res = resolve_songs([{"title": "Strongr", "artist": "Канье"}], FakeCatalog())
    assert res[0]["match"] == "fuzzy" and res[0]["track_id"] == "1"


@pytest.mark.unit
def test_fuzzy_picks_valid_candidate_from_top3():
    class MixedCatalog:
        def iter_songs(self):
            return []

        def search_tracks_fuzzy(self, q, limit=3):
            return [
                {"track_id": "bad", "title": "Completely Unrelated", "artist": "Nobody"},
                {"track_id": "good", "title": "Runaway", "artist": "Kanye West"},
            ]

    res = resolve_songs([{"title": "Runaway", "artist": "Kanye West"}], MixedCatalog())
    assert res[0]["match"] == "fuzzy" and res[0]["track_id"] == "good"
