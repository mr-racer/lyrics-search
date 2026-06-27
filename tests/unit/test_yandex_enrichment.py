"""Unit tests for Yandex metadata enrichment (feature #2)."""

import types

import pytest

from app.services.yandex import enrichment


def _ytrack(title="T", album="Alb", year=2021, genre="rock", duration_ms=200_000):
    ns = types.SimpleNamespace
    return ns(
        title=title, duration_ms=duration_ms,
        albums=[ns(title=album, year=year, genre=genre)],
    )


class _FakeClient:
    """Minimal stand-in: search() returns a best-match track."""

    def __init__(self, track):
        self._track = track
        self.calls = 0

    def search(self, text):
        self.calls += 1
        ns = types.SimpleNamespace
        best = ns(type="track", result=self._track)
        return ns(best=best, tracks=None)


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("MUSIX_YM_ENRICH", "1")
    # Avoid any real network if a test forgets to pass a client.
    monkeypatch.setattr(enrichment, "get_anonymous_client", lambda: None)


class TestApplyYandexTrack:
    def test_fills_only_empty_fields(self):
        meta = {"artist": "A", "title": "T", "album": "", "year": None, "genre": "Jazz"}
        enrichment.apply_yandex_track(meta, _ytrack(genre="rock"))
        assert meta["album"] == "Alb"
        assert meta["year"] == 2021
        assert meta["genre"] == "Jazz"  # already set → untouched

    def test_missing_fields_detection(self):
        assert enrichment._missing_fields(
            {"album": "x", "year": 2000, "genre": "g"}) == []
        assert set(enrichment._missing_fields(
            {"album": "", "year": None, "genre": None})) == {"album", "year", "genre"}


class TestDurationGuard:
    def test_within_tolerance_ok(self):
        assert enrichment._duration_matches({"duration": 200}, _ytrack(duration_ms=203_000))

    def test_outside_tolerance_rejected(self):
        assert not enrichment._duration_matches({"duration": 200}, _ytrack(duration_ms=240_000))

    def test_unknown_duration_does_not_block(self):
        assert enrichment._duration_matches({"duration": None}, _ytrack())


class TestEnrichMetadata:
    def test_enriches_via_client(self):
        meta = {"artist": "A", "title": "T", "duration": 200}
        client = _FakeClient(_ytrack())
        out = enrichment.enrich_metadata(meta, client=client)
        assert out["album"] == "Alb" and out["year"] == 2021 and out["genre"] == "rock"
        assert client.calls == 1

    def test_no_search_when_nothing_missing(self):
        meta = {"artist": "A", "title": "T", "album": "x", "year": 1, "genre": "g"}
        client = _FakeClient(_ytrack())
        enrichment.enrich_metadata(meta, client=client)
        assert client.calls == 0  # short-circuited

    def test_skips_without_artist_or_title(self):
        client = _FakeClient(_ytrack())
        enrichment.enrich_metadata({"artist": "", "title": "T"}, client=client)
        assert client.calls == 0

    def test_disabled_is_noop(self, monkeypatch):
        monkeypatch.setenv("MUSIX_YM_ENRICH", "0")
        client = _FakeClient(_ytrack())
        meta = {"artist": "A", "title": "T"}
        enrichment.enrich_metadata(meta, client=client)
        assert client.calls == 0 and "album" not in meta

    def test_duration_mismatch_skips_fill(self):
        meta = {"artist": "A", "title": "T", "duration": 200}
        client = _FakeClient(_ytrack(duration_ms=300_000))
        enrichment.enrich_metadata(meta, client=client)
        assert "album" not in meta or not meta.get("album")

    def test_client_error_is_swallowed(self):
        class Boom:
            def search(self, text):
                raise RuntimeError("yandex down")
        meta = {"artist": "A", "title": "T"}
        # Must not raise.
        out = enrichment.enrich_metadata(meta, client=Boom())
        assert out is meta
