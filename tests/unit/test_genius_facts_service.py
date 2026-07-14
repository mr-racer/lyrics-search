"""Tests for genius_facts_service: live-pipeline Genius fact fetch + storage."""
import asyncio
from unittest.mock import patch

import pytest

from app.resources.metadata_db import MetadataDB
from app.services.genius_facts_service import (
    fetch_genius_facts_for_song,
    fetch_genius_facts_for_songs,
)
from app.services.genius_service import GeniusParseError


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _parsed(**fields):
    base = {
        "url": "https://genius.com/Kanye-west-stronger-lyrics",
        "description": "A song about resilience, sampling Daft Punk.",
        "annotations": [("work it, make it", "A reference to gym discipline.")],
        "producers": ["Kanye West"],
        "label": "Roc-A-Fella",
    }
    base.update(fields)
    return base


class TestFetchGeniusFactsForSong:
    def test_happy_path_stores_facts_under_both_categories(self, isolated_db):
        with patch(
            "app.services.genius_facts_service.genius_service.fetch_song_data",
            return_value=_parsed(),
        ):
            found, producer, label = _run(
                fetch_genius_facts_for_song("Kanye West", "Stronger", "test_col")
            )

        assert found is True
        assert producer == "Kanye West"
        assert label == "Roc-A-Fella"
        facts = MetadataDB.get_song_facts("kanye-west-stronger", "test_col")
        assert any("resilience" in f for f in facts)
        assert any(f.startswith("Lyrics string: work it, make it. Fact:") for f in facts)

    def test_does_not_write_producer_or_label(self, isolated_db):
        """Live pipeline has no track_id yet at FACTS-stage time — must not touch
        track_metadata (the caller applies producer/label later, once ids resolve)."""
        with patch(
            "app.services.genius_facts_service.genius_service.fetch_song_data",
            return_value=_parsed(),
        ) as mock_fetch, patch(
            "app.services.genius_facts_service.MetadataDB.update_track_producer_label",
        ) as mock_update:
            _run(fetch_genius_facts_for_song("Kanye West", "Stronger", "test_col"))

        mock_fetch.assert_called_once()
        mock_update.assert_not_called()

    def test_returns_false_and_writes_nothing_on_parse_error(self, isolated_db):
        with patch(
            "app.services.genius_facts_service.genius_service.fetch_song_data",
            side_effect=GeniusParseError("404"),
        ):
            found, producer, label = _run(
                fetch_genius_facts_for_song("Nobody", "Unknown Song", "test_col")
            )

        assert (found, producer, label) == (False, None, None)
        assert MetadataDB.get_song_facts("nobody-unknown-song", "test_col") == []

    def test_returns_false_when_no_facts_found(self, isolated_db):
        with patch(
            "app.services.genius_facts_service.genius_service.fetch_song_data",
            return_value=_parsed(description=None, annotations=[]),
        ):
            found, _, _ = _run(
                fetch_genius_facts_for_song("Kanye West", "Stronger", "test_col")
            )

        assert found is False


class TestFetchGeniusFactsForSongs:
    def test_calls_progress_callback_per_song(self, isolated_db):
        with patch(
            "app.services.genius_facts_service.genius_service.fetch_song_data",
            side_effect=[_parsed(), GeniusParseError("404")],
        ):
            with patch("app.services.genius_facts_service.asyncio.sleep", return_value=None):
                calls = []
                _run(fetch_genius_facts_for_songs(
                    [("Kanye West", "Stronger"), ("Nobody", "Unknown Song")],
                    "test_col",
                    progress_callback=lambda idx, total, label, found: calls.append(
                        (idx, total, label, found)
                    ),
                ))

        assert calls == [
            (1, 2, "Kanye West — Stronger", True),
            (2, 2, "Nobody — Unknown Song", False),
        ]

    def test_returns_producer_label_keyed_by_artist_dash_title(self, isolated_db):
        """Key format must match IndexPipeline._resolve_track_ids's key so the
        caller can look up resolved track ids directly."""
        with patch(
            "app.services.genius_facts_service.genius_service.fetch_song_data",
            side_effect=[_parsed(), GeniusParseError("404")],
        ):
            with patch("app.services.genius_facts_service.asyncio.sleep", return_value=None):
                producer_label = _run(fetch_genius_facts_for_songs(
                    [("Kanye West", "Stronger"), ("Nobody", "Unknown Song")],
                    "test_col",
                ))

        assert producer_label == {
            "Kanye West — Stronger": ("Kanye West", "Roc-A-Fella"),
        }
