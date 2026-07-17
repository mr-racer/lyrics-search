"""Tests for song_facts_service.apply_song_relations — overlay of
producer/label/samples from the songs table onto TrackMetadata objects."""
import pytest

from app.domain.models import TrackMetadata
from app.resources.metadata_db import MetadataDB
from app.services.song_facts_service import apply_song_relations


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _track(artist="Kanye West", title="Stronger", **overrides):
    base = dict(
        track_id="t1", title=title, artist=artist,
        duration_sec=180.0, file_path="/x.mp3",
    )
    base.update(overrides)
    return TrackMetadata(**base)


class TestApplySongRelations:
    def test_overlays_credits_written_by_genius_import(self, isolated_db):
        MetadataDB.set_song_genius_credits(
            "kanye-west-stronger", ["Kanye West"], "Roc-A-Fella", "colA",
        )
        track = _track()
        apply_song_relations([track])
        assert track.label == "Roc-A-Fella"
        assert track.producer == "Kanye West"

    def test_overlays_merged_producers_from_both_sources(self, isolated_db):
        MetadataDB.set_song_genius_credits(
            "kanye-west-stronger", ["Kanye West"], None, "colA",
        )
        MetadataDB.set_song_relations(
            "kanye-west-stronger", '["KANYE WEST", "Mike Dean"]', "",
        )
        track = _track()
        apply_song_relations([track])
        assert track.producer == "Kanye West, Mike Dean"

    def test_overlays_producer_and_samples_from_extraction(self, isolated_db):
        MetadataDB.add_song_facts_batch(
            "kanye-west-stronger", "colA", ["Produced by Kanye West."],
        )
        MetadataDB.set_song_relations(
            "kanye-west-stronger", '["Kanye West"]',
            '{"samples": [{"song": "Harder Better Faster Stronger", "artist": "Daft Punk"}], "sampled_by": []}',
        )
        track = _track()
        apply_song_relations([track])
        assert track.producer == "Kanye West"
        assert track.samples == ["Daft Punk — Harder Better Faster Stronger"]
        assert track.sampled_by is None

    def test_track_without_extraction_left_untouched(self, isolated_db):
        track = _track(producer="Existing Producer", label="Existing Label")
        apply_song_relations([track])
        assert track.producer == "Existing Producer"
        assert track.label == "Existing Label"

    def test_collab_artist_tag_resolves_to_primary_artist_slug(self, isolated_db):
        """Slug at read time must match the slug the Genius import wrote under —
        both go through get_song_facts_key (primary artist only)."""
        MetadataDB.set_song_genius_credits(
            "kanye-west-stronger", None, "Roc-A-Fella", "colA",
        )
        track = _track(artist="Kanye West feat. Daft Punk")
        apply_song_relations([track])
        assert track.label == "Roc-A-Fella"
