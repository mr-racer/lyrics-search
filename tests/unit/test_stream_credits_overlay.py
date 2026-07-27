"""Regression: the wave / similar / axis endpoints must overlay
producer/samples credits (SQLite) onto their tracks, the same way
/metadata/tracks does for manual plays.

Before the fix, `recommend._stream_tracks` (and autoplay_service) built
StreamTracks straight from the Qdrant payload, so the producers drawer and
samples pill sat empty in the wave even when the extraction existed.
"""
import pytest

from app.api.routes.recommend import _stream_tracks
from app.resources.metadata_db import MetadataDB
from app.services.stream_service import StreamCandidate


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _candidate(artist="Kanye West", title="Stronger", **payload_extra):
    payload = {"artist": artist, "title": title, "duration": 180.0,
               "file_path": "/x.mp3"}
    payload.update(payload_extra)
    return StreamCandidate(track_id="t1", payload=payload, pool="anchor")


def test_stream_tracks_overlays_extraction_credits(isolated_db):
    # Extraction exists in SQLite but NOT in the Qdrant payload — exactly the
    # wave's situation. The overlay must still populate producer/samples.
    MetadataDB.add_song_facts_batch(
        "kanye-west-stronger", "colA", ["Produced by Kanye West."],
    )
    MetadataDB.set_song_relations(
        "kanye-west-stronger", '["Kanye West"]',
        '{"samples": [{"song": "HBFS", "artist": "Daft Punk"}], "sampled_by": []}',
    )

    tracks = _stream_tracks([_candidate()])

    assert len(tracks) == 1
    assert tracks[0].producer == "Kanye West"
    assert tracks[0].samples == ["Daft Punk — HBFS"]


def test_stream_tracks_keeps_payload_tag_when_no_extraction(isolated_db):
    # No extraction row → the ID3-tag producer from the payload survives.
    tracks = _stream_tracks([_candidate(producer="Tag Producer")])

    assert tracks[0].producer == "Tag Producer"


def test_stream_tracks_no_credits_when_neither(isolated_db):
    tracks = _stream_tracks([_candidate()])

    assert tracks[0].producer is None
    assert tracks[0].samples is None
