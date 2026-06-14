"""Track-number / disc-number extraction from audio tags (metadata_readers).

parse_track_index normalises the wildly different shapes mutagen returns across
containers (FLAC/MP3 strings like "5" or "5/12", MP4 (num, total) tuples) into a
positive int or None.
"""

from mutagen.flac import FLAC


class TestParseTrackIndex:
    def test_plain_number_string(self):
        from app.indexing.metadata_readers import parse_track_index
        assert parse_track_index("5") == 5

    def test_number_slash_total_string(self):
        from app.indexing.metadata_readers import parse_track_index
        assert parse_track_index("5/12") == 5

    def test_mp4_tuple(self):
        from app.indexing.metadata_readers import parse_track_index
        assert parse_track_index((5, 12)) == 5

    def test_plain_int(self):
        from app.indexing.metadata_readers import parse_track_index
        assert parse_track_index(7) == 7

    def test_zero_is_none(self):
        from app.indexing.metadata_readers import parse_track_index
        assert parse_track_index("0") is None
        assert parse_track_index(0) is None
        assert parse_track_index((0, 0)) is None

    def test_empty_and_none(self):
        from app.indexing.metadata_readers import parse_track_index
        assert parse_track_index("") is None
        assert parse_track_index(None) is None
        assert parse_track_index([]) is None

    def test_garbage_is_none(self):
        from app.indexing.metadata_readers import parse_track_index
        assert parse_track_index("abc") is None
        assert parse_track_index("--") is None


class TestFlacReaderExtractsTrackNumber:
    def test_tracknumber_and_disc_extracted(self, tmp_path, audio_path):
        from app.indexing.metadata_readers import get_flac_metadata

        flac_path = tmp_path / "song.flac"
        flac_path.write_bytes(audio_path("tiny.flac").read_bytes())
        audio = FLAC(str(flac_path))
        audio["tracknumber"] = "5/12"
        audio["discnumber"] = "1"
        audio.save()

        meta = get_flac_metadata(str(flac_path))
        assert meta["track_number"] == 5
        assert meta["disc_number"] == 1

    def test_missing_tracknumber_is_none(self, audio_path):
        from app.indexing.metadata_readers import get_flac_metadata

        # The tiny.flac fixture carries no tracknumber/discnumber tag.
        meta = get_flac_metadata(str(audio_path("tiny.flac")))
        assert meta["track_number"] is None
        assert meta["disc_number"] is None
