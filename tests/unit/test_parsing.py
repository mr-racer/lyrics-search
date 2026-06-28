"""Tag/HTML parsing helpers.

Consolidates:
- _parse_facts() / _parse_song_facts() HTML parsers (artist & song facts).
- parse_track_index() track/disc-number extraction from audio tags
  (metadata_readers), which normalises the wildly different shapes mutagen
  returns across containers (FLAC/MP3 strings like "5" or "5/12", MP4
  (num, total) tuples) into a positive int or None.
"""

from mutagen.flac import FLAC

from app.services.artist_facts_service import _parse_facts
from app.services.song_facts_service import _parse_song_facts


ARTIST_FACTS_HTML = """
<ul class="artistfacts-results">
    <li><div class="inner">Fact one about the artist</div></li>
    <li><div class="inner">Fact   two   with   spaces</div></li>
    <li><div class="inner">Fact &amp; entities with &lt;html&gt;</div></li>
    <li>No inner div here</li>
    <li><div class="inner"></div></li>
    <li><div class="inner">  </div></li>
</ul>
"""

SONG_FACTS_HTML = """
<ul class="songfacts-results">
    <li><div class="inner">Song fact one</div></li>
    <li><div class="inner">Song fact two</div></li>
</ul>
"""


class TestParseArtistFacts:
    def test_extracts_facts(self):
        facts = _parse_facts(ARTIST_FACTS_HTML)
        # Three facts: normal, whitespace-normalized, HTML-unescaped
        assert len(facts) == 3
        assert facts[0] == "Fact one about the artist"
        assert facts[1] == "Fact two with spaces"
        assert facts[2] == "Fact & entities with <html>"

    def test_normalizes_whitespace(self):
        facts = _parse_facts(ARTIST_FACTS_HTML)
        assert facts[1] == "Fact two with spaces"

    def test_unescapes_html_entities(self):
        facts = _parse_facts(ARTIST_FACTS_HTML)
        assert facts[2] == "Fact & entities with <html>"

    def test_skips_items_without_inner_div(self):
        facts = _parse_facts(ARTIST_FACTS_HTML)
        assert "No inner div here" not in "\n".join(facts)

    def test_skips_empty_facts(self):
        facts = _parse_facts(ARTIST_FACTS_HTML)
        assert all(f.strip() for f in facts)

    def test_no_container_returns_empty(self):
        facts = _parse_facts("<div>no facts here</div>")
        assert facts == []

    def test_empty_html(self):
        facts = _parse_facts("")
        assert facts == []


class TestParseSongFacts:
    """Song facts parser uses a different CSS class."""

    def test_extracts_facts(self):
        facts = _parse_song_facts(SONG_FACTS_HTML)
        assert len(facts) == 2
        assert facts[0] == "Song fact one"
        assert facts[1] == "Song fact two"

    def test_wrong_class_returns_empty(self):
        """Artist facts container should not match song facts parser."""
        facts = _parse_song_facts(ARTIST_FACTS_HTML)
        assert facts == []

    def test_no_container_returns_empty(self):
        facts = _parse_song_facts("<div>nothing</div>")
        assert facts == []

    def test_empty_html(self):
        facts = _parse_song_facts("")
        assert facts == []


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
