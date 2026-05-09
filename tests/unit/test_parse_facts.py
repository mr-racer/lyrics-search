"""Tests for _parse_facts() and _parse_song_facts() HTML parsers."""

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
