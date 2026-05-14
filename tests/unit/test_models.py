"""Tests for Pydantic domain models."""

import pytest

from app.domain.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Fact,
    IndexProgress,
    IndexRequest,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    TrackHit,
    TrackMetadata,
)


class TestTrackMetadata:
    def test_required_fields(self):
        track = TrackMetadata(
            track_id="1",
            title="Song",
            artist="Artist",
            duration_sec=200,
            file_path="/path.flac",
        )
        assert track.title == "Song"
        assert track.album is None
        assert track.year is None

    def test_all_fields(self):
        track = TrackMetadata(
            track_id="1",
            title="Song",
            artist="Artist",
            duration_sec=200,
            file_path="/path.flac",
            album="Album",
            year=2020,
            genre="Pop",
            lyrics="lyrics",
            cover_art_path="/cover.jpg",
            producer="P",
            label="L",
            samples=["A"],
            sampled_by=["B"],
        )
        assert track.year == 2020
        assert track.samples == ["A"]

    def test_missing_required_field_fails(self):
        with pytest.raises(Exception):
            TrackMetadata()


class TestTrackHit:
    def test_creation(self):
        track = TrackMetadata(
            track_id="1", title="S", artist="A", duration_sec=200, file_path="/f"
        )
        hit = TrackHit(track=track, score=0.9, matched_on="lyrics")
        assert hit.score == 0.9
        assert hit.matched_on == "lyrics"
        assert hit.artist_facts is None

    def test_matched_on_literal(self):
        track = TrackMetadata(
            track_id="1", title="S", artist="A", duration_sec=200, file_path="/f"
        )
        for mode in ("lyrics", "audio", "hybrid"):
            hit = TrackHit(track=track, score=0.5, matched_on=mode)
            assert hit.matched_on == mode


class TestSearchRequest:
    def test_basic(self):
        req = SearchRequest(query="hello")
        assert req.query == "hello"
        assert req.mode == "text"
        assert req.limit == 10

    def test_with_mode(self):
        for mode in ("text", "audio", "hybrid"):
            req = SearchRequest(query="q", mode=mode)
            assert req.mode == mode

    def test_with_filters(self):
        req = SearchRequest(query="q", filters=SearchFilters(artist="A"))
        assert req.filters.artist == "A"

    def test_invalid_mode(self):
        with pytest.raises(Exception):
            SearchRequest(query="q", mode="invalid")


class TestSearchResponse:
    def test_empty(self):
        resp = SearchResponse(hits=[], query="q", mode="text")
        assert resp.hits == []

    def test_with_hits(self):
        track = TrackMetadata(
            track_id="1", title="S", artist="A", duration_sec=200, file_path="/f"
        )
        hit = TrackHit(track=track, score=0.9, matched_on="lyrics")
        resp = SearchResponse(hits=[hit], query="q", mode="text")
        assert len(resp.hits) == 1


class TestSearchFilters:
    def test_all_none(self):
        f = SearchFilters()
        assert f.artist is None

    def test_partial(self):
        f = SearchFilters(artist="A", year_from=2000)
        assert f.artist == "A"
        assert f.year_from == 2000


class TestIndexRequest:
    def test_basic(self):
        req = IndexRequest(folder_path="/music")
        assert req.folder_path == "/music"
        assert req.collection_name == "music_explorer"

    def test_custom_collection(self):
        req = IndexRequest(folder_path="/music", collection_name="my_lib")
        assert req.collection_name == "my_lib"


class TestIndexProgress:
    def test_running(self):
        p = IndexProgress(status="running", progress=5, total=100)
        assert p.progress == 5
        assert p.total == 100

    def test_failed(self):
        p = IndexProgress(status="failed", progress=3, message="Error")
        assert p.status == "failed"


class TestFact:
    def test_basic(self):
        f = Fact(fact="Interesting fact")
        assert f.fact == "Interesting fact"
        assert f.lang == "en"

    def test_with_source(self):
        f = Fact(fact="F", category="trivia", source="web")
        assert f.category == "trivia"


class TestChatRequest:
    def test_basic(self):
        req = ChatRequest(message="Tell me about pop")
        assert req.message == "Tell me about pop"
        assert req.history == []
        assert req.auto_mode is True

    def test_with_history(self):
        msgs = [ChatMessage(role="user", content="Hi")]
        req = ChatRequest(message="More", history=msgs)
        assert len(req.history) == 1


class TestChatResponse:
    def test_basic(self):
        resp = ChatResponse(
            query="q", mode="text", hits=[], llm_response="Here you go"
        )
        assert resp.llm_response == "Here you go"


def test_sonic_tag_serializes():
    from app.domain.models import SonicTag
    t = SonicTag(tag="anxious", score=0.72)
    d = t.model_dump()
    assert d == {"tag": "anxious", "score": 0.72}


def test_sonic_descriptor_optional_class():
    from app.domain.models import SonicDescriptor, SonicTag
    d = SonicDescriptor(
        track_id="abc",
        tags=[SonicTag(tag="warm", score=0.6)],
        sonic_class=None,
        sonic_class_confidence=None,
    )
    assert d.sonic_class is None
    assert d.tags[0].tag == "warm"


def test_classifier_status_unready_state():
    from app.domain.models import ClassifierStatus
    s = ClassifierStatus(status="untrained", trained_at=None, accuracy=None, classes=[])
    assert s.status == "untrained"
    assert s.classes == []


def test_cluster_representative_shape():
    from app.domain.models import ClusterRepresentative
    r = ClusterRepresentative(
        cluster_id=0,
        size=12,
        representative_tracks=[
            {"track_id": "t1", "title": "Song", "artist": "Artist", "cover_art_path": None},
        ],
        current_label=None,
    )
    assert r.size == 12
    assert r.representative_tracks[0]["track_id"] == "t1"
