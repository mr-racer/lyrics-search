"""Tests for Pydantic domain models."""

import pytest
from pydantic import ValidationError

from app.domain.models import (
    AIEnabledRequest,
    AlbumSummary,
    ArtistAggregate,
    ArtistAlbum,
    AuthResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Fact,
    HomeTrack,
    IndexProgress,
    IndexRequest,
    InstanceConfigResponse,
    Invite,
    InviteResponse,
    LibraryAlbumsResponse,
    LikedSongTrack,
    LikedSongsResponse,
    ListeningStatsResponse,
    LLMStatusRequest,
    LLMStatusResponse,
    LoginRequest,
    PeakHour,
    PlaylistCreate,
    PlaylistDetail,
    PlaylistReorderRequest,
    PlaylistSummary,
    PlaylistTrack,
    PlaylistTrackAdd,
    PlaylistUpdate,
    PlaylistsResponse,
    RecentTrack,
    RecentTracksResponse,
    RediscoverResponse,
    RegisterRequest,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    TopArtistBrief,
    TopTrackBrief,
    TrackHit,
    TrackMetadata,
    User,
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
        # year_from dropped in B2.2; replaced by year_ranges list
        f = SearchFilters(artist="A", year_ranges=["2000-2009"])
        assert f.artist == "A"
        assert f.year_ranges == ["2000-2009"]


class TestIndexRequest:
    def test_basic(self):
        req = IndexRequest(folder_path="/music")
        assert req.folder_path == "/music"

    def test_collection_name_dropped_in_d_hard(self):
        # Phase D-hard: collection_name was removed from the request schema.
        # A stale client still sending it is NOT rejected (Pydantic ignores the
        # extra field — graceful, no extra='forbid'), but the model no longer
        # carries it; the server derives the collection from the JWT user.
        req = IndexRequest(folder_path="/music", collection_name="my_lib")
        assert req.folder_path == "/music"
        assert not hasattr(req, "collection_name")


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


class TestTrackChatModels:
    def test_track_chat_context_required_fields(self):
        from app.domain.models import TrackChatContext
        ctx = TrackChatContext(
            title="T", artist="A", album=None, year=None, genre=None,
            full_lyrics="line 1\nline 2",
        )
        assert ctx.title == "T"
        assert ctx.full_lyrics == "line 1\nline 2"

    def test_track_chat_request_song_mode_no_selected_line_ok(self):
        from app.domain.models import TrackChatContext, TrackChatRequest
        req = TrackChatRequest(
            track_context=TrackChatContext(
                title="T", artist="A", album=None, year=None, genre=None,
                full_lyrics="",
            ),
            mode="song",
            message="hi",
        )
        assert req.mode == "song"
        assert req.selected_line is None
        assert req.history == []

    def test_track_chat_request_lyric_explain_with_selected_line(self):
        from app.domain.models import TrackChatContext, TrackChatRequest
        req = TrackChatRequest(
            track_context=TrackChatContext(
                title="T", artist="A", album=None, year=None, genre=None,
                full_lyrics="",
            ),
            mode="lyric_explain",
            selected_line="Bring me water for my eyes",
            message="Explain this line",
        )
        assert req.mode == "lyric_explain"
        assert req.selected_line == "Bring me water for my eyes"

    def test_track_chat_response_shape(self):
        from app.domain.models import TrackChatResponse
        r = TrackChatResponse(message="hello", web_search_used=False)
        assert r.message == "hello"
        assert r.web_search_used is False


# --- merged from test_ai_mode_models.py -----------------------------------


class TestAiModeModels:
    """Unit tests for AI Mode pydantic models."""

    def test_llm_status_request_all_fields_optional(self):
        req = LLMStatusRequest()
        assert req.base_url is None
        assert req.model is None
        req = LLMStatusRequest(base_url="http://localhost:1234/v1", model="x")
        assert req.base_url == "http://localhost:1234/v1"

    def test_llm_status_response_required_available_field(self):
        r = LLMStatusResponse(available=True)
        assert r.available is True
        assert r.error is None
        r = LLMStatusResponse(available=False, error="connection refused")
        assert r.error == "connection refused"

    def test_ai_enabled_request_required_bool(self):
        r = AIEnabledRequest(enabled=True)
        assert r.enabled is True
        r = AIEnabledRequest(enabled=False)
        assert r.enabled is False


# --- merged from test_artist_models.py ------------------------------------


def _mk_track(track_id: str, title: str, year: int, album: str = "X") -> TrackMetadata:
    return TrackMetadata(
        track_id=track_id, title=title, artist="Dua Lipa",
        album=album, year=year, duration_sec=180.0,
        file_path=f"/music/{track_id}.flac",
    )


class TestArtistModels:
    """Unit tests for ArtistAggregate / ArtistAlbum pydantic models."""

    def test_artist_album_minimal(self):
        album = ArtistAlbum(title="Future Nostalgia", year=2020, tracks=[])
        assert album.title == "Future Nostalgia"
        assert album.cover_art_path is None  # optional
        assert album.liked_track_count == 0  # default when no reactions

    def test_artist_album_liked_track_count_set(self):
        album = ArtistAlbum(title="Future Nostalgia", year=2020, tracks=[],
                            liked_track_count=3)
        assert album.liked_track_count == 3
        assert album.model_dump()["liked_track_count"] == 3

    def test_artist_aggregate_with_bio(self):
        agg = ArtistAggregate(
            slug="dua-lipa", name="Dua Lipa", genre="pop",
            track_count=3, album_count=1,
            decade_range="2020s", bio="From London…",
            facts=["a", "b"],
            albums=[ArtistAlbum(title="Future Nostalgia", year=2020,
                                 cover_art_path="/covers/x.jpg",
                                 tracks=[_mk_track("t1", "Physical", 2020)])],
        )
        assert agg.bio == "From London…"
        assert agg.albums[0].tracks[0].title == "Physical"

    def test_artist_aggregate_no_bio(self):
        """bio=None must round-trip via .model_dump (frontend gates Bio tab on null)."""
        agg = ArtistAggregate(
            slug="x", name="X", track_count=0, album_count=0,
            facts=[], albums=[],
        )
        assert agg.bio is None
        assert agg.model_dump()["bio"] is None


# --- merged from test_auth_models.py --------------------------------------


class TestAuthModels:
    """Pydantic model contracts for Phase A auth."""

    def test_user_round_trip(self):
        u = User(id="uid-1", email="x@y.z", role="owner",
                 created_at=1700000000.0, last_login_at=None)
        assert u.id == "uid-1"
        assert u.role == "owner"
        assert "password_hash" not in u.model_dump()  # never exposed

    def test_user_role_must_be_owner_or_member(self):
        with pytest.raises(ValidationError):
            User(id="x", email="x@y.z", role="admin", created_at=1.0, last_login_at=None)

    def test_invite_round_trip(self):
        inv = Invite(code="abcdefghij12", created_by="uid-1",
                     created_at=1.0, expires_at=2.0,
                     consumed_by=None, consumed_at=None)
        assert inv.code == "abcdefghij12"

    def test_instance_config_response_modes(self):
        assert InstanceConfigResponse(mode="sharing").mode == "sharing"
        assert InstanceConfigResponse(mode="server").mode == "server"
        with pytest.raises(ValidationError):
            InstanceConfigResponse(mode="other")

    def test_login_request_requires_both_fields(self):
        with pytest.raises(ValidationError):
            LoginRequest(email="x@y.z")

    def test_register_request_requires_invite_code(self):
        with pytest.raises(ValidationError):
            RegisterRequest(email="x@y.z", password="pw12345678")

    def test_auth_response_carries_token_and_user(self):
        ar = AuthResponse(
            token="jwt.token.here",
            user=User(id="u", email="x@y.z", role="owner",
                      created_at=1.0, last_login_at=None),
        )
        assert ar.token.startswith("jwt.")

    def test_invite_response_omits_consumer_when_open(self):
        ir = InviteResponse(
            code="abcdefghij12", created_at=1.0, expires_at=2.0,
            consumed=False, consumed_at=None,
        )
        assert ir.consumed is False


# --- merged from test_home_models.py --------------------------------------


class TestHomeModels:
    def test_home_track_minimal(self):
        t = HomeTrack(track_id="1", title="T", artist="A")
        assert t.album is None and t.cover_art_path is None

    def test_rediscover_response_defaults(self):
        r = RediscoverResponse(collection_name="c")
        assert r.track is None and r.never_played is False and r.last_played is None


# --- merged from test_library_overhaul_models.py --------------------------


class TestLibraryOverhaulModels:
    """Tests that the Library Overhaul response models parse and serialise cleanly."""

    def test_album_summary_minimal(self):
        a = AlbumSummary(
            album_title="In Rainbows",
            primary_artist="Radiohead",
            primary_artist_slug="radiohead",
            feat_artists=[],
            year=2007,
            cover_art_path=None,
            track_count=1,
            duration_seconds=237,
            top_genres=["art rock"],
            tracks=[],
        )
        assert a.year_range is None
        assert a.feat_artists == []

    def test_album_summary_with_year_range(self):
        a = AlbumSummary(
            album_title="Live 2003",
            primary_artist="Sigur Rós",
            primary_artist_slug="sigur-ros",
            feat_artists=[],
            year=None,
            year_range="2002—2003",
            cover_art_path=None,
            track_count=3,
            duration_seconds=900,
            top_genres=["post-rock"],
            tracks=[],
        )
        assert a.year is None
        assert a.year_range == "2002—2003"

    def test_listening_stats_empty(self):
        r = ListeningStatsResponse(
            total_seconds_listened=0,
            since=None,
            top_track=None,
            top_artist=None,
            peak_hour=None,
        )
        assert r.total_seconds_listened == 0


# --- merged from test_playlist_models.py ----------------------------------


class TestPlaylistModels:
    """Smoke tests for Plan 19 Pydantic models."""

    def test_playlist_summary_minimal(self):
        s = PlaylistSummary(
            id=1, name="Mix", description=None,
            track_count=0, cover_track_ids=[], cover_art_paths=[],
            created_at="2026-05-21T10:00:00", updated_at="2026-05-21T10:00:00",
        )
        assert s.contains_track is None
        assert s.cover_track_ids == []

    def test_playlist_summary_with_contains_track(self):
        s = PlaylistSummary(
            id=1, name="Mix", description=None,
            track_count=2, cover_track_ids=["t1", "t2"], cover_art_paths=["/c1", "/c2"],
            created_at="x", updated_at="y", contains_track=True,
        )
        assert s.contains_track is True

    def test_playlist_track_required_fields(self):
        t = PlaylistTrack(track_id="x", position=1, added_at="z", title="A", artist="B")
        assert t.album is None and t.duration is None

    def test_playlist_detail_with_missing_track_ids(self):
        d = PlaylistDetail(
            id=1, name="M", description=None, collection_name="c",
            tracks=[], missing_track_ids=["orphan1"],
            created_at="x", updated_at="y",
        )
        assert d.missing_track_ids == ["orphan1"]

    def test_playlist_create_requires_name(self):
        with pytest.raises(ValidationError):
            PlaylistCreate(collection_name="c", name="", description=None)

    def test_playlist_create_strips_name_whitespace(self):
        p = PlaylistCreate(collection_name="c", name="  Mix  ", description=None)
        assert p.name == "Mix"

    def test_playlist_update_allows_both_none(self):
        PlaylistUpdate(name=None, description=None)

    def test_playlist_reorder_request_accepts_list_of_strings(self):
        r = PlaylistReorderRequest(track_ids=["a", "b", "c"])
        assert r.track_ids == ["a", "b", "c"]

    def test_playlists_response_wraps_list(self):
        r = PlaylistsResponse(playlists=[], collection_name="c")
        assert r.collection_name == "c"
