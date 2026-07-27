"""Domain models for Music Explorer."""

from typing import Literal, List, Optional, Dict
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


class Fact(BaseModel):
    """A single fact about an artist or song."""
    fact: str
    lang: str = "en"
    category: str | None = None
    source: str | None = None


class ArtistRef(BaseModel):
    """A single participant (display name + canonical slug) — used both for
    album feat lists and per-track artist credits so each collaborator links to
    their own artist page."""
    name: str
    slug: str
    # "feat" marks a featured guest (extracted from the title or credits);
    # "main" default keeps every pre-existing serializer back-compatible.
    role: Literal["main", "feat"] = "main"


class TrackMetadata(BaseModel):
    """Метаданные трека."""
    track_id: str  # хэш file_path или UUID, стабильный между рестартами
    title: str
    # Title with feat/with credits stripped («Bangarang (ft. Sirah)» →
    # «Bangarang»); None when the raw title is already clean.
    title_display: str | None = None
    artist: str
    album: str | None = None
    year: int | None = None
    genre: str | None = None
    duration_sec: float
    file_path: str
    lyrics: str | None = None
    cover_art_path: str | None = None  # путь к обложке /covers/{track_id}.{ext}
    producer: str | None = None
    label: str | None = None
    samples: list[str] | None = None
    sampled_by: list[str] | None = None
    reaction: Literal["like", "dislike"] | None = None
    bitrate_kbps: int | None = None
    # Canonical participants derived from the raw `artist` string at index time.
    artists: list[str] | None = None
    primary_artist_slug: str | None = None
    # Aligned name+slug pairs per participant — lets the UI render each
    # collaborator as its own clickable link. Populated by serializers via
    # artist_split.artist_refs(); empty when unknown.
    artist_refs: list[ArtistRef] = Field(default_factory=list)
    # Real position in the release, from file tags — drives album track ordering.
    track_number: int | None = None
    disc_number: int | None = None


class ScoreBreakdown(BaseModel):
    """Per-modality contributions to a TrackHit's final ranking score."""
    text_dense_score: Optional[float] = None  # cosine sim from sentence-transformer
    text_bm25_score: Optional[float] = None  # raw BM25 score
    audio_score: Optional[float] = None  # cosine sim from CLAP
    final_score: float  # combined score used for ranking
    weights: Dict[str, float] = Field(default_factory=dict)


class TrackHit(BaseModel):
    """Результат поиска с трек-метаданными, score и matched_on."""
    track: TrackMetadata
    score: float
    matched_on: Literal["lyrics", "audio", "hybrid"] = "lyrics"
    lyrics: str | None = None  # выдержка из лирики для lyrics-поиска
    matched_line: str | None = None  # строка лирики с максимальным перекрытием с запросом (подсветка в UI)
    artist_facts: str | None = None  # interesting facts about the artist
    song_facts: str | None = None  # interesting facts about the song
    score_breakdown: Optional[ScoreBreakdown] = None


class SearchFilters(BaseModel):
    """Фильтры для поиска."""
    # extra="ignore" — planner LLM may still emit legacy year_range (singular);
    # silently drop unknown fields instead of raising ValidationError.
    model_config = ConfigDict(extra="ignore")

    artist: str | None = None
    album: str | None = None
    genre: str | None = None

    # Decade / year-range chips. OR semantics across selected ranges.
    year_ranges: list[str] = []

    # Phase 1c — sonic descriptors. AND semantics across tags (track must carry every selected tag).
    sonic_tags: list[str] = []


class SearchRequest(BaseModel):
    """Запрос на поиск."""
    query: str
    mode: Literal["text", "audio", "hybrid"] = "text"
    text_model: Optional[str] = Field(None, description="Text embedding model to use")
    filters: SearchFilters | None = None
    limit: int = 10


class SearchResponse(BaseModel):
    """Ответ на поисковый запрос."""
    hits: List[TrackHit]
    query: str
    mode: Literal["text", "audio", "hybrid"]


class CatalogHit(BaseModel):
    """One result of the non-LLM catalog search (GET /search/catalog).

    Discriminated by ``type``; only the fields relevant to that type are set.
    """
    type: Literal["song", "album", "artist"]
    score: float
    # song
    track_id: str | None = None
    title: str | None = None
    # song + album
    album: str | None = None
    cover_art_path: str | None = None
    # all (album: primary artist; artist: the artist)
    artist: str | None = None
    artist_slug: str | None = None
    # album + artist
    track_count: int | None = None
    image: str | None = None            # artist photo (thumb_path / cutout_path)


class IndexRequest(BaseModel):
    """Запрос на индексацию папки с музыкой."""
    folder_path: str
    better_lyrics_quality: bool = False
    text_model: Optional[str] = None
    enhance_by_musicbrainz: bool = False


class IndexProgress(BaseModel):
    """Прогресс индексации."""
    status: Literal["pending", "running", "completed", "failed"]
    progress: int  # количество обработанных треков
    total: int | None = None  # общее количество треков
    message: str | None = None


# Chat types (LLM-assisted search)
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Запрос на LLM-чат с поиском."""
    message: str
    history: List[ChatMessage] = []
    mode: Literal["text", "audio", "hybrid"] = "hybrid"
    auto_mode: bool = True  # если True — LLM классифицирует запрос, если False — используется mode напрямую
    planner_enabled: bool = False  # если True — используется PydanticAI Planner вместо старой классификации
    # LLM connection — overrides env vars LLM_BASE_URL / LLM_MODEL if set
    llm_base_url: Optional[str] = Field(None, description="e.g. http://localhost:8000/v1")
    llm_model: Optional[str] = Field(None, description="e.g. openai/gpt-oss-20b")
    # UI language → language of the user-facing answer. None = no explicit directive.
    lang: Optional[str] = Field(None, description='"ru" | "en"')


class ChatResponse(BaseModel):
    """Ответ от LLM-чата."""
    query: str
    mode: Literal["text", "audio", "hybrid"]
    hits: List[TrackHit]
    llm_response: str


class TrackReactionRequest(BaseModel):
    """Запрос на установку реакции на трек."""
    reaction: Literal["like", "dislike"] | None = None  # None = remove reaction


class TrackReactionResponse(BaseModel):
    """Ответ с реакцией на трек."""
    track_id: str
    collection_name: str
    reaction: Literal["like", "dislike"] | None


class TrackChatContext(BaseModel):
    """Context about the track the user is chatting about. Backend resolves
    song_facts server-side (raw, not refined) — DO NOT include facts here."""
    title: str
    artist: str
    album: str | None = None
    year: int | None = None
    genre: str | None = None
    full_lyrics: str = ""


class TrackChatRequest(BaseModel):
    """Request body for POST /chat/track-chat."""
    track_context: TrackChatContext
    mode: Literal["song", "lyric_explain"]
    selected_line: str | None = None  # required when mode='lyric_explain'
    history: List[ChatMessage] = []
    message: str
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    # UI language → reply language. None falls back to "match the user's message".
    lang: Optional[str] = None


class TrackChatResponse(BaseModel):
    """Response body for POST /chat/track-chat."""
    message: str
    web_search_used: bool = False


class PlaylistDraft(BaseModel):
    """Playlist proposed by the playlist-builder agent (not yet persisted).

    ``track_ids`` reference library tracks the agent matched via ``get_songs``;
    ``missing`` lists songs the agent wanted but the library doesn't have —
    consumed by the recsys ``web_hits`` delegation (`_web_hits_playlist`).
    """
    title: str
    track_ids: List[str] = Field(default_factory=list)
    comment: str = ""
    missing: List[str] = Field(default_factory=list)

class BaseQueryItem(BaseModel):
    query: str


# ── PydanticAI Agent Models ────────────────────────────────────────────────────


class SearchPlan(BaseModel):
    """План поиска, сгенерированный PlannerAgent."""
    action: Literal["request_filter", "search"]
    query_type: Literal["text", "audio", "hybrid"]
    filters: SearchFilters | None = None
    filter_lookup: Dict[str, str | None] | None = None  # сырые значения для разрешения
    queries: List[BaseQueryItem] = Field(default_factory=list)
    search_mode: Literal["CONSERVATIVE", "AGGRESSIVE"] = "CONSERVATIVE"


class ScoreResult(BaseModel):
    """Результат оценки контекста от ScorerAgent."""
    action: Literal["search", "answer", "final_answer"]
    confidence: Literal["high", "medium", "low"]
    song: str | None = None
    artist: str | None = None
    filters: SearchFilters | None = None  # pass-through
    queries: List[BaseQueryItem] | None = None  # новые запросы, если action="search"
    message: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_booleans(cls, data: object) -> object:
        """Local LLMs sometimes return False/None for Literal string fields via tool calling.
        Coerce to safe defaults so PydanticAI doesn't retry and raise UnexpectedModelBehavior."""
        if not isinstance(data, dict):
            return data
        if not isinstance(data.get("action"), str):
            data["action"] = "search"
        if not isinstance(data.get("confidence"), str):
            data["confidence"] = "medium"
        if not isinstance(data.get("message"), str):
            data["message"] = ""
        return data




class ValidatorResult(BaseModel):
    """Решение ValidatorAgent: принять ответ или продолжить поиск."""
    valid: bool
    reason: str
    queries: List[BaseQueryItem] | None = None  # новые запросы если valid=False





# SONIC DESCRIPTOR MODELS


class SonicTag(BaseModel):
    """One adjective tag with similarity score from CLAP prompt-probing."""
    tag: str
    score: float


class SonicDescriptor(BaseModel):
    """Combined interpretable descriptor for a track."""
    track_id: str
    tags: list[SonicTag] = []
    sonic_class: str | None = None
    sonic_class_confidence: float | None = None


class PlaybackEventIn(BaseModel):
    """Request body for POST /playback/events."""
    session_id: str
    track_id: str
    played_sec: float
    total_dur: float | None = None
    # Stream RecSys idle rule: did the user touch any control during this track
    # (like/skip/pause/seek)? None = client doesn't report it (legacy frontend).
    interacted: bool | None = None
    # influence=False → event is stored for anti-repeat but excluded from the
    # "For You" taste profile (e.g. hand-queued tracks, Task 5 _noInfluence).
    influence: bool = True


class PlaybackEventOut(BaseModel):
    """Successful POST response."""
    id: int


class TasteSignalIn(BaseModel):
    """Request body for POST /recommend/taste-signal (огонёк/вода).

    fire = «больше такого» (strong ephemeral wave anchor); water = «остудить»
    (soft ephemeral demotion). Both fade on time × session-track-count.
    """
    session_id: str
    track_id: str
    kind: Literal["fire", "water"]


class TasteSignalOut(BaseModel):
    """Successful POST response."""
    id: int


class TasteSignalState(BaseModel):
    """Active огонёк/вода on a track: which kind, its remaining «заряд» (0..1)
    and whether the button is locked (charge still above the halfway unlock)."""
    kind: Literal["fire", "water"]
    contribution: float
    locked: bool


class TasteSignalStateOut(BaseModel):
    """GET /recommend/taste-signal/state — active reaction per requested track.
    Tracks with no active signal are omitted from ``states``."""
    states: dict[str, TasteSignalState]


class AutoplayQueueDiagnostics(BaseModel):
    """Counters from the autoplay filter pipeline — for telemetry / debug."""
    candidates_fetched: int
    dropped_excluded: int
    dropped_disliked: int
    dropped_diversity: int
    returned: int


class AutoplayQueueResponse(BaseModel):
    """Result of GET /recommend/autoplay-queue."""
    seed_track_id: str
    tracks: list[TrackMetadata]
    diagnostics: AutoplayQueueDiagnostics


class AIJobStatus(BaseModel):
    """Public surface for an AI Indexing job's state."""
    job_id: str
    task_type: str  # "sonic_vibe" | "refined_facts"
    collection_name: str
    lang: str
    status: str    # "queued" | "running" | "done" | "failed" | "cancelled"
    n_total: int
    n_done: int     # actually processed (LLM called OR cache hit)
    n_failed: int   # LLM / validation / persistence error
    n_skipped: int = 0  # default for forward-compat with rows from old schema
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


class ArtistAlbum(BaseModel):
    """One album in an artist's discography aggregate."""
    title: str
    year: Optional[int] = None
    cover_art_path: Optional[str] = None  # representative track's cover
    tracks: list[TrackMetadata] = Field(default_factory=list)
    liked_track_count: int = 0  # likes among this album's tracks (gold-star marker)
    # True when this artist appears on the album ONLY as a featured guest —
    # the Atlas screen shows such albums in a separate "featured on" rail.
    featured_only: bool = False


class ArtistAggregate(BaseModel):
    """Aggregate response for GET /artists/{slug} — drives the Atlas screen."""
    slug: str
    name: str
    genre: Optional[str] = None
    track_count: int
    album_count: int
    decade_range: Optional[str] = None  # e.g. "2010s-2020s"
    bio: Optional[str] = None           # None when artist_bio not yet indexed
    facts: list[str] = Field(default_factory=list)
    albums: list[ArtistAlbum] = Field(default_factory=list)
    # AudioDB-derived (out-of-band shipment 2026-05-20)
    mood: Optional[str] = None
    country_code: Optional[str] = None
    country: Optional[str] = None
    label: Optional[str] = None
    cutout_path: Optional[str] = None
    thumb_path: Optional[str] = None
    audiodb_mbid: Optional[str] = None


class LLMStatusRequest(BaseModel):
    """Body for POST /system/llm-status. All fields optional —
    server falls back to env LLM_BASE_URL / LLM_MODEL when omitted."""
    base_url: Optional[str] = None
    model: Optional[str] = None


class LLMStatusResponse(BaseModel):
    """Result of an LLM availability probe."""
    available: bool
    base_url: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None


class AIEnabledRequest(BaseModel):
    """Body for PATCH /library/collections/{name}/ai-enabled."""
    enabled: bool


# ── Library Overhaul (Phase 6 sub-plan #1) ──

class AlbumTrack(BaseModel):
    """Lightweight track inside an AlbumSummary — covers what AlbumModal renders."""
    track_id: str
    title: str
    artist: str
    duration: Optional[float] = None
    year: Optional[int] = None
    cover_art_path: Optional[str] = None


class AlbumSummary(BaseModel):
    album_title: str
    primary_artist: str
    primary_artist_slug: str
    feat_artists: list[ArtistRef] = Field(default_factory=list)
    year: Optional[int] = None
    year_range: Optional[str] = None
    cover_art_path: Optional[str] = None
    track_count: int
    duration_seconds: int
    top_genres: list[str] = Field(default_factory=list)
    tracks: list[AlbumTrack] = Field(default_factory=list)
    # Record labels across the album's tracks (deduped, most frequent first).
    labels: list[str] = Field(default_factory=list)


class LibraryAlbumsResponse(BaseModel):
    albums: list[AlbumSummary]
    collection_name: Optional[str] = None
    qdrant_available: bool = True


class ProducerCredit(BaseModel):
    """One name from a track's producer tag, resolved against the library."""
    name: str
    # Set when the library has this producer as an artist (canonical slug).
    artist_slug: Optional[str] = None
    # The artist's photo (thumb_path only — never the cutout: the producer
    # popup renders a portrait card). None → monogram fallback client-side.
    image: Optional[str] = None
    # How many OTHER tracks in the library credit them as producer.
    produced_count: int = 0


class ProducerResolveResponse(BaseModel):
    producers: list[ProducerCredit]
    collection_name: Optional[str] = None


class ProducerTrack(BaseModel):
    track_id: str
    title: str
    title_display: Optional[str] = None
    artist: str
    album: Optional[str] = None
    year: Optional[int] = None
    duration: Optional[float] = None
    cover_art_path: Optional[str] = None
    artist_refs: list[ArtistRef] = Field(default_factory=list)


class ProducerTracksResponse(BaseModel):
    producer: str
    tracks: list[ProducerTrack]
    collection_name: Optional[str] = None


class SamplesResolveRequest(BaseModel):
    """Sample-reference strings ("Artist — Title") to match against the library."""
    items: list[str] = Field(default_factory=list, max_length=50)


class SamplesResolveResponse(BaseModel):
    # Aligned with the request's ``items``; None → not in the library.
    matches: list[Optional[ProducerTrack]]
    collection_name: Optional[str] = None


class LikedSongTrack(BaseModel):
    track_id: str
    title: str
    title_display: Optional[str] = None
    artist: str
    album: Optional[str] = None
    year: Optional[int] = None
    duration: Optional[float] = None
    cover_art_path: Optional[str] = None
    genre: Optional[str] = None
    liked_at: str   # ISO datetime
    artist_refs: list[ArtistRef] = Field(default_factory=list)


class LikedSongsResponse(BaseModel):
    tracks: list[LikedSongTrack]
    collection_name: Optional[str] = None


class RecentTrack(BaseModel):
    track_id: str
    title: str
    title_display: Optional[str] = None
    artist: str
    album: Optional[str] = None
    year: Optional[int] = None
    duration: Optional[float] = None
    cover_art_path: Optional[str] = None
    genre: Optional[str] = None
    last_played: str   # ISO datetime
    play_count: int
    artist_refs: list[ArtistRef] = Field(default_factory=list)


class RecentTracksResponse(BaseModel):
    tracks: list[RecentTrack]
    collection_name: Optional[str] = None


class HomeTrack(BaseModel):
    """Lightweight track for Home (Discovery Magazine) blocks."""
    track_id: str
    title: str
    title_display: Optional[str] = None
    artist: str
    album: Optional[str] = None
    year: Optional[int] = None
    duration: Optional[float] = None
    cover_art_path: Optional[str] = None
    genre: Optional[str] = None
    artist_refs: list[ArtistRef] = Field(default_factory=list)


class RediscoverResponse(BaseModel):
    """Result of GET /library/rediscover — a long-unplayed track to revisit."""
    track: Optional[HomeTrack] = None
    last_played: Optional[str] = None   # ISO datetime, None when never played
    never_played: bool = False
    collection_name: Optional[str] = None


class StreamTrack(TrackMetadata):
    """One track in the personalized stream + per-track diagnostics.

    The diagnostics double as raw material for a future rationale chip
    («почему этот трек») — design §7. ``pool="axis"`` marks tracks picked by
    the axis-playlist knobs rather than the stream pools; ``pool="replay"`` marks
    a track resurfaced by the round-reset last-resort pass (design 2026-06-14).
    """
    pool: Literal["anchor", "explore", "liked", "axis", "replay"]
    anchor_track_id: Optional[str] = None   # closest anchor (pool=anchor only)
    axis_match: Optional[float] = None
    score: Optional[float] = None


class SessionAdaptationTrack(BaseModel):
    """A track that visibly steered this session (fire/replay)."""
    track_id: str
    title: str
    artist: str
    cover_art_path: Optional[str] = None


class SessionAdaptation(BaseModel):
    """«Подстроились под твой вайб» badge payload — present only when session
    contributions are distinguishable (a fire or a replay happened)."""
    active: bool
    tracks: List[SessionAdaptationTrack] = []


class StreamNextResponse(BaseModel):
    """Result of GET /recommend/stream/next."""
    session_id: str
    tracks: List[StreamTrack]
    diagnostics: dict
    session_adaptation: Optional[SessionAdaptation] = None


class StreamSettingsIn(BaseModel):
    """Body for PUT /recommend/stream/settings — persists the slider."""
    liked_share: float = Field(ge=0.0, le=1.0)


class SimilarTracksResponse(BaseModel):
    """Result of GET /recommend/similar — CLAP neighbors re-ranked by axes.

    Tracks reuse StreamTrack: ``anchor_track_id`` = the seed, ``axis_match`` =
    axis closeness to the seed, ``score`` = 0.7·cos + 0.3·axes blend.
    """
    seed_track_id: str
    tracks: List[StreamTrack]


class ProfileAxis(BaseModel):
    """One axis of the taste profile: z-score + discrete level label."""
    z: float
    level: Literal["very_low", "low", "medium", "high", "very_high"]


class ProfileIslandTrack(BaseModel):
    """Lightweight member of a taste island (cover wall)."""
    track_id: str
    title: str
    artist: str
    album: Optional[str] = None        # frontend dedupes covers per album
    cover_art_path: Optional[str] = None


class ProfileIsland(BaseModel):
    """One «музыкальная личность»: a merged anchor group from the stream model."""
    track_id: str                      # representative (strongest member)
    weight: float
    tracks: List[ProfileIslandTrack]   # representative first
    name: Optional[str] = None         # LLM-given name (cached, AI mode)


class StreamProfileResponse(BaseModel):
    """Result of GET /recommend/profile — the explainable long-term taste."""
    axes: Optional[Dict[str, ProfileAxis]] = None  # None until axis data exists
    confidence: float
    n_signals: int
    islands: List[ProfileIsland]
    vibes: List[ProfileIsland] = []    # «вайбики» — days-scale mood clusters
    portrait: Optional[str] = None     # cached LLM listener portrait (AI mode)
    headline: Optional[str] = None     # cached short hero title (AI mode)
    axis_stats_source: Optional[str] = None
    liked_share: float = 0.3           # persisted slider position (default 30%)


class VibeAlbumSuggestion(BaseModel):
    """One album pick for the library rail: sounds like a current vibe's mean
    CLAP but is not represented inside that vibe."""
    album_title: str
    primary_artist: str
    primary_artist_slug: str
    cover_art_path: Optional[str] = None
    track_count: int
    year: Optional[int] = None
    score: float                       # cosine(vibe centroid, album mean CLAP)
    vibe_track_id: str                 # representative track of the source vibe
    vibe_name: Optional[str] = None    # cached LLM vibe name, when fresh


class VibeAlbumSuggestionsResponse(BaseModel):
    """Result of GET /recommend/vibes/album-suggestions."""
    suggestions: List[VibeAlbumSuggestion] = []


class AxisPlaylistIn(BaseModel):
    """Body for POST /recommend/axis-playlist — the radar-knob targets.

    ``targets`` maps axis name → desired z-score (clamped server-side to ±3);
    omitted axes default to 0 (neutral).
    """
    targets: Dict[str, float] = Field(default_factory=dict)
    limit: int = Field(20, ge=1, le=100)


class AxisPlaylistResponse(BaseModel):
    tracks: List[StreamTrack]
    diagnostics: dict


class ProfileEnrichIn(BaseModel):
    """Body for POST /recommend/profile/ai-enrich (AI mode)."""
    lang: str = "en"
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None


class ProfileEnrichResponse(BaseModel):
    """LLM listener portrait + island names (also persisted server-side)."""
    portrait: Optional[str] = None
    island_names: Dict[str, str] = Field(default_factory=dict)
    headline: Optional[str] = None     # short hero title (AI mode)


class AIPlaylistIn(BaseModel):
    """Body for POST /recommend/ai-playlist — one wish → curated playlist."""
    prompt: str = Field(min_length=3, max_length=500)
    limit: int = Field(15, ge=1, le=40)
    lang: str = "en"
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None


class AIPlaylistStep(BaseModel):
    """One executed plan action — the frontend animates these."""
    tool: str
    query: str
    found: int


class AIPlaylistTrack(TrackMetadata):
    reason: Optional[str] = None       # why this track fits the wish (LLM, short)
    source_tool: Optional[str] = None  # which tool surfaced it


class AIPlaylistResponse(BaseModel):
    title: str
    steps: List[AIPlaylistStep]
    tracks: List[AIPlaylistTrack]


class TopTrackBrief(BaseModel):
    track_id: str
    title: str
    artist: str
    play_count: int
    cover_art_path: Optional[str] = None


class TopArtistBrief(BaseModel):
    name: str
    slug: str
    play_count: int
    # The artist's AudioDB photo (thumb_path / cutout_path), reused as the avatar
    # in the stats UI. None when nothing has been cached for the artist yet.
    image: Optional[str] = None


class PeakHour(BaseModel):
    hour: int   # 0-23
    label: str  # localised "вечера буднего" / "weekday evenings"


class ListeningStatsResponse(BaseModel):
    total_seconds_listened: int = 0
    since: Optional[str] = None
    top_track: Optional[TopTrackBrief] = None
    top_artist: Optional[TopArtistBrief] = None
    peak_hour: Optional[PeakHour] = None


class RhythmDay(BaseModel):
    date: str   # 'YYYY-MM-DD' in the user's local time
    count: int


class BusiestDay(BaseModel):
    date: str
    count: int
    top_track: Optional[TopTrackBrief] = None


class RhythmResponse(BaseModel):
    """Listening rhythm for the Library stats tab: per-day calendar heatmap,
    24h distribution, streaks and the busiest day. All times are bucketed in the
    caller's local time via ``tz_offset_minutes``."""
    days: List[RhythmDay] = []
    by_hour: List[int] = []        # 24 ints, index = local hour 0-23
    streak_current: int = 0
    streak_best: int = 0
    busiest_day: Optional[BusiestDay] = None


class WeeklyPulseResponse(BaseModel):
    """Light "this week" summary for the home page's bottom strip: listening
    time, top genre and first-time-heard track count for the current calendar
    week (Monday..now, caller's local time via ``tz_offset_minutes``)."""
    seconds_listened: int = 0
    top_genre: Optional[str] = None
    discoveries: int = 0
    # Monday..Sunday of the current week, seconds listened per local day
    # (future days are 0). Empty list = older server data, client hides bars.
    daily_seconds: List[int] = []


class EngagementTrack(BaseModel):
    track_id: str
    title: str
    artist: str
    cover_art_path: Optional[str] = None
    completion: float   # 0..1 mean play completion
    plays: int
    # "Loved" list: how many times the track was played through to the end.
    finish_count: int = 0
    # "Skipped" list: how often it was abandoned early, and the typical number
    # of seconds heard before the skip.
    skip_count: int = 0
    skip_seconds: Optional[float] = None


class EngagementResponse(BaseModel):
    """Honest-mirror engagement for the stats tab: how much you actually finish,
    the tracks you replay-and-finish ("loved") vs launch-often-but-skip ("guilty")."""
    overall_completion: float = 0.0   # 0..1 library-wide mean completion
    loved: List[EngagementTrack] = []
    guilty: List[EngagementTrack] = []


class TasteMapPoint(BaseModel):
    track_id: str
    x: float           # 2D projection, ~[-1, 1]
    y: float
    cluster: int       # cluster id (index into clusters[])
    title: str
    artist: str


class TasteMapCluster(BaseModel):
    id: int
    name: str          # plain-language name (dominant genre of the cluster)
    size: int
    cx: float          # centroid in projection space
    cy: float
    spread: float      # rms radius of the cluster (for the bloom size)
    sample_track_ids: List[str] = []


class TasteMapResponse(BaseModel):
    """«Сонар вкуса» — 2D projection of the library's audio vectors, grouped into
    sonic islands. Points/clusters are precomputed server-side; the client just
    paints them."""
    points: List[TasteMapPoint] = []
    clusters: List[TasteMapCluster] = []


# ─── Plan 19: Custom Playlists ───────────────────────────────────────────


class PlaylistTrack(BaseModel):
    """One track inside a playlist, with resolved Qdrant metadata."""
    track_id: str
    position: int
    added_at: str
    title: str
    title_display: Optional[str] = None
    artist: str
    album: Optional[str] = None
    year: Optional[int] = None
    duration: Optional[float] = None
    cover_art_path: Optional[str] = None
    artist_refs: list[ArtistRef] = Field(default_factory=list)


class PlaylistSummary(BaseModel):
    """List-view shape. cover_track_ids/cover_art_paths parallel-arrays for the mosaic."""
    id: int
    name: str
    description: Optional[str] = None
    track_count: int
    cover_track_ids: list[str]
    cover_art_paths: list[Optional[str]]
    created_at: str
    updated_at: str
    # Populated only when GET /playlists?include_track_id=... is used (otherwise None)
    contains_track: Optional[bool] = None


class PlaylistDetail(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    collection_name: str
    tracks: list[PlaylistTrack]
    missing_track_ids: list[str]
    created_at: str
    updated_at: str


class PlaylistCreate(BaseModel):
    name: str
    description: Optional[str] = None

    @field_validator("name")
    @classmethod
    def _strip_and_check_name(cls, v: str) -> str:
        stripped = (v or "").strip()
        if not stripped:
            raise ValueError("name must not be empty")
        if len(stripped) > 120:
            raise ValueError("name too long (max 120 chars)")
        return stripped

    @field_validator("description")
    @classmethod
    def _check_desc(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if len(v) > 1000:
            raise ValueError("description too long (max 1000 chars)")
        return v


class PlaylistUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    # If True, description is set to NULL (because Pydantic can't tell "field absent" from "field=null" in dict).
    clear_description: bool = False

    @field_validator("name")
    @classmethod
    def _check(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        stripped = v.strip()
        if not stripped:
            raise ValueError("name must not be empty when provided")
        if len(stripped) > 120:
            raise ValueError("name too long (max 120 chars)")
        return stripped


class PlaylistTrackAdd(BaseModel):
    track_id: str


class PlaylistReorderRequest(BaseModel):
    track_ids: list[str]


class PlaylistsResponse(BaseModel):
    playlists: list[PlaylistSummary]
    collection_name: str


# ─── Phase A: Auth Foundation ─────────────────────────────────────────────


class User(BaseModel):
    """Public-facing user shape. password_hash is intentionally absent."""
    id: str
    email: str
    role: Literal["owner", "member"]
    created_at: float
    last_login_at: Optional[float] = None
    # MusiX Premium tier — display-only flag (Yandex Music transfer, premium
    # metadata/cover sources). Set directly in SQLite by the operator; no
    # endpoint mutates it and nothing is gated server-side.
    premium: bool = False


class Invite(BaseModel):
    """Full invite row, including consumer fields. Owner-only surface."""
    code: str
    created_by: str
    created_at: float
    expires_at: float
    consumed_by: Optional[str] = None
    consumed_at: Optional[float] = None


class InstanceConfigResponse(BaseModel):
    """Body of GET /instance/config — public, no auth."""
    mode: Literal["sharing", "server"]
    # Whether AI features should be offered to members (admin enabled AI AND a
    # working LLM endpoint is configured). Exposed publicly so a member's
    # frontend can decide UI without ever seeing the endpoint/key.
    ai_available: bool = False
    # Server mode only: the mounted path (MEMBER_INDEX_ROOT) that members may
    # index by reference, or null when the opt-in is off. Lets the member UI
    # show/hide the "index mounted folder" action without exposing host details.
    member_index_root: str | None = None


class SetupRequest(BaseModel):
    """Body of POST /instance/setup — first-run bootstrap. Public while the
    instance is uninitialized; 409 forever after the first success."""
    email: str
    password: str
    mode: Literal["sharing", "server"]


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    invite_code: str


class AuthResponse(BaseModel):
    """Returned by /auth/login and /auth/register."""
    token: str
    user: User


class InviteResponse(BaseModel):
    """Returned by /auth/invites POST/GET. Slimmer than Invite — drops
    `created_by` since the only consumer of this surface IS the creator."""
    code: str
    created_at: float
    expires_at: float
    consumed: bool
    consumed_at: Optional[float] = None
    # Full registration URL (PUBLIC_BASE_URL or LAN-IP fallback + invite hash),
    # built server-side so the link works behind a reverse proxy. None on older
    # servers — the client then falls back to window.location.origin.
    link: Optional[str] = None


# ── Instance settings (owner-only; server-mode onboarding) ──────────────────
class SettingItem(BaseModel):
    """One resolved instance setting for the owner Settings UI. ``value`` is the
    effective value (DB > env > default); ``source`` says where it came from.
    For secrets, ``value`` is null and ``has_value`` indicates whether a key is
    configured."""
    value: Optional[str] = None
    source: Literal["db", "env", "default"]
    has_value: Optional[bool] = None


class InstanceSettingsResponse(BaseModel):
    """Body of GET /instance/settings (owner-only). Keyed by setting name
    (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, EMBED_MODEL, CLAP_ENABLED,
    AI_ENABLED)."""
    settings: Dict[str, SettingItem]


class InstanceSettingsPatch(BaseModel):
    """Body of PATCH /instance/settings (owner-only). Only fields explicitly
    present are written (handler uses exclude_unset). A field set to null clears
    the override (resolver falls back to env/default). For ``llm_api_key``, send
    the unchanged-sentinel to keep the stored secret untouched."""
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    embed_model: Optional[str] = None
    clap_enabled: Optional[bool] = None
    ai_enabled: Optional[bool] = None


class MemberResponse(BaseModel):
    """One row of GET /admin/members (owner-only). ``invite_code`` is the code a
    member registered with (None for the owner / pre-invite accounts). The stats
    triplet summarizes that account's library (collection ``acct_{id}``): how many
    tracks, total seconds listened, and how many liked tracks."""
    id: str
    email: str
    role: Literal["owner", "member"]
    created_at: float
    last_login_at: Optional[float] = None
    invite_code: Optional[str] = None
    songs: int = 0
    listened_sec: float = 0.0
    likes: int = 0


class DeleteAccountRequest(BaseModel):
    """Body of DELETE /admin/accounts/{user_id}. ``confirm_email`` must match the
    target's email — a defense-in-depth guard so a UI bug can't wipe the wrong
    account on a bare path id."""
    confirm_email: str


# ── Unified AI assistant ──────────────────────────────────────────────────────
# Facade over the three existing AI stacks (lyrics/audio search, playlist
# building, artist/song facts). Intent routing is done by GLiNER2 in
# app/services/assistant/router.py — never by the LLM — so the closed label set
# below is authoritative: the model picks from it, it cannot invent a fourth.

AssistantIntent = Literal["search", "playlist", "facts"]


class AssistantSlots(BaseModel):
    """Conversation state carried by the CLIENT across turns.

    The backend is stateless (every request rebuilds what it needs), so the
    server returns these in each terminal frame and the client echoes them back
    on the next message. Merge rule is unconditional: slots always carry
    forward, freshly extracted entities overwrite. That removes any need to
    decide "is this a follow-up?" — «ещё у этого артиста» simply finds no
    artist entity and falls back to ``last_artist``.
    """
    last_intent: Optional[AssistantIntent] = None
    last_artist: Optional[str] = None
    last_song: Optional[str] = None
    last_track_id: Optional[str] = None
    last_playlist_ids: List[str] = Field(default_factory=list)
    last_filters: Optional[SearchFilters] = None


class AssistantRequest(BaseModel):
    """Body for POST /assistant and /assistant/stream."""
    message: str = Field(min_length=1, max_length=1000)
    history: List[ChatMessage] = Field(default_factory=list)
    slots: AssistantSlots = Field(default_factory=AssistantSlots)

    # Set by the client when the user picked an option from a `clarify` frame:
    # routing is skipped entirely and this intent is used verbatim.
    intent: Optional[AssistantIntent] = None
    # Set when the user picked a card from a `disambiguate` frame.
    subject_track_id: Optional[str] = None
    subject_artist_slug: Optional[str] = None
    # What the player is on right now — lets "расскажи про этот трек" resolve
    # with no entity in the message at all.
    now_playing_track_id: Optional[str] = None

    limit: int = Field(15, ge=1, le=40)
    lang: str = "en"
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None


class AssistantRoute(BaseModel):
    """Outcome of the GLiNER2 routing pass."""
    intent: Optional[AssistantIntent] = None   # None → needs clarification
    confidence: float = 0.0
    margin: float = 0.0                        # top1 - top2, drives clarify
    source: Literal["gliner", "explicit", "sticky", "count_override", "unclear"] = "gliner"
    artist: Optional[str] = None
    song: Optional[str] = None
    count: Optional[int] = None
    era: Optional[str] = None


class AssistantClarifyOption(BaseModel):
    intent: AssistantIntent
    label: str


class AssistantSubjectOption(BaseModel):
    """One candidate subject for the facts intent (a `disambiguate` frame)."""
    kind: Literal["artist", "album", "song"]
    title: str
    subtitle: Optional[str] = None
    track_id: Optional[str] = None
    artist_slug: Optional[str] = None
    cover_art_path: Optional[str] = None


class AssistantFactItem(BaseModel):
    """One numbered item of the grounding pack, echoed to the UI as a chip.

    ``n`` matches the [n] index the LLM cites in ``used`` — the frontend
    highlights exactly the facts the answer leaned on.
    """
    n: int
    text: str
    source: Literal["facts", "bio", "credits", "gems", "lyrics", "catalog", "web"] = "facts"
    used: bool = False


class AssistantFactsPayload(BaseModel):
    """Result payload for intent="facts"."""
    subject_kind: Literal["artist", "album", "song"]
    subject_title: str
    subject_subtitle: Optional[str] = None
    artist_slug: Optional[str] = None
    track_id: Optional[str] = None
    image_path: Optional[str] = None
    answer: str
    # False when the LLM answer failed code verification and the deterministic
    # "here's what is known" fact rendering was served instead.
    grounded: bool = True
    web_search_used: bool = False
    items: List[AssistantFactItem] = Field(default_factory=list)
    related_tracks: List[TrackMetadata] = Field(default_factory=list)


class AssistantResponse(BaseModel):
    """Terminal payload — the non-streaming twin of the final NDJSON frame.

    Exactly one of ``search``/``playlist``/``facts`` is set, matching ``intent``.
    Deliberately NOT a merged union: the three results render as three different
    card types, and flattening them is what makes the UX muddy.
    """
    intent: Optional[AssistantIntent] = None
    human: str = ""
    slots: AssistantSlots = Field(default_factory=AssistantSlots)

    search: Optional[Dict] = None                      # shape of _run_chat_core
    playlist: Optional[AIPlaylistResponse] = None
    facts: Optional[AssistantFactsPayload] = None

    # Set instead of a payload when routing was inconclusive.
    clarify: Optional[List[AssistantClarifyOption]] = None
    # Set instead of a payload when the facts subject was ambiguous.
    disambiguate: Optional[List[AssistantSubjectOption]] = None


class DiscoveryTrackRef(BaseModel):
    """A track shown inside a discovery card (cover + caption, playable)."""
    track_id: str
    title: str
    artist: str
    cover_art_path: Optional[str] = None


class DiscoveryCard(BaseModel):
    """One hook card of the assistant page's rail.

    ``prompt`` is the message the card sends when tapped — the card is a
    pre-written turn, not a link. ``intent`` is passed alongside it so the
    router is skipped: the card already knows which branch answers it.
    """
    kind: Literal["relation", "producer", "artist"]
    intent: AssistantIntent
    prompt: str
    headline: str
    subline: Optional[str] = None
    badge: Optional[str] = None
    items: List[DiscoveryTrackRef] = Field(default_factory=list)
    track_id: Optional[str] = None
    artist_slug: Optional[str] = None
    count: Optional[int] = None


class DiscoveriesResponse(BaseModel):
    cards: List[DiscoveryCard] = Field(default_factory=list)
