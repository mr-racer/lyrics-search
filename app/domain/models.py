"""Domain models for Music Explorer."""

from typing import Literal, List, Optional, Annotated, Dict
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Fact(BaseModel):
    """A single fact about an artist or song."""
    fact: str
    lang: str = "en"
    category: str | None = None
    source: str | None = None


class TrackMetadata(BaseModel):
    """Метаданные трека."""
    track_id: str  # хэш file_path или UUID, стабильный между рестартами
    title: str
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
    collection_name: Optional[str] = Field(None, description="Qdrant collection to search in")


class SearchResponse(BaseModel):
    """Ответ на поисковый запрос."""
    hits: List[TrackHit]
    query: str
    mode: Literal["text", "audio", "hybrid"]


class IndexRequest(BaseModel):
    """Запрос на индексацию папки с музыкой."""
    folder_path: str
    collection_name: str = "music_explorer"
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
    collection_name: Optional[str] = Field(None, description="Qdrant collection to search in")


class ChatResponse(BaseModel):
    """Ответ от LLM-чата."""
    query: str
    mode: Literal["text", "audio", "hybrid"]
    hits: List[TrackHit]
    llm_response: str


class TrackReactionRequest(BaseModel):
    """Запрос на установку реакции на трек."""
    collection_name: str
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
    collection_name: Optional[str] = None


class TrackChatResponse(BaseModel):
    """Response body for POST /chat/track-chat."""
    message: str
    web_search_used: bool = False

# LLM MODELS
# TODO поменял QueryItem на BaseQueryItem, проверь что в нужных местах поменялось.

class BaseQueryItem(BaseModel):
    query: str

class RephrasedQuery(BaseModel):
    new_queries: list[str]

class SearchAction(BaseModel):
    action: Literal["search"]
    confidence: Literal["low", "medium", "high"]
    queries: list[BaseQueryItem]

class AnswerAction(BaseModel):
    action: Literal["answer"]
    confidence: Literal["high", "medium", "low"]
    song: str | None
    artist: str | None
    message: str


# ── PydanticAI Agent Models ────────────────────────────────────────────────────

class QueryType(BaseModel):
    """Тип запроса после классификации."""
    type: Literal["text", "audio", "hybrid"]
    reasoning: str = Field(description="Краткое объяснение, почему этот тип")


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


class AudioAnswer(BaseModel):
    """Ответ от AudioAgent."""
    message: str
    best_hit: dict | None = None
    hits: List[dict] = Field(default_factory=list)


class ValidatorResult(BaseModel):
    """Решение ValidatorAgent: принять ответ или продолжить поиск."""
    valid: bool
    reason: str
    queries: List[BaseQueryItem] | None = None  # новые запросы если valid=False


class PlannerOutput(BaseModel):
    action: Literal["request filter", "search"]
    filters: SearchFilters | None
    filter_lookup: SearchFilters | None
    # queries: 

# {
#   "action": "request_filter" | "search",
#   "filters": {
#     "Artist": "..." | null,
#     "Album": "..." | null,
#     "Genre": "..." | null,
#     "year_range": "YYYY-YYYY" | null
#   } | null,
#   "filter_lookup": {
#     "Artist": "raw user input to resolve" | null,
#     "Album": "..." | null,
#     "Genre": "..." | null
#   } | null,
#   "queries": [{"query": "..."}],
#   "search_mode": "CONSERVATIVE" | "AGGRESSIVE"
# }

LLMResponse = Annotated[SearchAction | AnswerAction, Field(discriminator="action")]


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


class ClassifierStatus(BaseModel):
    """Readiness state of the custom sonic-class MLP classifier."""
    status: Literal["untrained", "training", "ready", "failed"]
    trained_at: float | None = None
    accuracy: float | None = None
    classes: list[str] = []


class ClusterRepresentative(BaseModel):
    """One cluster's id, size, and top-N representative tracks (closest to centroid)."""
    cluster_id: int
    size: int
    representative_tracks: list[dict]  # [{track_id, title, artist, cover_art_path}, ...]
    current_label: str | None = None


class ClusterLabelsRequest(BaseModel):
    """Body for POST /library/clusters/labels — user-assigned cluster names."""
    collection: str
    labels: dict[int, str]  # {0: "Lo-fi indie", 1: "Cinematic drone", ...}


class PlaybackEventIn(BaseModel):
    """Request body for POST /playback/events."""
    session_id: str
    collection_name: str
    track_id: str
    played_sec: float
    total_dur: float | None = None


class PlaybackEventOut(BaseModel):
    """Successful POST response."""
    id: int


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

class ArtistRef(BaseModel):
    name: str
    slug: str


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


class LibraryAlbumsResponse(BaseModel):
    albums: list[AlbumSummary]
    collection_name: Optional[str] = None
    qdrant_available: bool = True


class LikedSongTrack(BaseModel):
    track_id: str
    title: str
    artist: str
    album: Optional[str] = None
    year: Optional[int] = None
    duration: Optional[float] = None
    cover_art_path: Optional[str] = None
    genre: Optional[str] = None
    liked_at: str   # ISO datetime


class LikedSongsResponse(BaseModel):
    tracks: list[LikedSongTrack]
    collection_name: Optional[str] = None


class RecentTrack(BaseModel):
    track_id: str
    title: str
    artist: str
    album: Optional[str] = None
    year: Optional[int] = None
    duration: Optional[float] = None
    cover_art_path: Optional[str] = None
    genre: Optional[str] = None
    last_played: str   # ISO datetime
    play_count: int


class RecentTracksResponse(BaseModel):
    tracks: list[RecentTrack]
    collection_name: Optional[str] = None


class TopTrackBrief(BaseModel):
    track_id: str
    title: str
    artist: str
    play_count: int


class TopArtistBrief(BaseModel):
    name: str
    slug: str
    play_count: int


class PeakHour(BaseModel):
    hour: int   # 0-23
    label: str  # localised "вечера буднего" / "weekday evenings"


class ListeningStatsResponse(BaseModel):
    total_seconds_listened: int = 0
    since: Optional[str] = None
    top_track: Optional[TopTrackBrief] = None
    top_artist: Optional[TopArtistBrief] = None
    peak_hour: Optional[PeakHour] = None
