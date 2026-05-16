"""Domain models for Music Explorer."""

from typing import Literal, List, Optional, Annotated, Dict
from pydantic import BaseModel, Field


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
    artist: str | None = None
    album: str | None = None
    genre: str | None = None
    year_range: str | None = None
    
    # TODO - проверить что пункты ниже нигде не применяются и удалить их
    year_from: int | None = None
    year_to: int | None = None
    duration_min_sec: float | None = None
    duration_max_sec: float | None = None


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

# LLM MODELS
# TODO поменял QueryItem на BaseQueryItem, проверь что в нужных местах поменялось.

class BaseQueryItem(BaseModel):
    query: str
    type: Literal["text", "audio", "hybrid"] = "hybrid"

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
