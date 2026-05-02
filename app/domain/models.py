"""Domain models for Music Explorer."""

from typing import Literal, List, Optional
from pydantic import BaseModel, Field


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


class TrackHit(BaseModel):
    """Результат поиска с трек-метаданными, score и matched_on."""
    track: TrackMetadata
    score: float
    matched_on: Literal["lyrics", "audio", "hybrid"] = "lyrics"
    lyrics: str | None = None  # выдержка из лирики для lyrics-поиска


class SearchFilters(BaseModel):
    """Фильтры для поиска."""
    artist: str | None = None
    album: str | None = None
    genre: str | None = None
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


class SearchResponse(BaseModel):
    """Ответ на поисковый запрос."""
    hits: List[TrackHit]
    query: str
    mode: Literal["text", "audio", "hybrid"]


class IndexRequest(BaseModel):
    """Запрос на индексацию папки с музыкой."""
    folder_path: str
    better_lyrics_quality: bool = False
    text_model: Optional[str] = None


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
    # LLM connection — overrides env vars LLM_BASE_URL / LLM_MODEL if set
    llm_base_url: Optional[str] = Field(None, description="e.g. http://localhost:8000/v1")
    llm_model: Optional[str] = Field(None, description="e.g. openai/gpt-oss-20b")


class ChatResponse(BaseModel):
    """Ответ от LLM-чата."""
    query: str
    mode: Literal["text", "audio", "hybrid"]
    hits: List[TrackHit]
    llm_response: str

