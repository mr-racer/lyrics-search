"""
Music Explorer — semantic music search platform.

Architecture:
- domain/: Pydantic models (TrackMetadata, TrackHit, SearchRequest...)
- resources/: ModelRegistry + DbClient + LyricsSearchEngine
- indexing/: folder scanning, metadata reading, lyrics fetching, cover art, audio
- services/: LibraryService + SearchService + IndexingService
- api/: FastAPI app with lifespan and routes
"""

__version__ = "0.1.0"

# Public API exports by layer
from .domain import (
    TrackMetadata,
    TrackHit,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    IndexRequest,
    IndexProgress,
    ChatMessage,
    ChatRequest,
    ChatResponse,
)

from .resources import ModelRegistry, DbClient

from .services import LibraryService, SearchService

from .api import app, create_app


__all__ = [
    # Domain models
    "TrackMetadata",
    "TrackHit",
    "SearchFilters",
    "SearchRequest",
    "SearchResponse",
    "IndexRequest",
    "IndexProgress",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",

    # Resources
    "ModelRegistry",
    "DbClient",

    # Services
    "LibraryService",
    "SearchService",

    # API
    "app",
    "create_app",
]
