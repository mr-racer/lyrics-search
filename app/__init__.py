"""
Music Explorer — semantic music search platform.

Architecture:
- domain/: Pydantic models (TrackMetadata, TrackHit, SearchRequest...)
- resources/: ModelRegistry + DbClient
- services/: LibraryService + SearchService
- api/: FastAPI app with lifespan and routes
- existing/: Wrappers for user's legacy code
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

# Legacy wrappers (optional)
from .existing import FileProcessor, LyricsDB

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

    # Legacy wrappers
    "FileProcessor",
    "LyricsDB",
]
