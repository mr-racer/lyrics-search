"""API routes."""

from .search import router as search_router
from .library import router as library_router
from .chat import router as chat_router
from .metadata import router as metadata_router
from .playback import router as playback_router
from .recommend import router as recommend_router
from .ai_indexing import router as ai_indexing_router
from .artists import router as artists_router
from .system import router as system_router
from .playlists import router as playlists_router
from .instance import router as instance_router

__all__ = ["search_router", "library_router", "chat_router", "metadata_router",
           "playback_router", "recommend_router", "ai_indexing_router",
           "artists_router", "system_router", "playlists_router",
           "instance_router"]
