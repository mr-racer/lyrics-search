"""API routes."""

from .search import router as search_router
from .search import stream_router as stream_router
from .library import router as library_router
from .chat import router as chat_router
from .assistant import router as assistant_router
from .metadata import router as metadata_router
from .playback import router as playback_router
from .recommend import router as recommend_router
from .ai_indexing import router as ai_indexing_router
from .artists import router as artists_router
from .system import router as system_router
from .playlists import router as playlists_router
from .instance import router as instance_router
from .auth import router as auth_router
from .admin import router as admin_router
from .imports import router as imports_router
from .quiz import router as quiz_router
from .quiz import stream_router as quiz_stream_router
from .models_public import router as models_public_router
from .models_public import install_model_error_handler

__all__ = ["search_router", "stream_router", "library_router", "chat_router", "assistant_router", "metadata_router",
           "playback_router", "recommend_router", "ai_indexing_router",
           "artists_router", "system_router", "playlists_router",
           "instance_router", "auth_router", "admin_router", "imports_router",
           "quiz_router", "quiz_stream_router",
           "models_public_router", "install_model_error_handler"]
