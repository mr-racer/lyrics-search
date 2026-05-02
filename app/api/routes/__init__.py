"""API routes."""

from .search import router as search_router
from .library import router as library_router
from .chat import router as chat_router

__all__ = ["search_router", "library_router", "chat_router"]
