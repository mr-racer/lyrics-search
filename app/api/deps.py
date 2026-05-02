"""
App lifecycle and dependencies.

- lifespan(context): setup DbClient, Services
- get_db(): yield DbClient
- get_search_service(): yield SearchService
- get_library_service(): LibraryService()
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Depends

from ..resources.db_client import DbClient
from ..services.search_service import SearchService
from ..services.library_service import LibraryService

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Setup on startup, cleanup on shutdown."""
    # Startup
    async with DbClient() as db:
        app.state.db_client = db
        app.state.search_service = SearchService(db.lyrics_db)
        app.state.library_service = LibraryService()
        
        yield  # application runs here
    
    # Shutdown — DbClient.__exit__ handles cleanup

def get_db() -> AsyncGenerator[DbClient, None]:
    """Dependency: yield DbClient from app.state."""
    # This needs FastAPI app instance — will be fixed in router
    pass  

def get_search_service():
    """Dependency: SearchService from app.state."""
    pass  

def get_library_service():
    """Dependency: LibraryService (stateless)."""
    return LibraryService()
