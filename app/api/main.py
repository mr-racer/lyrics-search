"""
Main FastAPI application.

- Lifespan: setup DbClient + Services on startup (including CLAP)
- Include search, library and chat routers
- CORS middleware
- Static files (frontend)
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..resources.db_client import DbClient
from ..resources.model_registry import ModelRegistry
from ..services.search_service import SearchService
from ..services.library_service import LibraryService
from .routes import search_router, library_router, chat_router

FRONTEND_INDEX = Path(__file__).parent.parent.parent / "frontend" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Setup on startup, cleanup on shutdown.

    Gracefully handles Qdrant being unavailable at startup — the app still
    starts so the frontend can show the onboarding screen and instruct the
    user to start Qdrant.
    """
    db: DbClient | None = None

    try:
        db = DbClient()
        db._connect()

        app.state.db_client = db
        app.state.search_service = SearchService(db.lyrics_db)
        app.state.library_service = LibraryService(search_service=app.state.search_service)

        # CLAP is optional — warn if unavailable, don't crash
        try:
            ModelRegistry.load_clap()
            print("✓ CLAP model loaded")
        except Exception as e:
            print(f"⚠  CLAP model not loaded (audio search unavailable): {e}")

        print("✓ Qdrant connected, services ready")

    except Exception as e:
        # Qdrant is down or model failed to load — start anyway with limited mode
        print(f"⚠  Startup warning: {e}")
        print("   App is running in limited mode (Qdrant unavailable).")
        app.state.db_client = None
        app.state.search_service = None
        app.state.library_service = None

    yield  # ← app serves requests here

    # Shutdown
    if db is not None:
        try:
            db._disconnect()
        except Exception:
            pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="Music Explorer",
        description="Semantic music search platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow everything for local development.
    # NOTE: allow_credentials=True is incompatible with allow_origins=["*"]
    # (invalid per the CORS spec; newer Starlette raises ValueError and the
    # middleware silently stops handling OPTIONS preflights → 404 on preflight).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.get("/", tags=["Health"])
    async def root():
        return {"name": "Music Explorer", "version": "0.1.0", "status": "running", "docs": "/docs"}

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status": "healthy",
            "qdrant": app.state.db_client is not None,
        }

    # Routers — MUST be registered BEFORE the SPA catch-all so Starlette
    # matches /api/v1/... routes first (routes are evaluated in order).
    app.include_router(search_router,  prefix="/api/v1")
    app.include_router(library_router, prefix="/api/v1")
    app.include_router(chat_router,    prefix="/api/v1")

    # SPA catch-all — must be LAST so it doesn't shadow API routes
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Try exact static file first
        file_path = Path(__file__).parent.parent.parent / "frontend" / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # SPA fallback → index.html
        if FRONTEND_INDEX.exists():
            return FileResponse(FRONTEND_INDEX)
        raise HTTPException(status_code=404, detail="Frontend not found")

    return app


app = create_app()
