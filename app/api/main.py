"""
Main FastAPI application.

- Lifespan: setup DbClient + Services on startup (models load in background)
- Include search, library and chat routers
- CORS middleware
- Static files (frontend)
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response

# Silence overly verbose third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

from ..resources.db_client import DbClient
from ..resources.model_registry import ModelRegistry
from ..services.search_service import SearchService
from ..services.library_service import LibraryService
from ..services.job_tracker import JobTracker
from .routes import search_router, library_router, chat_router, metadata_router
from .sse_utils import event_stream

logger = logging.getLogger(__name__)
FRONTEND_INDEX = Path(__file__).parent.parent.parent / "frontend" / "index.html"
COVERS_DIR = Path(__file__).parent.parent.parent / "frontend" / "covers"


async def _preload_models_in_background(db_client: DbClient):
    """Background task: find the largest collection and preload its text model + CLAP.

    This runs after the server is ready, so the user can start browsing immediately.
    Models are loaded into ModelRegistry cache; LyricsDB picks them up lazily.
    """
    try:
        await asyncio.sleep(1)  # give the event loop a moment

        # ── Step 1: Find the largest collection ──
        largest_col = None
        largest_count = 0
        try:
            cols = db_client.qdrant.get_collections().collections
            for col in cols:
                try:
                    info = db_client.qdrant.get_collection(col.name)
                    count = info.points_count or 0
                    if count > largest_count:
                        largest_count = count
                        largest_col = col.name
                except Exception:
                    pass
        except Exception as e:
            logger.warning("[preload] Could not query collections: %s", e)

        logger.info("[preload] Largest collection: %s (%d points)",
                    largest_col, largest_count)

        # Text models are loaded lazily on first use (per-collection, different collections
        # can use different embedding models). No background preload needed here.

        # ── Step 2: Load CLAP ──
        try:
            ModelRegistry.load_clap()
            logger.info("[preload] CLAP model loaded")
        except Exception as e:
            logger.warning("[preload] CLAP load failed: %s", e)

    except Exception as e:
        logger.error("[preload] Unexpected error during model preload: %s", e, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Setup on startup, cleanup on shutdown.

    Gracefully handles Qdrant being unavailable at startup — the app still
    starts so the frontend can show the onboarding screen and instruct the
    user to start Qdrant.

    Models are loaded lazily (on first use) AND preloaded in the background
    so the server starts fast but models are ready by the time the user
    searches or indexes.
    """
    db: DbClient | None = None

    try:
        db = DbClient()
        db._connect()

        app.state.db_client = db
        app.state.search_service = SearchService(db.lyrics_db)
        app.state.library_service = LibraryService(
            search_service=app.state.search_service,
            db_client=db,
        )
        app.state.job_tracker = JobTracker()

        logger.info("[OK] Qdrant connected, services ready — models will preload in background")

        # Initialise SQLite metadata store and migrate legacy .txt cache
        try:
            from ..resources.metadata_db import MetadataDB
            from ..services.migrate_facts import migrate_all
            MetadataDB.init()
            summary = migrate_all()
            MetadataDB.close()
            if summary["artists_migrated"] or summary["songs_migrated"]:
                logger.info(
                    "[OK] Facts migrated to SQLite: %d artists, %d songs",
                    summary["artists_migrated"],
                    summary["songs_migrated"],
                )
        except Exception as e:
            logger.warning("[WARN] Facts migration skipped: %s", e)

        # Start background model preload (non-blocking)
        asyncio.create_task(_preload_models_in_background(db))

    except Exception as e:
        # Qdrant is down or model failed to load — start anyway with limited mode
        logger.warning("[WARN] Startup warning: %s", e)
        logger.warning("   App is running in limited mode (Qdrant unavailable).")
        app.state.db_client = None
        app.state.search_service = None
        app.state.library_service = None
        app.state.job_tracker = JobTracker()

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

    # SSE endpoint for indexing progress
    @app.get("/api/v1/index/progress/{job_id}", tags=["Index"])
    async def get_index_progress(job_id: str):
        """Stream indexing progress via SSE."""
        job_tracker = app.state.job_tracker
        return StreamingResponse(
            event_stream(job_id, job_tracker),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # Cover art endpoint — serve extracted album covers with long cache
    @app.get("/api/v1/covers/{cover_file}", tags=["Covers"])
    async def serve_cover(cover_file: str):
        """Serve an extracted album cover image."""
        cover_path = COVERS_DIR / cover_file
        if not cover_path.exists() or not cover_path.is_file():
            raise HTTPException(status_code=404, detail="Cover not found")

        ext = cover_path.suffix.lower()
        content_type = "image/jpeg" if ext == ".jpg" else "image/png"

        return Response(
            cover_path.read_bytes(),
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",  # 1 year
            },
        )

    # Routers — MUST be registered BEFORE the SPA catch-all so Starlette
    # matches /api/v1/... routes first (routes are evaluated in order).
    app.include_router(search_router,  prefix="/api/v1")
    app.include_router(library_router, prefix="/api/v1")
    app.include_router(chat_router,    prefix="/api/v1")
    app.include_router(metadata_router, prefix="/api/v1")

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
