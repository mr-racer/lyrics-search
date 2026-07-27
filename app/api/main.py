"""
Main FastAPI application.

- Lifespan: setup DbClient + Services on startup (models load in background)
- Include search, library and chat routers
- CORS middleware
- Static files (frontend)
"""

import asyncio
import logging
import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

# Silence overly verbose third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

from ..resources.db_client import DbClient
from ..resources.model_registry import ModelRegistry
from ..services.search_service import SearchService
from ..services.library_service import LibraryService
from ..services.job_tracker import JobTracker
from ..services.sonic_descriptor_service import SonicDescriptorService
from ..services.auth_service import AuthService
# Side-effect import: each ai_tasks module calls register_task() at import
# time, populating the AI Indexing service registry. Without this, every
# POST /library/ai-index/{task_type} bails with HTTP 400 "unknown task_type"
# because the routes find the type in _TASK_TYPES but the service registry
# is empty.
from ..services import ai_tasks  # noqa: F401
from .routes import search_router, stream_router, library_router, chat_router, assistant_router, metadata_router, playback_router, recommend_router, ai_indexing_router, artists_router, system_router, playlists_router, instance_router, auth_router, admin_router, imports_router
from .dependencies import get_current_user
from .sse_utils import event_stream

logger = logging.getLogger(__name__)
# Tier 0: the frontend is now a Vite build. We serve the compiled assets from
# frontend/dist/ (produced by `npm run build`), not the source tree. covers/
# stays at frontend/covers (a runtime volume) and is served by its own routes.
# PWA: Windows не знает .webmanifest — без этого FileResponse отдал бы манифест
# как application/octet-stream и Chrome не предложил бы установку.
mimetypes.add_type("application/manifest+json", ".webmanifest")

FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
COVERS_DIR = Path(__file__).parent.parent.parent / "frontend" / "covers"

# Downscaled cover variants (?w=): mobile grids render covers at ~150px but
# used to download the full embedded art (often 1000px+/400KB). Generated
# lazily with Pillow, cached on disk next to the other runtime caches.
COVER_THUMBS_DIR = Path(__file__).parent.parent.parent / "cache" / "cover_thumbs"
COVER_THUMB_WIDTHS = {320}
COVER_HEADERS = {
    "Cache-Control": "public, max-age=31536000, immutable",  # 1 year
    # Unconditional, NOT left to CORSMiddleware: the middleware skips requests
    # without an Origin header (plain <img> / background-image), and that
    # ACAO-less response gets cached immutable for a year under the same key a
    # later crossOrigin='anonymous' canvas read will hit. The cached copy must
    # already be CORS-readable or useCoverColor falls back to the purple default.
    "Access-Control-Allow-Origin": "*",
}


def _make_cover_thumb(src: Path, dst: Path, width: int) -> bool:
    """Downscale a cover to ``width`` px (blocking — run in an executor).

    Keeps the source format: JPEG stays JPEG, PNG stays PNG so artist cutouts
    keep their alpha channel (the Atlas hero canvas-probes transparency).
    Writes tmp-then-rename so concurrent requests never see a partial file.
    """
    import uuid
    tmp = dst.with_suffix(dst.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
    try:
        from PIL import Image
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im.thumbnail((width, width))
            if dst.suffix.lower() in (".jpg", ".jpeg"):
                im.convert("RGB").save(tmp, "JPEG", quality=82, optimize=True)
            else:
                im.save(tmp, "PNG", optimize=True)
        tmp.replace(dst)
        return True
    except Exception:
        logger.warning("[covers] thumbnail generation failed for %s", src, exc_info=True)
        tmp.unlink(missing_ok=True)
        return False


async def _preload_models_in_background(db_client: DbClient):
    """Background task: find the largest collection and preload its text model + CLAP.

    This runs after the server is ready, so the user can start browsing immediately.
    Models are loaded into ModelRegistry cache; LyricsDB picks them up lazily.
    """
    try:
        await asyncio.sleep(1)  # give the event loop a moment

        # ── Step 1: Find the largest collection ──
        # The sync qdrant-client blocks — run the whole sweep in a thread so
        # early requests aren't stalled while we probe collections.
        def _find_largest() -> tuple[str | None, int]:
            largest_col, largest_count = None, 0
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
            return largest_col, largest_count

        largest_col, largest_count = await asyncio.to_thread(_find_largest)

        logger.info("[preload] Largest collection: %s (%d points)",
                    largest_col, largest_count)

        # Text models are loaded lazily on first use (per-collection, different collections
        # can use different embedding models). No background preload needed here.

        # ── Step 2: Load CLAP ──
        # load_clap() is heavily blocking (checkpoint load, possibly a ~2.3 GB
        # download). Calling it inline would freeze the event loop — and this
        # coroutine runs exactly when the user starts browsing — so it goes to
        # a worker thread; ModelRegistry._clap_lock keeps the load single.
        try:
            await asyncio.to_thread(ModelRegistry.load_clap)
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

    # Phase A: build AuthService from MUSIX_JWT_SECRET env. Use a developer
    # default in DEBUG dev runs so the repo is runnable out of the box, but
    # log a loud warning so prod ops don't accidentally ship it.
    jwt_secret = os.environ.get("MUSIX_JWT_SECRET", "")
    if not jwt_secret:
        jwt_secret = "DEV-ONLY-JWT-SECRET-set-MUSIX_JWT_SECRET-in-production-32+chars"
        logger.warning(
            "[AUTH] MUSIX_JWT_SECRET not set — using dev fallback. "
            "DO NOT DEPLOY this way; set the env var to a 32+ char random string."
        )
    app.state.auth_service = AuthService(jwt_secret=jwt_secret)

    try:
        db = DbClient()
        db._connect()

        app.state.db_client = db
        app.state.search_service = SearchService(db.lyrics_db)
        app.state.sonic_descriptor_service = SonicDescriptorService()
        app.state.library_service = LibraryService(
            search_service=app.state.search_service,
            db_client=db,
            sonic_descriptor_service=app.state.sonic_descriptor_service,
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

        # Start background model preload (non-blocking). Keep a reference on
        # app.state — the event loop holds tasks only weakly, so an anonymous
        # create_task() can be garbage-collected mid-preload.
        app.state._preload_task = asyncio.create_task(_preload_models_in_background(db))

    except Exception as e:
        # Qdrant is down or model failed to load — start anyway with limited mode
        logger.warning("[WARN] Startup warning: %s", e)
        logger.warning("   App is running in limited mode (Qdrant unavailable).")
        app.state.db_client = None
        app.state.search_service = None
        app.state.sonic_descriptor_service = None
        app.state.library_service = None
        app.state.job_tracker = JobTracker()

    # Phase C: background cleanup — sweep stale quarantine *.tmp + purge old
    # completed pending_uploads. Cheap, hourly, no Qdrant — so it runs even in
    # limited mode.
    async def _cleanup_loop():
        from app.services.uploads_service import sweep_old_quarantine
        from app.resources.metadata_db import MetadataDB as _MDB
        while True:
            try:
                await asyncio.sleep(3600)  # 1h
                nq = await asyncio.to_thread(sweep_old_quarantine)
                np_ = await asyncio.to_thread(_MDB.purge_old_pending_uploads)
                if nq or np_:
                    logger.info(
                        "[cleanup] swept %d quarantine, purged %d done uploads", nq, np_,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[cleanup] iteration failed: %s", e)

    app.state._cleanup_task = asyncio.create_task(_cleanup_loop())

    yield  # ← app serves requests here

    # Shutdown — cancel the background cleanup loop cleanly.
    _ct = getattr(app.state, "_cleanup_task", None)
    if _ct is not None and not _ct.done():
        _ct.cancel()
        try:
            await _ct
        except (asyncio.CancelledError, Exception):
            pass

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
    # :path converter so nested covers match too — artist avatars live under
    # covers/artists/<hash>.<ext>. With the default {cover_file} (no slashes)
    # those requests fell through to the SPA catch-all, were served index.html
    # as text/html, and the browser blocked them (ERR_BLOCKED_BY_ORB).
    @app.get("/api/v1/covers/{cover_file:path}", tags=["Covers"])
    async def serve_cover(cover_file: str, w: int | None = None):
        """Serve an extracted album or artist cover image.

        ``?w=320`` returns a lazily-generated downscaled variant (mobile grids).
        FileResponse (not an in-memory read): covers used to be read whole into
        RAM synchronously on the event loop, so a grid of covers loading during
        playback contended with audio byte delivery.
        """
        # :path matches slashes, so guard against escaping COVERS_DIR.
        covers_root = COVERS_DIR.resolve()
        cover_path = (covers_root / cover_file).resolve()
        if not cover_path.is_relative_to(covers_root):
            raise HTTPException(status_code=404, detail="Cover not found")
        if not cover_path.exists() or not cover_path.is_file():
            raise HTTPException(status_code=404, detail="Cover not found")

        ext = cover_path.suffix.lower()
        content_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

        if w in COVER_THUMB_WIDTHS:
            thumbs_root = COVER_THUMBS_DIR.resolve()
            thumb_path = (thumbs_root / str(w) / cover_file).resolve()
            if thumb_path.is_relative_to(thumbs_root):
                if not thumb_path.exists():
                    await asyncio.get_running_loop().run_in_executor(
                        None, _make_cover_thumb, cover_path, thumb_path, w,
                    )
                if thumb_path.exists():
                    return FileResponse(thumb_path, media_type=content_type, headers=COVER_HEADERS)
            # Thumbnail failed — fall through to the full-size original.

        return FileResponse(cover_path, media_type=content_type, headers=COVER_HEADERS)

    # Artist images (AudioDB cutouts / Deezer thumbs) live one directory
    # deeper: /covers/artists/<hash>.<ext>. serve_cover's single-segment path
    # param can't match the extra segment, so these URLs used to fall through
    # to the SPA catch-all and come back as index.html — a permanently broken
    # <img> in the Atlas hero.
    @app.get("/api/v1/covers/artists/{cover_file}", tags=["Covers"])
    async def serve_artist_cover(cover_file: str):
        """Serve a cached artist image with the same cache/ACAO contract as covers."""
        # Windows quirk: a URL-encoded backslash (%5C) survives into the path
        # param and Path() treats it as a separator — refuse anything that
        # isn't a plain filename.
        if Path(cover_file).name != cover_file:
            raise HTTPException(status_code=404, detail="Cover not found")
        cover_path = COVERS_DIR / "artists" / cover_file
        if not cover_path.exists() or not cover_path.is_file():
            raise HTTPException(status_code=404, detail="Cover not found")

        ext = cover_path.suffix.lower()
        content_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

        return FileResponse(cover_path, media_type=content_type, headers=COVER_HEADERS)

    # Routers — MUST be registered BEFORE the SPA catch-all so Starlette
    # matches /api/v1/... routes first (routes are evaluated in order).
    # Phase A: dependencies=[Depends(get_current_user)] gates every existing
    # surface on a valid JWT. The auth + instance routers stay open so the
    # frontend can log in / probe the mode before it holds a token.
    auth_gate = [Depends(get_current_user)]
    app.include_router(search_router,       prefix="/api/v1", dependencies=auth_gate)
    # Stream router carries its OWN auth (get_user_for_stream: Bearer or ?st=
    # stream token) because <audio> elements can't send Authorization headers.
    app.include_router(stream_router,       prefix="/api/v1")
    app.include_router(library_router,      prefix="/api/v1", dependencies=auth_gate)
    app.include_router(chat_router,         prefix="/api/v1", dependencies=auth_gate)
    app.include_router(assistant_router,    prefix="/api/v1", dependencies=auth_gate)
    app.include_router(metadata_router,     prefix="/api/v1", dependencies=auth_gate)
    app.include_router(playback_router,     prefix="/api/v1", dependencies=auth_gate)
    app.include_router(recommend_router,    prefix="/api/v1", dependencies=auth_gate)
    app.include_router(ai_indexing_router,  prefix="/api/v1", dependencies=auth_gate)
    app.include_router(artists_router,      prefix="/api/v1", dependencies=auth_gate)
    app.include_router(system_router,       prefix="/api/v1", dependencies=auth_gate)
    app.include_router(playlists_router,    prefix="/api/v1", dependencies=auth_gate)
    # Yandex import — auth-gated; each route also carries require_mode("server").
    app.include_router(imports_router,       prefix="/api/v1", dependencies=auth_gate)
    # Admin routes carry their own stricter gate (get_owner = JWT + role=owner),
    # so they don't need the blanket get_current_user dependency.
    app.include_router(admin_router,        prefix="/api/v1")
    # Public routes — NO auth gate (login / mode probe happen pre-token).
    app.include_router(instance_router,     prefix="/api/v1")
    app.include_router(auth_router,         prefix="/api/v1")

    # Machine-readable service info. Root `/` serves the SPA (via the catch-all
    # below), so expose the JSON status payload here for health-probing monitors.
    # MUST be registered before the catch-all so it isn't shadowed.
    @app.get("/api", tags=["System"])
    async def api_info():
        return {"name": "Music Explorer", "version": "0.1.0", "status": "running", "docs": "/docs"}

    # SPA catch-all — must be LAST so it doesn't shadow API routes
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Try exact static file first (hashed JS/CSS assets from the Vite build).
        # SECURITY: guard against path traversal (e.g. `/../../.env`, `/..%2f.env`)
        # — resolve the candidate and only serve it if it stays inside the build
        # dir. Without this, the catch-all is an arbitrary-file-read primitive.
        dist_root = FRONTEND_DIST.resolve()
        try:
            file_path = (dist_root / full_path).resolve()
        except (OSError, ValueError, RuntimeError):
            file_path = None
        if (
            file_path is not None
            and file_path.is_relative_to(dist_root)
            and file_path.is_file()
        ):
            return FileResponse(file_path)
        # SPA fallback → index.html
        if FRONTEND_INDEX.exists():
            return FileResponse(FRONTEND_INDEX)
        raise HTTPException(status_code=404, detail="Frontend not found")

    return app


app = create_app()
