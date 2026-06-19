"""DbClient — context manager for Qdrant + LyricsSearchEngine."""

from __future__ import annotations

import logging
import os

from qdrant_client import QdrantClient

from .lyrics_search_engine import LyricsSearchEngine

logger = logging.getLogger(__name__)


class DbClient:
    """
    Context manager providing:
    - qdrant: QdrantClient instance
    - lyrics_db: LyricsSearchEngine instance (models loaded lazily)

    Model loading is deferred to first use (search) or to a background
    preload task started by the FastAPI lifespan.
    """

    def __init__(self,
                 qdrant_url: str | None = None,
                 collection_name: str = "music_explorer",
                 model_name: str | None = None):
        # QDRANT_URL env override lets the app reach Qdrant by its Docker
        # Compose service name (http://qdrant:6333) inside a container, while
        # still defaulting to localhost for bare-metal/Windows runs.
        self.qdrant_url = qdrant_url or os.environ.get(
            "QDRANT_URL", "http://localhost:6333"
        )
        self.collection_name = collection_name
        self.model_name = model_name or os.environ.get(
            "TEXT_MODEL",
            "jinaai/jina-embeddings-v2-small-en",
        )

        self._qdrant_client: QdrantClient | None = None
        self._lyrics_db: LyricsSearchEngine | None = None

    def __enter__(self) -> "DbClient":
        return self._connect()

    def _connect(self) -> "DbClient":
        # Create Qdrant client (fast — just TCP connect).
        # HTTP_PROXY is cleared in docker-compose environment so QdrantClient
        # never routes through the proxy — internal Docker traffic stays direct.
        self._qdrant_client = QdrantClient(url=self.qdrant_url)

        # Create LyricsSearchEngine with lazy model loading (default)
        # Models are NOT loaded here — they load on first search access
        self._lyrics_db = LyricsSearchEngine(
            qdrant_client=self._qdrant_client,
            collection_name=self.collection_name,
            model_name=self.model_name,
            include_clap=True,
            lazy=True,  # defer model loading
        )
        logger.info("[DbClient] Connected to Qdrant, models will load lazily")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._disconnect()

    def _disconnect(self):
        if self._qdrant_client:
            try:
                self._qdrant_client.close()
            except Exception:
                pass

    # Async context manager support for FastAPI lifespan
    async def __aenter__(self) -> "DbClient":
        return self._connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._disconnect()

    @property
    def qdrant(self) -> QdrantClient:
        if self._qdrant_client is None:
            raise RuntimeError("DbClient not entered. Use 'with DbClient() as db: ...'")
        return self._qdrant_client

    @property
    def lyrics_db(self) -> LyricsSearchEngine:
        if self._lyrics_db is None:
            raise RuntimeError("DbClient not entered.")
        return self._lyrics_db

    @property
    def search_engine(self) -> LyricsSearchEngine:
        """Alias for ``lyrics_db`` — both now return a ``LyricsSearchEngine``."""
        return self.lyrics_db
