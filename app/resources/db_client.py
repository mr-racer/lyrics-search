"""DbClient — context manager for Qdrant + LyricsDB."""

import logging

from qdrant_client import QdrantClient
from ..existing.qdrant_db import LyricsDB
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class DbClient:
    """
    Context manager providing:
    - qdrant: QdrantClient instance
    - lyrics_db: LyricsDB instance (models loaded lazily)

    Model loading is deferred to first use (search/fit) or to a background
    preload task started by the FastAPI lifespan.
    """

    def __init__(self,
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "music_explorer",
                 model_name: str = "jinaai/jina-embeddings-v2-small-en"):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.model_name = model_name

        self._qdrant_client: QdrantClient | None = None
        self._lyrics_db: LyricsDB | None = None

    def __enter__(self) -> "DbClient":
        return self._connect()

    def _connect(self) -> "DbClient":
        # Create Qdrant client (fast — just TCP connect)
        self._qdrant_client = QdrantClient(url=self.qdrant_url)

        # Create LyricsDB with lazy model loading (default)
        # Models are NOT loaded here — they load on first search/fit access
        self._lyrics_db = LyricsDB(
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
    def lyrics_db(self) -> LyricsDB:
        if self._lyrics_db is None:
            raise RuntimeError("DbClient not entered.")
        return self._lyrics_db
