"""DbClient — context manager for Qdrant + LyricsDB."""

from qdrant_client import QdrantClient
from ..existing.qdrant_db import LyricsDB
from .model_registry import ModelRegistry


class DbClient:
    """
    Context manager providing:
    - qdrant: QdrantClient instance
    - lyrics_db: LyricsDB instance (using models from ModelRegistry)
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
        # Create Qdrant client
        self._qdrant_client = QdrantClient(url=self.qdrant_url)

        # Load text model via ModelRegistry
        model, vector_name, dim = ModelRegistry.load_text_model(self.model_name)
        
        # Create LyricsDB with the loaded model
        # Note: need to adapt LyricsDB to accept pre-loaded model
        self._lyrics_db = LyricsDB(
            qdrant_client=self._qdrant_client,
            collection_name=self.collection_name,
            model_name=self.model_name,  # will be refactored later
            include_clap=False
        )
        
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
