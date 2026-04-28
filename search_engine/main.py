from .utils import (
    load_model, prepare_metadata, build_filter,
    build_text_for_embedding, _encode_clap
)
from file_processor.utils import get_metadata

import gc
import uuid
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from pathlib import Path

from qdrant_client import QdrantClient, models


class LyricsDB:
    """Manage a Qdrant collection with hybrid dense and sparse (BM25) lyric embeddings."""

    def __init__(self, qdrant_client: QdrantClient, collection_name: str, model_name: str):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self._init_qdrant()

        self.model_name = model_name
        self.model, self.vector_name, self.vector_dim = load_model(self.model_name)

    def _init_qdrant(self):
        try:
            self.qdrant_client.get_collections()
        except Exception as e:
            raise ConnectionError(
                "Qdrant не запущен/не обнаружен. Пожалуйста, выключите VPN или перезапустите Docker"
            ) from e



    def _create_collection(self, include_clap: bool = False):
        collections = self.qdrant_client.get_collections().collections
        exists = any(c.name == str(self.collection_name) for c in collections)
        if exists:
            self.qdrant_client.delete_collection(self.collection_name)

        vectors_config = {
            self.vector_name: models.VectorParams(
                size=self.vector_dim,
                distance=models.Distance.COSINE,
            ),
        }
        if include_clap:
            vectors_config["clap"] = models.VectorParams(
                size=512,
                distance=models.Distance.COSINE,
            )

        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        print(f"Коллекция {self.collection_name} была успешно создана")


    def _upsert_in_batches(
        self,
        data: list[dict],
        text_vecs: np.ndarray,
        clap_map: dict | None = None,
        batch_size: int = 32,
    ):
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i : i + batch_size]
            vecs  = text_vecs[i : i + batch_size]

            points = []
            for song_info, vec in zip(batch, vecs):
                vector = {
                    "bm25": models.Document(text=build_text_for_embedding(song_info), model="Qdrant/bm25"),
                    self.vector_name: vec,
                }
                if clap_map:
                    key = (song_info.get("artist", "").lower(), song_info.get("title", "").lower())
                    clap_vec = clap_map.get(key)
                    if clap_vec is not None:
                        vector["clap"] = clap_vec

                points.append(models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vector,
                    payload={
                        "lyrics":     song_info["lyrics"],
                        "title":      song_info["title"],
                        "artist":     song_info["artist"],
                        "album":      song_info["album"],
                        "year":       song_info.get("year"),
                        "year_range": song_info.get("year_range"),
                        "genre":      song_info.get("genre"),
                        "duration":   song_info.get("duration"),
                    },
                ))

            self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

    def fit(self, data: list[dict], path: str | None = None):
        prepared_data = prepare_metadata(data)
        filtered = [s for s in prepared_data if len(s["lyrics"].split()) < 1300]

        if path:
            paths = [
                p for p in Path(path).rglob("*")
                if p.suffix.lower() in (".flac", ".m4a", ".mp3")
            ]

        self._create_collection(include_clap=paths is not None)

        # Pass 1: encode all lyrics at once (more efficient than per-batch)
        text_vecs = self.model.encode(
            [s["lyrics"] for s in filtered],
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Vacate GPU before loading CLAP
        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Pass 2: CLAP audio embeddings (GPU now free)
        clap_map = _encode_clap(paths) if paths is not None else {}

        # Restore text model to GPU for search
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self._upsert_in_batches(filtered, text_vecs, clap_map or None)
        print("Тексты песен были успешно проиндексированы в DB")


    def search(
        self,
        query: str,
        limit: int = 1,
        min_dense_score: float = 0.3,
        artist: str | None = None,
        album: str | None = None,
        title: str | None = None,
        genre: str | list[str] | None = None,
        year: int | None = None,
        year_range: str | None = None
    ) -> list[models.ScoredPoint]:

        query_filter = build_filter(
            artist=artist,
            album=album,
            title=title,
            genre=genre,
            year=year,
            year_range=year_range
        )

        query_vector = self.model.encode(query).tolist()

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using=self.vector_name,
                    limit=15,
                    score_threshold=min_dense_score,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="Qdrant/bm25",
                    ),
                    using="bm25",
                    limit=25,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        return results.points
