from pydantic import BaseModel
from typing import List

class Song(BaseModel):
    id: str
    title: str
    artist: str
    similarity_score: float | None = None

class SearchRequest(BaseModel):
    text: str

class SearchResponse(BaseModel):
    songs: List[Song]

class SimilarRequest(BaseModel):
    song_id: str

class SimilarResponse(BaseModel):
    songs: List[Song]

class LyricsRequest(BaseModel):
    prompt: str

class LyricsResponse(BaseModel):
    lyrics: str
