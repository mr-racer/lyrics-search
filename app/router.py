from fastapi import APIRouter, Query
from .models import SearchRequest, SearchResponse, SimilarResponse, LyricsRequest, LyricsResponse
from .services.search_service import search_async, get_similar_async
from .services.llm_client import ask_llm

router = APIRouter()

@router.post("/search", response_model=SearchResponse)
async def search_endpoint(
    req: SearchRequest,
    include_clap: bool = Query(False),
    min_dense_score: float = Query(0.3),
    limit: int = Query(5),
):
    songs = await search_async(req.text, include_clap=include_clap, min_dense_score=min_dense_score, limit=limit)
    return SearchResponse(songs=songs)

@router.get("/similar/{song_id}", response_model=SimilarResponse)
async def similar_endpoint(song_id: str):
    songs = await get_similar_async(song_id)
    return SimilarResponse(songs=songs)

@router.post("/generate_lyrics", response_model=LyricsResponse)
async def generate_lyrics(req: LyricsRequest):
    lyrics = await ask_llm(req.prompt)
    return LyricsResponse(lyrics=lyrics)
