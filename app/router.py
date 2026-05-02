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

    
# @router.post("/completions", response_model=ChatResponse)
# async def chat_completion(request: ChatRequest):
#     """
#     Handles incoming requests using the standard OpenAI/Chat Completions API format.
#     Processes the messages list and uses the underlying LLM client.
#     """

#     # --- 1. Parse Messages for ask_llm ---
    
#     system_prompt = None
#     user_message = ""
    
#     # Process the message list to find system/user content
#     messages: list[ChatMessage] = request.messages
#     if messages:
#         for i, msg in enumerate(messages):
#             role = msg.role.lower()
#             content = msg.content

#             if role == "system":
#                 # We only take the first system message as the prompt
#                 system_prompt = content
#             elif role == "user" and i > 0: # Assume user's query is not the very first message
#                 # The primary user message should be passed to the LLM core function
#                 user_message = content

#     if not user_message:
#         raise HTTPException(status_code=400, detail="Missing required 'user' message in chat history.")


#     # --- 2. Call the Core LLM Logic ---
#     try:
#         llm_response_content = await ask_llm(
#             user_message=user_message,
#             system_prompt=system_prompt, # Pass system prompt if available
#             model=request.model,          # Use model provided by client
#             temperature=request.temperature,
#             extra_body={"enable_thinking": request.enable_thinking}, # Passes extra fields like enable_thinking
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"LLM Communication Error: {e}")


#     # --- 3. Format the Output Response ---
#     # The client expects a specific structure (like OpenAI's response).
#     return ChatCompletionResponse(
#         id=f"chat-completion-{hash(''.join([m.content for m in messages]))}", # Generate a unique ID
#         object="chat.completion",
#         choices=[
#             {
#                 "message": {"role": "assistant", "content": llm_response_content}
#             }
#         ]
#     )
