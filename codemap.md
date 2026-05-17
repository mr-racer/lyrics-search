# MusiX — Architecture Codemap

> Semantic music search platform. Find music by meaning, not by name.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI (async), Python 3.11+, Pydantic |
| **Vector DB** | Qdrant (dense + BM25 sparse + CLAP audio vectors) |
| **Text Embeddings** | Sentence Transformers (jina-embeddings-v2-small-en, Qwen3-0.6B) |
| **Audio Embeddings** | CLAP (laion-clap, HTSAT-base) |
| **LLM** | OpenAI-compatible API (LM Studio, Ollama, etc.) |
| **Metadata** | Mutagen (audio tags), Syncedlyrics (lyrics), MusicBrainz (enrichment) |
| **Structured Data** | SQLite (facts, reactions) |
| **Frontend** | Single-file React SPA (Babel standalone, no build step) |
| **Testing** | pytest (unit, integration, slow markers) |

---

## Directory Structure

```
lyrics-search/
├── app/                          # Modular FastAPI application
│   ├── api/                      # API layer
│   │   ├── main.py               #   FastAPI app factory, lifespan, CORS, SPA serving
│   │   ├── sse_utils.py          #   Server-sent events helper
│   │   └── routes/               #   API route routers
│   │       ├── search.py         #     POST /search/, GET /tracks/{id}/stream, reactions
│   │       ├── library.py        #     GET /browse, /collections, /stats, POST /index
│   │       ├── chat.py           #     POST /chat/ (agentic LLM search loop)
│   │       └── metadata.py       #     GET/POST /metadata/artists/{slug}/facts
│   │
│   ├── domain/                   # Domain models (Pydantic)
│   │   └── models.py             #   SearchRequest, SearchResponse, TrackMetadata, TrackHit, etc.
│   │
│   ├── resources/                # External resource clients
│   │   ├── db_client.py          #   Qdrant connection wrapper + LyricsDB bridge
│   │   ├── model_registry.py     #   Lazy model loading (text + CLAP) with caching
│   │   └── metadata_db.py        #   SQLite store for facts, reactions
│   │
│   ├── services/                 # Business logic
│   │   ├── search_service.py     #   Core search (text/audio/hybrid modes)
│   │   ├── library_service.py    #   Folder indexing orchestration
│   │   ├── llm_client.py         #   OpenAI-compatible LLM client
│   │   ├── similarity_service.py #   Track similarity analysis
│   │   ├── job_tracker.py        #   SSE job progress tracking
│   │   ├── artist_facts_service.py  #   Artist facts fetching/caching
│   │   ├── song_facts_service.py    #   Song facts fetching/caching
│   │   └── migrate_facts.py      #   Legacy .txt → SQLite migration
│   │
│   ├── existing/                 # Legacy code (migrated, not deleted)
│   │   ├── qdrant_db.py          #   LyricsDB (Qdrant operations: upsert, search, scroll)
│   │   └── folder_processor.py   #   FileProcessor wrapper
│   │
│   └── main.py                   # Entry point shim (re-exports app.api.main:app)
│
├── search_engine/                # Legacy search module (Qdrant operations)
│   ├── main.py                   #   LyricsDB class (collection init, fit, search)
│   └── utils.py                  #   Model loading, metadata prep, CLAP encoding
│
├── file_processor/               # Legacy file processing
│   ├── main.py                   #   FileProcessor class
│   └── utils.py                  #   Audio file parsing (mutagen), lyric fetching
│
├── frontend/                     # Single-file React SPA
│   ├── index.html                #   Production build (served by FastAPI)
│   ├── MusiX Player Concept.html #   Design exploration
│   └── take_landing_concept.html #   Landing page concept
│
├── cache/                        # Runtime cache
│   ├── metadata.db               #   SQLite facts/reactions database
│   └── top_pairs/                #   Cached similarity pairs per collection
│
├── tests/                        # Test suite
│   ├── conftest.py               #   pytest configuration
│   ├── unit/                     #   Fast, no external deps
│   └── integration/              #   SQLite, file I/O
│
├── docs/                         # Documentation
├── notebooks/                    # Jupyter notebooks (exploration)
├── weights/                      # CLAP model checkpoint
│   └── music_audioset_epoch_15_esc_90.14.pt
│
├── pyproject.toml                # pytest config, markers
├── requirements.txt              # Python dependencies
├── logging.conf                  # Logging configuration
└── README.md                     # Project overview
```

---

## Architecture Layers

### 1. API Layer (`app/api/`)

FastAPI application with lifespan-managed dependencies.

```
Request → CORS → Router → Service → Resource → External
```

**Key patterns:**
- `lifespan()` sets up `app.state` with DbClient, SearchService, LibraryService, JobTracker
- Routes registered BEFORE SPA catch-all (Starlette matches in order)
- SSE streaming for indexing progress (`/api/v1/index/progress/{job_id}`)
- Static file serving for covers (`/api/v1/covers/{file}`)
- Graceful degradation when Qdrant is unavailable at startup

**Route prefix:** All API routes are mounted under `/api/v1`.

### 2. Services Layer (`app/services/`)

Business logic, no framework dependencies.

| Service | Responsibility |
|---------|---------------|
| `SearchService` | Dispatches to text/audio/hybrid search, merges hits |
| `LibraryService` | Orchestrates folder indexing (6 stages with progress) |
| `LLMClient` | OpenAI-compatible chat completions wrapper |
| `SimilarityService` | Analyzes track similarity, caches top pairs |
| `JobTracker` | Singleton tracking indexing jobs with SSE subscribers |
| `ArtistFactsService` | Fetches/caches artist facts (LLM-driven) |
| `SongFactsService` | Fetches/caches song facts (LLM-driven) |

### 3. Resources Layer (`app/resources/`)

External system clients and singletons.

| Resource | Responsibility |
|----------|---------------|
| `DbClient` | Qdrant connection + LyricsDB instance |
| `ModelRegistry` | Lazy model loading with GPU caching |
| `MetadataDB` | SQLite wrapper for facts, reactions |

### 4. Domain Layer (`app/domain/`)

Pydantic models shared across layers.

**Key models:**
- `SearchRequest` / `SearchResponse` — search API contract
- `TrackMetadata` — track payload (title, artist, album, year, genre, duration, file_path, cover_art_path, lyrics)
- `TrackHit` — search result with score, matched_on, facts
- `IndexRequest` / `IndexProgress` — indexing API contract
- `ChatRequest` / `ChatResponse` — LLM chat API contract

---

## Data Flow

### Search Flow

```
POST /api/v1/search/
  → SearchService.search(query, mode)
    → mode="text":  LyricsDB.search() → dense+BM25 RRF fusion
    → mode="audio": CLAP text embedding → Qdrant query_points(using="clap")
    → mode="hybrid": parallel text+audio, min-max normalize, weighted fusion (0.5/0.5)
    → _points_to_hits() → enrich with facts from MetadataDB
  → SearchResponse(hits, query, mode)
```

### Indexing Flow (6 stages)

```
POST /api/v1/index/
  → LibraryService.index_folder()
    → JobTracker.create_job()
    → asyncio.create_task(_run_indexing_job)
      
    Stage 1: LYRICS (25%)
      → FileProcessor.process_folder() → scan files, read tags, fetch lyrics (syncedlyrics)
      
    Stage 2: FACTS (10%)
      → fetch_facts_for_artists() + fetch_facts_for_songs() (parallel, LLM-driven)
      → cached to MetadataDB (SQLite)
      
    Stage 3: METADATA (5%)
      → MusicBrainz enrichment (currently disabled)
      
    Stage 4: DENSE (20%) + Stage 5: AUDIO (25%)
      → SearchService.index_tracks_with_progress()
        → LyricsDB.fit_with_progress()
          → encode lyrics (SentenceTransformer)
          → encode audio (CLAP)
          → upsert_in_batches() → Qdrant
          
    Stage 6: ANALYSIS (15%)
      → SimilarityService.analyze_collection()
        → compute top similar/dissimilar pairs, cache to disk
        
    → JobTracker.remove_completed_job()
```

### Chat Flow (agentic LLM loop)

```
POST /api/v1/chat/
  → Call 1: Classification (CLASSIFICATION_SYSTEM_PROMPT)
    → LLM returns {"type": "text"|"audio"|"hybrid", "reasoning": "..."}
    
  → Audio fast path (if type="audio"):
    → CLAP_REPHRASE_SYSTEM_PROMPT → 3 optimized prompts
    → Run each through audio search (10 results each)
    → Merge, pick top 5, generate answer (AUDIO_ANSWER_PROMPT)
    
  → Agentic loop (if type="text" or "hybrid"):
    → For attempt in 1..NUM_ATTEMPTS:
      → LLM receives DEVELOPER_PROMPT with {query, context, previous_queries, attempt}
      → If action="search": run queries, accumulate context, repeat
      → If action="answer": return message + hits
```

---

## Vector Space

Qdrant collection stores three vector types per track:

| Vector | Model | Dimension | Distance | Purpose |
|--------|-------|-----------|----------|---------|
| `text_*` | SentenceTransformer | 512 or 1024 | COSINE | Lyrics semantic search |
| `bm25` | Qdrant/bm25 (sparse) | — | IDF | Keywords BM25 |
| `clap` | CLAP (HTSAT-base) | 512 | COSINE | Audio cross-modal search |

**Text search:** RRF fusion of dense (`text_*`) + sparse (`bm25`) results.
**Audio search:** CLAP text embedding → query `clap` vector only.
**Hybrid search:** Parallel text + audio, min-max normalized, 0.5/0.5 weighted fusion.

---

## Model Loading Strategy

```
Server startup (lifespan):
  → DbClient._connect() (TCP only, no model loading)
  → asyncio.create_task(_preload_models_in_background)
    → sleeps 1s, then loads text model + CLAP from ModelRegistry

First search:
  → LyricsDB._ensure_model() → checks ModelRegistry cache → falls back to load_model()
  → LyricsDB._ensure_clap() → same pattern

GPU management during indexing:
  → Text model on GPU → encode lyrics → move to CPU
  → CLAP on GPU → encode audio → release
  → Restore text model to GPU
```

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `TEXT_MODEL` | `jinaai/jina-embeddings-v2-small-en` | Default text embedding model |
| `AUDIO_MODEL` | `laion/clap-htsat-base` | Default audio embedding model |
| `LLM_BASE_URL` | — | OpenAI-compatible LLM endpoint |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Default LLM model name |
| `OPENAI_API_KEY` | `lm-studio` | API key for LLM |
| `MUSIC_FOLDER` | — | Default music library path |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Health + Qdrant status |
| `POST` | `/api/v1/search/` | Search tracks (text/audio/hybrid) |
| `GET` | `/api/v1/search/models/text` | List available text models |
| `GET` | `/api/v1/search/models/loaded` | List loaded models |
| `GET` | `/api/v1/search/tracks/{id}/stream` | Stream audio file |
| `POST` | `/api/v1/search/tracks/{id}/reaction` | Set track reaction |
| `GET` | `/api/v1/search/tracks/{id}/reaction` | Get track reaction |
| `POST` | `/api/v1/chat/` | LLM-assisted chat search |
| `GET` | `/api/v1/browse` | Payload-only search (title/artist/album) |
| `GET` | `/api/v1/collections` | List collections with counts |
| `GET` | `/api/v1/stats` | Library statistics |
| `GET` | `/api/v1/top-pairs` | Cached similar/dissimilar pairs |
| `POST` | `/api/v1/index` | Start indexing |
| `GET` | `/api/v1/index/progress/{job_id}` | SSE progress stream |
| `GET` | `/api/v1/covers/{file}` | Serve album cover |
| `GET` | `/api/v1/metadata/random-facts` | Random trivia |
| `GET/POST` | `/api/v1/metadata/artists/{slug}/facts` | Artist facts CRUD |
| `GET/POST` | `/api/v1/metadata/songs/{slug}/facts` | Song facts CRUD |
| `DELETE` | `/api/v1/library/collection/{name}` | Delete collection |
| `GET` | `/api/v1/library/pick-folder` | Native folder picker dialog |

---

## Testing

```bash
pytest                          # All tests
pytest -m unit                  # Fast, no external deps
pytest -m integration           # SQLite, file I/O
pytest -m slow                  # >1s tests
pytest --cov=app                # Coverage report
```

**Markers:** `unit`, `integration`, `slow`

---

## Key Design Decisions

1. **Lazy model loading** — Server starts instantly, models load in background
2. **ModelRegistry caching** — Shared model instances across LyricsDB instances
3. **SSE progress tracking** — JobTracker with subscriber queues for real-time updates
4. **Graceful degradation** — App runs even if Qdrant is down at startup
5. **Dual storage** — Qdrant (vectors) + SQLite (structured facts/reactions)
6. **SPA catch-all ordering** — API routers registered before SPA fallback
7. **Collection-scoped facts** — Facts are namespaced by Qdrant collection
8. **CLAP rephrasing** — LLM transforms mood queries into acoustic prompts for better audio retrieval
9. **Agentic chat loop** — LLM iterates up to 4 times, refining queries based on context
10. **Min-max normalization** — Hybrid search normalizes scores before fusion

---

## Dependencies

**Core:** fastapi, uvicorn, pydantic, qdrant-client, sentence-transformers, laion-clap, mutagen, syncedlyrics
**LLM:** openai (AsyncOpenAI client)
**Data:** torch, numpy, librosa, scikit-learn, transformers
**Testing:** pytest, pytest-asyncio, pytest-cov, pytest-mock, responses
**Music:** musicbrainz
