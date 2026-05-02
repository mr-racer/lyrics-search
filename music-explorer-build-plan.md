# Music Explorer — план реализации для ИИ-кодерского агента

## Контекст и роль агента

Ты — ИИ-кодерский агент. Твоя задача — построить backend музыкального приложения **Music Explorer** на основе двух существующих классов пользователя:

1. `FolderProcessor` — обходит папку с музыкой (flac, mp3, m4a), достаёт метадату (title, artist, album, year, genre, duration), подгружает лирику из API (с опциональным fallback в MusixSearch при `better_lyrics_quality=True`), стандартизует жанры через mapping. Возвращает список объектов с полной метадатой включая лирику.
2. `QdrantMusicDB` — управляет подключением к Qdrant, грузит модель текстовых эмбеддингов (выбираемую через параметр) и CLAP для аудио-эмбеддингов, индексирует треки батчами, выполняет vector search с режимами `"lyrics"` / `"audio"` и фильтрами по метадате. Длительность кластеризуется через кастомный IQR-вариант на квартили, года — на десятилетия.

Эти классы **работают и не должны переписываться**. Ты строишь обвязку вокруг них: правильный lifecycle, FastAPI слой, Docker Compose, тесты, готовность к фронтенду.

## Цели проекта

- **MVP**: web-приложение для меломанов, индексирующее их локальную музыку и дающее три способа исследования:
  1. Семантический поиск по лирике (без LLM, прямой vector search).
  2. Поиск по звуку через текстовый запрос (CLAP текст→аудио).
  3. Чат-бот поверх библиотеки (LLM + RAG, LLM сам решает какие tool calls делать).
- **Не входит в MVP**: культурный контекст и переводы лирики через LLM. Это будет добавлено позже.
- **Деплой**: Docker Compose с тремя сервисами (qdrant + api + позже frontend). Аудио хранится у пользователя локально, монтируется в контейнер read-only.
- **Аудитория**: обычные меломаны со своей библиотекой.

## Архитектурные принципы — соблюдай неукоснительно

1. **Resources грузятся один раз**. Модели и Qdrant-клиент создаются в FastAPI lifespan, живут всё время работы приложения. Никаких повторных загрузок на запрос.
2. **Сервисы получают зависимости через конструктор**. Никаких глобальных переменных модуля, никакого импорта `app` внутри сервисов.
3. **API слой — тонкий**. Роуты только маршрутизируют HTTP-запросы и достают сервис из `app.state` через `Depends()`. Бизнес-логики в роутах нет.
4. **Существующие классы не переписываются**. Они переезжают в `src/music_explorer/existing/` как есть. Единственное допустимое изменение — добавление обратносовместимых параметров в `__init__` `QdrantMusicDB` для приёма уже загруженных инстансов моделей.
5. **Унифицированный формат результатов**. Все эндпоинты поиска возвращают `TrackHit` одного и того же типа — это упростит фронтенд.
6. **LLM не получает полные тексты песен**. В tool result отдаётся только метадата + короткий snippet. Это экономит токены, защищает от юридических проблем и предотвращает попытки LLM цитировать лирику.
7. **Тяжёлые синхронные операции** (загрузка моделей, индексация батчей, vector search через существующий sync-API класса) выполняются через `asyncio.get_event_loop().run_in_executor()`, чтобы не блокировать event loop.

## Структура репозитория

```
music-explorer/
├── docker-compose.yml
├── .env.example
├── README.md
├── backend/
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── src/music_explorer/
│   │   ├── __init__.py
│   │   ├── config.py                  # pydantic Settings
│   │   ├── domain/
│   │   │   └── models.py              # TrackMetadata, TrackHit, SearchFilters
│   │   ├── existing/                  # сюда переезжают 2 класса пользователя
│   │   │   ├── folder_processor.py
│   │   │   └── qdrant_db.py
│   │   ├── resources/                 # singletons с lifecycle
│   │   │   ├── model_registry.py
│   │   │   └── db_client.py
│   │   ├── services/                  # бизнес-логика
│   │   │   ├── library_service.py
│   │   │   ├── search_service.py
│   │   │   └── chat_service.py
│   │   ├── api/                       # FastAPI слой
│   │   │   ├── main.py
│   │   │   ├── deps.py
│   │   │   ├── schemas.py
│   │   │   └── routes/
│   │   │       ├── library.py
│   │   │       ├── search.py
│   │   │       └── chat.py
│   │   └── jobs/                      # in-memory job registry для индексации
│   │       └── indexing_job.py
│   └── tests/
│       ├── conftest.py
│       ├── test_search_service.py
│       └── test_api.py
└── frontend/
    └── .gitkeep                       # пока пусто
```

## Этап 1. Каркас проекта и зависимости

**Что делаешь:**

1. Создай дерево директорий как указано выше.
2. Создай `backend/pyproject.toml`:

```toml
[project]
name = "music-explorer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "qdrant-client>=1.12",
    "sentence-transformers>=3.0",
    "anthropic>=0.40",
    "python-multipart>=0.0.12",
    # сюда добавь существующие зависимости пользовательских классов:
    # mutagen, librosa, numpy, lyricsgenius, и т.д.
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio", "httpx", "ruff", "mypy"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

3. Перенеси существующие классы пользователя в `src/music_explorer/existing/folder_processor.py` и `src/music_explorer/existing/qdrant_db.py` **без изменений**.

**Definition of Done этапа 1:**
- Дерево создано.
- `pip install -e .` (или `uv pip install -e .`) проходит без ошибок.
- Существующие классы импортируются из новых путей.
- Старые скрипты пользователя продолжают работать.

## Этап 2. Доменные типы

**Что делаешь:** создай `src/music_explorer/domain/models.py`:

```python
from typing import Literal
from pydantic import BaseModel

class TrackMetadata(BaseModel):
    track_id: str           # хэш file_path или UUID, стабильный между рестартами
    title: str
    artist: str
    album: str | None = None
    year: int | None = None
    genre: str | None = None
    duration_sec: float
    file_path: str
    lyrics: str | None = None

class TrackHit(BaseModel):
    """Унифицированный результат поиска для всех режимов."""
    track: TrackMetadata
    score: float
    matched_on: Literal["lyrics", "audio", "hybrid"]
    snippet: str | None = None  # фрагмент лирики, если matched_on=lyrics

class SearchFilters(BaseModel):
    year_from: int | None = None
    year_to: int | None = None
    genres: list[str] | None = None
    duration_bucket: Literal["short", "medium", "long", "very_long"] | None = None
    artists: list[str] | None = None
```

**Что важно:**
- `TrackHit` — это контракт между сервисами и API. Если сигнатура поменяется, ломается всё.
- `track_id` должен быть детерминирован: вычисляй как `hashlib.sha256(file_path.encode()).hexdigest()[:16]`. При повторной индексации той же папки id остаются те же.

**Definition of Done этапа 2:**
- Файл создан, импортируется.
- Базовые тесты на валидацию (pydantic делает сам).

## Этап 3. Слой Resources — lifecycle моделей и БД

Это критический этап рефакторинга. Цель — модели грузятся один раз.

**Шаг 3.1.** Создай `src/music_explorer/resources/model_registry.py`:

```python
import asyncio
from sentence_transformers import SentenceTransformer

class ModelRegistry:
    """Singleton-обёртка над всеми ML-моделями.
    Создаётся один раз в lifespan FastAPI."""

    def __init__(self, text_model_name: str, audio_model_name: str):
        self._text_model_name = text_model_name
        self._audio_model_name = audio_model_name
        self._text_model: SentenceTransformer | None = None
        self._audio_model = None

    async def load(self) -> None:
        """Грузим модели в executor, чтобы не блокировать event loop."""
        loop = asyncio.get_event_loop()
        self._text_model = await loop.run_in_executor(
            None, SentenceTransformer, self._text_model_name
        )
        self._audio_model = await loop.run_in_executor(
            None, self._load_clap, self._audio_model_name
        )

    @staticmethod
    def _load_clap(name: str):
        # ВНИМАНИЕ: вставь сюда реальную логику загрузки CLAP, которую
        # пользователь использует в своём QdrantMusicDB.
        # Если он использует transformers ClapModel — импортируй его здесь.
        # Если laion-clap — соответственно.
        raise NotImplementedError("Adapt to user's actual CLAP loading code")

    @property
    def text(self) -> SentenceTransformer:
        if self._text_model is None:
            raise RuntimeError("Models not loaded. Call load() first.")
        return self._text_model

    @property
    def audio(self):
        if self._audio_model is None:
            raise RuntimeError("Models not loaded. Call load() first.")
        return self._audio_model

    def unload(self) -> None:
        self._text_model = None
        self._audio_model = None
```

**Шаг 3.2.** Адаптация `QdrantMusicDB` — единственное допустимое изменение существующего кода.

Найди в `src/music_explorer/existing/qdrant_db.py` метод `__init__`. Сейчас он, вероятно, грузит модели сам. Добавь параметры `text_model_instance=None` и `audio_model_instance=None`, и логику:

```python
# Внутри __init__ QdrantMusicDB:
if text_model_instance is not None:
    self.text_model = text_model_instance
else:
    self.text_model = SentenceTransformer(text_model_name)  # старый путь

if audio_model_instance is not None:
    self.audio_model = audio_model_instance
else:
    self.audio_model = self._load_clap(audio_model_name)  # старый путь
```

Это **обратносовместимое** изменение: существующие скрипты пользователя продолжают работать, потому что новые параметры опциональны.

**Шаг 3.3.** Создай `src/music_explorer/resources/db_client.py`:

```python
from music_explorer.existing.qdrant_db import QdrantMusicDB
from music_explorer.resources.model_registry import ModelRegistry

class DbClient:
    """Тонкая обёртка над QdrantMusicDB.
    Принимает уже загруженные модели через ModelRegistry."""

    def __init__(self, qdrant_url: str, models: ModelRegistry):
        self._models = models
        self.db = QdrantMusicDB(
            url=qdrant_url,
            text_model_instance=models.text,
            audio_model_instance=models.audio,
        )

    async def close(self) -> None:
        # если у QdrantMusicDB есть .close() — позови его здесь
        pass
```

**Definition of Done этапа 3:**
- `ModelRegistry` создаётся, `await registry.load()` грузит обе модели один раз.
- `QdrantMusicDB` принимает опциональные инстансы моделей. Старые скрипты не сломались.
- `DbClient` инстанцируется без повторной загрузки моделей.
- Простой smoke-тест: создал `ModelRegistry`, загрузил, передал в `DbClient`, проверил что `db.db.text_model is registry.text` (тот же объект в памяти).

## Этап 4. Слой Services — бизнес-логика

**Шаг 4.1.** `src/music_explorer/services/library_service.py`:

```python
import asyncio
from typing import Callable, Awaitable
from music_explorer.existing.folder_processor import FolderProcessor
from music_explorer.resources.db_client import DbClient

ProgressCallback = Callable[[int, int], Awaitable[None]]

class LibraryService:
    """Индексация музыкальной библиотеки."""

    def __init__(self, db: DbClient):
        self._db = db
        self._processor = FolderProcessor()

    async def index_folder(
        self,
        folder_path: str,
        better_lyrics: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> dict:
        loop = asyncio.get_event_loop()

        # 1. Метадата + лирика (синхронно, в executor)
        tracks = await loop.run_in_executor(
            None,
            lambda: self._processor.process_folder(folder_path, better_lyrics)
        )

        # 2. Индексация батчами с прогрессом
        batch_size = 32
        total = len(tracks)
        for i in range(0, total, batch_size):
            batch = tracks[i:i + batch_size]
            await loop.run_in_executor(None, self._db.db.index_batch, batch)
            if progress_callback:
                await progress_callback(i + len(batch), total)

        return {"indexed": total}

    async def get_stats(self) -> dict:
        loop = asyncio.get_event_loop()
        # АДАПТАЦИЯ: метод get_stats может отсутствовать в QdrantMusicDB.
        # Если его нет — добавь в QdrantMusicDB метод, который возвращает
        # {total_tracks, by_genre: dict[str, int], by_year: dict[int, int]}
        # через qdrant scroll API или хранимую агрегацию.
        return await loop.run_in_executor(None, self._db.db.get_stats)
```

**Шаг 4.2.** `src/music_explorer/services/search_service.py`:

```python
import asyncio
from typing import Literal
from music_explorer.resources.db_client import DbClient
from music_explorer.domain.models import TrackHit, TrackMetadata, SearchFilters

class SearchService:
    """Семантический поиск без LLM. Прямые vector search запросы."""

    def __init__(self, db: DbClient):
        self._db = db

    async def search_lyrics(
        self,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 20,
    ) -> list[TrackHit]:
        return await self._search_internal(query, "lyrics", filters, limit)

    async def search_audio_by_text(
        self,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 20,
    ) -> list[TrackHit]:
        return await self._search_internal(query, "audio", filters, limit)

    async def find_similar(
        self,
        track_id: str,
        mode: Literal["lyrics", "audio"],
        limit: int = 10,
    ) -> list[TrackHit]:
        loop = asyncio.get_event_loop()
        # АДАПТАЦИЯ: добавь в QdrantMusicDB метод search_by_id(track_id, mode, limit)
        # который делает recommendation-запрос в Qdrant используя вектор существующей точки.
        raw = await loop.run_in_executor(
            None, self._db.db.search_by_id, track_id, mode, limit
        )
        return [self._to_hit(r, matched_on=mode) for r in raw]

    async def _search_internal(
        self,
        query: str,
        mode: Literal["lyrics", "audio"],
        filters: SearchFilters | None,
        limit: int,
    ) -> list[TrackHit]:
        loop = asyncio.get_event_loop()
        filters_dict = filters.model_dump(exclude_none=True) if filters else None
        raw = await loop.run_in_executor(
            None, self._db.db.search, query, mode, filters_dict, limit
        )
        return [self._to_hit(r, matched_on=mode) for r in raw]

    @staticmethod
    def _to_hit(raw, matched_on: str) -> TrackHit:
        """Преобразование raw результата QdrantMusicDB в TrackHit.
        АДАПТАЦИЯ: подгони под реальный формат, который возвращает QdrantMusicDB.search.
        Предполагается, что raw — это объект с .payload (dict) и .score."""
        payload = raw.payload
        return TrackHit(
            track=TrackMetadata(
                track_id=payload["track_id"],
                title=payload["title"],
                artist=payload["artist"],
                album=payload.get("album"),
                year=payload.get("year"),
                genre=payload.get("genre"),
                duration_sec=payload["duration_sec"],
                file_path=payload["file_path"],
                lyrics=None,  # не отдаём полную лирику в API по умолчанию
            ),
            score=raw.score,
            matched_on=matched_on,
            snippet=SearchService._extract_snippet(payload.get("lyrics", ""))
            if matched_on == "lyrics" else None,
        )

    @staticmethod
    def _extract_snippet(lyrics: str, max_chars: int = 200) -> str | None:
        """Возвращает первые N символов лирики как preview."""
        if not lyrics:
            return None
        return lyrics[:max_chars] + ("..." if len(lyrics) > max_chars else "")
```

**Шаг 4.3.** `src/music_explorer/services/chat_service.py`:

```python
from typing import AsyncIterator, Any
from anthropic import AsyncAnthropic
from music_explorer.services.search_service import SearchService
from music_explorer.domain.models import TrackHit

SYSTEM_PROMPT = """Ты помощник, который помогает пользователю исследовать его музыкальную библиотеку.
У тебя есть инструменты search_lyrics для семантического поиска по текстам песен и search_audio для поиска по звуку (тембр, темп, настроение).
Когда пользователь просит найти музыку — используй инструменты, не выдумывай треки.
Если запрос комбинированный (например "грустные песни про любовь со спокойным звуком") — делай несколько tool calls и объединяй результаты.
Отвечай по-русски, кратко обосновывая, почему рекомендуешь именно эти треки.
Не цитируй полные тексты песен — только упоминай темы и настроение."""

TOOLS = [
    {
        "name": "search_lyrics",
        "description": "Семантический поиск по текстам песен в библиотеке пользователя. Используй для запросов про темы, эмоции, истории в текстах.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Семантический запрос"},
                "year_from": {"type": "integer"},
                "year_to": {"type": "integer"},
                "genres": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_audio",
        "description": "Поиск по звучанию/настроению трека (тембр, темп, атмосфера). Используй для запросов вроде 'медленная электронщина с мрачным басом'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
]

MAX_TOOL_ITERATIONS = 5  # защита от бесконечного цикла tool_use


class ChatService:
    def __init__(self, search: SearchService, llm_client: AsyncAnthropic, model: str):
        self._search = search
        self._llm = llm_client
        self._model = model

    async def stream_response(
        self,
        user_message: str,
        history: list[dict],
    ) -> AsyncIterator[dict]:
        """Стримит события для фронта.
        Типы событий:
          {"type": "tool_call", "name": str, "input": dict}
          {"type": "tool_result", "hits": list[dict]}
          {"type": "text", "content": str}
          {"type": "error", "message": str}
        """
        messages = list(history) + [{"role": "user", "content": user_message}]

        for iteration in range(MAX_TOOL_ITERATIONS):
            try:
                response = await self._llm.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )
            except Exception as e:
                yield {"type": "error", "message": f"LLM error: {e}"}
                return

            if response.stop_reason == "tool_use":
                tool_uses = [b for b in response.content if b.type == "tool_use"]
                tool_results = []

                for tu in tool_uses:
                    yield {"type": "tool_call", "name": tu.name, "input": dict(tu.input)}
                    try:
                        hits = await self._dispatch_tool(tu.name, tu.input)
                    except Exception as e:
                        yield {"type": "error", "message": f"Tool {tu.name} failed: {e}"}
                        hits = []

                    yield {
                        "type": "tool_result",
                        "hits": [h.model_dump() for h in hits],
                    }
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": self._format_hits_for_llm(hits),
                    })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            # финальный ответ
            for block in response.content:
                if block.type == "text":
                    yield {"type": "text", "content": block.text}
            return

        yield {"type": "error", "message": "Превышен лимит итераций tool calls"}

    async def _dispatch_tool(self, name: str, raw_input: Any) -> list[TrackHit]:
        params = dict(raw_input)
        if name == "search_lyrics":
            from music_explorer.domain.models import SearchFilters
            filters = SearchFilters(
                year_from=params.pop("year_from", None),
                year_to=params.pop("year_to", None),
                genres=params.pop("genres", None),
            )
            return await self._search.search_lyrics(
                query=params["query"],
                filters=filters,
                limit=params.get("limit", 10),
            )
        if name == "search_audio":
            return await self._search.search_audio_by_text(
                query=params["query"],
                limit=params.get("limit", 10),
            )
        raise ValueError(f"Unknown tool: {name}")

    @staticmethod
    def _format_hits_for_llm(hits: list[TrackHit]) -> str:
        """Компактный формат: только метадата + snippet, без полной лирики."""
        if not hits:
            return "Ничего не найдено."
        return "\n".join(
            f"- {h.track.artist} — {h.track.title} "
            f"({h.track.year or 'n/a'}, {h.track.genre or 'n/a'}). "
            f"score={h.score:.2f}. snippet: {h.snippet or '—'}"
            for h in hits
        )
```

**Definition of Done этапа 4:**
- Все три сервиса инстанцируются с `DbClient` (и `LLM client` для chat).
- Юнит-тесты на `SearchService` с моком `db.db.search` проходят.
- Юнит-тест на `ChatService`: мокаешь `AsyncAnthropic` и `SearchService`, проверяешь корректную диспетчеризацию tool_use.

## Этап 5. API слой — FastAPI с lifespan

**Шаг 5.1.** `src/music_explorer/config.py`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant:6333"
    text_model: str = "intfloat/multilingual-e5-large"
    audio_model: str = "laion/clap-htsat-unfused"
    llm_model: str = "claude-opus-4-7"
    anthropic_api_key: str = ""
    music_folder: str = "/music"
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

**Шаг 5.2.** `src/music_explorer/api/schemas.py`:

```python
from typing import Literal
from pydantic import BaseModel
from music_explorer.domain.models import SearchFilters, TrackHit

class IndexFolderRequest(BaseModel):
    folder_path: str
    better_lyrics_quality: bool = False

class IndexFolderResponse(BaseModel):
    job_id: str

class SearchRequest(BaseModel):
    query: str
    mode: Literal["lyrics", "audio"]
    filters: SearchFilters | None = None
    limit: int = 20

class SearchResponse(BaseModel):
    hits: list[TrackHit]
    query: str
    mode: str

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
```

**Шаг 5.3.** `src/music_explorer/jobs/indexing_job.py`:

```python
"""In-memory регистр прогресса индексации.
Для прода — заменить на Redis."""

_jobs: dict[str, dict] = {}

def create(job_id: str) -> None:
    _jobs[job_id] = {"status": "running", "progress": 0, "total": 0}

def update(job_id: str, **fields) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(fields)

def get(job_id: str) -> dict | None:
    return _jobs.get(job_id)
```

**Шаг 5.4.** `src/music_explorer/api/deps.py`:

```python
from fastapi import Request
from music_explorer.services.library_service import LibraryService
from music_explorer.services.search_service import SearchService
from music_explorer.services.chat_service import ChatService

def get_library(request: Request) -> LibraryService:
    return request.app.state.library

def get_search(request: Request) -> SearchService:
    return request.app.state.search

def get_chat(request: Request) -> ChatService:
    return request.app.state.chat
```

**Шаг 5.5.** `src/music_explorer/api/routes/library.py`:

```python
import asyncio
import json
import uuid
from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from music_explorer.api.schemas import IndexFolderRequest, IndexFolderResponse
from music_explorer.api.deps import get_library
from music_explorer.services.library_service import LibraryService
from music_explorer.jobs import indexing_job

router = APIRouter()

@router.post("/index", response_model=IndexFolderResponse)
async def index_folder(
    req: IndexFolderRequest,
    bg: BackgroundTasks,
    svc: LibraryService = Depends(get_library),
):
    job_id = str(uuid.uuid4())
    indexing_job.create(job_id)

    async def run():
        async def cb(done: int, total: int):
            indexing_job.update(job_id, progress=done, total=total)
        try:
            result = await svc.index_folder(
                req.folder_path, req.better_lyrics_quality, cb
            )
            indexing_job.update(job_id, status="done", **result)
        except Exception as e:
            indexing_job.update(job_id, status="error", error=str(e))

    bg.add_task(run)
    return IndexFolderResponse(job_id=job_id)


@router.get("/index/{job_id}/progress")
async def index_progress(job_id: str):
    async def gen():
        while True:
            job = indexing_job.get(job_id)
            if not job:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                return
            yield f"data: {json.dumps(job)}\n\n"
            if job.get("status") in ("done", "error"):
                return
            await asyncio.sleep(1)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/stats")
async def stats(svc: LibraryService = Depends(get_library)):
    return await svc.get_stats()
```

**Шаг 5.6.** `src/music_explorer/api/routes/search.py`:

```python
from fastapi import APIRouter, Depends, Query
from music_explorer.api.schemas import SearchRequest, SearchResponse
from music_explorer.api.deps import get_search
from music_explorer.services.search_service import SearchService

router = APIRouter()

@router.post("", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    svc: SearchService = Depends(get_search),
):
    if req.mode == "lyrics":
        hits = await svc.search_lyrics(req.query, req.filters, req.limit)
    else:
        hits = await svc.search_audio_by_text(req.query, req.filters, req.limit)
    return SearchResponse(hits=hits, query=req.query, mode=req.mode)


@router.get("/similar/{track_id}", response_model=SearchResponse)
async def find_similar(
    track_id: str,
    mode: str = Query("audio", regex="^(lyrics|audio)$"),
    limit: int = 10,
    svc: SearchService = Depends(get_search),
):
    hits = await svc.find_similar(track_id, mode, limit)
    return SearchResponse(hits=hits, query=f"similar_to:{track_id}", mode=mode)
```

**Шаг 5.7.** `src/music_explorer/api/routes/chat.py`:

```python
import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from music_explorer.api.schemas import ChatRequest
from music_explorer.api.deps import get_chat
from music_explorer.services.chat_service import ChatService

router = APIRouter()

@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    svc: ChatService = Depends(get_chat),
):
    history = [{"role": m.role, "content": m.content} for m in req.history]

    async def event_generator():
        async for event in svc.stream_response(req.message, history):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**Шаг 5.8.** `src/music_explorer/api/main.py`:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from anthropic import AsyncAnthropic

from music_explorer.config import Settings
from music_explorer.resources.model_registry import ModelRegistry
from music_explorer.resources.db_client import DbClient
from music_explorer.services.library_service import LibraryService
from music_explorer.services.search_service import SearchService
from music_explorer.services.chat_service import ChatService
from music_explorer.api.routes import library, search, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()

    # 1. Resources (heavy)
    models = ModelRegistry(settings.text_model, settings.audio_model)
    await models.load()
    db = DbClient(settings.qdrant_url, models)

    # 2. Services (light)
    library_svc = LibraryService(db)
    search_svc = SearchService(db)
    llm = AsyncAnthropic(api_key=settings.anthropic_api_key)
    chat_svc = ChatService(search_svc, llm, settings.llm_model)

    # 3. State
    app.state.library = library_svc
    app.state.search = search_svc
    app.state.chat = chat_svc
    app.state.settings = settings

    yield

    await db.close()
    models.unload()


app = FastAPI(lifespan=lifespan, title="Music Explorer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(library.router, prefix="/api/library", tags=["library"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Definition of Done этапа 5:**
- `uvicorn music_explorer.api.main:app` стартует, модели грузятся в lifespan один раз.
- `curl http://localhost:8000/health` возвращает `{"status":"ok"}`.
- `curl -X POST http://localhost:8000/api/search -H "Content-Type: application/json" -d '{"query":"одиночество","mode":"lyrics","limit":5}'` возвращает результаты.
- `/docs` показывает OpenAPI схему.

## Этап 6. Docker Compose

**Шаг 6.1.** `backend/Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && uv pip install --system -e .

COPY src ./src

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "music_explorer.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Шаг 6.2.** `docker-compose.yml`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD-SHELL", "bash -c '</dev/tcp/localhost/6333' || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 10

  api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      QDRANT_URL: http://qdrant:6333
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      MUSIC_FOLDER: /music
      TEXT_MODEL: ${TEXT_MODEL:-intfloat/multilingual-e5-large}
      AUDIO_MODEL: ${AUDIO_MODEL:-laion/clap-htsat-unfused}
      LLM_MODEL: ${LLM_MODEL:-claude-opus-4-7}
    volumes:
      - ${MUSIC_FOLDER:-./music}:/music:ro
      - models_cache:/cache/huggingface
    depends_on:
      qdrant:
        condition: service_healthy

volumes:
  qdrant_data:
  models_cache:
```

**Шаг 6.3.** `.env.example`:

```
ANTHROPIC_API_KEY=sk-ant-...
MUSIC_FOLDER=/absolute/path/to/your/music
TEXT_MODEL=intfloat/multilingual-e5-large
AUDIO_MODEL=laion/clap-htsat-unfused
LLM_MODEL=claude-opus-4-7
```

**Definition of Done этапа 6:**
- `docker compose up --build` поднимает оба сервиса.
- Qdrant healthcheck проходит до старта API.
- Модели кэшируются в volume `models_cache` — повторный старт не качает их заново.
- Музыка пользователя видна в `/music` внутри контейнера API.

## Этап 7. Тесты

**Шаг 7.1.** `backend/tests/conftest.py`:

```python
import pytest
from unittest.mock import MagicMock, AsyncMock

@pytest.fixture
def mock_db_client():
    db = MagicMock()
    db.db = MagicMock()
    return db

@pytest.fixture
def search_service(mock_db_client):
    from music_explorer.services.search_service import SearchService
    return SearchService(mock_db_client)
```

**Шаг 7.2.** `backend/tests/test_search_service.py`:

```python
import pytest
from unittest.mock import MagicMock

@pytest.mark.asyncio
async def test_search_lyrics_calls_db_with_lyrics_mode(search_service, mock_db_client):
    mock_db_client.db.search.return_value = []
    await search_service.search_lyrics("одиночество", limit=5)
    args, _ = mock_db_client.db.search.call_args
    assert args[0] == "одиночество"
    assert args[1] == "lyrics"
    assert args[3] == 5

@pytest.mark.asyncio
async def test_search_audio_by_text_calls_db_with_audio_mode(search_service, mock_db_client):
    mock_db_client.db.search.return_value = []
    await search_service.search_audio_by_text("медленный бас", limit=10)
    args, _ = mock_db_client.db.search.call_args
    assert args[1] == "audio"

@pytest.mark.asyncio
async def test_search_lyrics_returns_track_hits(search_service, mock_db_client):
    fake_raw = MagicMock()
    fake_raw.score = 0.85
    fake_raw.payload = {
        "track_id": "abc123",
        "title": "Test Song",
        "artist": "Test Artist",
        "duration_sec": 180.0,
        "file_path": "/music/test.mp3",
        "lyrics": "первая строка про осень и одиночество",
    }
    mock_db_client.db.search.return_value = [fake_raw]
    hits = await search_service.search_lyrics("осень")
    assert len(hits) == 1
    assert hits[0].track.title == "Test Song"
    assert hits[0].matched_on == "lyrics"
    assert hits[0].snippet is not None
```

**Шаг 7.3.** `backend/tests/test_api.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.mark.asyncio
async def test_health(monkeypatch):
    # лёгкий тест health без поднятия моделей
    from fastapi import FastAPI
    app = FastAPI()
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/health")
        assert r.status_code == 200
```

Полноценный интеграционный тест (с реальными моделями и Qdrant) делай отдельно, не в каждом CI запуске — только перед релизами.

**Definition of Done этапа 7:**
- `pytest` проходит локально.
- Тесты на `SearchService` и `ChatService` покрывают happy path и обработку ошибок.

## Этап 8. README и готовность к фронтенду

**Шаг 8.1.** Создай `README.md` с инструкциями по запуску:

```markdown
# Music Explorer

## Quick start

1. `cp .env.example .env` и впиши свои значения (`ANTHROPIC_API_KEY`, `MUSIC_FOLDER`).
2. `docker compose up --build`
3. Открой http://localhost:8000/docs для OpenAPI.
4. Индексируй библиотеку: `curl -X POST http://localhost:8000/api/library/index -H "Content-Type: application/json" -d '{"folder_path":"/music"}'`
5. Следи за прогрессом: `curl http://localhost:8000/api/library/index/<job_id>/progress`
6. Ищи: `curl -X POST http://localhost:8000/api/search -d '{"query":"осень","mode":"lyrics"}'`
```

**Шаг 8.2.** Подготовь промпт для фронтенд-агента (положи в `frontend/PROMPT.md`):

```markdown
# Промпт для фронтенд-агента

Я делаю фронтенд для Music Explorer на Next.js 14 (App Router) + TypeScript + TanStack Query + shadcn/ui. Бэкенд на FastAPI слушает http://localhost:8000.

## Эндпоинты

- `POST /api/library/index` body `{folder_path: string, better_lyrics_quality: bool}` → `{job_id: string}`
- `GET /api/library/index/{job_id}/progress` SSE: `{status, progress, total}`
- `GET /api/library/stats` → `{total_tracks, by_genre, by_year}`
- `POST /api/search` body `{query, mode: "lyrics"|"audio", filters?, limit}` → `{hits: TrackHit[], query, mode}`
- `GET /api/search/similar/{track_id}?mode=audio&limit=10` → `{hits: TrackHit[]}`
- `POST /api/chat/stream` body `{message, history}` SSE: `{type:"tool_call",name,input}`, `{type:"tool_result",hits}`, `{type:"text",content}`, `{type:"error",message}`

## Тип TrackHit

```ts
type TrackHit = {
  track: {
    track_id: string
    title: string
    artist: string
    album?: string
    year?: number
    genre?: string
    duration_sec: number
    file_path: string
  }
  score: number
  matched_on: "lyrics" | "audio" | "hybrid"
  snippet?: string
}
```

## Страницы

1. `/library` — форма индексации (путь + чекбокс better_lyrics) с прогресс-баром через SSE; статистика библиотеки
2. `/search` — два таба (Лирика/Звук), поле запроса, фильтры (год, жанр, длительность), список карточек
3. `/chat` — чат с историей, стриминг ответов через SSE, для tool_use показывать collapsible "Поиск: [параметры]" с найденными карточками
4. `/track/[id]` — детали + топ-10 похожих по звуку и по лирике (два списка рядом)

## Карточка трека

artist + title крупно, album/year/genre мелко, score справа, клик → /track/[id], кнопка "Похожие".
```

**Definition of Done этапа 8:**
- README понятен, команды работают копипастом.
- Промпт для фронта готов.

## Финальный чеклист готовности backend

- [ ] Этап 1: структура создана, существующие классы перенесены, `pip install -e .` проходит
- [ ] Этап 2: `domain/models.py` создан
- [ ] Этап 3: `ModelRegistry` грузит модели один раз, `QdrantMusicDB` адаптирован обратносовместимо
- [ ] Этап 4: три сервиса работают, юнит-тесты на SearchService проходят
- [ ] Этап 5: `uvicorn` стартует, `/health` отвечает, `POST /api/search` возвращает результаты, `/docs` показывает схему
- [ ] Этап 6: `docker compose up` поднимает qdrant + api, healthcheck работает, модели кэшируются
- [ ] Этап 7: `pytest` зелёный
- [ ] Этап 8: README написан, промпт для фронта готов

## Что НЕ надо делать в этом MVP

- Не реализуй культурный контекст и переводы лирики через LLM. Это будет следующая итерация.
- Не реализуй PCA/UMAP проекции для 2D/3D карты. Это тоже следующая итерация — после фронта MVP.
- Не реализуй поиск по загруженному аудио-файлу ("найди похожее на этот трек"). Аналогично — следующая итерация.
- Не пиши свой Qdrant-клиент или свою логику эмбеддингов. У пользователя уже есть рабочий `QdrantMusicDB`, используй его.
- Не превращай существующие классы в async. Они синхронные — оборачивай вызовы в `run_in_executor`.

## Важные оговорки про адаптацию

В нескольких местах плана я предполагал интерфейс пользовательских классов. Проверь и адаптируй:

1. **`QdrantMusicDB.__init__`**: добавь параметры `text_model_instance=None`, `audio_model_instance=None` и логику "если передан — используй, иначе грузи как раньше".
2. **`QdrantMusicDB.search`**: предполагается сигнатура `search(query, mode, filters_dict, limit) → list[ScoredPoint]`. Если другая — адаптируй вызов в `SearchService._search_internal`.
3. **`QdrantMusicDB.index_batch`**: предполагается `index_batch(tracks: list) → None`. Если возвращает что-то ещё — учти.
4. **`QdrantMusicDB.search_by_id`**: метод может отсутствовать. Добавь его — он должен делать `recommend` запрос в Qdrant с использованием вектора существующей точки по `track_id`.
5. **`QdrantMusicDB.get_stats`**: метод может отсутствовать. Если нет — добавь, возвращающий агрегацию через scroll API.
6. **CLAP loading**: загрузка CLAP в `ModelRegistry._load_clap` — поставь ту же логику, что у пользователя в `QdrantMusicDB`.
7. **Формат raw результата поиска**: `SearchService._to_hit` предполагает `raw.payload` (dict) и `raw.score` (float). Это стандарт для qdrant-client `ScoredPoint`. Если у пользователя обёртка возвращает что-то другое — адаптируй.

После каждого этапа делай smoke-тест и не двигайся дальше, пока DoD не выполнен.
