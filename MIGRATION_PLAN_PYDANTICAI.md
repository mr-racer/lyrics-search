# Миграция на PydanticAI Agents — Подробный план

> Создан: 2026-05-17
> Статус: Planning
> Целевая версия: pydantic-ai (уже в requirements.txt)

---

## 1. Текущая архитектура и проблемы

### 1.1 Как работает сейчас

```
chat.py endpoint (~720 строк)
│
├── Call 1: LLM классификация (raw prompt → JSON)
│   └── CLASSIFICATION_SYSTEM_PROMPT → {"type": "text|audio|hybrid"}
│
├── Audio fast path (когда type == "audio")
│   ├── CLAP_REPHRASE_SYSTEM_PROMPT → 3 English prompts
│   ├── 3× search_db(mode="audio", limit=10)
│   ├── merge + sort by score → top 5
│   └── AUDIO_ANSWER_PROMPT → conversational reply
│
└── Agentic loop (4 attempts, text/hybrid)
    ├── DEVELOPER_PROMPT (сокращённый, ~30 строк)
    │   ├── LLM → {"action": "search", "queries": [...]}
    │   └── LLM → {"action": "answer", "message": "..."}
    │
    ├── _run_searches() ← ПРОБЛЕМА ЗДЕСЬ
    │   ├── Для каждого query: query_type = q.get("type", "hybrid")
    │   │   └── Баг: LLM не возвращает type → default "hybrid"
    │   ├── Если query_type in ("audio", "hybrid") → CLAP rephrase
    │   │   └── Баг: ВСЕ запросы получают CLAP рефрейзинг!
    │   └── search_db(query, mode)
    │
    └── Контекст растёт с каждой итерацией
```

### 1.2 Проблемы текущей архитектуры

| # | Проблема | Симптомы | Приоритет |
|---|----------|----------|-----------|
| 1 | **CLAP рефрейзит ВСЕ запросы** | Текстовый запрос "песня про любовь" → LLM не возвращает `type` → default `"hybrid"` → CLAP rephrase → поиск по звуку вместо текста | CRITICAL |
| 2 | **Монолитный endpoint** | 720 строк в одном файле, сложно тестировать, сложно менять | HIGH |
| 3 | **Нет извлечения фильтров** | "песня Канье Уэста про любовь" → артист не используется как фильтр | HIGH |
| 4 | **DEVELOPER_PROMPT урезан** | Старый промпт содержал STEP 3A/3B с инструкциями по типам запросов. Новый (~30 строк) не содержит | HIGH |
| 5 | **Ручной agentic loop** | `for attempt in range(1, 5)` с ручным управлением контекстом, без типизации | MEDIUM |
| 6 | **Нет разделения ответственности** | Один промпт делает: классификацию + поиск + оценку + ответ | MEDIUM |
| 7 | **Нет веб-поиска в чате** | Есть `llm_web_search.py` для био, но нет tool для поиска фактов о песнях | LOW |

### 1.3 Что уже есть в проекте

- ✅ `pydantic-ai` в requirements.txt
- ✅ `llm_web_search.py` — пример pydantic_ai Agent с tool `web_search`
- ✅ `ai_tasks/artist_bio.py` — пример использования agent.run()
- ✅ `_WIP_llm_agents.py` — наброски идей по архитектуре агентов
- ✅ `_get_client()` в `llm_web_search.py` — фабрика OpenAI клиента
- ✅ `SearchService.search()` — унифицированный интерфейс поиска
- ✅ `SearchFilters` модель — artist, album, genre, year_range

---

## 2. Целевая архитектура

### 2.1 Обзор

```
┌─────────────────────────────────────────────────────────────┐
│  chat.py (endpoint, ~120 строк)                             │
│  ├── Роутинг: auto_mode? → planner, иначе → прямой поиск   │
│  ├── Audio path: AudioAgent (CLAP rephrase + search)        │
│  └── Text/Hybrid path: Planner ↔ Scorer loop               │
└───────────────────┬─────────────────────────────────────────┘
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────┐
│  Planner    │ │   Scorer    │ │  AudioAgent  │
│   Agent     │ │   Agent     │ │              │
└─────────────┘ └─────────────┘ └──────────────┘
       │               │                 │
       ▼               ▼                 ▼
 tool: classify   tool: evaluate      tool: clap_rephrase
 tool: extract    → search/answer     tool: search_db (×3)
   _filters                              tool: audio_answer
 tool: rephrase
 tool: search_db
 tool: search_web (опционально)
```

### 2.2 Роли агентов

| Агент | Отвечает за | Вход | Выход |
|-------|-------------|------|-------|
| **Planner** | Классификация + фильтрация + планирование | user_query, previous_queries, resolved_filters | `SearchPlan` (type, filters, queries, mode) |
| **Scorer** | Оценка контекста + решение search/answer | user_query, context, attempt, filters | `ScoreResult` (action, confidence, message, queries) |
| **AudioAgent** | Аудио-быстрый путь | user_query | `AudioAnswer` (message, top5 hits) |

### 2.3 Pydantic модели (новые)

```python
# app/domain/models.py — добавить

class QueryType(BaseModel):
    """Тип запроса после классификации."""
    type: Literal["text", "audio", "hybrid"]
    reasoning: str = Field(description="Краткое объяснение, почему этот тип")


class SearchPlan(BaseModel):
    """План поиска, сгенерированный PlannerAgent."""
    action: Literal["request_filter", "search"]
    query_type: Literal["text", "audio", "hybrid"]
    filters: Optional[SearchFilters] = None
    filter_lookup: Optional[Dict[str, str]] = None  # сырые значения для разрешения
    queries: List[str] = Field(default_factory=list)
    search_mode: Literal["CONSERVATIVE", "AGGRESSIVE"] = "CONSERVATIVE"


class ScoreResult(BaseModel):
    """Результат оценки контекста."""
    action: Literal["search", "answer"]
    confidence: Literal["high", "medium", "low"]
    song: Optional[str] = None
    artist: Optional[str] = None
    queries: Optional[List[str]] = None  # новые запросы, если action="search"
    message: str


class AudioAnswer(BaseModel):
    """Ответ от AudioAgent."""
    message: str
    best_hit: Optional[dict] = None
    hits: List[dict] = Field(default_factory=list)
```

### 2.4 Dependency Injection

```python
# app/services/agent_deps.py — новый файл

from dataclasses import dataclass
from app.services.search_service import SearchService
from app.existing.qdrant_db import LyricsDB

@dataclass
class SearchDeps:
    """Зависимости, передаваемые в агенты через deps."""
    service: SearchService
    collection_name: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    lyrics_db: LyricsDB | None = None  # для resolve_filters

    async def search(
        self, query: str, mode: str, limit: int = 10, filters: SearchFilters | None = None
    ) -> list[TrackHit]:
        return await self.service.search(
            query=query, mode=mode, limit=limit,
            filters=filters, collection_name=self.collection_name,
        )

    async def resolve_filter_values(
        self, filter_key: str, raw_value: str
    ) -> list[str]:
        """Ищет валидные значения фильтра в Qdrant."""
        # scroll по collection, fuzzy match по ключу
        ...
```

---

## 3. Поэтапный план реализации

### Фаза 1: Инфраструктура (не ломает ничего)

**Цель:** Подготовить базу для агентов, не затрагивая существующий функционал.

#### Шаг 1.1: Добавить Pydantic модели

**Файл:** `app/domain/models.py`

Добавить модели: `QueryType`, `SearchPlan`, `ScoreResult`, `AudioAnswer`

```python
class SearchPlan(BaseModel):
    action: Literal["request_filter", "search"]
    query_type: Literal["text", "audio", "hybrid"]
    filters: Optional[SearchFilters] = None
    filter_lookup: Optional[Dict[str, str]] = None
    queries: List[str] = Field(default_factory=list)
    search_mode: Literal["CONSERVATIVE", "AGGRESSIVE"] = "CONSERVATIVE"
```

**Риск:** Минимальный. Только добавление новых моделей, ничего не меняется в существующем коде.

#### Шаг 1.2: Создать `app/services/agent_deps.py`

**Файл:** `app/services/agent_deps.py` (новый)

```python
from dataclasses import dataclass
from app.services.search_service import SearchService
from app.domain.models import SearchFilters, TrackHit

@dataclass
class SearchDeps:
    service: SearchService
    collection_name: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None

    async def search(self, query: str, mode: str, limit: int = 10, filters: SearchFilters | None = None):
        return await self.service.search(query=query, mode=mode, limit=limit,
                                         filters=filters, collection_name=self.collection_name)

    async def resolve_filter_values(self, filter_key: str, raw_value: str) -> list[str]:
        # TODO: реализовать scroll по Qdrant + fuzzy match
        return []
```

**Риск:** Минимальный. Файл не используется до Фазы 2.

#### Шаг 1.3: Создать `app/services/agents.py` (пустой каркас)

**Файл:** `app/services/agents.py` (новый)

```python
"""PydanticAI agents for music search.

- PlannerAgent: классификация + фильтрация + планирование запросов
- ScorerAgent: оценка контекста + решение search/answer
- AudioAgent: аудио-быстрый путь (CLAP rephrase + search)
"""

from __future__ import annotations

# TODO: Фаза 2+ — реализовать агенты
__all__ = ["planner_agent", "scorer_agent", "audio_agent"]
```

**Риск:** Нет.

---

### Фаза 2: PlannerAgent (самый ценный)

**Цель:** Реализовать агента, который классифицирует запрос, извлекает фильтры и планирует поиск.

#### Шаг 2.1: Промпт PlannerAgent

**Файл:** `app/services/agents.py`

```python
PLANNER_PROMPT = """
You are a music search planner. Analyze the user's query and prepare a search plan.

INPUTS:
<user_query>{query}</user_query>
<previous_queries>{previous_queries}</previous_queries>
<resolved_filters>{resolved_filters}</resolved_filters>
<search_filter_query>{search_filter_query}</search_filter_query>

STEP 1 — CLASSIFY:
Determine query type:
- "text" → user asks about concrete details in lyrics (words, phrases, themes)
- "audio" → user describes feelings, vibe, vocals, production, atmosphere
- "hybrid" → mix of both, or unclear

STEP 2 — EXTRACT FILTERS:
If user mentions artist, album, genre, or era → extract as filters (always ENGLISH).
- If <resolved_filters> has values → pick best match (fuzzy OK)
- If <resolved_filters> is empty AND filters needed → action="request_filter"
- Only request filters ONCE per session

STEP 3 — GENERATE QUERIES:
Generate 2-3 search queries in English (3-10 words each):
- CONSERVATIVE (first attempt): close to user's literal words
- AGGRESSIVE (previous_queries not empty): use imagery, metaphors, synonyms

OUTPUT FORMAT:
{{
  "action": "request_filter" | "search",
  "query_type": "text" | "audio" | "hybrid",
  "filters": {{"Artist": "..." | null, "Album": "..." | null, ...}} | null,
  "filter_lookup": {{"Artist": "raw input"}} | null,
  "queries": ["query 1", "query 2"],
  "search_mode": "CONSERVATIVE" | "AGGRESSIVE"
}}
"""
```

#### Шаг 2.2: Фабрика PlannerAgent

**Файл:** `app/services/agents.py`

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from app.domain.models import SearchPlan
from app.services.agent_deps import SearchDeps

def _create_model(base_url: str | None, model_name: str | None) -> OpenAIModel:
    from app.services.llm_client import _get_client
    resolved_model = (model_name or os.getenv("LLM_MODEL", "openai/gpt-oss-20b")).strip()
    openai_client = _get_client(base_url)
    provider = OpenAIProvider(openai_client=openai_client)
    return OpenAIModel(resolved_model, provider=provider)

async def create_planner_agent(deps: SearchDeps) -> Agent:
    model = _create_model(deps.llm_base_url, deps.llm_model)
    agent = Agent(model, result_type=SearchPlan, system_prompt=PLANNER_PROMPT)
    return agent
```

**Риск:** `OpenAIModel` → `OpenAIChatModel` (deprecation warning в pydantic-ai). Нужно проверить актуальное API.

#### Шаг 2.3: Реализовать `resolve_filters()`

**Файл:** `app/services/agent_deps.py`

```python
async def resolve_filter_values(self, filter_key: str, raw_value: str) -> list[str]:
    """Scroll по Qdrant collection, собрать уникальные значения поля, fuzzy match."""
    from rapidfuzz import process, fuzz_ratio  # или использовать встроенный fuzzy

    if not self.lyrics_db:
        return []

    collection = self.collection_name or self.lyrics_db.collection_name
    seen = set()
    candidates = []

    offset = None
    while True:
        points, offset = self.lyrics_db.qdrant_client.scroll(
            collection_name=collection, limit=128, offset=offset,
            with_payload=[filter_key], with_vectors=False,
        )
        for pt in points:
            val = (pt.payload or {}).get(filter_key)
            if val:
                seen.add(val)
        if offset is None:
            break

    # Fuzzy match
    threshold = 70
    matches = process.extract(raw_value, seen, scorer=fuzz_ratio, limit=5, score_threshold=threshold)
    return [m[0] for m in matches]
```

**Риск:**
- `rapidfuzz` не в requirements.txt → нужно добавить или использовать другой подход
- Scroll по всей коллекции может быть медленным для больших библиотек → нужно кэшировать
- **Митигация:** кэшировать результаты в `MetadataDB` или `cache/`

#### Шаг 2.4: Подключить Planner к endpoint (без замены agentic loop)

**Файл:** `app/api/routes/chat.py`

На этом этапе endpoint вызывает Planner для получения `SearchPlan`, но использует старый agentic loop для поиска. Это позволяет отладить Planner изолированно.

```python
# В chat endpoint, после classification:

# Экспериментально: использовать Planner для генерации запросов
from app.services.agents import create_planner_agent
deps = SearchDeps(service=service, collection_name=req.collection_name,
                  llm_base_url=llm_kw.get("base_url"), llm_model=llm_kw.get("model"))
planner = await create_planner_agent(deps)
plan = await planner.run(req.message)
# plan.data → SearchPlan с query_type, filters, queries
```

**Риск:** Planner может вернуть пустые queries или некорректный JSON. Нужно fallback на старый behavior.

---

### Фаза 3: ScorerAgent + новый agentic loop

**Цель:** Заменить ручной agentic loop на Planner ↔ Scorer цикл.

#### Шаг 3.1: Промпт ScorerAgent

**Файл:** `app/services/agents.py`

```python
SCORER_PROMPT = """
You are a music search evaluator. Evaluate search results and decide whether to answer or search again.

INPUTS:
<user_query>{query}</user_query>
<context>{context}</context>
<previous_queries>{previous_queries}</previous_queries>
<active_filters>{active_filters}</active_filters>
<attempt>{attempt_number}</attempt>

EVALUATE:
- Does any song in <context> match the user's query?
- If filters are active, does the result satisfy them?
- Confidence levels:
  HIGH → lyrics clearly match specific details
  MEDIUM → plausible match, but key detail missing
  LOW → nothing fits, or context is empty

DECIDE:
- HIGH → action="answer"
- MEDIUM AND attempt < max → action="search"
- MEDIUM AND attempt == max → action="answer" (best guess)
- LOW AND attempt < max → action="search"
- LOW AND attempt == max → action="answer" (admit failure)

OUTPUT:
{{
  "action": "search" | "answer",
  "confidence": "high" | "medium" | "low",
  "song": "Title" | null,
  "artist": "Artist" | null,
  "queries": ["new query 1", "new query 2"] | null,
  "message": "Conversational reply to user"
}}
"""
```

#### Шаг 3.2: Новый agentic loop в endpoint

**Файл:** `app/api/routes/chat.py`

```python
# Заменить старый for-цикл на:

for attempt in range(1, NUM_ATTEMPTS + 1):
    # 1. Scorer оценивает контекст
    scorer = await create_scorer_agent(deps)
    score = await scorer.run(req.message, context=context, attempt=attempt)

    if score.data.action == "answer":
        return _format_answer(score.data, all_hits)

    # 2. Если search → выполнить поиск по запросам скорера
    if score.data.queries:
        for q in score.data.queries:
            hits = await deps.search(
                query=q, mode=plan.data.query_type,
                limit=SEARCH_LIMIT, filters=plan.data.filters,
            )
            all_hits = _merge_hits(all_hits, hits)
        context = _format_context(all_hits)

    # 3. Если всё ещё search → Planner генерирует новые запросы
    planner = await create_planner_agent(deps)
    plan = await planner.run(
        req.message,
        previous_queries=score.data.queries,
        context=context,
    )
```

**Риск:** Двойной вызов LLM на итерацию (Scorer + Planner). Можно оптимизировать, объединив в один агент с tool-вызовами.

---

### Фаза 4: AudioAgent

**Цель:** Инкапсулировать аудио-быстрый путь в отдельный агент.

#### Шаг 4.1: AudioAgent с tools

**Файл:** `app/services/agents.py`

```python
AUDIO_AGENT_PROMPT = """
You are a music search assistant for audio-based queries.
The user described a song by mood or vibe. Use CLAP rephrasing to find matches.

Steps:
1. Rephrase the user's query into 3 CLAP-friendly English prompts
2. Search the database with each prompt
3. Return the best match with a conversational answer
"""

async def create_audio_agent(deps: SearchDeps) -> Agent:
    model = _create_model(deps.llm_base_url, deps.llm_model)
    agent = Agent(model, result_type=AudioAnswer, system_prompt=AUDIO_AGENT_PROMPT)

    @agent.tool
    async def clap_rephrase(deps: SearchDeps, user_query: str) -> list[str]:
        """Rephrase user's mood/vibe query into 3 CLAP-friendly English prompts."""
        try:
            rephrase_prompt = CLAP_REPHRASE_SYSTEM_PROMPT.format(user_query=user_query)
            result = await ask_llm(
                user_query, system_prompt=rephrase_prompt,
                parse_json=True, base_url=deps.llm_base_url, model=deps.llm_model,
            )
            return result if isinstance(result, list) and result else [user_query]
        except Exception:
            return [user_query]

    @agent.tool
    async def search_db(deps: SearchDeps, query: str) -> str:
        """Search the music database using audio (CLAP) mode."""
        hits = await deps.search(query=query, mode="audio", limit=10)
        return _format_context(hits)

    return agent
```

**Риск:** PydanticAI tools с async + deps. Нужно проверить совместимость с текущей версией pydantic-ai.

---

### Фаза 5: Чистка и оптимизация

**Цель:** Удалить старый код, упростить endpoint.

#### Шаг 5.1: Удалить старый код

Из `chat.py` удалить:
- Закомментированный `DEVELOPER_PROMPT` (~150 строк)
- `_run_searches()` → заменить на `deps.search()`
- `_merge_hits()` → перенести в `SearchService` или `agent_deps`
- Старый agentic loop → заменить на Planner ↔ Scorer

#### Шаг 5.2: Упростить endpoint

Целевой размер `chat.py`: ~120 строк

```python
@router.post("/")
async def chat(req: ChatRequest, request: Request) -> dict:
    service = request.app.state.search_service
    deps = SearchDeps(service=service, collection_name=req.collection_name, ...)

    # 1. Planner: классификация + фильтры + запросы
    planner = await create_planner_agent(deps)
    plan = await planner.run(req.message)

    # 1.5. Resolve фильтры если нужно
    if plan.data.action == "request_filter":
        resolved = await _resolve_all_filters(plan.data.filter_lookup, deps)
        plan = await planner.run(req.message, resolved_filters=resolved)

    # 2. Роутинг по типу
    if plan.data.query_type == "audio":
        return await _audio_path(req.message, deps)

    # 3. Agentic loop: search → score → repeat
    return await _agentic_loop(req.message, plan, deps)
```

#### Шаг 5.3: Исправить баг с CLAP рефрейзингом

Теперь `query_type` типизирован через Pydantic модель. CLAP рефрейзинг применяется только когда `query_type == "audio"`.

---

## 4. Риски и митигация

### 4.1 Риски миграции

| Риск | Вероятность | Влияние | Митигация |
|------|-------------|---------|-----------|
| **PydanticAI API change** | Средняя | Высокое | `OpenAIModel` → `OpenAIChatModel`, проверить версию в requirements |
| **Контекст между итерациями** | Высокая | Среднее | Передавать контекст через промпт, не через message_history |
| **Токен-бюджет** | Средняя | Среднее | 2 агента вместо 1 = +50% вызовов LLM. Оптимизировать объединением |
| **resolve_filters медленно** | Высокая | Среднее | Кэшировать результаты, использовать scroll limit |
| **Фейл LLM парсинга** | Средняя | Высокое | Fallback на старый behavior, валидация через Pydantic result_type |
| **Совместимость с Windows** | Низкая | Среднее | Тестировать на Windows, особенно async/await паттерны |

### 4.2 Стратегия отката

На каждой фазе:
- **Фаза 1:** Только новые файлы → откат = удалить файлы
- **Фаза 2:** Planner подключён опционально → откат = убрать вызов planner.run()
- **Фаза 3:** Новый loop параллельно старому → откат = переключить флаг
- **Фаза 4:** AudioAgent изолирован → откат = вернуть старый audio path
- **Фаза 5:** Удаление старого кода → откат = git revert

### 4.3 Тестирование

| Что тестировать | Как |
|----------------|-----|
| Planner возвращает валидный SearchPlan | Unit test с mock LLM |
| Scorer правильно оценивает контекст | Unit test с фиксированным контекстом |
| AudioAgent не ломает аудио поиск | Integration test с реальным CLAP |
| Endpoint не регрессирует | E2E test с фиксированными запросами |
| CLAP рефрейзинг применяется только к audio | Unit test: query_type="text" → нет CLAP |
| Фильтры извлекаются корректно | Unit test: "песня Канье Уэста" → artist="Kanye West" |

---

## 5. Изменения по файлам

### Новые файлы

| Файл | Описание | Фаза |
|------|----------|------|
| `app/services/agents.py` | PydanticAI агенты + промпты | 2 |
| `app/services/agent_deps.py` | Dependency injection для агентов | 1 |

### Изменённые файлы

| Файл | Что изменить | Фаза |
|------|-------------|------|
| `app/domain/models.py` | Добавить `SearchPlan`, `ScoreResult`, `AudioAnswer`, `QueryType` | 1 |
| `app/api/routes/chat.py` | Упростить endpoint, подключить агенты | 3-5 |
| `requirements.txt` | Добавить `rapidfuzz` (для resolve_filters) | 2 |

### Удалённый код

| Файл | Что удалить | Фаза |
|------|------------|------|
| `app/api/routes/chat.py` | Закомментированный DEVELOPER_PROMPT, `_run_searches()`, старый loop | 5 |
| `app/services/_WIP_llm_agents.py` | Переместить в agents.py | 2 |

---

## 6. Критические решения

### 6.1 Контекст между итерациями

**Проблема:** `agent.run()` создаёт новый диалог. Как передать контекст между итерациями?

**Решение:** Контекст передаётся через форматирование промпта, не через message_history:

```python
filled_prompt = SCORER_PROMPT.format(
    query=req.message,
    context=context or "(empty)",
    previous_queries=previous_queries or "(none)",
    attempt=attempt,
)
score = await scorer_agent.run(req.message, system_prompt=filled_prompt)
```

**Альтернатива:** Использовать `agent.run()` с `message_history`, но это требует сохранения состояния между вызовами.

### 6.2 Один агент или два?

**Вариант A:** Planner + Scorer (текущий план)
- ✅ Чёткое разделение ответственности
- ❌ +50% вызовов LLM

**Вариант B:** Один агент с tool-вызовами
- ✅ Меньше вызовов LLM
- ❌ Сложнее промпт, сложнее тестировать

**Рекомендация:** Начать с Варианта A (2 агента). Если токены станут проблемой → объединить.

### 6.3 Когда применять CLAP рефрейзинг?

**Правило:** Только когда `plan.data.query_type == "audio"`. Никогда не применять к `text` запросам.

**Реализация:**
```python
if plan.data.query_type == "audio":
    rephrased = await clap_rephrase(query)
    hits = await deps.search(rephrased[0], mode="audio")
else:
    hits = await deps.search(query, mode=plan.data.query_type)
```

---

## 7. Чек-лист готовности

### Фаза 1
- [ ] Модели `SearchPlan`, `ScoreResult`, `AudioAnswer`, `QueryType` добавлены
- [ ] `agent_deps.py` создан и работает
- [ ] `agents.py` создан (пустой каркас)
- [ ] Тесты проходят, ничего не сломано

### Фаза 2
- [ ] PlannerAgent возвращает валидный `SearchPlan`
- [ ] `resolve_filters()` работает и кэшируется
- [ ] Endpoint использует Planner для генерации запросов (опционально)
- [ ] Фильтры извлекаются корректно из запроса пользователя

### Фаза 3
- [ ] ScorerAgent правильно оценивает контекст
- [ ] Новый agentic loop работает (Planner ↔ Scorer)
- [ ] Старый agentic loop удалён
- [ ] Баг с CLAP рефрейзингом исправлен

### Фаза 4
- [x] AudioAgent инкапсулирует аудио-быстрый путь (tools: clap_rephrase, search_db)
- [x] CLAP рефрейзинг применяется только к audio запросам (через Planner классификацию)
- [x] AudioAgent возвращает top-5 результатов (черз кэш _audio_hits_cache)

### Фаза 5
- [x] Закомментированный DEVELOPER_PROMPT удалён (~150 строк)
- [x] Дублирующиеся промпты CLAP_REPHRASE и AUDIO_ANSWER удалены из chat.py (импорт из agents.py)
- [x] chat.py сокращён с 845 → 623 строк (-26%)
- [x] Все тесты проходят (11 chat/agent/audio тестов)
- [ ] `_WIP_llm_agents.py` — оставлен по запросу пользователя (не удалять)

---

## 8. Приоритет и порядок

**Рекомендуемый порядок:**

1. **Фаза 1** — безопасно, подготовка инфраструктуры
2. **Фаза 2** — самый ценный (фильтры + классификация)
3. **Фаза 3** — замена agentic loop (исправляет баг с CLAP)
4. **Фаза 4** — чистка аудио пути
5. **Фаза 5** — удаление старого кода

**Минимально жизнеспособная миграция:** Фазы 1 + 2 + 3 (без Фазы 4 и 5)

Это даёт:
- ✅ Исправленный баг с CLAP рефрейзингом
- ✅ Извлечение фильтров из запроса
- ✅ Типизированные результаты через Pydantic
- ✅ Разделение ответственности (Planner + Scorer)

Без:
- ❌ AudioAgent (аудио путь остаётся старым)
- ❌ Чистка старого кода
