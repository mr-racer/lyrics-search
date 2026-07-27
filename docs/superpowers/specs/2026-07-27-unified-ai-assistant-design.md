# Единый ИИ-ассистент — дизайн

Дата: 2026-07-27 · Ветка: `genius-addition`

## Проблема

Две ИИ-фичи живут на разных страницах и говорят на разных протоколах:

- **Поиск по текстам** — `POST /chat/stream` (SSE), агентный цикл
  Planner → Scorer → Validator поверх `_run_chat_core` в `app/api/routes/chat.py`.
- **Сборка плейлиста** — `POST /recommend/ai-playlist/stream` (NDJSON),
  детерминированный plan → execute → select в `recsys_ai_service.ai_playlist`,
  с делегированием hits/OST в `playlist_agent/`.

Юзер должен заранее знать, какая страница решает его задачу. Плюс отдельный
LLM-вызов на классификацию в чате тратит время и иногда ошибается.

Ключевое наблюдение: **оба стека уже построены по одному принципу** — LLM
отдаёт закрытый JSON, исполняет код. Объединение — это тонкий детерминированный
роутер спереди и общий конверт стрима, а не переписывание.

## Решение

Новая страница + новый фасад `POST /api/v1/assistant[/stream]`, который
маршрутизирует запрос в один из трёх исполнителей. Старые эндпоинты и страницы
остаются рабочими — путь отката мгновенный.

```
POST /api/v1/assistant/stream
   └─ router.route(message, slots)        ← GLiNER2, без LLM, ~200 мс CPU
        ├─ intent=search   → chat_search_service.run()      (нынешний _run_chat_core)
        ├─ intent=playlist → recsys_ai_service.ai_playlist() (без правок)
        └─ intent=facts    → assistant/facts_executor.py     (новое)
```

## 1. Роутинг интентов — GLiNER2

Один вызов уже загруженного в процессе GLiNER2
(`fact_relations.extractor.get_model()`, `fastino/gliner2-multi-v1`,
многоязычная) с комбинированной схемой — классификация, сущности и структура
за один forward-проход:

```python
schema = model.create_schema()
schema.classification("intent", [
    "find one specific song by its lyrics, words or sound",
    "build a playlist or a collection of many songs",
    "learn facts, history or biography about an artist or a song",
], multi_label=False)
schema.entities({
    "artist": "name of a music artist, band or performer",
    "song":   "title of a specific song or album",
})
req = schema.structure("request")
req.field("count", dtype="str", description="how many songs the user asks for")
req.field("era",   dtype="str", description="decade, year or year range mentioned")
```

Регулярки отвергнуты сознательно: их пришлось бы вести на двух языках и
дополнять под каждую новую формулировку. GLiNER классифицирует кросс-язычно
из коробки и физически не может вернуть метку вне переданного списка.

Ненадёжность гасится **кодом**:

| Условие | Действие |
|---|---|
| `confidence ≥ 0.45` и `top1 − top2 ≥ 0.15` | маршрутизируем |
| низкая уверенность, сообщение < 5 слов, есть `last_intent` | липкость: тот же интент («а побыстрее?», «ещё такого же») |
| иначе | фрейм `clarify` с тремя кнопками; ответ приходит с явным `intent` в body, GLiNER не вызывается |
| `count ≥ 2` распознан | жёсткий оверрайд → `playlist` (сигнал не языковой) |

Старый `CLASSIFICATION_SYSTEM_PROMPT` в search-ветке больше не вызывается:
интент уже известен, `planner_enabled=True` ставится принудительно, а выбор
text/audio/hybrid остаётся за планнером (`SearchPlan.query_type`).

## 2. Слоты и многоходовость

Проект stateless — состояние носит клиент. К существующему `history`
(`ChatMessage`) добавляется структурный `slots`, который сервер возвращает в
каждом терминальном фрейме, а клиент эхом шлёт назад:

```python
class AssistantSlots(BaseModel):
    last_intent: str | None = None
    last_artist: str | None = None
    last_song: str | None = None
    last_track_id: str | None = None
    last_playlist_ids: list[str] = []
    last_filters: SearchFilters | None = None
```

Правило слияния одно и без исключений: **слоты всегда переносятся, новые
сущности перезаписывают**. Решать «это follow-up или нет» не нужно — если в
«ещё у этого артиста» GLiNER не нашёл артиста, берётся `last_artist`.

`_history_preamble` остаётся для LLM-исполнителей, но интент от текста истории
больше не зависит.

## 3. Интент «расскажи про трек / артиста»

Единственная по-настоящему новая ветка. Принцип: **на 12b LLM не решает ничего,
кроме формулировки**. Пять шагов, четыре из них — код.

**Шаг 1. Резолв субъекта (код).** `catalog_search_service.search_catalog`
(entity-режим) по спанам `artist`/`song` от GLiNER, иначе по всему сообщению.
Тонкий отрыв top-1 от top-2 → фрейм `disambiguate` с карточками-кнопками. Если
сущностей нет, но есть `now_playing_track_id` или `slots.last_track_id` —
субъект оттуда.

**Шаг 2. Нумерованный контекст-пак (код, ноль LLM).**

- трек: `get_refined_facts(scope="song")` → фоллбэк `get_song_facts` (тот же
  порядок, что в `track_chat_service`), `get_song_relations_bulk`
  (продюсеры/сэмплы/лейбл), `get_track_gems`, выдержка лирики. Genius-описание
  и построчные аннотации уже лежат в song_facts.
- артист: `get_artist_bio(slug, collection, lang)`,
  `get_refined_facts(scope="artist")` → фоллбэк `get_artist_facts`, AudioDB
  (страна, лейбл, mood), топ-треки и альбомы из `build_artist_aggregate`.

Каждый пункт нумеруется `[1] … [2] …` — это рычаг против галлюцинаций.

**Шаг 3. Один `ask_llm(parse_json=True)`** — не pydantic-ai, не tool-calling
(на маленьких моделях проект на этом уже обжигался дважды). Выход:
`{"answer": "...", "used": [1, 3, 4]}`.

**Шаг 4. Код верифицирует.** `used` пуст, содержит индексы вне диапазона, или
`answer` пуст → ответ отбрасывается, отдаётся детерминированный рендер
«вот что известно» + пункты пака как есть. Негрунтованный текст до юзера не
доходит физически.

**Шаг 5. Веб — по решению кода.** Пак пуст или тоньше порога →
`llm_web_search.smart_web_search` (тот же, что у `playlist_agent`), 1–2 запроса
с потолком в коде, сниппеты дописываются в пак нумерованными, шаг 3 повторяется.
Модель не решает «искать ли» — она на этом и ломается.

`track_chat_service` не трогаем: в плеере у него есть контекст трека, он работает.

## 4. Единый конверт стрима

Приводим к **NDJSON поверх POST fetch-streaming**: EventSource не умеет POST,
плейлист уже доказал паттерн, а SSE-поверх-POST в чате — рудимент. У фронта
останется одна утилита вместо `apiStream` + `apiStreamNdjson`.

```jsonc
{"type":"route","intent":"playlist","confidence":0.82,"human":"Понял — собираю подборку"}
{"type":"clarify","options":[{"intent":"search","label":"Найти трек"}, …]}
{"type":"disambiguate","subject_options":[…]}
{"type":"status","stage":"web_search","human":"Ищу в интернете: Kanye West greatest hits"}
{"type":"result","intent":"playlist","payload":{…},"slots":{…}}
{"type":"ping"}
{"type":"error","message":"…"}
```

Payload **тегирован интентом, не слит в общую схему** — UI рисует три типа
карточек. Слияние контрактов `best_hit/hits` и `title/steps/tracks` в union —
ровно то, от чего UX становится мутным.

### Живые статусы

Большинство уже emit-ится сегодня, просто в двух разных словарях:

| Ветка | События | Откуда |
|---|---|---|
| search | `classify`, `plan`, `search`(+found), `validate`(+valid), `retry`, `answer` | `_emit()` в `chat.py`, у каждого готов `human` на ru/en |
| playlist | `plan`/`plan_done`, `action`(tool,query)/`action_done`(found), `select`/`select_done` | `_emit()` в `ai_playlist` |
| playlist → web_hits | `filters`/`filters_done`, `web_search`(query), `auto_matched`(found), `matching`/`matching_done` | `playlist_agent` через `_web_hits_playlist.on_status` |

Добавляем:

- **`thinking`** — ни один стек не сообщает, что LLM прямо сейчас генерирует.
  Обёртка в assistant-слое эмитит до вызова и снимает после.
- **`route`** — GLiNER отработал за ~200 мс, юзер сразу видит «Понял — собираю
  подборку», ещё до первого LLM-вызова. Сейчас первые 3–8 секунд экран пуст.
- **facts-ветка** — `resolving` → `collecting`(n) → `web_search`(query) →
  `thinking`.
- **`ping` каждые 15 с** — у SSE-чата heartbeat есть, у NDJSON-плейлиста нет;
  за nginx на VPS длинная тишина рвёт соединение.

**Техническая деталь, обязательная к учёту:** колбэки двух стеков разной
природы — в чате `emit` асинхронный, в плейлисте `on_status` синхронный и
может вызываться из worker-треда (инструменты агента крутятся в
`asyncio.to_thread`). В `routes/recommend.py` это решено через
`loop.call_soon_threadsafe`. Адаптер в `assistant/service.py` нормализует оба в
одну очередь тем же приёмом — иначе события из тредов теряются или
перепутываются местами с финальным `result`.

Все `human` строит один `assistant/humanize.py` (расширенный `_human()` из
`chat.py`) — фронт ничего не формулирует, только анимирует.

## Файлы

**Новые**
- `app/services/assistant/{__init__,router,service,humanize,facts_executor}.py`
- `app/api/routes/assistant.py`
- `app/services/chat_search_service.py` (вынос `_run_chat_core`)

**Правим**
- `app/api/routes/chat.py` → тонкая обёртка, контракт не меняется
- `app/domain/models.py` → `AssistantRequest`, `AssistantSlots`, result-модели
- `app/api/main.py` → регистрация роутера **до** SPA catch-all

**Не трогаем**
- `recsys_ai_service.py`, `playlist_agent/`, `agents.py`,
  `track_chat_service.py`, `routes/recommend.py`

## Тесты

- `tests/unit/test_assistant_router.py` — таблица «фраза → интент» на
  застабленном GLiNER, пороги, липкость, `count`-оверрайд.
- `tests/unit/test_assistant_slots.py` — перенос и перезапись слотов.
- `tests/unit/test_facts_executor.py` — отбраковка ответа с пустым/выходящим за
  диапазон `used`, детерминированный фоллбэк.
- `tests/docker/test_assistant_routing.py` — настоящий GLiNER, ~30 фраз ru/en,
  порог точности. Это и есть реальная проверка, что выбор GLiNER вместо LLM
  оправдан.

## Что осознанно не делаем

- Не мержим старые эндпоинты в новый — откат должен быть мгновенным.
- Не тащим на страницу вкус/портрет, gems и статистику: больше интентов →
  классификатор на 12b начнёт путаться, а UX размывается.
- Не используем tool-calling в новой ветке фактов.
