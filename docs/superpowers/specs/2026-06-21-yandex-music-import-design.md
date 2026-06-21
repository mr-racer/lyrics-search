# Импорт из Яндекс.Музыки + авто-обогащение метаданных — Design

**Дата:** 2026-06-21
**Ветка:** `feature/yandex-music-import`
**Статус:** утверждён к реализации

---

## 1. Цель и контекст

Дать пользователю мигрировать свою библиотеку с Яндекс.Музыки в MusiX:

1. **Импорт плейлиста** — юзер логинится в Яндекс через device-flow (вводит код на
   стороне Яндекса), выбирает источник («Мне нравится» или конкретный плейлист), и
   модуль скачивает все треки источника, прогоняя их через **существующий** конвейер
   индексации без изменений в его сути.
2. **Авто-обогащение метаданных** — для любого трека (folder / upload / импорт), у
   которого есть `artist`+`title`, но не хватает остальных полей, ищем трек в каталоге
   Яндекса и **дозаполняем только пустые** поля (album, year, genre, обложка, label и
   т.д.), не перезаписывая уже заполненные.

Библиотека API: [`yandex-music`](https://github.com/MarshalX/yandex-music-api)
(MarshalX), документация: <https://ym.marshal.dev/>.

### Ключевые проектные решения (утверждены)

| # | Решение |
|---|---------|
| Хранение токена | Персистить per-account в SQLite, токен зашифрован (Fernet). |
| Охват фичи №2 | Все треки (folder + upload + импорт), отдельный этап обогащения. |
| Обложка/текст при импорте | Встраивать данные Яндекса прямо в файл перед конвейером. |
| Недоступные треки | Пропускать и собрать отчёт; импорт не падает. |
| Креды для поиска (№2) | Токен аккаунта, если привязан; иначе анонимный клиент. |

---

## 2. Область и режим работы

- Импорт пишет файлы в `media/<account_id>/audio/<sha256>.<ext>` — это **server-режим**
  (та же модель, что и `POST /library/upload`). В sharing-режиме (single-tenant)
  модуль не активен: там используется folder-индексация.
- Реализуется **только бэкенд-логика + REST-эндпоинты**. Фронт-страница «первый логин
  → индексация» подключится позже к этим эндпоинтам.
- Все эндпоинты под `Depends(get_current_user)` и `Depends(require_mode("server"))`,
  как `/library/upload`.

---

## 3. Принцип переиспользования конвейера

Существующая цепочка загрузки:

```
POST /library/upload         → файл в media/<acct>/audio/<sha>.<ext> + строка pending_uploads
POST /library/upload/batch-commit → LibraryService.enqueue_upload_indexing(...)
                                  → IndexingService.index_uploads(...)
                                     (теги → текст → обложка → эмбеддинги → facts)
```

Импорт встаёт **источником файлов** для этой же цепочки. Раннер импорта:

1. скачивает трек,
2. пишет теги + встраивает обложку/текст,
3. кладёт файл через `uploads_service.atomic_promote_to_managed`,
4. создаёт строку `pending_uploads` (status='uploaded'),
5. в конце вызывает `LibraryService.enqueue_upload_indexing(account_id, upload_ids)`.

Так «флоу обработки файлов ровно такой же, как и был» — `index_uploads` не меняется по
сути (см. §6 про точку подключения обогащения).

---

## 4. Раскладка модуля

```
app/services/yandex/
  __init__.py
  client_factory.py     # build_client(token|None) + ThrottledClient (rate-limit обёртка)
  auth.py               # device-flow: start_session, poll_session, finalize → токен
  token_store.py        # Fernet шифрование + CRUD токена per-account (yandex_accounts)
  playlists.py          # list_playlists(client) + «Мне нравится» (likes) как источник
  downloader.py         # download_track: flac→mp3→aac, теги, обложка, текст; tag-writer
  importer.py           # оркестратор импорта: source → файлы → pending_uploads → enqueue
  enrichment.py         # фича №2: search(artist+title) → дозаполнение пустых полей

app/api/routes/imports.py   # REST-эндпоинты импорта/авторизации Яндекса
```

Зависимости (в `requirements.txt`):
- `yandex-music` — клиент API.
- `cryptography` — Fernet для шифрования токена.

---

## 5. Яндекс-авторизация (device flow) и хранение токена

### 5.1 Device-flow

Поток (MarshalX `Client.device_auth` — использует креды мобильного приложения
Яндекса; проверить на этапе реализации):

1. `POST /import/yandex/auth/start`
   → создаёт device-сессию, **хранится in-memory** в `app.state.ym_auth_sessions`
     (короткоживущая), возвращает `{session_id, verification_url, user_code, expires_in}`.
   → запускает фоновый поллинг (`asyncio.create_task`), который ждёт подтверждения.
2. Юзер открывает `verification_url`, вводит `user_code` на стороне Яндекса.
3. `GET /import/yandex/auth/status?session_id=...`
   → `{status: "pending" | "authorized" | "expired" | "error"}`.
4. При `authorized` фоновый поллинг уже сохранил токен (см. 5.2) и пометил привязку.

### 5.2 Хранение токена

- Таблица `yandex_accounts`:
  ```
  account_id   TEXT PRIMARY KEY     -- FK на users.id
  enc_token    TEXT NOT NULL        -- Fernet(access+refresh+expires JSON)
  yandex_uid   TEXT                 -- uid аккаунта Яндекса (для справки)
  expires_at   REAL                 -- epoch, для проактивного refresh
  linked_at    REAL NOT NULL
  ```
- Шифрование: **Fernet**. Ключ из env `MUSIX_YM_TOKEN_KEY` (base64 32 байта); если не
  задан — детерминированно деривируется из `MUSIX_JWT_SECRET`
  (`base64(sha256(secret + "ym-token"))`). Ротация ключа инвалидирует токены —
  допустимо (требует повторного логина).
- `token_store`: `save(account_id, token)`, `load(account_id) -> token|None`,
  `delete(account_id)`. Только финальный токен персистится; device-сессия временная.
- Refresh: если `expires_at` близко/прошёл — пробуем `refresh_token` через клиент;
  при неудаче помечаем привязку «протухшей» (юзеру предложить перелогиниться).

### 5.3 Эндпоинты статуса привязки

- `GET  /import/yandex`        → `{linked: bool, expires_at, yandex_uid}`.
- `DELETE /import/yandex`      → стереть токен (отвязать аккаунт).

---

## 6. Список плейлистов

`GET /import/yandex/playlists` (требует привязки):
- `client.users_playlists_list()` → плейлисты юзера.
- Плюс синтетический пункт **«Мне нравится» (likes)** — это `users_likes_tracks`, у
  него нет `kind`, поэтому это отдельный псевдо-источник.
- Ответ: `[{source, title, track_count, cover}]`, где `source` это `"likes"` или
  `{"kind": <int>}`.

---

## 7. Импорт-джоба (главная фича)

`POST /import/yandex/start {source: "likes" | {"kind": <int>}}` → `{job_id}`.

Прогресс — через существующий `JobTracker` + `/library/status` + SSE (новых сущностей
прогресса не вводим). Конкуренция — под `_INDEX_SEMAPHORE` и per-account слот (как
upload-индексация: второй импорт того же аккаунта во время RUNNING → 409).

### 7.1 Шаги раннера (`importer.run_import`)

1. **Резолв источника** → список треков (`TrackShort.fetch_track()` → полный `Track`).
2. **Дедуп**: отфильтровать треки, уже импортированные этим аккаунтом, по
   `yandex_imports(account_id, yandex_track_id)`.
3. **Скачка** (фаза «download»):
   - `ThreadPoolExecutor(max_workers=2)`.
   - Общий троттл: **не чаще 1 запроса в 0.5 c** (глобальный `min-interval` lock,
     потоко-безопасный).
   - Кодеки по приоритету: **`flac` → `mp3@320` → `aac`** (через
     `track.get_download_info()` / `get_specific_download_info`). FLAC — best-effort
     (зависит от версии API/подписки); если нет — следующий по списку.
   - На трек, который не скачался (нет Плюс, регион, недоступен) — **пропуск** +
     запись причины в отчёт; джоба продолжается.
4. **Тегирование + встраивание** (на каждый успешный файл, mutagen-writer):
   - Теги: `artist`, `title`, `album`, `year`, `genre`, `track_number`, `disc_number`.
   - **Обложка Яндекса** (скачать по `cover_uri`, размер ~600x600) встроить в файл.
   - **Текст Яндекса** (`client.tracks_lyrics`, если есть) встроить в файл/сохранить
     рядом так, чтобы существующий конвейер его подхватил.
5. **Промоут в managed-хранилище**: SHA-256 → `uploads_service.atomic_promote_to_managed`
   → `media/<account>/audio/<sha>.<ext>`; идемпотентность по SHA (как в upload).
6. **Регистрация**: `MetadataDB.create_pending_upload(...)` (status='uploaded') +
   строка в `yandex_imports` (yandex_track_id → upload_id, status).
7. **Индексация**: после фазы скачки — `LibraryService.enqueue_upload_indexing(
   account_id, upload_ids)`. Дальше **существующий** `index_uploads`: теги → текст →
   обложка → эмбеддинги → facts. Обложка/текст уже в файле → текущие фетчеры
   подхватывают/не перетирают.
8. **Итог джобы**: `{downloaded: N, skipped: [{artist,title,reason}], indexed: M}`.

### 7.2 Замечание про объединение фаз

Скачка и индексация — две фазы одной пользовательской операции. MVP: импорт-раннер
выполняет фазу скачки, затем вызывает `enqueue_upload_indexing` (которая стартует свою
джобу индексации). Прогресс импорта = «download X/Y», далее статус индексации читается
тем же `/library/status`. (Возможная унификация в один job со стадиями — вне MVP.)

---

## 8. Авто-обогащение метаданных (фича №2)

Модуль `enrichment.py`, функция `enrich(song_info: dict, client) -> dict`.

- **Триггер**: есть `artist`+`title`, но пусты какие-то из
  `album / year / genre / cover_art_path / label / producer / ...`.
- **Поиск**: `client.search(f"{artist} {title}")` → `best`/первый трек.
  **Защита от мисматча**: сверяем длительность (если известна) с допуском **±5 c**;
  при сильном расхождении — обогащение пропускаем.
- **Заполнение**: только пустые поля. Заполненные не трогаем.
- **Клиент**: токен аккаунта, если привязан; иначе анонимный
  (`client_factory.build_client(None)`).
- **Точки подключения**:
  - `app/indexing/folder_scanner.scan_and_enrich_folder` — после получения lyrics.
  - `app/services/indexing_service.index_uploads._scan_one` — после `process_file`.
- **Уровень записи**: обогащаем **payload** (`song_info`-dict). Для folder-треков
  файлы на хосте **не переписываем**. Для managed-файлов (upload/импорт) теги и так
  корректны (импорт уже записал их из Яндекса).
- **Устойчивость**: любые сбои Яндекса/сети — `try/except`, обогащение не валит
  индексацию (паттерн уже принят в кодовой базе).
- **Флаг**: глобальный включатель через env (`MUSIX_YM_ENRICH`, по умолчанию on) +
  тот же троттл, что у скачки.

---

## 9. Данные (новые таблицы в `cache/metadata.db`)

```sql
CREATE TABLE IF NOT EXISTS yandex_accounts (
  account_id  TEXT PRIMARY KEY,
  enc_token   TEXT NOT NULL,
  yandex_uid  TEXT,
  expires_at  REAL,
  linked_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS yandex_imports (
  account_id       TEXT NOT NULL,
  yandex_track_id  TEXT NOT NULL,
  upload_id        TEXT,
  track_id         TEXT,
  status           TEXT NOT NULL,     -- pending|downloaded|indexed|skipped|failed
  reason           TEXT,
  imported_at      REAL NOT NULL,
  PRIMARY KEY (account_id, yandex_track_id)
);
```

Device-сессии авторизации — **in-memory** (`app.state`), в БД не пишутся.

---

## 10. Эндпоинты (server-mode, `get_current_user`)

| Метод | Путь | Назначение |
|-------|------|-----------|
| POST | `/import/yandex/auth/start` | старт device-flow → код + url |
| GET | `/import/yandex/auth/status` | статус device-сессии |
| GET | `/import/yandex` | статус привязки аккаунта |
| DELETE | `/import/yandex` | отвязать (стереть токен) |
| GET | `/import/yandex/playlists` | список плейлистов + «Мне нравится» |
| POST | `/import/yandex/start` | старт импорт-джобы → job_id |
| GET | `/import/yandex/status` | сводка импорта (или reuse `/library/status`) |

---

## 11. Тестирование

Юнит (мок Yandex `Client`, без сети):
- `token_store`: шифр/дешифр round-trip; чтение несуществующего → None; delete.
- `downloader`: codec-fallback (flac нет → mp3 → aac); тегирование/встраивание обложки.
- `enrichment`: заполняет **только** пустые поля; duration-guard отбрасывает мисматч;
  сбой клиента не бросает исключение.
- `importer`: дедуп по `yandex_imports`; пропуск недоступного с записью в отчёт;
  successful path → `pending_uploads` + вызов `enqueue_upload_indexing`.

Интеграция:
- фейковый Client → файлы появляются в `media/<acct>/audio/`, строки `pending_uploads`
  и `yandex_imports` корректны.

Маркеры: `unit`, `integration` (как в проекте).

---

## 12. Риски и ограничения

- **Неофициальный API**: ломкость, антибот/лимиты → закладываем троттл (0.5 c, 2
  воркера) и устойчивость к ошибкам.
- **Скачка требует подписки Плюс**; без неё — только 30-сек превью (не качаем, в отчёт).
- **FLAC** не гарантирован версией API/правами → fallback на mp3@320 / aac.
- **Device-flow** MarshalX завязан на креды мобильного приложения Яндекса — проверить
  доступность на этапе реализации; при проблемах — план Б: вставка готового OAuth-токена.
- **Юридически**: серая зона (реверс-инжиниринг); приемлемо для личной миграции.
```
