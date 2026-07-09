# PWA-упаковка MusiX

**Дата:** 2026-07-10 · **Статус:** одобрено · **Ветка:** container-version

## Цель

Сделать MusiX устанавливаемым PWA: иконка на домашнем экране телефона,
standalone-окно без адресной строки, мгновенная загрузка оболочки. Фоновое
воспроизведение и Media Session (метаданные/управление на локскрине и в
шторке) уже работают — их не трогаем и не ломаем.

## Вне скоупа (отложено)

Очередь треков на Wear OS часах. Выяснено при проработке: штатные
медиа-контролы Pixel Watch показывают «Up Next» только если приложение на
телефоне выставляет очередь в **нативную Android MediaSession**
(`setQueue`); веб-API Media Session понятия очереди не имеет, поэтому из
PWA/Chrome это недостижимо. Реализация потребует companion-приложение на
телефоне или своё приложение на часах — отдельный проект. Что PWA даёт
часам уже сейчас: мост Chrome-уведомления (название/исполнитель/скип) —
включается в настройках компаньона часов (уведомления от Chrome) без кода.

## Решение

`vite-plugin-pwa` (Workbox) в `frontend/vite.config.js`. Отвергнутые
альтернативы: ручные манифест+SW (больше кода, легче ошибиться в кэше);
манифест без SW (нет офлайн-оболочки и контроля обновлений).

### 1. Манифест

- `name`: «MusiX», `short_name`: «MusiX»
- `display: standalone`, `start_url: /`, `scope: /`, `lang: ru`
- `background_color` / `theme_color`: тёмные, под палитру Studio Console
- Иконки: `pwa-192x192.png`, `pwa-512x512.png`, `pwa-512x512-maskable.png`
  (`purpose: maskable` отдельным файлом с safe-zone). Кладутся в
  `frontend/public/` (Vite копирует в корень `dist`). Генерируются
  one-off скриптом на Pillow (`scripts/gen_pwa_icons.py`): тёмный фон,
  простой глиф; заменяемы на настоящий логотип позже.

### 2. Service worker (generateSW, Workbox)

- `registerType: 'autoUpdate'` — новая сборка активируется сама, без
  «застрявшего» старого кэша.
- Прекэш: JS/CSS/HTML/иконки из сборки Vite.
- Runtime-кэш: только Google Fonts (`fonts.googleapis.com` /
  `fonts.gstatic.com`) — CacheFirst с лимитом записей.
- `navigateFallback: 'index.html'`, `navigateFallbackDenylist: [/^\/api\//,
  /^\/docs/, /^\/covers\//]` — SPA-роуты работают в standalone-окне, API не
  затягивается в фоллбек.
- **`/api/**` не перехватывается и не кэшируется.** Никаких runtime-роутов
  на API: аудио-стрим (`/api/v1/stream`, Range-запросы, `?st=` токены) идёт
  мимо SW нетронутым. Запрос, не совпавший ни с одним правилом Workbox,
  уходит в сеть как есть — этого достаточно, отдельный «исключатель» не нужен.
- Регистрация SW: `virtual:pwa-register` в `main.jsx` (одна строка,
  `immediate: true`).

### 3. index.html

`<meta name="theme-color">`, `apple-touch-icon` (180×180). Ссылку на
манифест и регистрацию SW инжектит плагин.

### 4. Бэкенд (одна правка)

В `app/api/main.py` зарегистрировать MIME-тип:
`mimetypes.add_type("application/manifest+json", ".webmanifest")` — иначе
на Windows `FileResponse` отдаст манифест как octet-stream. Catch-all
`GET /{full_path:path}` уже отдаёт реальные файлы из `dist` до фоллбека —
`sw.js` и манифест обслуживаются без других изменений.

### 5. HTTPS через Tailscale (инфраструктура, без кода)

PWA ставится только с secure context. На машине с сервером:

```bash
tailscale serve --bg 8000        # https://<host>.<tailnet>.ts.net → :8000
tailscale serve status           # проверить
```

Телефон в том же tailnet (приложение Tailscale, тот же аккаунт). Сертификат
валидный (Let's Encrypt через ts.net), ничего доверять вручную не надо.
HTTP-доступ по LAN продолжает работать как раньше.

## Ошибки и деградация

- Нет HTTPS (открыли по LAN-IP) → SW не регистрируется, установки нет,
  приложение работает как обычный сайт. Регистрация SW обёрнута так, что
  отказ молча игнорируется.
- Обновление фронта: `autoUpdate` перезапишет прекэш; API-запросы не
  кэшируются, поэтому рассинхрон фронт/бэк не дольше одной перезагрузки.
- Офлайн: оболочка откроется, API-запросы упадут как сейчас при потере
  сети — отдельного офлайн-UI не делаем (YAGNI).

## Проверка

1. `npm --prefix frontend run build` → в `dist` есть `manifest.webmanifest`,
   `sw.js`, иконки.
2. С телефона по `https://….ts.net`: Chrome предлагает установку; standalone
   запуск с иконки.
3. Регресс плеера в установленном PWA: перемотка (Range), переключение
   треков при погашенном экране, `?st=` стрим-токены.
4. Lighthouse PWA-аудит без ошибок installability.
5. `pytest -m unit` — бэкенд-правка тривиальна, но прогнать.
