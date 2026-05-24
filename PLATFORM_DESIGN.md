# MusiX → Smart Companion Platform · Design Document

**Дата:** 2026-05-13
**Брейнсторм session:** `.superpowers/brainstorm/410-1778623356/content/` (6 итераций мокапов)

---

## 1. Context

**Текущее состояние**: приложение `lyrics-search` (MusiX) сейчас — это **аналитический инструмент** для музыкальной библиотеки. Пять разделов (Home/Search/Recommend/Library/Player) организованы как **инструменты аналитика**: "поиск", "рекомендации", "статистика". Плеер был добавлен последним коммитом (28425b4) и пока живёт как ещё одна "вкладка".

**Проблема**: бэкенд готов на ~70% к полноценной музыкальной платформе (facts harvesting, similarity engine, CLAP audio embeddings, LLM-driven chat), но **UI этого не раскрывает**. Пользователь не получает погружения "в исполнителя и в песню", потому что архитектура экранов — инструментальная, а не story-driven.

**Цель**: трансформировать MusiX из "library analytics" в **"music platform for deep listening"**. Сохранить тактильный premium-character существующего скевоморфного UI, но добавить атмосферность и сделать каждую песню "open book" с фактами, прозрачным ранжированием похожих, LLM-обсуждением и автоплеем.

**Чем отличается от Spotify**: не социал и не daily mix, а **глубокое знание про эту конкретную песню**: факты из SongFacts, прозрачное ранжирование (видно ПОЧЕМУ трек похож), Sonic Sibling (ближайший CLAP-сосед из другой эпохи/артиста), LLM-объяснения.

---

## 2. Vision

> **"Listen smart"** — каждый трек = open book: факты, истории, related треки, lyrics-объяснения, поэтичные характеристики звука. AI помогает, но в центре — погружение.

**Архитектурный принцип**: **Artist as universe**. Клик по треку → попадаешь в "вселенную артиста" (Artist Atlas). Песня — это динамический контекст (Player screen), артист — статический контекст (Atlas).

**Scope**: **Full platform** = Smart Companion (Artist Atlas + Player + autoplay + transparent ranking + LLM-объяснения + Sonic Vibe + Sonic Sibling) **+** Home (Discovery Magazine) **+** Library Search (mode-aware semantic) **+** Library Stats (Sonic Map + catalog) **+** Recommendations (Prompt-to-Playlist + For You stream + Quick-Rate cold start) **+** Spotify-like MVP (Recently Played, Liked Songs, Custom Playlists CRUD, Manual Queue).

Это полное замещение текущего приложения, а не аддитивное расширение.

**Платформа PC-only** в этой итерации. Mobile — отдельный design pass позже.

---

## 3. Style Language: Hybrid v3

Гибрид двух направлений: **Atmospheric base** (C) + **Skeuomorphic depth** (A). Пропорция ~40/60 — атмосфера как акцент, скевоморф как tactile база.

### 3.1 Cвет
- **Base**: `linear-gradient(180deg, #161420 0%, #0c0a14 70%)` (близко к текущему `#0d0d10`)
- **Hue accents**: radial-gradients в углах:
  - Top-right: `rgba(124,91,255,0.22)` (purple, ~10% canvas)
  - Bottom-left: `rgba(255,120,200,0.08-0.10)` (pink)
- **Accent**: `oklch(60% 0.18 270)` (purple), `oklch(72% 0.13 75)` (amber для секондари)
- **Текст**: `#eeeef3` базовый, `rgba(238,238,243,0.6)` muted, `#d8ccff` accent text

### 3.2 Типографика
- **Body / UI**: `system-ui` / `Geist` sans-serif
- **Display / Quotes**: `'Noto Serif Display'`, Georgia, serif italic (для Sonic Vibe и LLM-фраз)
- **Labels / Mono**: `ui-monospace` / `JetBrains Mono` с letter-spacing 0.18–0.22em (CAPS labels)

### 3.3 Materials

**Panel (default)** — glass + skeuomorphic depth:
```css
background: linear-gradient(180deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.02) 60%);
backdrop-filter: blur(22px) saturate(1.1);
border: 1px solid rgba(255,255,255,0.07);
box-shadow:
  inset 0 1px 0 rgba(255,255,255,0.13),   /* top light edge */
  inset 0 -1px 0 rgba(0,0,0,0.28),         /* bottom dark edge */
  0 5px 20px rgba(0,0,0,0.3);              /* outer lift */
```

**CTA button** — gradient + 3D press:
```css
background: linear-gradient(180deg, oklch(72% 0.2 275) 0%, oklch(52% 0.24 282) 100%);
box-shadow:
  inset 0 1px 0 rgba(255,255,255,0.42),
  inset 0 -1px 0 rgba(0,0,0,0.4),
  0 8px 22px oklch(60% 0.18 270 / 0.5);
```

**Ask AI button** (special: gradient × glass):
```css
background:
  radial-gradient(ellipse at 25% 20%, rgba(255,235,200,0.32) 0%, transparent 60%),
  linear-gradient(180deg, oklch(70% 0.16 75 / 0.55) 0%, oklch(48% 0.18 75 / 0.4) 100%),
  rgba(255,255,255,0.02);
backdrop-filter: blur(20px) saturate(1.2);
animation: askGlow 3.8s infinite;
```

### 3.4 Анимации
- `coverBreath` 4.2s — лёгкое расширение outer shadow вокруг обложки (имитация "alive playback")
- `eq1-4` 0.75–1.05s — анимация EQ-баров рядом с "NOW PLAYING"
- `askGlow` 3.8s — лёгкое усиление glow на Ask AI кнопке
- `vinylSpin` — переиспользовать из существующего кода (есть в `frontend/index.html`)

---

## 4. Screens

### 4.1 Global navigation: Floating Icon Sidebar

Заменяет текущий sidebar (232px wide с label-ами) на **64px-узкую плавающую полосу иконок**:
- Без правой границы (нет "rigid column")
- Glass material фоном (наследует atmospheric gradient)
- Иконки: `⌂ HOME`, `🔍 SRCH`, `▣ LIB`, `📊 STAT`, `✨ REC`, `♫ PLAY`, `⚙ SET`
- Активный пункт в glass-капсуле с oklch purple glow
- Под иконкой — тонкий 8.5px моно-label (HOME/SRCH/LIB/STAT/REC/...)

**At the bottom of the sidebar — Now Playing pebble**:
- 40×40 круглая обложка текущего трека (если плеер активен)
- Лёгкая золотая обводка + breath-animation (slow pulse если playing)
- Hover/click → выезжает мини-floating panel справа от sidebar (~320×80):
  - Mini cover 60×60, title + artist (truncated)
  - Кнопки: `⏮ ⏯ ⏭` (3 cherry icons)
  - Scrubber: `1:24 ──●── 4:21`
  - Action: `↕ Open Player` → expand в полный Player screen
- Если no track playing → pebble показывает placeholder ("MusiX" mono label), без panel

**Global keyboard shortcuts** (работают везде, если focus не в input/textarea):
- `Space` — play/pause
- `→` / `←` — skip ±10s
- `Shift+→` / `Shift+←` — next/prev track
- `M` — mute toggle
- `L` — like current
- `D` — dislike current
- `/` — focus Search bar (если на Search screen, иначе noop)

### 4.2 Artist Atlas (revised L2)

**Когда открывается**: клик по любому артисту/треку в результатах поиска / library / related artists.

**Layout** (сверху вниз):
1. **Breadcrumb**: `LIBRARY / ARTISTS / RADIOHEAD` (10px моно, opacity 0.4)
2. **Hero row**: cover 138px + название (44px serif weight 300) + контекст (Oxford · alt-rock · 9 albums) + CTA `▶ Spin from here`
3. **Tab pills**: `Bio` (active) `Discography` `Facts ·27` `Related` `Eras`
4. **Bio panel** (full-width glass panel): абзац текста + source link `→ читать полностью`
5. **Discography rail**: горизонтальный scroll, карточки 138×138 (обложка + название + год · трекс)

**Behavior**:
- Click трека → начинает играть, mini-player активируется
- Click `▶ Spin from here` → запуск autoplay queue от seed-трека/артиста
- Mini-player снизу с кнопкой `↕ EXPAND TO PLAYER`

**Что НЕТ на этом экране**: lyrics, song facts, similar — это всё на Player screen.

### 4.3 Player screen (v6)

**Самый info-плотный экран платформы.** Открывается клик-ом на `↕ EXPAND TO PLAYER` в mini-bar или на иконке `♫ PLAY` в sidebar.

**Layout**:

```
┌─ [floating icon sidebar] ──────────────────────────────────────────┐
│  ← BACK TO RADIOHEAD                                                │
│                                                                      │
│  ┌─────────────────────────── HERO ──────────────────────────────┐ │
│  │ [Cover 180px      [NOW PLAYING + EQ bars]      ❝ vibe        │ │
│  │  3D tilt           Karma Police                  phrase ❞    │ │
│  │  breathing]        Radiohead                     — SONIC...]  │ │
│  │                    OK COMPUTER · 1997 · 4:21                  │ │
│  │ [📜 LYRICS]        ♥ Liked  ⨯ Skip  ↻ Loop                    │ │
│  │                    [✨ ASK AI ABOUT THIS SONG]                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌─ BODY: 2 columns ──────────────────────────────────────────────┐ │
│  │  LEFT (1.4fr): FACTS              RIGHT (1fr): NETWORK         │ │
│  │  ABOUT THIS SONG                  SONIC SIBLING (hero)         │ │
│  │  • Yorke's "karma cops"...        [J.D. — Atmosphere 1980]     │ │
│  │  • Recorded at St. Catherine's... ✨ "twin tonal landscape..."  │ │
│  │  • The line "for a minute..."     [▶ PLAY] · 87% sonic match   │ │
│  │  • The piano riff was originally..                              │ │
│  │  • Yorke originally intended...   [▽ SEE OTHER SIMILAR · 4]    │ │
│  │  → 2 more facts                                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ⏮ ▶ ⏭  1:24 ───●──── 4:21    🔊 ──●──    ≡                       │
└──────────────────────────────────────────────────────────────────────┘
```

**Tier-структура default state**:
- **Tier 1 (always visible)**: Hero (cover/meta/Sonic Vibe quote), Facts panel, Sonic Sibling card, Ask AI button
- **Tier 2 (one click away)**:
  - `📜 LYRICS` под обложкой → раскрывает lyrics-панель в левой колонке под Facts (inline accordion). Каждая строка `click → ✨ explain` мини-пилюля → popover с LLM-ответом.
  - `▽ SEE OTHER SIMILAR · 4 tracks` под Sonic Sibling → разворачивает список other similar (cover-mini + title + score%)
- **Tier 3 (AI chat)**: клик по `✨ ASK AI` → drawer выезжает справа (400px), с pre-filled context, suggested prompts, chat thread, input

**Sonic Vibe** оформлен как **pull-quote**:
- Большие ❝ ❞ в Noto Serif (56px, розовый glow)
- Центрированный 22px italic phrase
- Attribution `— SONIC VIBE · CLAP × LLM` справа внизу
- Без panel-chrome, только лёгкая hue-wash

**NOW PLAYING signals**:
- 4 анимированных EQ bars (eq1-4 keyframes)
- Label `NOW PLAYING` в oklch purple с text-shadow glow
- Обложка дышит (`coverBreath` 4.2s)

### 4.4 Home (Landing) — Discovery Magazine

Открывается при запуске приложения. Концепция: **discovery magazine, не библиотечный resume**. Это первый touchpoint, и он формирует identity "Listen smart".

**Layout** (сверху вниз):
```
┌──────────────────────────────────────────────────────────────┐
│ [floating sidebar]                                            │
│                                                                │
│ ┌─ HERO ROW (двойной блок) ─────────────────────────────────┐│
│ │ TODAY'S REDISCOVERY (60%)    │  FEATURED ARTIST (40%)      ││
│ │ ┌──── ┐                       │  [Cover stack 220x140]      ││
│ │ │COVER│ Track Title           │  ARTIST NAME                ││
│ │ │ XL  │ Artist Name           │  genre · X tracks · era     ││
│ │ │160px│ ❝ teaser fact ❞      │  "one-line bio summary..."   ││
│ │ └──── ┘ [▶ PLAY]  [+ queue]   │  [→ OPEN ATLAS]             ││
│ └────────────────────────────────────────────────────────────┘│
│                                                                │
│ ┌─ SHELF: Recently played ──────────────────── more →────────┐│
│ │ [■][■][■][■][■][■][■][■][■] (horizontal scroll)             ││
│ └────────────────────────────────────────────────────────────┘│
│ ┌─ SHELF: Your liked songs ─────────────────── more →────────┐│
│ │ [■][■][■][■][■][■][■][■][■]                                  ││
│ └────────────────────────────────────────────────────────────┘│
│ ┌─ SHELF: Try something different ──────────── more →────────┐│
│ │ [■][■][■][■][■][■][■][■][■]                                  ││
│ └────────────────────────────────────────────────────────────┘│
│ ┌─ CTA STRIP: ▶ Start "For You" personalized stream ─────────┐│
│ └────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────┘
```

**Hero — двойной блок**:
- **Left (60%) — Today's Rediscovery**: рандомный трек из библиотеки, который юзер давно не слушал (longest gap since last play по playback_history). Big 160px cover, title, artist, teaser-фраза из существующих SongFacts (первый интересный fact из таблицы, например *"Yorke wrote the lyrics after a bad encounter with paparazzi..."*). CTA `[▶ PLAY]` — immediately начинает + adds to queue.
- **Right (40%) — Featured Artist of the Day**: ротируемый артист (deterministic per-date hash). Cover stack из 3 albums, artist name, library stats ("5 tracks in your library, oldest 1971"), one-line bio summary. CTA `[→ OPEN ATLAS]` → Artist Atlas.

**Shelves under Hero** (horizontal-scroll rails):
1. **Recently played** — из `playback_history` (limit 12)
2. **Your liked songs** — из `track_reactions WHERE reaction='like'`
3. **Try something different** — top-pairs dissimilar entries (anti-similar to user's top-listened)

Каждая shelf-карточка ~120×120 cover + title + artist (2 lines truncated). Hover показывает quick-play overlay. Click cover → Player. Click artist text → Artist Atlas.

**Bottom CTA strip**: persistent `▶ Start For You stream` — главный entry point в personalized ranking без предварительной конфигурации. Если `playback_history` пустой → CTA меняется на `Start with Quick-Rate (2 min) → personalized stream`.

**Empty state** (свежая библиотека, no playback_history yet):
- Hero остаётся активным (использует random track + featured artist)
- Shelves заменяются единым "Quick-Rate session" prompt: "Tell us what you like in 2 minutes — rate 10 tracks → get a personalized starter mix."

**Backend dependencies**:
- `GET /library/rediscover?collection=...` — least-recently-played random pick (new)
- `GET /library/featured-artist?collection=...&date=YYYY-MM-DD` — deterministic daily rotation (new)
- `GET /playback/recent?limit=12` — recent plays (existing in MVP plan)
- `GET /library/liked-songs?limit=12` — liked tracks (existing in MVP plan)
- `GET /library/top-pairs` — dissimilar pairs (already exists, repurpose)

### 4.5 Library Search

Открывается через иконку `🔍` в sidebar. Замещает текущий SearchSection в `frontend/index.html`.

**Layout**:
```
┌──────────────────────────────────────────────────────────────┐
│ ┌─ MODE TOGGLE ──────────────────────────────────────────────┐│
│ │ [Название] [Текст песни] [Схожий звук] [Hybrid]            ││
│ └────────────────────────────────────────────────────────────┘│
│                                                                │
│ ┌─ SEARCH FIELD ─────────────────────────────────────────────┐│
│ │ 🔍 placeholder per mode...                                  ││
│ └────────────────────────────────────────────────────────────┘│
│   hint (per mode): "e.g. Yacht Holiday" / "about loneliness"  │
│                                                                │
│ recent: [chip] [chip] [chip] [chip]                            │
│ ▼ filters (collapsed)                                          │
│                                                                │
│ ┌─ RESULTS GRID (4 cols) ───────────────────────────────────┐ │
│ │ [Cover]    [Cover]    [Cover]    [Cover]                  │ │
│ │ Title      Title      Title      Title                    │ │
│ │ Artist     Artist     Artist     Artist                   │ │
│ │ ● 0.87     ● 0.82     ● 0.79     ● 0.74                  │ │
│ │ (hover → breakdown tooltip text/audio/bm25)               │ │
│ └────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

**Mode toggle** (segmented control в Hybrid v3 стиле, panel-v3 + active pill):
- **Название** — uses `GET /library/browse?q=...` (relevance scoring по title/artist/album, exact-match boost)
- **Текст песни** — uses `POST /search/ {mode: "text"}` (CLAP disabled, dense lyrics + BM25 only)
- **Схожий звук** — uses `POST /search/ {mode: "audio"}` (CLAP text-prompt → audio space)
- **Hybrid** — uses `POST /search/ {mode: "hybrid"}` (все три)

Каждый mode меняет placeholder, hint, и иконку (магнифай / pencil / waveform / chain).

**Filters row** (collapsed by default, expandable accordion с smooth transition):
- Genre multi-select (chips)
- Decade range slider
- Duration range slider (0:30 – 10:00)
- Liked-only toggle
- **Sonic class** multi-select (chips) — из user-curated taxonomy (§5.7). Доступно только если classifier натренирован; иначе hidden.
- **Sonic tags** multi-select (chips) — из adjective vocabulary; "match all selected tags" semantics

**Recent searches** — chip-row над field-ом, сохраняется в `localStorage` под ключом `recent_searches:<collection>`. Click chip = restore query + mode.

**Results grid**: 4-column responsive (adapts: 3 cols at <1100px), cards ~140px wide. Поля:
- Cover (90×90, rounded 8px)
- Title (1 line, truncated, 14px)
- Artist (1 line, 12px muted)
- Score badge `● 0.87` (только в semantic modes; в "Название" mode — score не показывается)
- Hover: card lift + reveal quick-play + quick-like icons; tooltip с breakdown bars (text / audio / bm25 contribution)

Click card → запускается Player (auto-start playback). Click artist name (отдельно от карточки) → Artist Atlas.

**Empty state**: chips с примерами per mode — `"rainy guitars"` / `"about regret"` / `"punchy drums"` / `"yacht holiday"`.

**Backend dependencies**:
- `GET /library/browse` (existing) — для Название mode
- `POST /search/` (existing, extended with `score_breakdown` per MVP Section 6) — для остальных modes

### 4.6 Library Stats — Sonic Map

Открывается через иконку `📊` в sidebar.

**Layout** (сверху вниз):
```
┌──────────────────────────────────────────────────────────────┐
│ ┌─ SONIC MAP ─────────────────────────────────────────────── ┐│
│ │ Color: [by genre ▼] | View: [scatter | clusters]           ││
│ │ ┌────────────────────────────────────────────────────────┐ ││
│ │ │       . . :⋆.⋆⋆⋆ ⋆ . .         <Canvas scatter>        │ ││
│ │ │    . . :⋆.⋆⋆⋆. ⋆ . .                                   │ ││
│ │ │    . . ⋆ ⋆ :⋆ . . .                                    │ ││
│ │ │    . . . . . . . . .  (~1500 dots)                     │ ││
│ │ │  liked = ⋆ (golden)    other = · (muted)               │ ││
│ │ └────────────────────────────────────────────────────────┘ ││
│ │ hover dot → tooltip [cover, title, artist, genre]          ││
│ │ click dot → Player                                          ││
│ └────────────────────────────────────────────────────────────┘│
│                                                                │
│ ┌─ KPI TILES (4 column row) ─────────────────────────────────┐│
│ │ 1480       286         1965–2024      14 genres            ││
│ │ tracks     artists     era            unique               ││
│ └────────────────────────────────────────────────────────────┘│
│                                                                │
│ ┌─ DECADES TIMELINE ─────────────────────────────────────────┐│
│ │ ████ ██████ █████████ █████████ ██████ █████ ███████        ││
│ │ 60s    70s    80s       90s     00s   10s   20s             ││
│ └────────────────────────────────────────────────────────────┘│
│                                                                │
│ ┌─ TOP GENRES (bar) ──┐ ┌─ TOP ARTISTS (list) ─┐              │
│ │ rock     ████████   │ │ 1. Radiohead    47   │              │
│ │ pop      █████      │ │ 2. Beatles      32   │              │
│ │ jazz     ████       │ │ ...                  │              │
│ └─────────────────────┘ └──────────────────────┘              │
└────────────────────────────────────────────────────────────────┘
```

**Sonic Map** — якорная фича. HTML5 `<canvas>` (2D context) scatter ~1500 точек. Каждая точка = трек, координаты x/y = UMAP проекция CLAP audio embedding.

**Color modes** (dropdown):
- **by genre** (default) — palette {rock=red, pop=pink, jazz=amber, electronic=cyan, ...}
- **by decade** — chronological gradient (1960s→2020s navy→amber)
- **by reaction** — liked=gold, disliked=red, neutral=muted

**Liked tracks** — слегка увеличенные точки с золотой обводкой (всегда поверх остальных).

**Interactions**:
- Hover dot → tooltip floating panel (cover thumb 60×60, title, artist, genre). Spatial index (grid bucket) для fast hit-test.
- Click dot → play track в Player + navigate to Player screen.
- Drag pan, scroll-wheel zoom (clamped 0.5–4x).
- Reset view button bottom-right corner.

**View toggle**:
- **Scatter** — все точки видны (default)
- **Clusters** — HDBSCAN кластеры из Sonic Descriptor Layer (§5.7), labels = user-curated names через cluster curator. Подсвечены полупрозрачными convex hulls с floating labels рядом с centroids. View disabled если classifier ещё не натренирован.

**KPI tiles** — 4 компактных `panel-v3` тайла. Из существующего `/library/stats`.

**Decades timeline** — bar chart, ширина column proportional to track count per decade.

**Top genres** — bar chart с labels. **Top artists** — нумерованный list. Click genre/artist → Search screen с pre-filled query в Название mode.

**Backend dependencies**:
- `GET /library/sonic-map?collection=...` — new endpoint. Возвращает `[{track_id, x, y, genre, year, reaction}]`. Computed once via UMAP at indexing completion, cached at `cache/sonic_map/<collection>.json`. Recomputed when library changes by >5% (new collection point count vs cached point count).
- `GET /library/stats` (existing) — KPI tiles, decades, top genres/artists.
- `GET /library/sonic-clusters?collection=...` — HDBSCAN cluster centroids + user-curated labels (см. §5.7). Empty list if curator не запускался.

### 4.7 Recommendations

Открывается через иконку `✨` в sidebar.

**Layout**:
```
┌──────────────────────────────────────────────────────────────┐
│ ┌─ MODE PICKER (3 cards row) ────────────────────────────────┐│
│ │ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        ││
│ │ │ ▶ FOR YOU    │ │ ✍ DESCRIBE   │ │ ⚡ QUICK-RATE │        ││
│ │ │ endless mix  │ │ a playlist   │ │ session       │        ││
│ │ │ from your    │ │ in words     │ │ rate 10 tracks│        ││
│ │ │ history      │ │              │ │ get a mix     │        ││
│ │ │ [▶ START]    │ │ [start]      │ │ [start]       │        ││
│ │ └──────────────┘ └──────────────┘ └──────────────┘        ││
│ └────────────────────────────────────────────────────────────┘│
│                                                                │
│ ┌─ ACTIVE PANEL (varies by selected mode) ───────────────────┐│
│ │ (e.g. Prompt-to-Playlist):                                  ││
│ │  ┌─────────────────────────────────────────────────────┐    ││
│ │  │ ✍ rainy afternoon, intricate guitars, melancholic   │    ││
│ │  └─────────────────────────────────────────────────────┘    ││
│ │  Length: [10 ▼] tracks    [✨ GENERATE PLAYLIST]            ││
│ │                                                              ││
│ │ ┌─ Generated playlist ─────────────────────────────────┐   ││
│ │ │ "rainy afternoon mix"                  ▶ Play all     │   ││
│ │ │ ✨ Why this set: "These tracks share a slow-burn      │   ││
│ │ │   tempo and finger-picked guitar palette..."          │   ││
│ │ │ ─────                                                  │   ││
│ │ │ 1. [■] Track Title — Artist  3:45    ● breakdown      │   ││
│ │ │ 2. [■] Track Title — Artist  4:21    ● breakdown      │   ││
│ │ │ ...                                                    │   ││
│ │ │ [💾 SAVE AS PLAYLIST]                                  │   ││
│ │ └────────────────────────────────────────────────────────┘   ││
│ └────────────────────────────────────────────────────────────┘│
│                                                                │
│ ┌─ SAVED RECOMMENDATIONS ─────────────────────────────────────┐│
│ │ history of generated playlists / saved For You snapshots    ││
│ └────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────┘
```

**Three primary modes** (cards в верхней row, click → switches active panel):

#### Mode 1: For You — Endless Personalized Stream
- One-click action: `[▶ START]` → играет всю библиотеку, отсортированную по personalized score, начиная с самого релевантного.
- **Algorithm**:
  ```
  user_vector = normalize(mean(liked_vectors) − 0.3 * mean(skipped_vectors))
  score(track)  = cosine(track.dense_vector, user_vector) − recency_penalty
  queue         = sorted(library, by=-score) filtered (not recently played)
  ```
- **Rationale chip**: каждый трек в For You queue имеет clickable `i` icon → popover "Почему этот трек: matches your usual sonic palette + lyrically near recent likes" (transparent ranking value prop). LLM lazy-generated, cached per `(track_id, user_vector_hash)`.
- Если playback_history пустой → CTA блокирован, активируется альтернативный CTA "Start with Quick-Rate first".
- Skip в For You queue marks track as `skipped` в current session и recomputes downstream ordering.

#### Mode 2: Prompt-to-Playlist
- Free-text field: "rainy afternoon, intricate guitars, melancholic"
- Length selector: 5 / 10 / 15 / 20 tracks
- `[✨ GENERATE PLAYLIST]` button.
- **Pipeline**:
  1. LLM разбивает prompt на параллельные queries (audio mood + text theme + genre hint)
  2. Hybrid search извлекает кандидатов из library
  3. LLM делает diversification + финальный set
  4. LLM пишет "Why this set"-пояснительную фразу
- Result: list треков с score breakdown + одна "Why this set" фраза (как Sonic Sibling LLM, в Noto Serif italic).
- Action `[💾 SAVE AS PLAYLIST]` → создаёт named playlist в Custom Playlists.

#### Mode 3: Quick-Rate Session (cold start)
- Опросная мини-сессия. Card-stack или wizard step.
- Каждая карточка: обложка + 30-сек snippet (autoplay middle of track) + кнопки `👍 норм` / `👎 не`.
- N карточек (default 10), sampled diversely (по жанрам / decades / sonic clusters).
- После N → `[✨ GENERATE PLAYLIST]` → playlist из похожих на 👍 + анти-похожих на 👎.
- **Side effect**: ratings сохраняются в `track_reactions` (как обычные likes/dislikes) → обогащают For You signal.

**Saved Recommendations** — внизу page, list of ранее сгенерированных playlists и сохранённых For You snapshots. Click → play или edit (если playlist saved).

**Backend dependencies**:
- `POST /recommend/prompt-to-playlist` — body: `{prompt, length, collection}` → response: `{title, why_phrase, tracks: [TrackHit with breakdown]}`
- `GET /recommend/for-you?collection=...&limit=...&offset=...` — paginated ranked traversal
- `GET /recommend/for-you/rationale?track_id=...&collection=...` — lazy per-track rationale (на `i` click)
- `POST /recommend/quick-rate/batch?collection=...&size=10` — returns 10 diverse candidate tracks
- `POST /recommend/quick-rate/finish` — body: `{ratings: [{track_id, rating}], collection}` → playlist response

### 4.8 Cross-cutting: where can user pick a track to play?

Track-picker — не отдельный экран, а UX-pattern, повторяемый везде где видны треки:

| Source | Action |
|--------|--------|
| Home hero card | `[▶ PLAY]` button — auto-start |
| Home shelves | Click cover anywhere — auto-start |
| Search results grid | Click card — auto-start |
| Stats Sonic Map | Click dot — auto-start |
| Recommendations (any mode) | `[▶ Play all]` / click track in list |
| Artist Atlas discog rail | Click cover — auto-start |
| Player screen Sonic Sibling / Other Similar | Click card — auto-start |
| Playlists detail | Click track row — auto-start from that position |
| Liked Songs view | Click track row — auto-start from that position |
| Keyboard | `Space` (resume), `Shift+→` (next in queue) |

Все эти actions делают одно: starts the track в Player + navigates to Player screen (либо просто triggers playback если уже в Player). Double-click на covers — same as single-click.

---

## 5. Features

### 5.1 Smart Companion core (новое)

#### `▶ Spin from here` — One play button / Autoplay queue
**Где**: Hero Artist Atlas (CTA), а также context-menu любого трека.
**Что делает**: запускает алгоритмический autoplay queue от seed-трека/артиста. Каждый следующий трек выбирается через гибридный CLAP+text similarity с user-feedback factor (likes boost, dislikes filter).
**Backend**: новый endpoint `GET /recommend/autoplay-queue?seed_track_id=...&collection=...&limit=20`. Использует Qdrant gibrid search, фильтрует disliked, кэширует на 24h.

#### Transparent ranking — Score breakdown
**Где**: в Other Similar list (когда expanded).
**Что делает**: каждый similar трек = title + score% + **breakdown bars** (розовый `text`, голубой `audio`). Юзер видит "лирика похожа 72%, звук похож 91%".
**Backend**: расширить `TrackHit.score_breakdown: ScoreBreakdown` в `app/domain/models.py`. Поля: `text_dense_score`, `text_bm25_score`, `audio_score`, `final_score`, `weights`. Модифицировать `_merge_hits()` в `app/services/search_service.py:236-294` чтобы сохранять intermediate scores.

#### LLM "Explain this lyric" + AI Chat
**Где**:
- Inline `✨ explain` на конкретной строке lyrics (line-level)
- Кнопка `✨ ASK AI ABOUT THIS SONG` под реакциями (song-level conversation)
**Что делает**:
- Line-level: popover с LLM-ответом, контекст = строка + facts + artist info
- Song-level: drawer справа с pre-filled context, suggested prompts ("What is this song about?", "Compare to Atmosphere by Joy Division", "What was Radiohead doing in 1997?"), chat history, input
**Backend**: переиспользует существующий `POST /chat/` endpoint (`app/api/routes/chat.py`) с двумя новыми system prompts: `LYRIC_EXPLAIN_PROMPT` и `SONG_DISCUSS_PROMPT`.

#### Sonic Vibe (новый ML-параметр #1)
**Где**: правая часть hero Player screen, как pull-quote.
**Что делает**: одна поэтичная фраза, генерируемая LLM из **Sonic Descriptor** (top-K adjective tags) + lyrics + facts. Пример: *"anxious hypnotic drift, with piano spirals that loop until they unravel"*.
**Зависимость**: требует Sonic Descriptor Layer (см. **§5.7**) — LLM получает на вход interpretable tags вроде `["anxious"=0.72, "atmospheric"=0.68, "piano-led"=0.61]`, а **не** raw CLAP vector (LLM не способен прочитать opaque embedding).
**Backend**: новое поле `TrackMetadata.audio_signature: str | None` (LLM-summary). Генерируется при индексации после descriptor computation, или lazy on first request. Endpoint: `GET /metadata/tracks/{track_id}/sonic-vibe`. Кэширован в `MetadataDB` (новая колонка в `songs` table).

#### Sonic Sibling (новый ML-параметр #2)
**Где**: правая колонка Player screen, верхняя hero-карточка.
**Что делает**: показывает ближайший трек по CLAP embedding **из другой эпохи и/или артиста**. Это превращает похожесть в discovery. Пример: твой *Karma Police* 1997 → *Atmosphere* Joy Division 1980 (87% sonic match).
**Зависимость**: для LLM-фразы "почему похож" нужны descriptors двух треков. LLM получает: `track_A.tags ∩ track_B.tags` (общие черты, например `["atmospheric", "drifting"]`) + diff (`track_A.sonic_class="indie melancholic"`, `track_B.sonic_class="post-punk gloom"`). Из этого LLM пишет фразу про "сходство в текстуре, разная эпоха".
**Backend**: новый endpoint `GET /recommend/sonic-sibling?track_id=...&collection=...`. Один Qdrant query с payload-фильтрами `artist != current_artist AND year_range != current_year_range`. Плюс LLM-фраза "почему похож" (одним вызовом, descriptors-driven). Кэширование 30 days.

### 5.2 Spotify-like MVP additions (выбраны)

#### Recently Played
**Backend**: новая SQLite-таблица `playback_history(collection_name, track_id, played_at, duration_played)`. Endpoints:
- `POST /playback/record` — записать начало проигрывания
- `GET /playback/recent?limit=20` — последние N
**Frontend**: блок на Home / Library с горизонтальным rail обложек.

#### Liked Songs view
**Backend**: использует существующий `track_reactions` в `MetadataDB`. Новый endpoint:
- `GET /library/liked-songs?collection=...&limit=...&offset=...`
**Frontend**: новый раздел внутри Library — карточки/строки треков с `reaction=='like'`.

#### Custom Playlists CRUD
**Backend**: новая SQLite модель `Playlist` и `PlaylistTrack`:
```sql
playlists(id, name, description, collection_name, created_at, cover_track_id)
playlist_tracks(playlist_id, track_id, position, added_at)
```
Endpoints:
- `POST /playlists` (create)
- `GET /playlists?collection=...` (list)
- `GET /playlists/{id}` (detail with tracks)
- `PUT /playlists/{id}` (rename/edit)
- `POST /playlists/{id}/tracks` (add track)
- `DELETE /playlists/{id}/tracks/{track_id}` (remove)
- `POST /playlists/{id}/reorder` (drag-drop)
- `DELETE /playlists/{id}` (delete)
**Frontend**: новый раздел в Library — список плейлистов + детальная страница + UI для добавления (context-menu "Add to playlist").

#### Manual Queue (Up Next override)
**Frontend-only feature** (нет нужды в backend storage — сессионное состояние).
**Что делает**: поверх autoplay queue, пользователь может вручную добавлять треки в "Up Next" через context-menu "Add to queue". Drag-drop reorder в Queue panel.
**UI**: панель queue выезжает из иконки `≡` в playback bar.

### 5.3 Prompt-to-Playlist

**Где**: Recommendations → Mode 2. Якорная фича раздела.
**Что делает**: пользователь описывает плейлист в свободной форме → LLM декомпозирует prompt на audio/text queries → hybrid search извлекает кандидатов → LLM делает diversification и пишет "why this set"-фразу.

**Pipeline detail**:
1. **Decompose**: LLM с system-prompt "extract sonic / lyrical / genre / era hints from this query" → структурированный JSON `{audio_prompt, text_prompt, genre_hint?, decade_hint?}`
2. **Search**: parallel `/search/ {mode: audio, query: audio_prompt}` + `{mode: text, query: text_prompt}` + payload-filter по genre/decade если есть hints. Берём top-50 от каждого, объединяем по track_id.
3. **Re-rank**: LLM получает объединённый набор + original prompt → выбирает top-N (`length` parameter), оптимизируя для diversity (не два подряд трека одного артиста / одной эпохи).
4. **Why-phrase**: LLM генерирует одну italic-фразу "что объединяет этот набор".

**Backend**: новый endpoint `POST /recommend/prompt-to-playlist`. Internal pipeline в `app/services/prompt_to_playlist_service.py` (new).

**Edge cases**:
- Empty library → graceful "No tracks match" + suggest indexing
- Слишком узкий prompt (single match) → still returns 1 track + "we only found one — try broadening"
- Слишком широкий prompt → fallback к top-genre + recency

### 5.4 For You — Personalized Stream

**Где**: Recommendations → Mode 1. Также CTA strip на Home bottom.
**Что делает**: one-click → играет всю библиотеку, отсортированную по personalized score from listening history + reactions.

**Algorithm**:
```python
liked_vectors    = qdrant.retrieve(track_ids=liked_track_ids, with_vectors=["dense"])
skipped_vectors  = qdrant.retrieve(track_ids=skipped_track_ids, with_vectors=["dense"])
user_vector      = normalize(mean(liked_vectors) − 0.3 * mean(skipped_vectors))

for track in library:
    base_score    = cosine(track.dense_vector, user_vector)
    recency_pen   = 0.5 * exp(-hours_since_played(track) / 24)
    score         = base_score − recency_pen

queue = sorted(library, by=-score)
queue = filter(queue, not_played_within(window=1h))
```

**Backend**:
- `GET /recommend/for-you?collection=...&limit=50&offset=0` — paginated ordered list
- `GET /recommend/for-you/rationale?track_id=...&collection=...` — lazy LLM call. **Inputs to LLM**: top-3 пересекающихся descriptor tags между current track и aggregated user-profile descriptors (computed как top tags по liked-треках). Plus shared sonic_class между current track и user's most-liked sonic_class. Из этого LLM пишет фразу: *"matches your usual lush + warm + acoustic palette; same Lo-fi indie cluster as 12 of your liked tracks."*
- `user_vector` cached in-memory per collection с TTL 1h. Invalidated на каждый `track_reactions` insert/update + каждый `playback_history` insert.
- `app/services/personalization_service.py` (new) — handles user_vector compute + cache + queue generation + aggregate descriptor profile.

**Skip behavior**: Если юзер skip-ает трек в For You queue:
- Track marked as `skipped` in current session
- Triggers user_vector recompute (если набралось >5 skips since last recompute)
- Downstream queue resorted

### 5.5 Quick-Rate Session (cold start)

**Где**: Recommendations → Mode 3. Также empty-state CTA на Home если no playback_history.
**Что делает**: 10-track snippet rating mini-session → generates starter playlist + populates initial user_vector signal.

**UX flow**:
1. User clicks `[Start Quick-Rate session]`
2. Modal/wizard opens, blocking sidebar (для focus)
3. Card 1 of 10: обложка 280×280, title, artist, autoplay 30-сек snippet (from middle of track, `audio currentTime = duration / 2`)
4. Two big buttons: `👍 норм` / `👎 не`. После клика — переход к следующей карточке (fade transition).
5. Progress indicator ("4 of 10") + skip button (rate as neutral).
6. После всех 10 — `[✨ GENERATE PLAYLIST]` button, generates playlist using same Prompt-to-Playlist pipeline + user_vector bootstrapped из ratings.

**Sampling strategy для batch**:
- Distribute по genre (один трек per top-5 genres, остальные random)
- Distribute по decade (минимум 1 трек из каждого присутствующего decade)
- Exclude already-rated tracks (track_reactions)

**Backend**:
- `POST /recommend/quick-rate/batch?collection=...&size=10` — returns 10 diverse candidate tracks
- `POST /recommend/quick-rate/finish` — body: `{ratings: [{track_id, rating}], collection}` → playlist response. Side-effect: writes ratings as `track_reactions`.

### 5.6 Sonic Map visualization (Stats core)

**Где**: Stats screen, основной блок сверху.
**Что делает**: 2D UMAP проекция CLAP audio embeddings всей библиотеки → interactive scatter, color-encoded.

**Backend**:
- `app/services/sonic_map_service.py` (new) — computes UMAP at indexing completion + on library change. Использует `umap-learn` (~150MB depend) или `pacmap` (lighter). Saves `cache/sonic_map/<collection>.json` со списком `{track_id, x, y, genre, year, reaction, sonic_class}`.
- `GET /library/sonic-map?collection=...` returns full point list (~1500 entries × ~80 bytes = ~120KB — fine для one-shot fetch).
- `GET /library/sonic-clusters?collection=...` — возвращает manually-curated cluster labels из **Sonic Descriptor Layer** (см. §5.7). НЕ k-means + LLM-naming автоматически — labels приходят из user-curated cluster taxonomy. Если curator ещё не запускался — endpoint возвращает empty list, и Sonic Map работает без cluster overlay (только scatter).

**Cluster overlay rendering** (когда labels доступны):
- Каждая точка окрашивается по своему `sonic_class` (из custom classifier)
- Convex hull или soft-blob fill за группами точек одного class (полупрозрачный, не блокирует точки)
- Class labels плавают рядом с centroid'ом каждой группы

**Frontend**:
- HTML5 canvas (2D context, не WebGL — overkill для 1500 точек)
- Spatial index (uniform grid, ~50×50 cells) для fast hover hit-test
- Smooth pan via mouse-drag, zoom via wheel
- Animated entrance (точки fade-in поочерёдно)
- Color-mode toggle: `by sonic_class` (default if available, иначе `by genre`) / `by decade` / `by reaction`

### 5.7 Sonic Descriptor Layer (foundation for §5.1 Vibe/Sibling, §5.4 rationale, §5.6 clusters)

**Проблема**: CLAP даёт 512-dim opaque vector. LLM не может прочитать его и описать словами. Все features которые делают взаимодействие "Listen smart" (Sonic Vibe фраза, Sonic Sibling "почему похож", For You rationale, cluster labels на Sonic Map) — требуют **interpretable descriptors**, не вектора.

**Решение**: Sonic Descriptor Layer — промежуточный слой между CLAP embedding и LLM/UI. Состоит из **двух independent моделей** на одном входе (CLAP vector):

#### 5.7.1 Track 1: Prompt-Probing Tags (zero-shot)

**Что**: получаем top-K descriptive tags для каждого трека через CLAP cross-modal text-encoder.

**Как работает**:
```python
# offline (once per app start)
prompt_vocab = ["a sad song", "punchy drums", "lush strings", "lo-fi production",
                "atmospheric ambient", "acoustic guitar", "warm vocals", ...]  # ~30-50 prompts
prompt_embeddings = CLAP.encode_text(prompt_vocab)  # shape: (N_prompts, 512)

# per track at indexing
track_emb = CLAP.encode_audio(audio)  # shape: (512,)
sims = cosine(track_emb, prompt_embeddings)  # shape: (N_prompts,)
top_K = top-K by sims  # e.g. K=5
# Result: [("anxious", 0.72), ("atmospheric", 0.68), ("piano-led", 0.61), ...]
```

**Преимущества**:
- Zero training — работает сразу
- Iterable — можно менять prompt vocabulary без re-индексации (только пересчитать sims)
- Adjective-уровень (для Sonic Vibe phrasing)

**Vocabulary file**: `cache/sonic_prompts.json` — список prompts. Стартовый словарь предлагается ~30-50 prompts, организованных по группам:
- **Energy**: explosive, driving, mid-tempo, languid, ambient, drone
- **Valence**: euphoric, hopeful, neutral, melancholy, anxious, dark
- **Density**: minimal, sparse, lush, wall-of-sound
- **Texture**: clean, warm, raw, lo-fi, polished, saturated, crystalline
- **Instrumentation**: acoustic guitar, piano-led, orchestral, synth-heavy, electronic
- **Vocal**: instrumental, sparse vocals, lead vocals prominent, harmony-rich
- **Rhythm**: 4/4 steady, swung, syncopated, free-time, motorik
- **Era hints**: vintage, contemporary, 80s synth, 90s indie, post-punk

Юзер может редактировать словарь — `cache/sonic_prompts.json` watchable, пересчёт sims триггерится сам.

#### 5.7.2 Track 2: Custom Sonic Class Classifier

**Что**: твоя личная таксономия music classes — обученный MLP, который выдаёт one-of-N cluster name по CLAP vector.

**Как обучается** (one-time setup, опционально повторяется):
1. **Cluster discovery** (`sonic_descriptor_service.cluster_library()`):
   - Clustering CLAP-vectors всей библиотеки. Default — `HDBSCAN` (auto-detects k, не требует guess; альтернатива — hierarchical с silhouette tuning).
   - Output: assignment `track_id → cluster_id`, + representative tracks per cluster (top-5 closest to centroid).
2. **Cluster curator tool** — UI или CLI, в котором юзер:
   - Видит представителей каждого cluster (cover + title + artist + play button)
   - Слушает 30-сек snippet
   - Даёт имя кластеру (например "Lo-fi indie", "Cinematic drone")
   - Может merge'ить близкие clusters / разделить confused ones
   - Saves labels → `cache/sonic_clusters/<collection>_labels.json`
3. **Train classifier** (`sonic_descriptor_service.train_classifier()`):
   - Input: CLAP vectors треков с known cluster labels (после curator)
   - sklearn `MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200)` — простая 2-layer MLP, ~30 sec training на 1500 треков
   - Output: trained model → `cache/sonic_classifier/<collection>.joblib`
4. **Apply at indexing**:
   - Для каждого нового трека: `model.predict_proba(clap_vector)` → softmax над all classes
   - Top-1 class = `sonic_class`, его score = `sonic_class_confidence`

**Преимущества**:
- Personal taxonomy — твоя карта music, не generic genres
- Cluster-aware visualization for Sonic Map
- Discrete labels for Search facets ("show me Lo-fi indie only")
- Stable assignments (для curator tool — re-label не требует re-clustering всей библиотеки)

#### 5.7.3 Combined output

Финальный record для трека (хранится в MetadataDB):
```python
{
    "sonic_tags_json": [
        {"tag": "anxious",     "score": 0.72},
        {"tag": "atmospheric", "score": 0.68},
        {"tag": "piano-led",   "score": 0.61},
        {"tag": "warm",        "score": 0.54},
        {"tag": "mid-tempo",   "score": 0.49}
    ],
    "sonic_class": "Lo-fi indie",
    "sonic_class_confidence": 0.81
}
```

**Consumers**:
- **Sonic Vibe** (§5.1) — top tags + facts + lyrics → LLM phrase
- **Sonic Sibling** (§5.1) — common tags + class diff → LLM "почему похож"
- **For You rationale** (§5.4) — aggregated user descriptor profile → LLM personalized phrase
- **Sonic Map** (§5.6) — `sonic_class` for color/overlay, tags as hover-tooltip
- **Search facets** (§4.5) — filter by `sonic_class` или присутствие конкретного `tag`

#### 5.7.4 Cluster Curator Tool

**MVP delivery**: CLI script `scripts/cluster_curator.py` (interactive prompts), достаточен для one-time setup.

**Future enhancement**: встроенный screen в Stats — `Stats → Curate Clusters` tab. Показывает cluster grid (один блок на cluster: covers grid + name input + merge/split actions + play button per representative). Будет реализован после MVP когда core platform работает.

**Backend endpoints для curator** (used by both CLI and future UI):
- `POST /library/cluster-discovery?collection=...` — запускает HDBSCAN, returns clusters + representatives. Long-running, returns job_id.
- `GET /library/cluster-jobs/{job_id}` — статус discovery
- `GET /library/clusters/representatives?collection=...` — listing of clusters + their top tracks (после discovery)
- `POST /library/clusters/labels` — сохранить mapping `cluster_id → label`
- `POST /library/sonic-classifier/train?collection=...` — train MLP, returns job_id
- `GET /library/sonic-classifier/status?collection=...` — `untrained` / `training` / `ready` + last training date

#### 5.7.5 Empty/incremental states

- **Свежая библиотека, classifier не натренирован**:
  - Sonic Vibe фраза генерируется только из prompt-probing tags (LLM получает `["anxious", "atmospheric", ...]` без cluster context). Quality OK.
  - Sonic Sibling — "почему похож" фраза генерируется только из common tags, без class diff.
  - For You rationale — только descriptor tags aggregated, без class match.
  - Sonic Map работает в "by genre" / "by decade" color mode (cluster mode disabled).
- **Pacrtially curated** (некоторые clusters labeled, classifier trained):
  - All features работают, но некоторые tracks могут иметь `sonic_class=null` если confidence < threshold (e.g. 0.4).
- **Fully curated**:
  - Full functionality.

---

## 6. Backend Changes

### 6.1 New / Updated endpoints

| Endpoint | Method | File | Покрывает |
|----------|--------|------|-----------|
| `/artists/{slug}` | GET | new route file `app/api/routes/artists.py` | Artist universe aggregate (bio, discog, facts, related). AudioDB-enriched fields (optional): `mood`, `country_code`, `country`, `label`, `cutout_path`, `thumb_path`, `audiodb_mbid`. |
| `/recommend/autoplay-queue` | POST | `app/api/routes/recommend.py` (new) | Spin from here |
| `/recommend/sonic-sibling` | GET | `app/api/routes/recommend.py` | Sonic Sibling |
| `/metadata/tracks/{id}/sonic-vibe` | GET | `app/api/routes/metadata.py` | Sonic Vibe (lazy LLM) |
| `/search/` (modified) | POST | `app/api/routes/search.py` | Add `score_breakdown` to response |
| `/playback/record` | POST | `app/api/routes/playback.py` (new) | Recently Played |
| `/playback/recent` | GET | `app/api/routes/playback.py` | Deduped playback events by track_id (latest first), with play_count (non-skipped). limit=1..200. |
| `/library/albums` | GET | `app/api/routes/library.py` | Albums grouped from Qdrant payload (majority-vote primary artist + feat list + year/year_range + top genres + embedded tracks). Sort: alphabetical/year_desc/year_asc/track_count_desc. |
| `/library/liked-songs` | GET | `app/api/routes/library.py` | Tracks with reaction='like' in collection, ordered by liked_at DESC, enriched via Qdrant payload. |
| `/library/listening-stats` | GET | `app/api/routes/library.py` | Total seconds listened, since (first play), top track + top artist + peak hour (excludes skipped_early plays). lang=en\|ru. |
| `/playlists/*` | various | `app/api/routes/playlists.py` (new) | Playlists CRUD |
| `/chat/` (modified) | POST | `app/api/routes/chat.py` | Add `LYRIC_EXPLAIN` and `SONG_DISCUSS` modes |
| `/library/sonic-map` | GET | `app/api/routes/library.py` | Sonic Map (Stats) |
| `/library/sonic-clusters` | GET | `app/api/routes/library.py` | Optional cluster names |
| `/library/rediscover` | GET | `app/api/routes/library.py` | Today's Rediscovery (Home Hero) |
| `/library/featured-artist` | GET | `app/api/routes/library.py` | Featured Artist of the Day (Home Hero) |
| `/recommend/prompt-to-playlist` | POST | `app/api/routes/recommend.py` | Prompt-to-Playlist |
| `/recommend/for-you` | GET | `app/api/routes/recommend.py` | Personalized stream |
| `/recommend/for-you/rationale` | GET | `app/api/routes/recommend.py` | Lazy per-track rationale |
| `/recommend/quick-rate/batch` | POST | `app/api/routes/recommend.py` | Quick-Rate session start |
| `/recommend/quick-rate/finish` | POST | `app/api/routes/recommend.py` | Quick-Rate finish + generate |
| `/recommend/snapshots` | POST | `app/api/routes/recommend.py` | Save generated playlist snapshot |
| `/recommend/snapshots` | GET | `app/api/routes/recommend.py` | List saved snapshots |
| `/recommend/snapshots/{id}` | GET | `app/api/routes/recommend.py` | Get snapshot detail |
| `/recommend/snapshots/{id}` | DELETE | `app/api/routes/recommend.py` | Delete snapshot |
| `/library/sonic-descriptor/{track_id}` | GET | `app/api/routes/library.py` | Per-track tags + sonic_class (lazy compute if missing) |
| `/library/cluster-discovery` | POST | `app/api/routes/library.py` | Trigger HDBSCAN on collection (returns job_id) |
| `/library/cluster-jobs/{job_id}` | GET | `app/api/routes/library.py` | Cluster discovery job status |
| `/library/clusters/representatives` | GET | `app/api/routes/library.py` | Cluster grid (id → top tracks) for curator |
| `/library/clusters/labels` | POST | `app/api/routes/library.py` | Persist cluster_id → label mapping |
| `/library/sonic-classifier/train` | POST | `app/api/routes/library.py` | Train MLP on labeled clusters (returns job_id) |
| `/library/sonic-classifier/status` | GET | `app/api/routes/library.py` | Classifier readiness state |
| `/library/sonic-prompts` | GET | `app/api/routes/library.py` | Current prompt vocabulary |
| `/library/sonic-prompts` | PUT | `app/api/routes/library.py` | Update vocabulary, triggers re-tagging |

### 6.2 Updated models (`app/domain/models.py`)

```python
class ScoreBreakdown(BaseModel):
    text_dense_score: float = 0.0
    text_bm25_score: float = 0.0
    audio_score: float = 0.0
    final_score: float
    weights: dict[str, float] = {"text": 0.5, "audio": 0.5}

class TrackHit(BaseModel):
    track: TrackMetadata
    score: float
    matched_on: Literal["lyrics", "audio", "hybrid"]
    score_breakdown: Optional[ScoreBreakdown] = None  # NEW
    lyrics: str | None = None
    artist_facts: str | None = None
    song_facts: str | None = None

class TrackMetadata(BaseModel):
    # ... existing fields ...
    audio_signature: str | None = None  # NEW (Sonic Vibe LLM phrase)
```

### 6.3 New services

- `app/services/sonic_vibe_service.py` — генерация LLM-фразы (CLAP vector + facts → italic phrase)
- `app/services/playlist_service.py` — CRUD для плейлистов
- `app/services/playback_history_service.py` — запись и retrieval play history
- `app/services/sonic_map_service.py` — UMAP computation + cache management (`cache/sonic_map/<collection>.json`)
- `app/services/personalization_service.py` — user_vector compute (mean liked − 0.3 × mean skipped), in-memory cache TTL 1h, For You queue generation
- `app/services/prompt_to_playlist_service.py` — LLM-driven playlist generation (decompose → search → re-rank → why-phrase)
- `app/services/quick_rate_service.py` — diversity-sampled batch creation + post-rating playlist gen
- `app/services/sonic_descriptor_service.py` — CLAP-prompt-probing (zero-shot tags) + HDBSCAN clustering + sklearn MLP train/predict. Управляет `cache/sonic_prompts.json`, `cache/sonic_classifier/<col>.joblib`, `cache/sonic_clusters/<col>_labels.json`. Per-track output stored в SQLite (`songs.sonic_tags_json`, `songs.sonic_class`).
- `scripts/cluster_curator.py` (CLI tool, не сервис) — interactive cluster labeling. Запускает discovery → показывает clusters + представителей → принимает labels от пользователя → инициирует training.

### 6.4 MusicBrainz: scaffolded, populating deferred

**Status**: MusicBrainz API нестабилен — на больших библиотеках matching конкретной записи неконсистентен, samples-relations плохо populated. Поэтому активное обогащение откладывается. Но **концепция полей сохраняется** в schema, чтобы будущий harvesting не требовал миграции.

**Что делаем сейчас**:
1. Оставить `app/services/_WIP_musicbraniz_search.py` отключённым (no Phase 1 work).
2. Добавить в `MetadataDB` пустые scaffold-колонки/таблицы для будущего использования:
   ```sql
   -- songs: дополнительные nullable поля
   ALTER TABLE songs ADD COLUMN producers TEXT;       -- JSON list or null
   ALTER TABLE songs ADD COLUMN label TEXT;           -- null by default
   ALTER TABLE songs ADD COLUMN samples_json TEXT;    -- JSON list of {sampled_track_id?, raw_text} or null

   -- artists: scaffolded
   ALTER TABLE artists ADD COLUMN mbid TEXT;          -- MusicBrainz ID for future re-lookup

   -- artists: AudioDB enrichment (out-of-band ship 2026-05-20 — see Phase 6 status block)
   ALTER TABLE artists ADD COLUMN bio TEXT;            -- AudioDB biography (seeds artist_bio AI task)
   ALTER TABLE artists ADD COLUMN mood TEXT;           -- AudioDB mood tag
   ALTER TABLE artists ADD COLUMN country_code TEXT;   -- ISO country code
   ALTER TABLE artists ADD COLUMN country TEXT;        -- Human-readable country
   ALTER TABLE artists ADD COLUMN label TEXT;          -- AudioDB record label
   ALTER TABLE artists ADD COLUMN cutout_path TEXT;    -- Local cached cutout image path under /covers/artists/
   ALTER TABLE artists ADD COLUMN thumb_path TEXT;     -- Local cached thumbnail path under /covers/artists/
   ALTER TABLE artists ADD COLUMN audiodb_mbid TEXT;   -- MBID reported by AudioDB
   ```
3. В Artist Atlas UI и Player Facts panel: рендеринг producers/samples блоков **только if non-null** (graceful empty state — секция просто отсутствует).
4. Все backend endpoints возвращают эти поля как nullable — frontend не делает на них hard dependency.

**Что НЕ делаем**:
- Активный MusicBrainz API matching на этапе indexing
- Парсинг samples-of связей
- Сетка producers / labels в Artist Atlas (Hero context остаётся короткий: city · genre · album count, без producer list)

**Future migration path** (когда API стабилизируется или появится альтернативный источник):
- Включить parser → стабилизировать на test-batch
- Backfill script `app/services/_WIP_musicbraniz_search.py:backfill_all()` — fill scaffold columns по существующей библиотеке
- Frontend сам подхватит данные без code-changes (graceful conditional rendering уже на месте)

### 6.5 MetadataDB schema additions

```sql
-- New columns on songs
ALTER TABLE songs ADD COLUMN audio_signature TEXT;       -- Sonic Vibe LLM phrase cache
ALTER TABLE songs ADD COLUMN sonic_tags_json TEXT;       -- top-K {tag, score} from prompt-probing
ALTER TABLE songs ADD COLUMN sonic_class TEXT;           -- predicted cluster label
ALTER TABLE songs ADD COLUMN sonic_class_confidence REAL; -- 0..1

-- New tables
CREATE TABLE playback_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  collection_name TEXT NOT NULL,
  track_id TEXT NOT NULL,
  played_at REAL NOT NULL,
  duration_played REAL DEFAULT 0
);
CREATE INDEX idx_playback_recent ON playback_history (collection_name, played_at DESC);

CREATE TABLE playlists (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  collection_name TEXT NOT NULL,
  created_at REAL NOT NULL,
  cover_track_id TEXT
);

CREATE TABLE playlist_tracks (
  playlist_id TEXT NOT NULL,
  track_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  added_at REAL NOT NULL,
  PRIMARY KEY (playlist_id, track_id),
  FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE CASCADE
);

-- Cache table for quick-rate / for-you generated playlists (saved snapshots)
CREATE TABLE recommendation_snapshots (
  id TEXT PRIMARY KEY,
  collection_name TEXT NOT NULL,
  source TEXT NOT NULL,         -- 'prompt' | 'for_you' | 'quick_rate'
  prompt TEXT,                  -- original prompt (если source='prompt')
  why_phrase TEXT,              -- LLM why-this-set фраза
  track_ids_json TEXT NOT NULL, -- JSON array of track_ids
  created_at REAL NOT NULL
);
```

### 6.6 New file system caches

- `cache/sonic_map/<collection>.json` — UMAP scatter (one-time compute per collection, re-computed when track count drifts >5%)
- `cache/personalization/<collection>.json` — user_vector snapshot + computed_at (TTL 1h, also invalidated on reactions/playback changes)
- `cache/sonic_prompts.json` — adjective vocabulary для prompt-probing (editable by user). Шаблон стартового vocabulary поставляется при первой инициализации.
- `cache/sonic_prompts_embeddings.npy` — pre-computed CLAP text embeddings для prompts (re-compute если vocabulary меняется)
- `cache/sonic_clusters/<collection>_assignments.json` — HDBSCAN result: `{track_id: cluster_id, ...}`
- `cache/sonic_clusters/<collection>_labels.json` — user-assigned labels: `{cluster_id: "Lo-fi indie", ...}` (persisted from curator)
- `cache/sonic_classifier/<collection>.joblib` — trained sklearn MLP (predicts sonic_class from CLAP vector)
- `cache/sonic_classifier/<collection>_meta.json` — training metadata (training date, accuracy on held-out, class list)

### 6.7 New Python dependencies

- `umap-learn` (~150MB indirect via numpy/scikit-learn — уже частично) **OR** `pacmap` (~lighter) для Sonic Map UMAP projection. Choose `umap-learn` для quality, `pacmap` для cold-start speed.
- `hdbscan` (~30MB) — для cluster discovery в Sonic Descriptor Layer (auto k-detection, не требует guess как у k-means).
- Existing `scikit-learn` уже доступен — use it для MLPClassifier (custom sonic classifier).
- Existing `numpy` — для cosine-similarity bulk operations в prompt-probing.

---

## 7. Frontend Changes

### 7.1 Critical file

`frontend/index.html` — single-file React via in-browser Babel. **ВСЕ изменения тут**, без разделения на отдельные файлы (так требует текущая архитектура).

### 7.2 Component changes

**Modified**:
- `App` (lines ~4957–5145) — добавить новые routes (artist atlas, liked songs, playlists)
- `Sidebar` — заменить на `<FloatingIconNav>` (64px, без border, glass)
- `PlayerSection` (lines 4487–4954) — рестрактурировать на v6 layout
- `SearchSection` — клик на трек → перейти на Artist Atlas (не Detail Panel)

**New components** (внутри `index.html`):

**Core (Smart Companion)**:
- `ArtistAtlasSection` — hero + tabs (Bio/Discog/Facts/Related/Eras) + content + mini-player
- `AIChatDrawer` — выезжающая правая панель с suggested prompts, history, input
- `LyricsPanel` — collapsible панель с lyrics + click-to-explain
- `SonicSiblingCard` — янтарная hero-карточка с LLM-фразой
- `SonicVibeQuote` — pull-quote с декоративными ❝❞
- `BreakdownBars` — две цветные мини-полоски (text + audio)
- `PlaylistsSection` — CRUD UI
- `LikedSongsView` — фильтр reactions=='like'
- `RecentlyPlayedRail` — горизонтальный rail
- `QueuePanel` — drag-drop Up Next

**Home (Discovery Magazine)**:
- `HomeSection` — основной layout (Hero row + shelves + CTA strip)
- `HeroRediscoveryCard` — левый Hero блок с teaser-фактом
- `HeroFeaturedArtistCard` — правый Hero блок с cover stack + bio summary
- `Shelf` — generic horizontal-scroll rail (используется для Recently played / Liked / Try different)
- `ForYouCTAStrip` — bottom-of-Home gradient bar с одним button

**Search redesign**:
- `SearchSectionV2` — replaces existing SearchSection
- `SearchModeToggle` — segmented control (Название / Текст / Звук / Hybrid)
- `SearchFiltersAccordion` — collapsible filters row (genre / decade / duration / liked)
- `SearchResultsGrid` — 4-column responsive grid с hover-breakdown tooltip
- `RecentSearchesChips` — chip-row из localStorage

**Stats (Sonic Map + KPI)**:
- `StatsSection` — главный layout
- `SonicMapCanvas` — HTML5 canvas с pan / zoom / hover / click (uses spatial-grid hit-test)
- `SonicMapTooltip` — floating cover+title tooltip on hover
- `SonicMapControls` — color-mode dropdown + view toggle (scatter/clusters)
- `KPITiles` — 4-column compact panel-v3 tiles
- `DecadesTimeline` — horizontal bar chart (SVG или CSS bars)
- `TopGenresBar` — vertical bars list
- `TopArtistsList` — нумерованный list с counts

**Recommendations**:
- `RecommendationsSection` — главный layout (mode picker + active panel + saved snapshots)
- `RecommendModePicker` — 3 cards row (For You / Prompt / Quick-Rate)
- `ForYouPanel` — endless queue + rationale popover
- `PromptToPlaylistPanel` — text input + length selector + generate + result list + save action
- `WhyThisSetQuote` — italic pull-quote (style like Sonic Vibe)
- `QuickRateSessionWizard` — card-stack rating (modal overlay)
- `QuickRateCard` — single track snippet + 👍/👎 buttons
- `RecommendationSnapshotsList` — saved generated playlists list

**Sidebar enhancements**:
- `FloatingIconNav` (replaces current sidebar) — иконки + active glow
- `NowPlayingPebble` — sidebar bottom playback indicator (40×40 cover circle)
- `MiniPlaybackPopout` — floating ~320×80 panel from pebble hover (cover/title/⏮⏯⏭/scrubber/expand)
- `GlobalKeyboardShortcuts` — invisible global key listener (hook)

### 7.3 Style updates (CSS)

Добавить в `<style>` блок:
- Все `.h3-*` / `.v3-*` / `.pv6-*` классы из мокапов
- Keyframes: `eq1`–`eq4`, `coverBreath`, `askGlow`
- Materials: `.panel-v3`, `.cta-v3`, `.pill-v3`, `.ask-ai-btn`, `.vibe-quote`

Сохранить существующие `.ske-*` классы — они теперь base layer для tactile depth.

### 7.4 Reactions row update

Текущие like/dislike pills (`frontend/index.html` PlayerSection) → расширить:
- `♥ Liked` / `⨯ Skip` / `↻ Loop` (pills)
- Под ними отдельной строкой: `✨ ASK AI ABOUT THIS SONG` (gradient × glass)

### 7.5 useChatHistory hook

Существующий `useChatHistory(collectionName)` (для chat в SearchSection) — переиспользовать в AIChatDrawer но с **per-song context**: ключ `chat_history:song:${track_id}` вместо `chat_history:${collectionName}`.

---

> **Refactor (2026-05-23)**: Legacy packages `file_processor/`, `search_engine/`,
> and the `app/existing/` facade were consolidated into the `app/` tree:
>
> - `app/indexing/` — 5 modules from `file_processor/utils.py` (metadata_readers,
>   cover_art, lyrics_fetchers, audio_optimization, folder_scanner)
> - `app/resources/lyrics_search_engine.py` — canonical search-side class (was
>   half of `search_engine.main.LyricsDB`); ModelRegistry is now the sole entry
>   point for text + CLAP loading
> - `app/resources/qdrant_filters.py`, `qdrant_payload.py`, `clap_features.py` —
>   the 3 mixed-concern halves of the old `search_engine/utils.py`
> - `app/services/indexing_service.py` — orchestrates folder scan → encode →
>   upsert (replaces FileProcessor + LyricsDB.fit pipeline)
>
> See `refactor/legacy-to-app-modules` branch for the 6-commit incremental
> migration.

## 8. Critical Files Reference

| Файл | Что меняется |
|------|--------------|
| `frontend/index.html` | Основная UI-перестройка (sidebar, ArtistAtlas, Player v6, AIChatDrawer, и все остальные новые компоненты + styles) |
| `app/api/routes/search.py` | TrackHit с score_breakdown |
| `app/api/routes/chat.py` | LYRIC_EXPLAIN_PROMPT, SONG_DISCUSS_PROMPT |
| `app/api/routes/metadata.py` | sonic-vibe endpoint |
| `app/api/routes/library.py` | liked-songs endpoint |
| `app/api/routes/artists.py` (new) | Artist universe aggregate |
| `app/api/routes/recommend.py` (new) | autoplay-queue, sonic-sibling |
| `app/api/routes/playlists.py` (new) | Playlists CRUD |
| `app/api/routes/playback.py` (new) | Playback history |
| `app/domain/models.py` | ScoreBreakdown, audio_signature |
| `app/services/search_service.py:236-294` | _merge_hits возвращает breakdown |
| `app/services/sonic_vibe_service.py` (new) | LLM-фраза generation |
| `app/services/playlist_service.py` (new) | Playlists business logic |
| `app/services/playback_history_service.py` (new) | History recording |
| `app/services/sonic_map_service.py` (new) | UMAP computation + cache management |
| `app/services/personalization_service.py` (new) | user_vector compute + For You queue |
| `app/services/prompt_to_playlist_service.py` (new) | LLM-driven playlist generation pipeline |
| `app/services/quick_rate_service.py` (new) | Diversity-sampled batch + post-rate generation |
| `app/services/sonic_descriptor_service.py` (new) | Prompt-probing + HDBSCAN + MLP train/predict; populates `songs.sonic_tags_json` / `sonic_class` |
| `scripts/cluster_curator.py` (new, CLI) | Interactive cluster labeling tool (one-time setup, future UI in Stats) |
| `app/services/_WIP_musicbraniz_search.py` | Оставить отключённым; future re-enable когда API/source стабилизируется |
| `app/resources/metadata_db.py` | New tables (playback_history, playlists, playlist_tracks, recommendation_snapshots) + audio_signature column |
| `app/main.py` | Register new routes (`artists`, `recommend`, `playlists`, `playback`) |
| `cache/sonic_map/<collection>.json` (new) | UMAP scatter cache |
| `cache/personalization/<collection>.json` (new) | user_vector cache |
| `cache/sonic_prompts.json` (new) | Editable prompt vocabulary (adjective set) |
| `cache/sonic_prompts_embeddings.npy` (new) | Pre-computed CLAP text embeddings for prompts |
| `cache/sonic_clusters/<collection>_assignments.json` (new) | HDBSCAN cluster assignments |
| `cache/sonic_clusters/<collection>_labels.json` (new) | User-curated cluster labels |
| `cache/sonic_classifier/<collection>.joblib` (new) | Trained sklearn MLP classifier |

---

## 9. Build Sequence

Рекомендуемый порядок имплементации (каждый шаг можно проверить независимо).

> ### Status snapshot (updated 2026-05-16)
>
> | Phase | Status | Reference |
> |---|---|---|
> | **1**  Backend foundations            | ✅ Shipped (Plan 3)                       | `docs/superpowers/plans/2026-05-14-plan-3-backend-foundations.md` |
> | **1b** Additional backend services    | 🛠 **1b.2 partial** — `/library/rediscover` + `/library/featured-artist` shipped (Home plan); For-You uses placeholder `/recommend/for-you-seed` (full personalization 1b.3 still pending). 1b.1/1b.4/1b.5/1b.6 not started | `docs/superpowers/plans/2026-05-24-home-discovery-magazine.md` |
> | **1c** Sonic Descriptor Layer         | ✅ Shipped (merged from `feature/sonic-descriptor-layer`) — unblocks Sonic Sibling, Sonic Map cluster overlay, For You rationale | `app/services/sonic_descriptor_service.py`, `scripts/cluster_curator.py` |
> | **2**  Frontend foundation            | ✅ Shipped (out-of-band — landed alongside Plan 4 timeframe) | inline in `frontend/index.html` |
> | **3**  Artist Atlas                   | ✅ Shipped (Plan 5)                       | `docs/superpowers/plans/2026-05-16-plan-5-artist-atlas.md` |
> | **4**  Player v6 redesign             | ✅ Shipped (Plan 4) + post-plan polish round | `docs/superpowers/plans/2026-05-16-plan-4-player-redesign.md` |
> | **5**  AI Chat & lyrics-explain       | ✅ **Shipped** — AIChatDrawer (slide-up panel replacing queue, FactsRail stays visible; slim 32px header; persistent suggested prompts above input; ↺ session-clear) + Inline ✨ explain (draw-under panel). Single endpoint POST /chat/track-chat with web_search tool fallback (pydantic-ai). Plan: `docs/superpowers/plans/2026-05-19-plan-5-ai-chat-lyrics-explain.md`, polish: `docs/superpowers/plans/2026-05-20-chat-drawer-replaces-queue.md` |
> | **6**  Spotify-like MVP               | 🛠 **Library Overhaul + Playlists CRUD shipped (sub-plans #1, #2 of 3)** | feature/library-overhaul, feature/plan-19-playlists-crud |
> | **6a** Home (Discovery Magazine)      | ✅ **Shipped** — cinematic, embedded/boundaryless landing. **For-You-first reordering** vs §4.4 (For-You autoplay hero → Rediscovery interlude → Featured Artist → shelves Recently/Liked/Different/Playlists → Library/Search "another way to listen"). Components: `ForYouHero` (cover-deck + iridescent aurora start-control), `RediscoveryBand` (ambient cover-color wash + gap badge + teaser fact), `FeaturedArtistCard` (hover-spread stack), generic `Shelf`/`ShelfCard`, `AlternativeModes`. Old "Studio Console" launcher (LandingPlayer/doors/ticker/marquee) removed. | `docs/superpowers/plans/2026-05-24-home-discovery-magazine.md` |
> | **6b-B1** Search visual redesign      | ✅ **Shipped** — backend (SearchFilters + Qdrant payload + /sonic-facets) + frontend (Hybrid v3 visuals + RecentSearchesChips + hover breakdown + SonicFiltersChips). Plan: `docs/superpowers/plans/2026-05-19-plan-b1-search-section-redesign.md` |
> | **6b-B2** Search AI gating + functional | ✅ **Shipped** — chat tab gated on aiStatus.aiActive, DecadeFiltersChips (OR) backed by /library/year-facets, card-click → Player auto-play (detail panel removed), hover play+like overlay extends B1's ScoreBreakdownTooltip, autocomplete typeahead fix, SearchFilters dead-field cleanup (year_from/year_to/duration_* dropped; year_range singular → year_ranges plural). Plan: `docs/superpowers/plans/2026-05-19-plan-b2-search-section-ai-gating.md` |
> | **6c** Stats redesign (Sonic Map)     | ⏳ Not started — partly blocked on 1c    | — |
> | **6d** Recommendations                | ⏳ Not started — partly blocked on 1b    | — |
> | **7**  Polish                         | ⏳ Not started                            | — |
>
> **Next to execute — Plan B1 then B2** (specs + plans drafted 2026-05-16; refs in table above). Plan B was split because the original SearchSection still uses old skeuomorphic materials (`ske('inset')`, plain Segmented) while Player and Atlas already shipped Hybrid v3. **B1** is a pure visual redesign + three opinionated additions (recent searches chips, hover breakdown tooltip, sonic class/tags filter chips backed by new `/library/sonic-facets` endpoint that surfaces Phase 1c data). **B2** layers behavior on top: aiActive gating of the Chat tab, decade filter, card-click → Player auto-play, hover play/like buttons added to B1's overlay, autocomplete fix, and SearchFilters dead-field cleanup. Plan B1 also extends `SearchFilters` with `sonic_class: list[str]` (OR semantics) + `sonic_tags: list[str]` (AND semantics) and wires both into `LyricsDB.search()`. **Original Plan B scope (single-PR with both visual + functional)** was rejected on 2026-05-16 because (a) the visual redesign benefits from brainstorm-style iteration that B2's behavioral work would obscure, and (b) PR diffs stay smaller / reviewable when materials land before behavior.
> - Split today's chat-only SearchSection into two modes gated on `aiStatus.aiActive`:
>   - **AI on (`aiActive === true`)** — keep the existing natural-language chat experience.
>   - **AI off / LLM offline** — render a non-chat results UI: query input → `SearchResultsGrid` of cards (cover + title + artist), no LLM dependency.
> - **Filters** (MVP scope agreed during brainstorm): Artist, Album, genre, decade. Duration + liked-only filters explicitly cut from MVP.
> - **Hover-actions on cards**: play + like + breakdown (the "full spec" option), not minimal.
> - **Card click → Player auto-play** (replaces Plan 5's hybrid card→Atlas semantics in Search; clicking the artist name in the subtitle still routes to Atlas per current Player/Landing convention).
> - **Autocomplete fix** — Artist/Album typeahead suggestions are currently broken in chat-search; bundle this fix into Plan B.
>
> **Deferred from shipped plans** (slots reserved, will return as smaller plans):
> - **Sonic Sibling** (Phase 1 item 4 + Phase 4 item 12) — `/recommend/sonic-sibling` returns 501. Dependencies on Phase 1c.1-1c.2 (sonic tags) and 1c.5-1c.6 (classifier for class-diff phrase) are **now satisfied** (Phase 1c shipped 2026-05-16); endpoint awaits its own plan to wire the Qdrant payload-filter query + LLM `common_tags ∩ class_diff` phrasing.
> - **Sonic class filter follow-up** — `sonic_class: list[str]` field + OR-chip group in SearchSection. Waiting on operator-trained classifier (separate dataset; no user-curator pass needed in this project's workflow).
> - **Related artists tab** + **Eras tab** (Phase 3) — Related needs CLAP artist centroids (Phase 1c products); Eras is a small standalone follow-up.
> - **Click-to-artist routing in Library / RecentlyPlayed** — those sections haven't been redesigned yet; routing will be added when each ships.
> - **OnboardingScreen 3-phase wizard** (Plan 6 follow-up) — `OnboardingScreen.handleIndex` still ships its own pre-wizard copy; lift `startIndexing` into a shared helper so first-run users get the same ai-setup → indexing → ai-bootstrap flow as SettingsPanel.
>
> **Post-plan polish round shipped outside formal plans** (after Plan 4 merge, before Plan 5):
> - Player: audio-reactive spectrum bars flanking the cover (FFT via Web Audio AnalyserNode singleton, dominant-color extraction from cover via canvas sampling, blurred-wave mirrored layout with edge-fade mask).
> - Player: vinyl-stack door-swing track transition (sequential exit/enter via CSS animation-delay), glassy play/pause indicator (CSS mask + backdrop-filter so the icon shape itself is the glass surface), cover-tap → toggle play + first-3-clicks hint, scope toggle on FactsRail (SONG | ARTIST segmented pill with sticky user choice + auto-fallback).
> - AI Indexing reliability: `n_skipped` accounting through the whole stack (DB column + JobState + AIJobStatus + frontend warning banner when job completed with zero real work), default `_SYSTEM_PROMPT` for refined_facts with fail-fast on empty, slug resolution from `payload[artist]+payload[title]` (Qdrant never wrote `song_slug`/`artist_slug` payload fields), smart-quote (U+2019 et al) stripping in `_slugify` so iTunes-encoded titles round-trip.
> - AI Mode Infrastructure (Plan 6 — `docs/superpowers/plans/2026-05-16-plan-6-ai-mode-infrastructure.md`): `useAIStatus(activeCollection)` hook in App (LLM probe every 60s + per-collection ai_enabled), threaded as `aiStatus` prop. IndexingModal refactored into 3-phase wizard (ai-setup → indexing → ai-bootstrap). PlayerSection Ask AI button + AIIndexingCard Run buttons + chat-search (planned Plan B) all gate on `aiStatus.aiActive`. Cached AI artifacts (sonic_vibe / refined_facts / artist_bio rows) stay accessible regardless of LLM status.

> **Out-of-plan shipment (2026-05-20)**: AudioDB enrichment in indexing FACTS stage. New columns on `artists` table (bio/mood/country/label/cutout/thumb/MBID), local image cache under `/covers/artists/`, integrated with `artist_bio` AI task via new `seed_bio` parameter. Frontend Atlas integration deferred to a follow-up plan. Spec: `docs/superpowers/specs/2026-05-20-audiodb-enrichment-design.md`. Plan: `docs/superpowers/plans/2026-05-20-audiodb-enrichment.md`.

### Phase 1: Backend foundations (без UI-изменений)

> **Status (2026-05-16)**: ✅ **Plan 3 ships ScoreBreakdown + MusicBrainz scaffold + AI Indexing (Sonic Vibe + Refined Facts) + Autoplay queue + Playback history**. Plan + spec: `docs/superpowers/{plans,specs}/2026-05-14-plan-3-backend-foundations*`. Single notable revision vs the bullets below: **Sonic Vibe is now an AI Indexing task** (user-triggered batch, gated by opt-in in Settings → AI Indexing card) — not a lazy on-demand endpoint. Cached output keyed by `(track_id, collection, lang)`. **Refined Facts** is a sibling AI-Indexing task that batch-filters and shortens song/artist facts; `/metadata/tracks/{id}/facts` transparently prefers refined over originals when present. **Sonic Sibling deferred** to a later plan — `/recommend/sonic-sibling` returns 501.
>
> **Post-plan reliability fixes (2026-05-16)**: AI-Indexing tasks were silently completing with zero LLM calls because `_TASK_TYPES` whitelisted them but `register_task()` never fired (no production import path for the `ai_tasks` package) — fixed by side-effect-importing in `app/api/main.py`. Slug resolution patched to read `artist+title` from the Qdrant payload (which never carried `song_slug`/`artist_slug`) and run them through `_slugify_artist` / `get_song_facts_key` for deterministic match against the SQLite `*_facts` tables. `n_skipped` column added so the UI can honestly report "completed with 0 real work" (vs the previous "16/16 done" false success). `refined_facts._SYSTEM_PROMPT` got a sensible default + `RuntimeError` fail-fast on empty. `_slugify` in both `song_facts_service` and `artist_facts_service` now strips U+2019 (curly apostrophe) and other smart quotes — fixes silent 404s from songfacts.com URL builders for titles like "We're Good".

1. **ScoreBreakdown в TrackHit** — модифицировать `_merge_hits()` в `search_service.py`, добавить breakdown в response. Verify: вызов `/search/` возвращает breakdown поля.
2. **MusicBrainz scaffolding (data-only)** — добавить nullable columns в `songs`/`artists` tables (`producers`, `label`, `samples_json`, `mbid`) **без** активного парсинга. UI рендерит conditionally if non-null. Verify: schema migration работает, существующие данные не повреждены, columns пустые.
3. **AI Indexing subsystem** [depends on Phase 1c.1-1c.2 for sonic_vibe task] — user-triggered batch job runner (`POST /library/ai-index/{task_type}`, `GET /status`, `DELETE /cache`). Two task types ship in Plan 3: **Sonic Vibe** (one-sentence atmospheric phrase per track via LLM, cached in `sonic_vibes` table); **Refined Facts** (batch-filter+shorten song/artist facts via LLM, replacement semantics on `/metadata/tracks/{id}/facts`, system prompt operator-filled). Settings UI card surfaces Run/Reset/status per task. Verify: запрос возвращает разумные phrase-ы; refined facts перекрывают originals; lang капчурится из текущего фронта при запуске.
4. **Sonic Sibling endpoint** [DEFERRED — slot reserved, returns 501] [**unblocked** — Phase 1c.1-1c.2 + 1c.5-1c.6 shipped 2026-05-16; awaits its own plan] — `/recommend/sonic-sibling`. Qdrant query с payload filters. Plus LLM-фраза из common_tags ∩ + class_diff.
5. **Autoplay queue endpoint** — `/recommend/autoplay-queue`. Pure CLAP CLAP neighbors + reaction/session filtering + diversity demotion. Verify: returns up to 20 диверсифицированных треков.
6. **Playback history** [lifted from Phase 6 because needed by future Rediscover / Personalization] — `playback_events` table with full session tracking (`session_id`, `played_sec`, `total_dur`, server-derived `skipped_early`). Frontend mints `session_id` in `sessionStorage` and POSTs on track end / track switch / `beforeunload` (via `sendBeacon`).

### Phase 1c: Sonic Descriptor Layer (prerequisite для Sonic Vibe / Sibling / Map cluster overlay / For You rationale)

> **Status (2026-05-16)**: ✅ **Shipped** via `feature/sonic-descriptor-layer` (22 commits, `c4b5493` → `c9ac6ec`, fully landed in `main`). All seven checklist items (1c.1 vocab + embeddings cache → 1c.2 prompt-probing tagger → 1c.3 HDBSCAN discovery → 1c.4 cluster curator CLI → 1c.5 MLP classifier training → 1c.6 classifier-at-indexing → 1c.7 empty-state guards) are live. `SonicDescriptorService` is wired into the indexing pipeline (commit `abc151d`); per-track output is persisted to `songs.sonic_tags_json` / `sonic_class` / `sonic_class_confidence` and surfaced via `GET /library/sonic-descriptor/{slug}`. **Follow-ups still pending**: (a) Sonic Vibe task does not yet ingest `sonic_tags_json` (still facts-only) — needs `ai_tasks/sonic_vibe.py` extension; (b) Sonic Sibling endpoint still returns 501; (c) Sonic Map cluster overlay still hidden behind unshipped Phase 6c. Plan for (a) is small and can ship standalone.

1c.1. **Prompt vocabulary scaffolding** — создать `cache/sonic_prompts.json` со стартовым набором ~30-50 prompts по группам (energy/valence/density/texture/instrumentation/vocal/rhythm/era). Pre-compute их CLAP text embeddings в `.npy`. Verify: vocab loadable, embeddings shape correct.
1c.2. **Prompt-probing tagger** — `sonic_descriptor_service.compute_tags(track_id)`: cosine между track_emb и prompt_embeddings, top-K. Persist в `songs.sonic_tags_json`. Trigger automatically на indexing (incremental: new tracks get tagged immediately, existing — bulk script). Verify: 5 tracks возвращают sensible top-5 tags каждый.
1c.3. **HDBSCAN clustering** — `sonic_descriptor_service.cluster_library(collection)`: clusterise все CLAP-vectors. Save assignments → `cache/sonic_clusters/<col>_assignments.json` + representatives (top-5 closest to centroid per cluster). Endpoint `POST /library/cluster-discovery`. Verify: returns sensible cluster count (5-30 для test library of 500 tracks), representatives визуально похожи.
1c.4. **Cluster curator CLI** — `scripts/cluster_curator.py`: interactive prompts (list clusters → show representatives — `[1] play 'Track A by Artist'`, etc. — accept name input, merge/split commands). On finish — writes `<col>_labels.json`. Verify: end-to-end на test cluster set, labels persisted.
1c.5. **MLP classifier training** — `sonic_descriptor_service.train_classifier(collection)`: sklearn `MLPClassifier`, hidden=(256, 128), trained on (CLAP_vec, cluster_label) pairs. Save → `cache/sonic_classifier/<col>.joblib`. Endpoint `POST /library/sonic-classifier/train`. Verify: training completes <60 sec on 1500 tracks, accuracy >0.7 on 80/20 split.
1c.6. **Apply classifier at indexing** — для каждого нового трека после CLAP embedding: `predict_proba` → top-class + confidence → write to `songs.sonic_class` / `sonic_class_confidence`. Verify: existing library re-classified в bulk script, new tracks classified inline.
1c.7. **Empty/incremental state guards** — if classifier not trained, all consumers (Sonic Vibe, Sibling, Map overlay) gracefully fall back (tags only, no class). Verify: features работают на свежей библиотеке без curator pass.

### Phase 1b: Additional backend foundations (Home / Stats / Recommendations)

> **Status (2026-05-16)**: ⏳ **Not started.** Endpoints scaffolded by Plan 3 spec but no service implementations yet. Blocks 6a (Home Hero needs `/library/rediscover` + `/library/featured-artist`), 6c (`/library/sonic-map`), 6d (`/recommend/prompt-to-playlist` + `/recommend/for-you` + `/recommend/quick-rate/*`).

1b.1. **Sonic Map service** — `sonic_map_service.py`, UMAP computation, cache file `cache/sonic_map/<col>.json`. Endpoint `/library/sonic-map`. Trigger compute on indexing completion. Verify: запрос возвращает ~N points для test collection.
1b.2. **Home Hero endpoints** — `/library/rediscover` (least-recently-played; requires `playback_history` table populated), `/library/featured-artist` (date-deterministic rotation hash). Verify: rediscover varies per call, featured-artist stable per date.
1b.3. **Personalization service** — `personalization_service.py`, user_vector compute, in-memory cache TTL 1h, invalidation hooks. Endpoint `/recommend/for-you` + `/recommend/for-you/rationale`. Verify: меняется при new like/dislike, queue order reflects user_vector.
1b.4. **Prompt-to-Playlist pipeline** — `prompt_to_playlist_service.py`, LLM decompose → parallel hybrid search → LLM re-rank+why-phrase. Endpoint `/recommend/prompt-to-playlist`. Verify: end-to-end на 3 разных prompts (sad / energetic / specific era), результаты sensible.
1b.5. **Quick-Rate endpoints** — `quick_rate_service.py`, diversity-sampled batch + finish. Endpoints `/recommend/quick-rate/batch` + `.../finish`. Verify: 10 diverse candidates returned (spread по genre/decade); finish persists ratings + generates playlist.
1b.6. **Recommendation snapshots** — SQLite table + save endpoint (`POST /recommend/snapshots`) + list endpoint (`GET /recommend/snapshots`). Verify: save и retrieve работают, восстановление playlist по id.

### Phase 2: Frontend foundation

> **Status (2026-05-16)**: ✅ **Shipped (out-of-band).** All four bullets landed in `frontend/index.html` over the Plan 4 timeframe without a dedicated plan doc — `FloatingIconNav` + `NowPlayingPebble` + `MiniPlaybackPopout` + `useGlobalKeyboardShortcuts` are all wired and visible, and v3 hybrid CSS classes (`.panel-v3`, `.pill-v3`, `.cta-v3`, `.ask-ai-btn`, `.vibe-quote`) ship in the `<style>` block. New screens (Atlas, etc.) consume them directly.

6. **Floating icon sidebar** — заменить current sidebar на `FloatingIconNav`. Сохранить routing к существующим разделам + добавить новые pivot-ы (Home/Search/Lib/Stats/Recom/Player/Set). Verify: visual change без regression в navigation.
7. **Style updates** — добавить v3 hybrid CSS (panels, pills, CTAs, animations) в `<style>`. Существующие компоненты постепенно мигрируют на новые классы.
7a. **Now Playing Pebble + MiniPlaybackPopout** — sidebar bottom indicator с hover-popout. Verify: pebble виден везде, popout появляется при hover, controls работают.
7b. **Global keyboard shortcuts** — Space / arrows / Shift+arrows / M / L / D / `/`. Verify: shortcuts работают на любом экране, не вызывают конфликта когда focus в input.

### Phase 3: Artist Atlas

> **Status (2026-05-16)**: ✅ **Plan 5 ships v1 — `ArtistAtlasSection` (Hero + Bio/Discog/Facts tabs, Spin from here CTA), `GET /artists/{slug}` aggregate (slug-tolerant via `_slugify_artist`), new `artist_bio` AI-Indexing task (operator-fill `_SYSTEM_PROMPT`), AIIndexingCard 3rd row for `artist_bio`.** Plan: `docs/superpowers/plans/2026-05-16-plan-5-artist-atlas.md`. Deferred to later plans: Related tab (needs CLAP artist centroids), Eras tab, click-to-artist routing in Library / RecentlyPlayed (those sections haven't been redesigned yet).
>
> **Click-to-artist routing (2026-05-16 update)** — Plan 5 originally shipped a hybrid card click in Search (card → Atlas, ▶ → play), but reverted because the inline detail panel (lyrics + score breakdown) is still the more useful default until Search is redesigned (Phase 6b). The artist link now lives on **the artist name itself** (under the song title), with `e.stopPropagation` so it doesn't fire the card click. Same artist-name-click pattern was added to PlayerSection (full Player tab) and LandingPlayer (Home block) — both surfaces now navigate to Artist Atlas on click of the artist name in the song subtitle. Helper: `slugifyArtistName` at the top of `frontend/index.html`.

8. **ArtistAtlasSection component** — новый view. Использует существующий `/library/browse` + новый `/artists/{slug}` aggregate. Hero, tabs, bio panel, discog rail.
9. **Click-to-artist routing** — из SearchSection и других мест клик на трек/артиста → Artist Atlas, не Detail Panel.

### Phase 4: Player v6 redesign

> **Status (2026-05-16)**: ✅ **Plan 4 ships Player Redesign — cover↔lyrics 3D flip (Shift+L / Esc), action pills row (Like / Skip / Lyrics / Ask AI), Vibe Line (Sonic Vibe phrase), Facts Rail (player variant), queue with autoplay divider, score-bars hover tooltip on queue items, Ask AI toast stub (Plan 5 placeholder), theme-token parity sweep + custom purple scrollbar.** Plan: `docs/superpowers/plans/2026-05-16-plan-4-player-redesign`. Notable delta vs bullets below: Sonic Sibling card (item 12) remains deferred (backend returns 501); flip is cover↔lyrics rather than separate lyrics-btn.
>
> **Post-plan polish round (2026-05-16)** — landed on `main` outside a formal plan, all on top of Plan 4:
> - **Vinyl-stack track transition** — door-swing rotateY around Y, sequential exit (~420ms ease-in) + entry (~600ms bounce, 320ms delay). Mirrored for prev. Disables tilt during transition. Auto-fires on `audio.ended` too.
> - **Glassy play/pause indicator + cover-tap toggle** — old hover overlay removed; cover-click toggles play. On toggle: press-scale 0.96 + glassy play/pause icon (CSS `mask-image` + `backdrop-filter: blur(22px)` so the icon shape itself is the glass surface) + frosted blur over cover, all fading over 1200ms. First-3-clicks hint above cover (localStorage-backed).
> - **Side-flanking prev/next buttons** + title row hosts like/dislike/lyrics/AI icons inline (replaces the previous bottom button row).
> - **FactsRail scope toggle** — segmented pill SONG | ARTIST replaces the text header in player variant. Sticky user choice across tracks with auto-fallback when the chosen scope is empty for the current track. 5 random artist facts per track (Fisher-Yates, stable within track). No more random-collection fallback in the player variant.
> - **Audio-reactive spectrum bars** — mirrored blurred-wave strips behind the cover row. Web Audio AnalyserNode (FFT, 128 bins) wired through a module-level singleton that survives PlayerSection unmount/remount. Dominant color extracted from cover via 32×32 off-screen canvas sampling. ~32 bars per side, `filter: blur(12px) saturate(1.35)` with `mask-image` linear-gradient fading outer 35% so the wave dissolves into the column edge. Bars collapse fully (scaleY=0) when audio is paused.
> - **AI Indexing card** surfaces `n_skipped` alongside `n_done/n_failed` + amber warning banner when a job completed with zero real work, with scope-specific remediation hint.

10. **PlayerSection restructure** — hero с 3 zones, breathing cover, EQ bars, Sonic Vibe quote, Ask AI inline.
11. **Facts panel** — slot для song-facts (через `/metadata/tracks/{id}/facts`).
12. **Sonic Sibling card** — янтарная hero-карточка с LLM "why similar" phrase.
13. **Progressive disclosure**: lyrics-btn под cover (toggle), see-other-similar btn (toggle).
14. **Breakdown bars** в other similar list.

### Phase 5: AI Chat & lyrics-explain

> **Status (2026-05-16)**: ⏳ **Not started.** The Player's "Ask AI" button currently shows a `showToast` stub ("AI-чат появится в следующем плане (Plan 5)" — note: this stub message is misleading since Plan 5 turned out to be Artist Atlas, not AI Chat; the stub copy will be updated when this phase ships).
>
> **Status (2026-05-19)**: ✅ **Shipped** on `feature/plan-5-ai-chat-lyrics-explain`. Ships:
>   - `POST /chat/track-chat` endpoint with `mode: 'song' | 'lyric_explain'` discriminator (single endpoint, two prompts)
>   - `TrackChatAgent` (pydantic-ai 1.61.0) with `web_search` tool wrapping `smart_web_search` from `llm_web_search.py`
>   - Backend resolves raw `song_facts` server-side (NOT refined) — agent always sees original facts regardless of AI-Indexing state
>   - `AIChatDrawer` (420px right-side glass slide, backdrop blur, Hybrid v3 panel-v3) with 3 static suggested prompts (the third "Какие песни семплирует?" probes `web_search` since we have no local sample data — natural acceptance test)
>   - `InlineLyricExplain` (per-line draw-under panel inside `LyricsBackFace`, multiple concurrent expansions OK, state resets on track change)
>   - Both surfaces gated on `aiStatus.aiActive` (the Plan 6 signal)
>   - Per-track `localStorage` chat history via `useTrackChat(trackId)` hook
>
> **Schema discovery during implementation**: `song_facts` table is `(id PK, song_slug FK, lang, fact TEXT)` not `(slug, notes)` as the plan assumed — `resolve_song_facts` adapted to `SELECT fact FROM song_facts WHERE song_slug = ?` with `\n\n` concatenation of multiple fact rows.
>
> **Deferred follow-ups**: global drawer (Atlas/Search), LLM-generated suggested prompts, SSE streaming, per-line explanation cache, multi-turn agent history support (current `answer_track_chat` ignores `req.history` — first-pass simplification, will revisit if engagement shows it matters).
>
> **Polish (2026-05-20)**: ✅ **Shipped** — drawer placement rewired. Was anchored to the entire right column (covered both FactsRail and queue); now anchored to a new `.queue-chat-area` wrapper that contains only the queue, so FactsRail stays visible while chat is open. Header slimmed from 70px (cover+title+artist+year) to 32px ("✨ Чат по треку" label + ↺ new-chat + ✕). Suggested prompts moved from "show once when chat empty" to "always visible above input" (Perplexity/ChatGPT quick-actions pattern). New `clearChat` UI (↺ button, conditional on `messages.length > 0`, with "Чат очищен" toast). Animation switched from `translateX(110% → 0)` to `translateY(100% → 0) + opacity` (~280ms) with `backdrop-filter: blur(14px) saturate(140%)` for glassy effect during transition. Spec: `docs/superpowers/specs/2026-05-20-chat-drawer-replaces-queue-design.md`.

15. **AIChatDrawer component** — слайд из правой стороны. Pre-filled context current song. Suggested prompts. Reuse `useChatHistory` с per-song ключом.
16. **Inline ✨ explain** на lyric line — popover с LLM-ответом. Reuse `/chat/` endpoint с `LYRIC_EXPLAIN_PROMPT`.

### Phase 6: Spotify-like MVP

> **Status (2026-05-20)**: 🛠 **In progress.** Sub-plan #1 (Library Overhaul) shipped via `feature/library-overhaul`: full LibrarySection rewrite as track-picker hub with Albums browser (grouped by album_title, majority-vote primary artist, feat-artist pills, drill-into-modal with tracklist + Play All + per-track like-toggle) + Liked Songs glassy-row view + Recently Played glassy-row view (relative time + N× play count + sort by last_played/play_count) + compressed hero (5 values: tracks/albums/artists/genres/year-range) + expandable Distributions panel (decades / top-5 genres / top-5 artists) + 4 listening-stats widgets (∑ Listened / Top Track / Top Artist / Peak Hour). Backend adds 4 endpoints — `/library/albums`, `/library/liked-songs`, `/playback/recent`, `/library/listening-stats` — all with TDD coverage. No DB schema changes. Plan: `docs/superpowers/plans/2026-05-20-library-overhaul.md`. Spec: `docs/superpowers/specs/2026-05-20-library-overhaul-design.md`. Remaining sub-plans: (19) Playlists CRUD, (20) Manual Queue.

> **Status (2026-05-21)**: ✅ **Sub-plan #2 (Playlists CRUD) shipped** via `feature/plan-19-playlists-crud`. Two SQLite tables (`playlists`, `playlist_tracks`) added to `MetadataDB._SCHEMA_SQL`; 8 endpoints under `/api/v1/playlists` (create / list / detail / rename / delete / add-track / remove-track / reorder) with full TDD coverage (26 integration + 20 unit DB + 9 model + 10 service tests). `list_playlists?include_track_id=Y` annotates each summary with `contains_track` for popover use without N+1 queries. Orphan tracks (in `playlist_tracks` but not in Qdrant) silently filtered on read into `missing_track_ids`; re-indexing restores them. Frontend: 4th tab in LibrarySection with mosaic 2×2 covers (adaptive 1/2/3/4 layout), inline-editable hero name + serif italic description, native HTML5 drag-drop reorder with purple-glow drop-target indicator. `AddToPlaylistPopover` (membership-aware, inline-create) wired to `＋` button in `LibraryGlassyRow` (Liked / Recently surfaces). Player-style icon buttons (`.player-icon-btn`) with `.player-icon-burst` radial pulse animation reused for ♥/＋/⨯ row actions. Plan: `docs/superpowers/plans/2026-05-21-plan-19-playlists-crud.md`. Spec: `docs/superpowers/specs/2026-05-21-plan-19-playlists-crud-design.md`. Mockup: `docs/superpowers/mockups/2026-05-21-plan-19-playlists.html`. **Deferred to follow-up**: `＋` button in PlayerSection title row + SearchSection hover overlay (avoiding prop-drilling explosion through 3+ components; Library surfaces cover MVP entry points).

17. ✅ **Playback history backend** + `RecentlyPlayedRail` на Home/Library — shipped as part of Library Overhaul (recently-played tab in Library; rail-on-Home deferred to Phase 6a).
18. ✅ **Liked Songs view** — фильтр + UI — shipped as part of Library Overhaul.
19. ✅ **Playlists CRUD** — shipped 2026-05-21 in `feature/plan-19-playlists-crud`. 2 tables + 8 endpoints + 4th Library tab + AddToPlaylistPopover on Liked/Recently track-rows. Player+Search `＋` deferred to follow-up.
20. ⏳ **Manual Queue panel** — frontend-only, drag-drop reorder, выезжает из ≡ кнопки.

### Phase 6a: Home (Discovery Magazine)

> **Status (2026-05-24)**: ✅ **Shipped** on `feature/home-discovery-magazine`. Spec: `docs/superpowers/specs/2026-05-24-home-discovery-magazine-design.md`; Plan: `docs/superpowers/plans/2026-05-24-home-discovery-magazine.md`.
>
> **Design revisions vs the §4.4 sketch below** (decided in the 2026-05-24 brainstorm, visual companion):
> - **For-You-first reordering**: the For-You autoplay stream is now the cinematic hero #1 (was a bottom CTA strip in §4.4). Order: ① For-You hero → ② Today's Rediscovery (centered ambient interlude) → ③ Featured Artist → ④ shelves (Recently / Liked / Try different / **Playlists** if any) → ⑤ "another way to listen" (Library / Search secondary entry points).
> - **Visual language**: cinematic + **embedded / boundaryless** ("clean air" — zones separated by spacing + mono labels, no glass-card chrome). For-You and Rediscovery carry an ambient wash derived from the **dominant color of the cover** (reuses `useCoverColor`), fading smoothly at edges (mask-gradient).
> - **For-You start control**: iridescent colored-glass aurora orb (`.fy-hybrid.tint-irid`) with hover (scale + faster swirl + halo) — distinct from Yandex "Моя волна".
> - **For-You seed**: new placeholder endpoint `/recommend/for-you-seed` (weighted likes+recency) → existing `/recommend/autoplay-queue`. **Explicitly a placeholder** for the full personalization service (1b.3, §5.4) — to be swapped later.
> - **Empty/no-history**: blocks degrade gracefully (each hides when its data is absent); hero still works from a random seed. Quick-Rate (§5.5) was **not** built (out of scope).
>
> **Backend shipped (Phase 1b.2):** `/library/rediscover` (least-recently-played / never-played), `/library/featured-artist` (deterministic daily rotation, reuses extracted `build_artist_aggregate`), `/recommend/for-you-seed` (placeholder). New `MetadataDB.get_play_recency_map`, `LibraryService.get_rediscover` + `list_distinct_artist_slugs`, `personalization_service.pick_for_you_seed`. No schema migration.
>
> **Deferred polish:** loading skeletons + the no-history albums-rail fallback (components currently hide empty shelves rather than substituting an albums rail); FFT "breathing" of the aurora control while a stream plays.

6a.1. **HomeSection layout** — Hero row (двойной блок) + 3 shelves + bottom CTA.
6a.2. **HeroRediscoveryCard** — uses `/library/rediscover` + facts из existing `/metadata/tracks/{id}/facts`. `[▶ PLAY]` запускает Player.
6a.3. **HeroFeaturedArtistCard** — uses `/library/featured-artist`. Cover stack, bio summary, CTA → Artist Atlas.
6a.4. **Shelf components** для Recently played / Your liked / Try different. Re-uses `RecentlyPlayedRail` pattern из Phase 6.
6a.5. **ForYouCTAStrip** — gradient bar внизу, обрабатывает empty-state (no playback_history → Quick-Rate suggestion).
6a.6. Verify: Home рендерится на свежей библиотеке (empty state), на богатой библиотеке, при отсутствии playback_history.

### Phase 6b: Search redesign

> **Status (2026-05-16)**: 🔜 **Next plan — Plan B (SearchSectionV2).** Plan 6 (AI Mode Infrastructure) shipped the `aiStatus.aiActive` signal that Plan B needs to gate chat-search; Plan B is now unblocked and is the immediate next plan to brainstorm → spec → write.
>
> **Status (2026-05-19)**: ✅ **B1 shipped** on `feature/plan-b1-search-section-redesign`. Ships: `SearchFilters.sonic_tags` + Qdrant payload write + `/library/sonic-facets` endpoint + `MetadataDB.get_sonic_facets` aggregate + `scripts/backfill_sonic_payload.py` migration + SearchSection rewritten to Hybrid v3 materials with `RecentSearchesChips`, `ScoreBreakdownTooltip` (hover), `SonicFiltersChips` (AND-semantics tag filter). **sonic_class deferred** — operator-trained classifier required first; will return as a separate small follow-up plan (add `sonic_class: list[str]` OR-chip group). B2 (`aiActive` gating + decade + card-click → Player + hover play/like + autocomplete fix + dead-field cleanup) is next.
>
> **Operator action after merge**: re-index a representative collection to verify `sonic_tags` populates the Qdrant payload (B1.2) and that `year_range` is also present (already shipped pre-B2). For collections already indexed before B1.2, run `python -m scripts.backfill_sonic_payload --collection <name>` per collection. `year-facets` aggregates Qdrant payload directly so no separate backfill is needed for decade chips.
>
> **Status (2026-05-19, later)**: ✅ **B2 shipped** on `feature/plan-b2-search-section-ai-gating`. Ships:
>   - chat-tab gated on `aiStatus.aiActive` (snaps to search tab when LLM offline);
>   - `year_ranges: list[str]` OR-semantics filter + `GET /library/year-facets` aggregate endpoint (reads Qdrant payload directly) + `DecadeFiltersChips` component;
>   - card body click → Player auto-play (artist-name link still routes to Atlas via `stopPropagation`); inline detail panel removed (–109 lines);
>   - hover play + like action overlay extends B1's `ScoreBreakdownTooltip` (reaction state lazy-fetched per-card on hover);
>   - autocomplete typeahead fix (`/browse` → `/library/browse` path correction);
>   - `SearchFilters` dead-field cleanup (`year_from`, `year_to`, `duration_min_sec`, `duration_max_sec` dropped; singular `year_range` replaced by plural `year_ranges`; `extra="ignore"` added to silently absorb legacy planner-LLM emissions).
>
> **Phase 6b complete.** Next candidate: 6c (Stats / Sonic Map — blocked on Phase 1b.1) or the deferred `sonic_class` follow-up once operator-trained classifier ships.
>
> **Plan B scope (decided during 2026-05-16 brainstorm, awaiting formal spec):**
> - Two-mode UI gated on `aiStatus.aiActive`:
>   - **AI on** → keep current chat-search.
>   - **AI off** → non-chat results UI: query input + `SearchResultsGrid` of cards (cover + title + artist), no LLM dependency. Plan 5's hybrid card-click (card → Atlas, ▶ → play) is **replaced** in this mode — card click → Player auto-play. Artist-name link in the subtitle still routes to Atlas (per current Player/Landing convention added in the Plan 5 follow-up polish).
> - **Filters (MVP)**: Artist, Album, genre, decade. Duration and liked-only filters cut from MVP per brainstorm.
> - **Hover-actions on each card**: play + like + breakdown (full spec).
> - **Autocomplete fix** — Artist/Album typeahead is currently broken in chat-search; bundle the fix into Plan B.
>
> Components from the §8 inventory still apply (`SearchSectionV2`, `SearchModeToggle`, `SearchFiltersAccordion`, `SearchResultsGrid`, `RecentSearchesChips`) but the mode-toggle semantics differ from the original §8 sketch (which assumed 4 modes Название / Текст / Звук / Hybrid) — Plan B's mode set is driven by `aiActive`, not by query-type pickers.

6b.1. **SearchSectionV2** with `aiActive`-driven mode split — replace existing SearchSection. Preserve backward-compat for existing search params (deep-links) if any.
6b.2. **SearchFiltersAccordion** — Artist / Album / genre / decade only; collapsed by default, smooth expand.
6b.3. **RecentSearchesChips** — localStorage persistent.
6b.4. **SearchResultsGrid** with hover-action overlay (play + like + breakdown) and card-click → Player auto-play.
6b.5. Fix Artist/Album autocomplete typeahead (regression — currently doesn't fire).
6b.6. Verify: AI-on mode keeps chat behavior; AI-off mode returns grid results, filters work, hover-actions fire, card-click plays in Player.

### Phase 6c: Stats redesign (Sonic Map)

> **Status (2026-05-16)**: ⏳ **Not started.** Blocks on Phase 1b.1 (`/library/sonic-map` UMAP service). Cluster overlay (6c.4) additionally blocks on Phase 1c.3-1c.4.

6c.1. **SonicMapCanvas** — компонент с pan/zoom/hover/click. Performance test: 1500+ points smooth at 60fps.
6c.2. **SonicMapControls** — color-mode dropdown (by genre / by decade / by reaction).
6c.3. **KPITiles + DecadesTimeline + TopGenres/Artists** — auxiliary visualizations.
6c.4. (Optional) **Cluster overlay** — if `/library/sonic-clusters` endpoint ready.
6c.5. Verify: map renders на 100/1000/3000 points, click-to-play работает, KPI tiles tally с `/library/stats`.

### Phase 6d: Recommendations

> **Status (2026-05-16)**: ⏳ **Not started.** Blocks on Phase 1b.3-1b.6 (`/recommend/for-you`, `/recommend/prompt-to-playlist`, `/recommend/quick-rate/*`, snapshots).

6d.1. **RecommendationsSection** + **RecommendModePicker** — 3-card mode picker.
6d.2. **PromptToPlaylistPanel** — text input, length selector, generate, result list с breakdown + WhyThisSetQuote. Save-as-playlist action.
6d.3. **ForYouPanel** — endless queue с per-track `i` icon → rationale popover. Recompute trigger.
6d.4. **QuickRateSessionWizard** — modal overlay, card-stack, 30-сек snippet autoplay, 👍/👎 buttons, progress, finish-generate.
6d.5. **RecommendationSnapshotsList** — bottom of screen, list of saved generated playlists.
6d.6. Verify: каждый mode end-to-end работает; ratings из Quick-Rate переходят в track_reactions; saved playlists появляются в Custom Playlists.

### Phase 7: Polish

> **Status (2026-05-16)**: ⏳ **Not started.** Player has had targeted polish rounds (see Phase 4 status), but a cross-screen consistency pass + a11y / performance audit are still pending.

21. Анимации (coverBreath, eq, askGlow, pebble pulse, hero fade-in) — финальная настройка.
22. Empty states, loading states, error handling по всем screens (Home, Search, Stats, Recommend, Artist Atlas, Player).
23. Visual consistency pass — проверить что все screens используют panel-v3 materials uniformly, без regression в существующей navigation/playback.
24. Accessibility pass — focus rings, keyboard navigation в каждом screen, semantic HTML, aria-labels на иконках.
25. Performance pass — Sonic Map smooth at 60fps, no jank при переходах между screens, lazy-load обложек в shelves.

---

## 10. Verification

### Smoke tests (manual)
1. Запустить dev server, открыть приложение.
2. **Floating sidebar**: иконки видны, активный пункт светится, navigation работает.
3. **Artist Atlas**:
   - Search "Radiohead" → клик на любой трек/артист → попадаешь на Artist Atlas
   - Видно hero с обложкой, биографией, дискографией rail
   - Click `▶ Spin from here` → начинается воспроизведение, autoplay queue заполнена
4. **Player screen**:
   - Click `↕ EXPAND TO PLAYER` в mini-bar
   - Cover дышит (subtle pulse), EQ bars анимированы, NOW PLAYING label виден
   - Sonic Vibe фраза отображается как quote с декоративными ❝❞
   - Facts panel показывает 5 фактов из songfacts + "more"
   - Sonic Sibling card показывает реальный different-era трек с LLM "why similar"
   - Click `📜 LYRICS` → lyrics-панель раскрывается в левой колонке
   - Click строку lyrics → `✨ explain` пилюля → popover с LLM-ответом
   - Click `▽ SEE OTHER SIMILAR` → раскрывается список с breakdown bars
   - Click `✨ ASK AI ABOUT THIS SONG` → drawer выезжает справа, видны suggested prompts
5. **Spotify-like features**:
   - Like несколько треков → `♥ Liked Songs` секция в Library показывает их
   - Recently Played обновляется после прослушивания
   - Создать playlist, добавить трек через context-menu, удалить
   - Add to queue → manual queue panel показывает их в Up Next

### API smoke tests
```bash
# ScoreBreakdown
curl -X POST localhost:8000/api/v1/search/ -d '{"query":"sad", "mode":"hybrid"}' | jq '.hits[0].score_breakdown'

# Sonic Sibling
curl localhost:8000/api/v1/recommend/sonic-sibling?track_id=...

# Autoplay queue
curl -X POST localhost:8000/api/v1/recommend/autoplay-queue -d '{"seed_track_id":"...", "limit":20}'

# Sonic Vibe
curl localhost:8000/api/v1/metadata/tracks/.../sonic-vibe

# Playlists
curl -X POST localhost:8000/api/v1/playlists -d '{"name":"Late night"}'
curl localhost:8000/api/v1/playlists

# Liked songs
curl localhost:8000/api/v1/library/liked-songs

# Recently played
curl localhost:8000/api/v1/playback/recent

# Sonic Map
curl 'localhost:8000/api/v1/library/sonic-map?collection=music_explorer'

# Home Hero
curl 'localhost:8000/api/v1/library/rediscover?collection=music_explorer'
curl 'localhost:8000/api/v1/library/featured-artist?collection=music_explorer&date=2026-05-13'

# For You
curl 'localhost:8000/api/v1/recommend/for-you?collection=music_explorer&limit=20'
curl 'localhost:8000/api/v1/recommend/for-you/rationale?track_id=...&collection=...'

# Prompt-to-Playlist
curl -X POST localhost:8000/api/v1/recommend/prompt-to-playlist \
  -d '{"prompt":"rainy afternoon with guitars", "length":10, "collection":"music_explorer"}'

# Quick-Rate
curl -X POST 'localhost:8000/api/v1/recommend/quick-rate/batch?collection=music_explorer&size=10'
curl -X POST localhost:8000/api/v1/recommend/quick-rate/finish \
  -d '{"ratings":[{"track_id":"...","rating":"up"}], "collection":"..."}'

# Snapshots
curl -X POST localhost:8000/api/v1/recommend/snapshots \
  -d '{"source":"prompt", "prompt":"...", "why_phrase":"...", "track_ids":[...], "collection":"..."}'
curl 'localhost:8000/api/v1/recommend/snapshots?collection=music_explorer'

# Sonic Clusters (after curator pass)
curl 'localhost:8000/api/v1/library/sonic-clusters?collection=music_explorer'

# Sonic Descriptor per track
curl 'localhost:8000/api/v1/library/sonic-descriptor/abc-123'
# expected: {"sonic_tags_json":[...], "sonic_class":"Lo-fi indie", "sonic_class_confidence":0.81}

# Prompt vocabulary
curl 'localhost:8000/api/v1/library/sonic-prompts'

# Trigger cluster discovery
curl -X POST 'localhost:8000/api/v1/library/cluster-discovery?collection=music_explorer'
# expected: {"job_id":"..."}

# Curator workflow
curl 'localhost:8000/api/v1/library/clusters/representatives?collection=music_explorer'
curl -X POST localhost:8000/api/v1/library/clusters/labels \
  -d '{"collection":"music_explorer", "labels":{"0":"Lo-fi indie", "1":"Cinematic drone"}}'

# Train classifier
curl -X POST 'localhost:8000/api/v1/library/sonic-classifier/train?collection=music_explorer'
curl 'localhost:8000/api/v1/library/sonic-classifier/status?collection=music_explorer'
# expected (after training): {"status":"ready", "trained_at":..., "accuracy":0.78, "classes":["Lo-fi indie", ...]}

# Playlists CRUD (Plan 19)
curl -X POST localhost:8000/api/v1/playlists -d '{"collection_name":"music_explorer","name":"Late night","description":"ночной дрифт"}'
curl 'localhost:8000/api/v1/playlists?collection_name=music_explorer'
curl 'localhost:8000/api/v1/playlists?collection_name=music_explorer&include_track_id=abc-123'
curl localhost:8000/api/v1/playlists/1
curl -X PUT localhost:8000/api/v1/playlists/1 -d '{"description":"updated"}'
curl -X POST localhost:8000/api/v1/playlists/1/tracks -d '{"track_id":"abc-123"}'
curl -X POST localhost:8000/api/v1/playlists/1/reorder -d '{"track_ids":["abc-123","def-456"]}'
curl -X DELETE localhost:8000/api/v1/playlists/1/tracks/abc-123
curl -X DELETE localhost:8000/api/v1/playlists/1
```

### What NOT to verify (explicit non-goals в MVP)
- Synced lyrics (mp3 не имеет .lrc, отложено)
- Sleep timer (не выбрано в MVP)
- Mobile responsive — отложено, PC-only
- Equalizer / audio quality / crossfade
- Daily Mix / автогенерация persona-плейлистов (заменён нашим For You stream)
- Social / sharing / collaborative playlists

---

## 11. Open questions для будущих итераций

После завершения этого MVP, кандидаты для следующего pass-а:
- **MusicBrainz populating** — когда найдётся стабильный matching pipeline (или альтернативный источник: Discogs, AcousticBrainz, custom scraper), включить backfill в scaffolded columns (`producers`, `label`, `samples_json`, `mbid`). UI уже готов рендерить conditionally — данные сами подхватятся.
- **Sonic Descriptor refinement** — после первого MVP с initial vocabulary, проанализировать какие prompts наиболее differentiating (variance across library), какие redundant (correlated). Tune `cache/sonic_prompts.json` по результатам. Возможно ввести **multi-level taxonomy** (super-classes над sonic_class — e.g. "Indie/Folk/Acoustic" → ["Lo-fi indie", "Bedroom pop", "Folk revival"]).
- **Cluster Curator screen (UI)** — когда CLI MVP проверен, перенести функционал в `Stats → Curate Clusters` tab: cluster grid с covers, inline play, label editing, merge/split actions.
- **Sonic Descriptor confidence as quality signal** — tracks с низким `sonic_class_confidence` (<0.4) могут означать outlier треки (хорошие кандидаты для Featured Artist of the Day или "Try something different" shelf).
- **Synced lyrics**: интеграция syncedlyrics + timeline highlight (требует `.lrc` parsing)
- **Mobile responsive** — second design pass
- **Sleep timer** — простой add
- **Multi-user / accounts** — если когда-то понадобится shareability
- **Cluster auto-naming improvement** — LLM иногда даёт generic labels ("Mixed group"), стоит experimental tune
- **For You diversity floor** — sometimes ranking даёт repetitive bunches (one artist подряд); может потребоваться MMR-style diversity at fetch time
- **Stats: listening patterns** — week/hour heatmap, скип-распределение по song length, "когда ты больше всего слушаешь" (требует обогащения playback_history)
- **Recommendation snapshots search** — full-text search по prompt history
- **Sonic Map sub-views** — drill-down в кластер (open cluster → mini-map с N точками)

---

## 12. Appendix: брейнсторм-артефакты

Все 6 итераций мокапов сохранены в `.superpowers/brainstorm/410-1778623356/content/`:
- `style-direction.html` — выбор A/B/C/D direction (C выбран)
- `style-hybrid.html` / `-v2.html` / `-v3.html` — три итерации гибрида C+A (v3 финал)
- `artist-atlas-layout.html` — L1/L2/L3 layouts (L2 выбран)
- `artist-atlas-revised.html` — L2 с лёгкой границей, без правой панели
- `player-screen.html` / `player-facts-centric.html` / `player-v2/v3/v4/v5/v6.html` — 7 итераций player layout
- `style-hybrid-v3.html` — финальный визуальный baseline

Эти файлы можно открывать в браузере (server в `.superpowers/brainstorm/410-1778623356/` или просто как static HTML) — они полезны как visual reference при имплементации.
