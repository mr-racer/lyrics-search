# Chat UX v2 — GPT-style layout, lg-island glass, history rail, big agent card

**Date:** 2026-07-06
**Scope:** Chat tab of `SearchSection` in `frontend/src/main.jsx` + chat CSS in
`frontend/src/index.css` + `matched_line` in `app/api/routes/chat.py` /
`app/domain/models.py`. Regular (non-AI) search tab is **out of scope**.
Iterates on `2026-07-06-chat-redesign-design.md`.

## Problems being fixed

1. `.liquid-glass` reads "cheap" next to the nav rail's `.lg-island` glass.
2. Chat layout is old-fashioned — not the modern LLM pattern (centered hero
   composer + slogan; narrow centered conversation column).
3. Chat history is hidden behind a tiny icon.
4. Agent-step timeline is small, abrupt between steps, and not where the eye
   expects (should read as "the assistant is working" in the answer's place).
5. Results under-accented; matched lyric lines never come from the backend.
6. On wide screens content stretches edge-to-edge.
7. Section background hard-cuts vertically at the nav rail's edge — the rail
   must float **above** a continuous background.

## Design

### 1. Glass v2 (`.liquid-glass` rebuilt on the `.lg-island` recipe)

Same layered recipe as the nav islands: base tint layer
(`rgba(24,24,32,0.36)` dark / `rgba(244,243,250,0.35)` light) under a 165°
white gradient, `blur(28px) saturate(1.7)`, bright top+left "lens edge" inner
strokes, soft double drop shadow, and the `::after` specular sweep on
hover/focus. Composer keeps a focus glow ring (accent oklch). Applied to the
composer and best-hit card automatically (same class).

### 2. GPT-style layout (chat tab only)

- **Hero state** (no user messages yet): vertically centered stack — serif
  slogan («Что послушаем?» / "What shall we listen to?") + one-line subtitle
  (absorbs the old greeting bubble; initial messages array becomes empty) +
  glass composer `max-width: 640px` + 3–4 suggestion-chip prompts. No bottom
  dock.
- **Conversation state**: messages in a centered column `max-width: 760px`
  (fixes wide-screen sprawl); composer docks bottom-center, same width. Hero →
  dock switch animates via fade/slide (no FLIP complexity).
- **Mode controls move inside the composer**: bottom row of the capsule holds
  ✦ AUTO pill + TEXT/AUDIO/HYB segmented (compact), like a GPT tools row. The
  old mode bar above the composer is removed; «новый чат» moves to the history
  rail header.

### 3. History rail (right side)

- Desktop (`≥1100px`): persistent `~260px` right column, glass-lite panel
  (panel-v3-like, lighter than composer), listing sessions: title (1 line,
  ellipsis) + relative date; hover reveals delete. Header row: «Чаты» + «+
  Новый» button. Collapsible to a thin edge tab (state in localStorage).
- `<1100px`: rail hidden; the existing slide-over stays, opened from a
  **visible labeled button** (not a bare icon) in the chat header area.
- Clicking a session loads it (existing behavior).

### 4. Agent activity card (replaces the small timeline while streaming)

A large `liquid-glass` card in the message flow where the answer will appear
(inside the 760px column, full width):

- Active step: large text (~17px), crossfaded on change (two absolutely
  stacked labels, outgoing fades up+out, incoming fades up+in, ~0.35s) — never
  a hard swap.
- Completed steps: stack above, shrink to 13px muted rows with green checks;
  a connector line grows smoothly (existing keyframes reused).
- Card carries a subtle animated shimmer strip while active.
- On `answer`: card collapses (grid-rows transition) into the existing
  summary pill («Нашёл за N шагов», expandable) and the answer bubble fades in
  below. `prefers-reduced-motion` degrades all of it to instant.

### 5. Result accents

- **Best-hit card**: cover 96px (72px on mobile), title ~18px/700, artist
  14px, confidence bar + score pill kept, `LyricSnippet` below (unchanged
  collapsed/expand behavior).
- **Secondary hits**: covers 56px, same row layout otherwise.

### 6. Backend `matched_line`

- `TrackHit.matched_line: str | None = None` (additive, `app/domain/models.py`).
- In `_run_searches` (`chat.py`): for `mode in ("text","hybrid")`, after
  collecting each query's hits, pick the lyric line with max token overlap
  against the **executed** query text (the server knows planner-generated
  queries; the client does not). Only set when overlap > 0; cheap pure-python,
  no extra LLM/embedding calls. Frontend already reads `hit.matched_line`
  with its word-overlap heuristic as fallback. Schema stays open for future
  LLM-quoted lines.

### 7. Full-bleed chat ambient (no seam at the rail)

An `ambient` layer for the search section rendered at **app-shell level**
(next to `PlayerAmbient`, `section === 'search'`): fixed/absolute full-viewport
soft radial glows (violet top-left, faint amber bottom-right; dark+light
variants), sitting behind both the transparent nav rail and the section
content. SearchSection root becomes transparent over it, so the background is
continuous and the rail floats above it. Subtle enough not to fight the
results grid of the regular search tab (shared section).

## Testing

- **Backend**: unit test for the matched-line picker (overlap, no-overlap →
  None, empty lyrics); integration assert `/chat/stream` answer hits carry
  `matched_line` when lyrics search is stubbed. Import-safe under conftest
  stubs.
- **Frontend**: manual `/verify` pass — hero state, dock transition, rail
  collapse, step crossfade, snippet highlight, both themes, narrow/wide
  widths.

## Rollout / risk

- Pure UI + additive API field; no agent-logic changes, old sessions in
  localStorage still render (messages array shape unchanged).
