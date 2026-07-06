# Chat Redesign — Streaming Agent Steps, Liquid Glass, Lyric Snippets

**Date:** 2026-07-06
**Scope:** Chat/lyrics-search UI in `frontend/src/main.jsx` + streaming layer in
`app/api/routes/chat.py`. Regular search grid is **out of scope**.

## Goal

Modernize the chat search into a GPT-style experience that stays in the app's
visual language (skeuomorphism as the base, **liquid glass** as an accent on key
elements). Three concrete improvements:

1. **Real-time agent steps** — stream what the agent is doing, in plain language
   for non-technical users, with smooth transitions between steps.
2. **Liquid glass** on the input bar and the best-result card.
3. **Lyric snippets** — show the matched lyric line with surrounding context,
   highlighted, collapsed-by-default with expand.

## Non-goals (YAGNI)

- No changes to the regular search grid, secondary hit cards' base layout, or the
  per-track `AIChatDrawer`.
- No new agent logic — the agent loop's behavior is unchanged; we only **emit**
  what it already computes.
- No persisted step history in localStorage (steps are ephemeral; only the final
  answer is saved to sessions as today).

---

## 1. Backend — SSE streaming (`app/api/routes/chat.py`)

### Refactor

The body of the current `run_chat` handler is refactored into an **async
generator** `chat_events(req, request, current_user) -> AsyncGenerator[dict]`
that `yield`s typed event dicts as the agent progresses, and finally yields the
`answer` event carrying the exact payload the old handler returned.

Two endpoints wrap it:

- **`POST /api/v1/chat/stream`** — new. Returns
  `StreamingResponse(media_type="text/event-stream")`, serializing each yielded
  event as an SSE frame `data: {json}\n\n` (reuse `sse_utils.sse_data`). Standard
  Bearer auth (POST body carries `message`, `history`, `mode`, LLM settings), so
  it stays under the normal auth gate — **not** an EventSource/job-capability
  endpoint. A `: heartbeat\n\n` is emitted on long LLM waits to keep the
  connection alive.
- **`POST /api/v1/chat/`** — unchanged contract. Becomes a thin wrapper that
  consumes `chat_events` to completion and returns the final `answer` payload's
  body. Zero breakage for any existing caller.

### Event schema

Every event is `{"type": <str>, "human": <localized str>, ...fields}`. The
`human` string is built server-side in `req.lang` (ru/en) so the frontend renders
it verbatim — no client-side phrasing logic. Emission points map to existing log
sites in the agent loop:

| type       | emitted when                                   | extra fields                          | human (ru example)                          |
|------------|------------------------------------------------|---------------------------------------|---------------------------------------------|
| `classify` | after query classification                     | `mode` (text/audio/hybrid)            | «Понял запрос — ищу по тексту»              |
| `plan`     | planner produced queries (attempt 1, if any)   | `queries: [str]`                      | «Составил план поиска»                      |
| `search`   | after each `_run_searches` call                | `attempt`, `queries: [str]`, `found`  | «Ищу по тексту… нашёл 7 совпадений»         |
| `validate` | after ValidatorAgent runs                      | `valid: bool`                         | «Проверяю, точно ли это тот трек»           |
| `retry`    | validator rejected → new queries injected      | `attempt`                             | «Уточняю и пробую снова»                    |
| `answer`   | final                                          | full result payload (see below)       | — (payload only)                            |

`answer` payload = today's return dict verbatim: `message`, `song`, `artist`,
`confidence`, `best_hit`, `hits`, `attempts`, `classification`.

The audio fast-path (non-agentic) emits `classify` → `search` → `answer`. The
agentic path emits the full sequence per attempt.

### Optional (cheap) — `matched_line`

Each `TrackHit` may optionally gain a `matched_line: str | None` set from the
retrieval layer when the search was lyrics/hybrid (the specific line that scored
highest). If present, the frontend uses it directly for snippet highlighting;
otherwise the frontend falls back to a word-overlap heuristic. This is additive
and non-breaking; may be deferred to a follow-up if it complicates the retrieval
path — the frontend works without it.

---

## 2. Frontend — components (`frontend/src/main.jsx`)

### 2.0 Streaming consumption

`handleChat` is rewritten to POST `/chat/stream` and read the response body as a
stream: `res.body.getReader()` + `TextDecoder`, buffering on `\n\n`, parsing each
`data:` frame to an event object. A small helper `readSSE(response, onEvent)`
encapsulates this (reused nowhere else, but kept as a named function for clarity).

The in-flight assistant message gains a `steps: [event]` array and a
`streaming: true` flag. Each incoming non-`answer` event appends/updates a step;
the `answer` event fills the message body, `hits`, `best_hit`, etc., clears
`streaming`, and the session is saved (as today).

### 2.1 `AgentSteps` (skeuomorphic, NOT glass)

Vertical timeline rendered inside the assistant bubble while `streaming`:

- Each step: a status dot (spinner on the active step → checkmark when done) + the
  `human` label. Completed steps are muted; the active step is accent-colored.
- **Smooth transitions (hard requirement):**
  - New step enters with `fadeIn + slideUp` (~0.3s).
  - The **active step's label crossfades** on change — outgoing text fades out,
    incoming fades in (position-absolute overlap), never a hard swap.
  - The connector line between dots grows via a `height`/`scaleY` transition.
  - Spinner → checkmark morphs through `scale` + `opacity`, not an instant icon
    replace.
- On `answer`, the whole block **collapses** GPT-style into a single summary row:
  «✓ Нашёл за N шагов · M попыток» / «✓ Found in N steps · M attempts», expandable
  on click via a smooth `max-height`/opacity collapse. Collapsed by default once
  the answer arrives.

Reduced-motion: honor `prefers-reduced-motion` — transitions degrade to instant.

### 2.2 Liquid glass

Add a reusable `liquidGlass(isDark)` style helper (and/or a `.liquid-glass` CSS
class in the existing style block):

```
backdrop-filter: blur(20px) saturate(1.8);
background: translucent tint (dark/light variants);
box-shadow: inset 0 1px 0 rgba(255,255,255,.5) (specular top highlight)
          + soft outer glow;
border: 1px hairline tinted edge;
```

Applied to:

- **Input bar** — the floating chat composer becomes a glass capsule
  (`border-radius: 20px`) that lifts off the background. On focus, glow/tint
  intensify via a 0.25s transition. Send button stays the existing `cta-v3`.
- **Best-result card** (`best_hit`) — larger than secondary hits, glass backing
  with an accent tint keyed to `confidence` (green = high, amber = medium),
  larger cover, and it hosts the `LyricSnippet`. Secondary hits keep their
  current flat treatment.

Both themes styled; no external assets (CSP-safe, all inline).

### 2.3 `LyricSnippet` (inline, collapsed + expand)

Rendered inside the best-result card, source = `hit.lyrics` (excerpt already
returned by the backend):

- Determine the matched line: use `hit.matched_line` if present, else the line
  with maximum query-word overlap (frontend heuristic).
- Collapsed view: **matched line + one neighbor above/below**. Matched line has an
  accent highlight (background `accentBg`, bold); neighbors are muted. Lyric
  typography (comfortable line-height, vertical rhythm).
- **«показать больше» / «show more»** expands to the full `hit.lyrics` with the
  same highlighting, via a smooth `height` transition.

---

## Testing

- **Backend (`tests/integration`):** a test hitting `/chat/stream` asserts the SSE
  frame sequence for a lyrics query (`classify` → `search`(≥1) → `answer`) and
  that the `answer` payload equals what `POST /chat/` returns for the same input
  (parity test — guards the wrapper refactor). LLM/search are stubbed per the
  existing conftest stubs.
- **Backend (`tests/unit`):** `chat_events` yields well-formed events with a
  non-empty `human` string in both `lang=ru` and `lang=en`; import-safe under the
  conftest heavy-dep stubs.
- **Frontend:** manual verification via the running app (no JS test harness in the
  repo) — drive a lyrics query, confirm streamed steps animate and collapse, glass
  renders in dark+light, snippet highlights and expands. Covered by the `/verify`
  flow at implementation time.

## Rollout / risk

- Old `/chat/` endpoint preserved → safe fallback; if `/chat/stream` fails on the
  client, `handleChat` can fall back to the non-streaming endpoint and render the
  answer without steps.
- Single-file frontend (`main.jsx`) — new components are added inline next to the
  existing chat code, following current in-file conventions (no new files).
