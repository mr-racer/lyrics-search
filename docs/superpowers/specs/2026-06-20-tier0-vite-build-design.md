# Tier 0 — Vite build around the existing frontend

**Date:** 2026-06-20
**Branch:** `tier0-vite-build`
**Status:** Approved, implementing

## Goal

Eliminate the main cause of frontend lag — in-browser Babel transpilation of a
~13.5k-line JSX block on every page load — by introducing a Vite build step.
Switch React from the `development` UMD build to a bundled production build,
minify and code-split the bundle, and serve the built assets from FastAPI.

**Hard constraint:** this is infrastructure scaffolding only. The UI logic is
NOT changed. The body of the current `<script type="text/babel">` block moves
into `main.jsx` essentially byte-for-byte; the only code edits are ~5 prepended
`import` lines. No router (Tier 1), no file splitting (Tier 2), no CSS rewrite.

## Current state (verified)

- `frontend/index.html`: single file, 15,828 lines, 800 KB.
  - `<style>` block: lines ~18–2278 (~2200 lines of CSS, inline).
  - `<script type="text/babel">`: lines 2279–15826 (~13,550 lines of JSX).
- Dependencies loaded via CDN `<script>` tags: `react@18.3.1`
  (`react.development.js`), `react-dom@18.3.1`, `@babel/standalone@7.29.0`,
  `marked@13.0.0`, `dompurify@3.1.6`.
- The babel block starts with
  `const { useState, useEffect, useRef, useCallback, useMemo, useLayoutEffect, Fragment } = React;`
  and references only the globals `React`, `ReactDOM`, `marked`, `DOMPurify`.
- `const API = window.location.protocol === 'file:' ? 'http://localhost:8000/api/v1' : '/api/v1';`
  — every backend call (including `/covers/...` and audio streams) goes through
  the `/api/v1` prefix. A single dev-proxy rule on `/api` covers everything.
- FastAPI serves the frontend via a catch-all in `app/api/main.py` (~line 349):
  exact static file from `frontend/{full_path}`, else fallback to
  `frontend/index.html` (`FRONTEND_INDEX`, line 44).
- Dockerfile: `COPY frontend/ ./frontend/`. Single-stage Python image.
- `.gitignore` already contains `/node_modules` and `dist/`.
- Node 24 / npm 11 available locally.

## Decisions (from brainstorming)

1. **Layout:** Vite project rooted at `frontend/`. Build output → `frontend/dist/`.
2. **Dev workflow:** Vite dev server with HMR + proxy `/api` → `:8000`.
3. **Dependencies:** bundle React, marked, dompurify from npm (no CDN).
4. **Backup:** keep the current monolith as `frontend/index.html.bak` for the
   life of the branch (in addition to git history) for instant rollback.

## Target layout

```
frontend/
  index.html          # NEW Vite entry: fonts <head> + <div id=root> + module script
  src/
    main.jsx          # the entire old babel block + ~5 import lines on top
    index.css         # the entire old <style> content
  package.json
  package-lock.json
  vite.config.js
  dist/               # build output (gitignored)
  covers/             # unchanged (runtime volume)
  index.html.bak      # backup of the old monolith
```

## Changes

### 1. `frontend/src/main.jsx`
Prepend exactly:
```js
import React from 'react';
import * as ReactDOM from 'react-dom/client';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import './index.css';
```
- `import * as ReactDOM` preserves the existing `ReactDOM.createRoot(...)` call.
- Named `marked` / default `DOMPurify` imports preserve the existing bare
  identifier references (`marked.parse`, `DOMPurify.sanitize`).
- Everything after these imports is the old babel-block body, unchanged.

### 2. `frontend/src/index.css`
The full contents of the old `<style>` block, verbatim.

### 3. `frontend/index.html` (Vite entry)
- `<head>`: keep `charset`, `viewport`, `<title>`, the Google Fonts
  `preconnect` + stylesheet `<link>`s.
- Remove all CDN `<script>` tags and the inline `<style>` / `<script type=text/babel>`.
- `<body>`: `<div id="root"></div>` + `<script type="module" src="/src/main.jsx"></script>`.

### 4. `frontend/package.json`
- deps: `react`, `react-dom`, `marked`, `dompurify`.
- devDeps: `vite`, `@vitejs/plugin-react`.
- scripts: `dev` (vite), `build` (vite build), `preview` (vite preview).
- Pin to the currently-used versions where reasonable (react 18.3.1, marked 13,
  dompurify 3.1.x).

### 5. `frontend/vite.config.js`
```js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
});
```

### 6. `app/api/main.py`
- Point `FRONTEND_INDEX` and the catch-all base directory at `frontend/dist/`
  instead of `frontend/`. Catch-all logic (exact-file → SPA fallback) unchanged.
- Note: `covers/` is currently served via dedicated routes (`/api/v1/covers/...`),
  not the catch-all, so moving the catch-all base to `dist/` does not affect
  cover serving. Verify no other asset depends on the old `frontend/<path>`
  catch-all behaviour.

### 7. `Dockerfile` (multi-stage)
- New first stage `node:24-alpine`: copy `frontend/` (minus dist), `npm ci`,
  `npm run build`.
- Python stage: copy the built `frontend/dist` from the node stage instead of
  `COPY frontend/`. `covers/` remains a volume.

### 8. `.dockerignore` / `.gitignore`
- Ensure `frontend/node_modules` and `frontend/dist` are excluded from the
  Docker build context where appropriate (dist is built inside the image).

## Out of scope (Tier 0 boundary)

- No `react-router` / URL routing (Tier 1).
- No splitting `main.jsx` into modules (Tier 2).
- No TypeScript (Tier 3).
- No CSS refactor — moved verbatim.
- No UI/behaviour changes.

## Verification

1. `cd frontend && npm install` succeeds.
2. `npm run build` completes without errors; `frontend/dist/index.html` +
   hashed JS/CSS assets produced.
3. Run uvicorn, load the app from `dist/`, manually exercise the key surfaces:
   login → library → player (audio playback) → search/chat → recommendations.
   Confirm no console errors and that Babel is gone from the network tab.
4. (Optional) `docker compose build` succeeds with the multi-stage Dockerfile.

## Rollback

`frontend/index.html.bak` holds the old monolith; restoring it (and reverting
the FastAPI/Docker edits) returns to the CDN+Babel setup immediately. Git
history on `main` is the durable fallback.
