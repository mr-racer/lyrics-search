# Genius facts + producer/label backfill — design

## Context

MusiX already has an artist/song facts pipeline (songfacts.com scrape → SQLite
`song_facts`/`artist_facts` → consumed by the "chat about track" and "explain
this line" LLM prompts in `track_chat_service.py`). It also has unused
`producer`/`label` columns in `track_metadata` that nothing has ever populated.

Genius.com carries three things songfacts.com doesn't: a song description, a
per-line annotation system ("referents") explaining specific lyric fragments,
and structured producer/label credits. A working Genius scraper already exists
at the repo root (`parse_genius.py` + `cli_genius.py`, built and tested ad hoc
against real Genius pages) but isn't wired into the app. This spec turns it
into a one-off backfill script that fills both gaps for an existing library,
reusing the exact slug scheme the chat/explain-line prompts already read from
so no prompt-building code needs to change.

## Scope

One-off backfill script + a small service module, run manually per account
against an already-indexed library. Not part of the live indexing pipeline
(out of scope — could be added later as an `ai_tasks` stage, but that's a
separate change).

## Components

### 1. `app/services/genius_service.py` (new)

Mirrors the shape of `song_facts_service.py` / `artist_facts_service.py`.

**Genius slug builder**
```
build_genius_url(artist: str, title: str) -> str
```
- Normalize artist via existing `artist_split.normalize_artist_name`, then
  reduce to the primary performer via `artist_split.primary_artist` (feat./
  collab artists are dropped — matches how `song_facts_service._artist_query`
  already handles collab tags).
- Replace `&` with `and` **only** in the primary artist name (not the title).
- Genius-specific slugification (distinct from `song_facts_service._slugify`,
  which targets songfacts.com's lowercase-dash convention): title-case each
  word, strip apostrophes/quotes/commas/`?`/`!` (reuse the same punctuation
  set as `_slugify`), join words with `-`, join artist-slug and title-slug
  with `-`, append `-lyrics`.
- Example: `("Dr. Dre", "Still D.R.E.")` → primary artist "Dr. Dre" →
  `https://genius.com/Dr-dre-still-dre-lyrics` (matches the example in the
  original request).

**Page fetch + parse** — adapted from `parse_genius.py`, not a verbatim copy:
- `_fetch_page`, `_parse_preloaded_state`, `_extract_song_info`,
  `_extract_producers`, `_extract_credits`, `_get_referent_links`,
  `_fetch_single_referent`, `_fetch_all_annotations` carry over largely as-is.
- The one behavioral change: outbound requests route through
  `app.services.proxy_config.get_proxy()` (matching `song_facts_service`),
  instead of `parse_genius.py`'s bare `curl_cffi.requests.get`.
- Wrapped for async callers via `asyncio.to_thread`, matching
  `song_facts_service.fetch_song_facts`'s `await asyncio.to_thread(_fetch_song_facts_html, ...)`
  pattern — the backfill script is async like `backfill_account_enrichment.py`.
- A 404 or a page that doesn't resolve to the expected song raises/returns a
  sentinel (`None`) that the caller logs and skips — no retries, no fallback
  search API (per user decision: log-and-skip, not a Genius-search fallback).

**Facts extraction**
```
build_song_facts(parsed: dict) -> list[tuple[str, str]]  # (fact_text, category)
```
- Description → **one fact**, `category="genius_description"`, taken as-is
  (no paragraph splitting — per user decision).
- Each annotation with `status == "ok"` → **one fact**, `category="genius_annotation"`,
  formatted as the literal string:
  ```
  Lyrics string: {fragment_text}. Fact: {annotation_text}
  ```
  where `annotation_text` has markdown/HTML stripped to plain text (links →
  their visible text, no `**`/`_`/`[]()` markup) so it reads cleanly inside an
  LLM prompt.
- This formatting is baked into the stored fact string itself — deliberately,
  so `track_chat_service.resolve_song_facts_list()` (which just does
  `SELECT fact FROM song_facts ...`) picks these up unchanged for **both**
  `mode="song"` chat and `mode="lyric_explain"`, with zero changes to
  `track_chat_service.py`.

**Producer/label extraction**
```
build_producer_label(parsed: dict) -> tuple[str | None, str | None]  # (producer, label)
```
- `producer`: join names from `parsed["producers"]["producer_artists"]` (fall
  back to `producer_roles` values if `producer_artists` is empty) with `", "`.
- `label`: `parsed["credits"]["label"]` as-is. If `None`, stays `None` — no
  fallback through contributors/roles (per user decision).

### 2. `MetadataDB` changes (`app/resources/metadata_db.py`)

- `add_song_facts_batch(slug, collection_name, facts, source=None, category=None)`
  — add the `category` param (currently hardcoded to omit the column, so every
  batch-inserted fact has `category=NULL`). Applies to the whole batch; existing
  callers (`song_facts_service`, `artist_facts_service`-style call sites) are
  unaffected since it defaults to `None`.
- New `update_track_producer_label(collection_name: str, track_id: str, producer: str | None, label: str | None) -> None`
  — a plain
  ```sql
  UPDATE track_metadata SET producer = ?, label = ?
  WHERE collection_name = ? AND track_id = ?
  ```
  Deliberately **not** reusing `upsert_track_metadata`: that method's `INSERT
  ... ON CONFLICT DO UPDATE` writes every column from a full payload dict, so
  calling it with just `{"producer": ..., "label": ...}` would blank out
  `title`/`artist`/`album`/etc. on the existing row. The backfill only ever
  has producer/label to contribute, so it needs a narrow update.

### 3. `scripts/backfill_genius_facts.py` (new)

Modeled directly on `scripts/backfill_account_enrichment.py` — reuses its
account-resolution (`resolve_account`, `list_accounts`, `_VALID_USER_ID`),
Qdrant scroll (`_iter_points`), proxy diagnostic (`log_proxy_diagnostic`), and
CLI/logging conventions.

**CLI flags**: `--email` / `--account-id` / `--list-accounts`, `--qdrant-url`
(env-default), `--delay` (default 0.5s, applied between both song-page and
per-annotation requests), `--dry-run`, `--force` (bypass the already-processed
skip check), `--sample N` (print the first N built Genius URLs and exit
without fetching — the manual slug-check step agreed on before a full run).

**Flow**:
1. `MetadataDB.init()`, resolve account → collection, log proxy diagnostic
   (all copied verbatim from `backfill_account_enrichment.py`).
2. Scroll the collection. Unlike `collect_artists_and_songs` (which only needs
   unique artist/song pairs), this script also needs **track_id** per point
   to write `track_metadata`, so it builds
   `dict[(artist, title) -> list[track_id]]` directly from the payloads
   (`artist`/`title` fields), not deduplicated away.
3. `--sample N`: build Genius URLs for the first N unique `(artist, title)`
   pairs via `genius_service.build_genius_url`, print them, exit. No network
   calls. This is the manual-check step — run once, eyeball the URLs, then run
   for real.
4. `--dry-run`: print counts (tracks/unique songs) and what would run; no
   network calls (matches `backfill_account_enrichment.py`'s dry-run shape).
5. For each unique `(artist, title)`:
   - Compute `song_slug = get_song_facts_key(artist, title)` (imported from
     `song_facts_service` — same function, so the slug matches what
     `track_chat_service.resolve_song_facts_list` queries).
   - Unless `--force`: check `SELECT 1 FROM song_facts WHERE song_slug = ? AND
     source = 'genius.com' LIMIT 1` — skip (log) if already present.
   - Build the Genius URL, fetch + parse (via `genius_service`, wrapped so one
     failure doesn't abort the run — `try/except` per song, matching the
     per-item resilience in `backfill_track_metadata.py`).
   - On fetch/parse failure (404, no `__PRELOADED_STATE__`, wrong song): log a
     warning and add to a `failed: list[str]` accumulator; continue to the next
     song.
   - On success: write facts via `MetadataDB.add_song_facts_batch(...)` (twice
     — once per category) and, for every `track_id` mapped to this
     `(artist, title)`, call `MetadataDB.update_track_producer_label(...)`.
   - `await asyncio.sleep(args.delay)` between songs (mirrors
     `fetch_facts_for_songs`'s inter-request delay).
6. Print a final summary: total songs processed, succeeded, skipped (already
   processed), failed — with the failed `(artist, title)` pairs listed so they
   can be investigated or retried with `--force`.

## Out of scope

- Wiring Genius facts into the live indexing pipeline (`ai_tasks`) — this is a
  manual backfill script only.
- A Genius-search-API fallback for unresolved URLs — log-and-skip only, per
  user decision.
- Deduplication/merging between songfacts.com facts and Genius facts — they
  coexist in `song_facts` distinguished by `source` (`songfacts.com` vs
  `genius.com`) and `category`.
- UI display of `producer`/`label` — those columns aren't read by the frontend
  today; this backfill only populates the data for future use.

## Testing

- Manual: `--sample 20` against a real account/collection, eyeball the URLs
  against actual Genius pages (per the agreed slug-sanity-check step).
- Manual: `--dry-run` against a small test collection to confirm counts.
- Manual: full run against a small account, then verify via `track-chat`
  (`mode="song"` and `mode="lyric_explain"`) that Genius facts appear in the
  LLM context, and check `track_metadata.producer`/`label` in `cache/metadata.db`
  for a couple of known tracks.
- No automated test suite changes planned — this is an offline one-off script
  in the same category as the existing `backfill_*.py` scripts, none of which
  have unit tests today.
