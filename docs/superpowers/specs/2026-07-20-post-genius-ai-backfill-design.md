# Post-Genius AI enrichment backfill — design

## Context

The library on this instance was indexed before several pipeline stages
existed or were reliable:

- `613e3f2` fixed a curl_cffi fingerprint block that could make Genius
  fetches silently fail — songs indexed before that fix may be missing
  Genius facts/producer/label even though a fetch was attempted.
- `0c604c4` moved `fact_relations` (GLiNER2: producer/sample extraction) and
  `lyric_gems` (GLiNER2: named-drop/pop-culture/sample-source "gems") into
  the automatic `library_service._run_ai_tasks` pipeline. Tracks indexed
  before this commit never ran either task.
- `sonic_vibe`, `refined_facts`, `artist_bio` are older stages already in
  `_run_ai_tasks`, but any tracks/artists added or re-indexed outside a full
  run (or added before these stages existed) can still be missing them.

This is a one-off script to catch the existing library up on all of the
above, without re-touching the expensive/stable parts: text embeddings, CLAP
embeddings, and the raw AudioDB/songfacts.com fact-scraping stage
(`fetch_facts_for_artists`, `fetch_facts_for_songs`, `fetch_audiodb_for_artists`
in `library_service._run_facts_stage`) — those are left exactly as they are.

## Scope

One-off script, run manually inside the container
(`docker compose exec musix python -m scripts.backfill_gliner_genius ...`),
against **one account** (`--email` / `--account-id`, same convention as
`backfill_genius_facts.py`). Not wired into the live indexing pipeline.

Two phases, run in order:

1. **Genius force-refresh** — re-fetch Genius facts + producer/label for
   every song in the collection, ignoring the "already processed" skip
   (default `--force`, since the whole point is picking up songs a prior,
   possibly-blocked run silently skipped or missed).
2. **All five `ai_tasks`** (`sonic_vibe`, `refined_facts`, `fact_relations`,
   `lyric_gems`, `artist_bio`) — run via the exact same production code path
   `library_service._run_ai_tasks` uses, in its existing order. Each task's
   own idempotency (skip-if-cached) is left untouched — no cache-busting for
   these five. This means a song where `fact_relations` already ran and
   found nothing stays skipped even though Genius just added new facts to
   it; re-running with `--force-genius` again after this script (or a future
   incremental run) is how that gap eventually closes, and is out of scope
   for this pass.

## Why phase 2 delegates to `LibraryService._run_ai_tasks`

`_run_ai_tasks(collection_name, n_total, lang, job=None)` already
encapsulates exactly what's needed: the LLM-reachability gate
(`is_llm_available()` — skips all five tasks with a log line if no LLM is
configured, matching what a real indexing run would do), the fixed task
order (`sonic_vibe, refined_facts, fact_relations, lyric_gems, artist_bio`),
and per-task job creation/waiting via `ai_indexing_service.start_job` +
`wait_for_job`. `job=None` is already a supported call shape (`_publish()`
no-ops when `job is None`) — progress is still visible via the existing
`logger.info("[enrich] AI task '%s' finished ...")` lines per task.

Reimplementing that loop in the script would duplicate the exact ordering/
gating logic this codebase already has to keep in sync in one place; calling
the private method directly is the pragmatic choice here, same spirit as
scripts already reaching into service internals (`backfill_track_metadata.py`
etc.). Constructing it needs only `LibraryService(db_client=db_client)` — the
method never touches `self.search_service`.

## Components

### 1. `scripts/backfill_gliner_genius.py` (new)

**Account resolution**: same `resolve_account`/`list_accounts`/
`_VALID_USER_ID` pattern as `backfill_genius_facts.py` (small duplicated
block, matching this repo's existing script conventions — no shared helper
module today).

**CLI flags**:
- `--email` / `--account-id` / `--list-accounts`
- `--qdrant-url` (env `QDRANT_URL` default)
- `--lang` (default `ru`) — passed to `fact_relations`/`lyric_gems`/
  `refined_facts`/`sonic_vibe`/`artist_bio`
- `--delay` (default 0.3s) — inter-request delay for the Genius phase only
- `--no-force-genius` — switch phase 1 to skip-already-processed instead of
  forcing (for a future incremental re-run of this same script)
- `--skip-genius` / `--skip-ai-tasks` — run only one phase
- `--dry-run` — print counts for both phases (unique songs for Genius,
  track count for the AI tasks), no network/LLM calls

**Flow**:
1. `MetadataDB.init()`, resolve account → collection, connect
   `QdrantClient(url=..., trust_env=False)`, verify the collection exists.
2. **Phase 1 (Genius)**, unless `--skip-genius`:
   - Reuse `collect_songs`, `process_song`, `log_proxy_diagnostic` imported
     directly from `scripts.backfill_genius_facts` (no copy-paste — this
     script imports that module).
   - `--dry-run`: print the unique-song count and skip straight to phase 2's
     own dry-run branch (no network calls) — `--dry-run` reports both
     phases' workload in one run, it never aborts the script early.
   - Otherwise loop `process_song(..., force=not args.no_force_genius)` over
     every unique `(artist, title)`, same per-song try/except resilience and
     end-of-phase summary (ok/skipped/failed) as the existing script.
3. **Phase 2 (AI tasks)**, unless `--skip-ai-tasks`:
   - `--dry-run`: print the collection's track count (`qdrant.count(...,
     exact=True)` — this is the `n_total` phase 2 would use) and the five
     task names in their run order, then return.
   - `is_llm_available()` check up front: if unreachable, print a clear
     message ("LLM unreachable — skipping sonic_vibe/refined_facts/
     fact_relations/lyric_gems/artist_bio; Genius phase above still ran
     (unless skipped)") and exit phase 2 without calling `_run_ai_tasks`
     (saves the trip through it just to log the same skip internally).
   - Build `DbClient(qdrant_url=args.qdrant_url, collection_name=collection)`
     via `with DbClient(...) as db_client:` (lazy-loads no models — only
     `.qdrant` is touched by these tasks).
   - `service = LibraryService(db_client=db_client)`
   - `n_total = db_client.qdrant.count(collection_name=collection,
     exact=True).count`
   - `await service._run_ai_tasks(collection, n_total, lang=args.lang,
     job=None)`
4. Final summary line: which phases ran, which were skipped by flag, whether
   phase 2 was skipped due to no LLM.

### 2. No `MetadataDB` / `ai_tasks` changes

Everything phase 2 needs already exists (`_run_ai_tasks`, the five
registered tasks, `ai_indexing_service.start_job`). Phase 1 reuses
`backfill_genius_facts.py` as a library import, unchanged.

## Out of scope

- Any change to `_run_facts_stage` (AudioDB / songfacts.com fetching) — left
  exactly as-is, never called by this script.
- Text/CLAP embedding, Qdrant upsert, similarity recompute — untouched.
- Cache-busting `fact_relations`/`lyric_gems`/`sonic_vibe`/`refined_facts`/
  `artist_bio` to force-reprocess already-cached songs/tracks/artists — all
  five keep their existing skip-if-cached behavior in this pass.
- Multi-account / "all accounts" mode — single account per run, matching the
  existing `backfill_*` scripts.
- Live progress bars / SSE — plain `logger.info` lines per task (same as a
  real indexing run without a `job`), plus the phase-1 summary block already
  in `backfill_genius_facts.py`.

## Testing

- Manual: `--dry-run` against the real account to confirm song/track counts
  before committing to a full run.
- Manual: `--skip-ai-tasks` first (Genius-only pass), spot-check a few
  previously-blocked songs now have `song_facts` rows with
  `source='genius.com'` and `track_metadata.producer`/`label` populated.
- Manual: full run (or `--skip-genius` if phase 1 already done), then check
  `ai_indexing_jobs` rows (`status='done'`) for all five task types against
  the collection, and spot-check `songs.producers`/`samples_json`,
  `track_gems`, `sonic_vibes`, `refined_facts`, `artist_bio` tables have new
  rows for tracks/artists that lacked them before.
- No automated tests — matches the existing `backfill_*.py` scripts, none of
  which have unit test coverage today.
