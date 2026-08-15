# Porting the lab assistant into production

**Date:** 2026-08-15
**Status:** approved, implementation in progress
**Supersedes (in the assistant tab only):** `2026-07-27-unified-ai-assistant-design.md`,
`2026-08-03-assistant-llm-routing-and-fact-explain.md`

## Goal

`lab/agent/` is a deterministic music assistant that reads the web and the library and
makes no decision a model can talk it out of. It works, its thresholds were calibrated by
hand against real runs, and it now has to become the engine behind the **Assistant** tab —
replacing `assistant/router.py` + `assistant/intent_llm.py` + `assistant/facts_executor.py`
and the `chat_search_service` delegation.

The notebook that drove the lab is retired. Nothing in this design needs to stay
line-comparable with `lab/`; the code is dissolved into the production layering instead of
landing as a foreign package.

## What ships

Four intents behind one input field:

| intent | what it does | model calls |
|---|---|---|
| `general` | question about an artist, a song, an incident — library facts + web, numbered evidence pack, citations verified in code | plan, answer, (next queries) |
| `playlist` | "songs from X" / "hits of Y" — structured tracklists + web prose, matched against the library, weighted, triaged, curated | plan, extract, triage, curate |
| `lyrics_search` | "the line about concrete and sky" — Qdrant dense+BM25 over lyrics, cross-encoder rerank over lyric windows | plan, answer |
| `audio_search` | "calm Sade songs" — CLAP text→audio, four rephrasings, RRF | plan, CLAP rephrase |

Plus the pinned-subject entry points the current UI already has (`focus_fact`,
`subject_track_id`, `subject_artist_slug`), re-targeted at the new engine.

## Tuning knobs

The config values this ships with, all at the top of `app/services/assistant/config.py`:

```
ce_threshold_docs     = 0.35   # download this page? (title + snippet)
ce_threshold_chunks   = 0.65   # put this passage in front of the model?
ce_threshold_facts    = 0.25   # library facts
fetch_refill_attempts = 4      # replace pages that failed to fetch
dedup_chunks          = True
dedup_pool_factor     = 3      # pool deeper than the pack, so a freed slot refills
dedup_thresholds      = {"dense": 0.95, "milco": 0.90}   # duplicate = BOTH agree
dedup_prefer_longer   = 1.2    # a copy takes the slot when it is this much longer
```

New knobs added by this work:

```
facts_use_refined     = False  # raw song_facts/artist_facts, not the AI-rewritten ones
clap_queries          = 4
clap_limit_per_query  = 10
clap_result_count     = 15
clap_rrf_k            = 60
reddit_cooldown       = 60.0   # seconds; NOT a wait — see "Reddit"
lyrics_pool           = 40     # Qdrant candidates before the reranker
lyrics_ctx_hits       = 12     # tracks in the answer prompt (was MAX_CTX_HITS)
lyrics_window_words   = 24     # rerank window over a lyric
lyrics_window_stride  = 12
```

Everything else keeps the lab default.

## Layering

Dissolved into the production layers rather than kept as a package.

### `app/resources/` — wrappers over external systems

- **`model_registry.py`** — gains MILCO and the cross-encoder alongside Octen, plus
  `encode_sparse()` and `ce_probabilities()`. Same double-checked-locking pattern as
  `get_text_model()`. It becomes the single owner of every resident model; nothing else
  instantiates a `SentenceTransformer` or an `AutoModel`.
- **`web_fetch.py`** — the fetcher cascade, bot-wall detection, the table-aware markdown
  extractor, MediaWiki `api.php`, and the Reddit Atom reader. From `lab/websearch_lab.py`
  (fetch + extract halves), `lab/agent/fetch.py`, `mediawiki.py`, `reddit_rss.py`.
- **`searxng_client.py`** — the raw SearXNG call: pacing, engine whitelist, host-takeover
  and per-host caps, spam attribution, `unresponsive_engines` diagnostics. From the lower
  half of `lab/agent/sources.py` plus `spam.py`.

### `app/services/` — business logic

- **`text_normalize.py`** gains `similar()`, `strip_qualifiers()`, `title_key()`,
  `to_latin()`. Its existing `fold()` is byte-for-byte the lab's `fold()` and its
  `_CYR_TO_LAT` map is identical, so nothing is duplicated.
- **`retrieval/`** — `hybrid.py` (`HybridRetriever`, `rrf`), `bm25.py`, `diversity.py`,
  `chunking.py`.
- **`library_catalog.py`** — `LibraryCatalog`, `Subject`, track matching. Reads through
  `MetadataDB`; the lab's `sqlite_ro` read-only-DSN machinery does not travel (it existed
  for reading a dump off a read-only share).
- **`assistant/`** — `config.py`, `contracts.py`, `planner.py`, `prompts.py`, `clarify.py`,
  `web_sources.py`, `tracklists.py`, `selection.py`, `facts_source.py`, `timing.py`,
  `agent.py` (orchestrator), `branches/{general,playlist,lyrics,audio}.py`.

Not ported: `sqlite_ro.py`, `lab/agent/cleanup.py` and `spam.py` as separate modules (they
fold into `searxng_client`), `apple_music_playlist.py`, `dump_catalog.py`, and the
`Catalog` / `load_prod_bm25` halves of `websearch_lab.py` — all notebook scaffolding.

## Models

Three residents on the GPU, all fp16 (the box has 12–16 GB):

```
Octen-Embedding-0.6B   fp16  ~1.2 GB   (already resident)
milco-650m             fp16  ~1.3 GB   new
bge-reranker-v2-m3     fp16  ~1.1 GB   new
CLAP                   CPU, resident, unchanged
```

Preloaded in the existing `_preload_models_in_background`, after Octen and before CLAP.

Three deliberate deltas from the lab:

1. **MILCO in fp16, not fp32.** `dedup_thresholds` were calibrated on fp32 scores. fp16
   moves cosines in the fourth decimal; at 0.90 that is noise, but it is a real difference
   and is recorded here rather than discovered later.
2. **`keep_reranker_resident` and `release_reranker` do not travel.** They existed for a
   tight laptop card. Production keeps all three resident.
3. **Degradation stays.** A leg that fails to load is reported by `ModelHub.status()`, the
   retriever ranks on whatever signals remain, and `select_pack` logs that the two-signal
   duplicate rule is not being applied. Startup never fails on it — same posture as Qdrant
   being down.

## Library catalog, facts, multi-tenancy

**A porting bug, fixed here.** `songs` and `artists` are keyed by a **global** slug PK with
`collection_name` as one mutable column; `metadata_db.py` documents that whichever account
indexed a slug last used to steal every other account's visibility, which is why
`fact_visibility` exists. `lab/agent/catalog.py` filters `songs`/`artists` by
`collection_name`. On a single-account dump that works; in production it would hand every
account but one an empty artist index. The production catalog gates on `fact_visibility`
instead:

- tracks — `track_metadata WHERE collection_name = ?` (composite PK, correct as-is);
- song rows — `songs JOIN fact_visibility ON kind='song' AND slug=songs.slug AND collection_name=?`;
- artists — same with `kind='artist'`.

`collection_name` is always `derive_collection_for_user(user)`; a client-supplied
collection name is never accepted.

**Caching.** The catalog builds a token index over the whole library — real work at 5–6k
tracks, and the fuzzy leg degrades to quadratic without it. Cached per `collection_name`
in a module-level cache with a TTL plus explicit invalidation from `library_service` when
indexing completes, mirroring `qdrant_utils.light_points`. The root `conftest` must reset
it, exactly as it resets `light_points`.

**Facts.** `FactsRetriever` pools `song_facts` + `artist_facts` for the subject's slugs,
dedupes on text, ranks with `HybridRetriever` + the cross-encoder, keeps everything above
`ce_threshold_facts`. Two production deltas:

1. `fact_visibility` is applied here too. The lab skips it ("the subject came out of this
   library"), which is sound but costs one join against an invariant this project treats as
   load-bearing.
2. **Raw facts, not refined.** Production prefers AI-rewritten facts elsewhere; the lab
   ranks raw ones and the threshold was measured on those. `facts_use_refined = False`
   makes it a one-word change.

Deliberately **not** in the pool: credits, gems, bios, catalog rows — everything the old
`facts_executor` mixed in. Narrow homogeneous pool is what the threshold was calibrated
against.

## Branches

### `general`

Unchanged from the lab: subject facts + web, up to two iterations, the stop/continue veto
owned by code (the model says "answered", the code checks the best chunk actually cleared
`WEAK_CONTEXT_PROB`), a numbered evidence pack, and an answer with no valid citations
thrown away whole. Reddit is the parachute, deployed once per run and only when nothing
cleared the chunk threshold.

### `playlist`

Unchanged: host-pinned structured sources → table parsing → LLM extraction from prose →
library matching → source weights → discography rescue → LLM triage → curation. Added:
`cover_art_path` enrichment from `track_metadata` before the payload leaves the service.

### `lyrics_search` (new)

Deterministic, shaped like the playlist branch. The retired `chat_search_service` loop
(four attempts, a validator that re-asks) is not reproduced — that architecture is what
this work exists to leave — but its best parts are ported verbatim.

1. **The planner emits two different queries.** `search_query` reads like a line as it
   would appear in a song (it goes to the embedding). `ce_query` states what the line is
   *about* and **carries no artist name** — a name in the cross-encoder pair pulls the
   score toward any text by that artist instead of toward the line. Both forms are shown as
   examples in the prompt, because a 12b model without an example writes the same string
   into both fields.
2. **Qdrant** through `SearchService.search(mode="text")` — RRF dense+BM25, planner filters
   passed through, pool `lyrics_pool` deep.
3. **Rerank over lyric windows.** `TrackHit.track.lyrics` carries the full text with line
   breaks. Split into overlapping windows, score `(ce_query, window)` pairs, take the max
   per track — that is both the track's rank and its `matched_line` for highlighting.
4. **`lyrics_ctx_hits = 12` tracks into the answer prompt** (the old `MAX_CTX_HITS`).
5. **Verification, ported as-is from `chat_search_service`:** `_match_best_hit` (three
   passes — title+artist, title, containment, so "(Remastered)" survives),
   `_quoted_fragments` + `_pick_matched_line` (re-anchor the highlight on the line the
   answer actually quoted, not on the executed query), and "the model named no song → ship
   an empty hit list" so near-misses are never presented as an answer.
6. **Result shape unchanged:** `{message, song, artist, confidence, best_hit, hits}`. The
   existing search card is not rewritten.

### `audio_search` (new)

`Filters.style` — reserved in the lab contracts with the note "the CLAP branch that will
consume it is not built yet" — is that branch's input. The LLM is called twice and neither
call is a judgement.

1. **Planner** → `intent=audio_search`, `filters.artist="Sade"`, `filters.style="спокойные"`
   kept as the user wrote it.
2. **Artist resolved against the library** via the same `_library_artist` helper the
   playlist branch uses («Канье» → "Kanye West"), or the Qdrant filter matches nothing.
3. **CLAP rephrase → four prompts.** The existing `_CLAP_REPHRASE_SYSTEM_PROMPT` (template
   `"This song is a "`, emotions replaced by acoustic proxies, names/titles/lyrics
   stripped) extended from three variants to four: tempo+dynamics, timbre+texture,
   production+key, instrumentation+vocal delivery. The artist name never enters the prompt
   text — it is already doing its work as a filter.
4. **Four searches** through `SearchService.search(mode="audio")`, the artist filter pushed
   into Qdrant via `_build_qdrant_filter_models` rather than applied afterwards, so the
   per-query limit is not spent on other artists.
5. **Dedupe across result sets by `track_id`, RRF over positions** (k = 60, the constant
   used everywhere else here), then truncate to `clap_result_count`.
6. **No validation, and the caption is written by code.** There is nothing for a model to
   judge, and an LLM round trip for one sentence costs seconds.

**Degradation:** when `CLAP_ENABLED` is off or CLAP is unavailable, the intent is not
offered to the planner and "calm Sade songs" falls through to `playlist`, i.e. to the web.

**Payload:** the playlist card, not the search card — this is a list to play and save, and
tracks carry no `reason` (nobody wrote one, and having a model invent them is exactly what
this branch avoids).

## Pinned subject and `focus_fact`

The request fields are unchanged; what changes is who serves them.

**Structural resolution, no name matching.** `focus_fact` arrives from a card where the
track is already known by id; running it through `resolve_subject` would re-guess a settled
fact (the failure mode that once resolved "Amerie" to "Fergie").

- `subject_track_id` → `track_metadata` row (collection-gated) → `artist` + `title` →
  `song_slug = get_song_facts_key(artist, title)`, `artist_slug = primary_artist_slug`.
  `get_song_facts_key` is the one of the three `_slugify` variants that wrote
  `songs`/`song_facts`.
- `subject_artist_slug` → `artists` row → display name.
- Neither resolves → fall back to resolving from the message text.

**The planner does not guess in focus mode.** Intent forced to `general`; `ce_query` is the
fact itself; the model may phrase the web queries but **code injects the artist and title**,
the same discipline as `quote_work` — a 12b model regularly drops the performer and goes off
to explain someone else's sample.

**The web is allowed.** The `general` branch runs normally: the subject's library facts and
fetched pages land in one numbered pack.

**Honest silence survives.** No valid citation → `explained=false` and no answer is written.

**Disambiguate.** `resolve_subject` already returns `how="shortlist"` and
`_pick_from_shortlist` asks the model and checks its answer against the list. When the model
does not pick, a `ClarifyRequest(kind="subject", …)` becomes the existing `disambiguate`
frame and the current UI renders it unchanged.

## API and frontend

`AssistantIntent` becomes `"lyrics_search" | "audio_search" | "playlist" | "general"`.
Breaking, with exactly one consumer, changed in the same commit.

New `answer` payload replacing `facts`:

```
answer: str            grounded: bool         iterations: int
evidence: [{ n, kind: "fact"|"chunk", text, source, url, ce_prob, used }]
subject:  { kind, title, subtitle, artist_slug, track_id, image_path } | null
focus_fact: str|null   explained: bool|null   follow_ups: [str]   notes: [str]
```

- `subject` is **optional** — a pure web answer has none, which is why the card is new
  rather than the old one with a hidden header.
- `related_tracks` (in-library sample/cover links) does **not** carry over; the lab has no
  such stage and inventing one is a feature, not a port. That section disappears from the
  card.
- `follow_ups` survives as one extra key in the JSON the answer call already returns — no
  extra round trip, and the chips under the answer keep working.

The `search` payload is unchanged and serves `lyrics_search`; the `playlist` payload is
unchanged and serves both `playlist` and `audio_search`.

**`AsxAnswerCard` (new).** The hover-the-citation-see-the-chunk behaviour has to keep
working, and today it would not: `asxEnrichCitations` truncates `data-tip` at 240
characters and `.asn-cite::after` is `max-width: 280px`, which fits a one-sentence library
fact but not a web chunk of up to `chunk_max_chars = 1200`. So:

- hover stays — new `.asx-cite`, `max-width: 460px`, ~420-character preview;
- **clicking a citation expands that source in full in the list below and highlights it**,
  so the hover is no longer the only way to read a chunk;
- the source list shows number, a `library`/`web` badge, the domain and page title, a link
  that opens in a new tab, the cross-encoder `p=`, and the full text under a disclosure.

Styles go into `frontend/packages/musix-ui/styles.css` next to the existing `asn-*` rules.
`ASX_INTENT_COLOR`, the intent dots and `ASX_PHRASES` extend from three intents to four.

## Network

**Proxy.** `curl_cffi` and `httpx` take `proxies=get_proxy()` per call.
`trafilatura.fetch_url` cannot: it builds its own `urllib3.PoolManager`, and a `PoolManager`
(unlike a `ProxyManager`) does not read proxy env vars — which compose blanks in the
container anyway, on purpose. So **in production trafilatura is the extractor only and
`curl_plain` does the downloading**: the cascade becomes `curl_plain → curl_chrome124 →
httpx`. Behaviourally near-identical (trafilatura's fetcher is urllib3 with a UA, i.e.
roughly `curl_plain`), and the proxy applies uniformly instead of to two fetchers out of
three.

**Reddit.** `fetch_thread` holds a process-global gate with `time.sleep(wait)`. The gate is
right for a long-lived process; the sleep is not. It becomes "cooldown has not elapsed →
return `None`, log, move on", and `reddit_min_interval` is renamed `reddit_cooldown`
because the field now means something else. The parachute only deploys when the pack is
empty, so a skip means "the parachute did not open", never "the request hung".

## Dependencies

Added to `requirements.txt`: `trafilatura`, `lxml_html_clean`. Already present:
`curl_cffi`, `readability-lxml`, `beautifulsoup4`, `httpx`, `transformers`.

## Tests

- `lab/tests/*` port into `tests/unit/` under the new paths — they are pure, with no
  network and no models: table parsing, chunking, URL canonicalisation, diversity, spam,
  planner validation, catalog matching, sources, timing, bot walls, `title_key`.
- New: lyric-window reranking and `_match_best_hit` for `lyrics_search`; four queries +
  dedupe + RRF + artist filter for `audio_search`; **a two-account regression on the
  `fact_visibility` gate** (the bug found above); sparse/reranker loading in
  `ModelRegistry` under the conftest stubs.
- Every new module must import with `torch` stubbed — heavy imports inside functions only.

## Order of work

1. `app/resources/`: models, `web_fetch`, `searxng_client`.
2. `app/services/retrieval/`, `text_normalize`, `library_catalog`.
3. `assistant/`: contracts, config, planner, `general` + `playlist`, route.
4. `lyrics_search` and `audio_search`.
5. `focus_fact` and pinned subject.
6. Frontend: the new card, four intents, hover + click on sources.
7. `npm run build` + unit tests. After a live check on production, a separate commit
   deletes `facts_executor.py`, `assistant/router.py`, `assistant/intent_llm.py` and `lab/`.

## Out of scope

- `/search` and `/chat` keep using `chat_search_service` — untouched.
- `/recommend/ai-playlist` keeps using `recsys_ai_service` — untouched.
- No overall wall-clock deadline on a run: the lab budgets (iterations, searches, pages)
  are kept as they are, by decision.
