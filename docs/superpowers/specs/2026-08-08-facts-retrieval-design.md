# Facts retrieval — design

Date: 2026-08-08
Branch: `facts-retrieval` (off `genius-addition`)

## The problem

A listener taps a fact on the home strip and the assistant restates it:

> **FACT** Michael Jackson признался в своей автобиографии, что при написании этой
> песни он думал о своей давней любви и наставнице Diana Ross.
>
> **ANSWER** При написании этой песни Michael Jackson вдохновлялся мыслями о своей
> давней любви и наставнице Diana Ross.

Four independent causes, found by reading the explain branch
(`app/services/assistant/facts_executor.py`):

1. **`_fact_evidence` (`:903`) throws the material away.** It filters the pack by
   content-token overlap between the *Russian* refined fact and the *English* raw
   sources, threshold `EXPLAIN_MIN_OVERLAP = 2`, with the subject's own name
   subtracted (`:913`). Cross-script overlap only fires on Latin proper names, so
   almost every raw source is dropped as "not about this fact".
2. **The tapped fact is a refined one-liner and has no back-link to its source.**
   `refined_facts.refined_json` is `[{"text", "confirmed"}]` (`metadata_db.py:192`);
   the editorial refine path batches facts in and out with no ids, so the English
   paragraph the one-liner grew from is not recoverable by key.
3. **`_EXPLAIN_SYSTEM` (`:239`) is twelve lines of prohibitions and three of
   guidance, with no positive definition of what an explanation is.** A 12b model
   under that pressure picks the safest legal output: a paraphrase.
4. **`_verify` (`:1024`) accepts it.** A non-empty answer citing one valid number
   passes — and the paraphrase honestly cites `[1]`, which is the fact itself.

The default embedding model makes (1) structural rather than incidental:
`jinaai/jina-embeddings-v2-small-en` (`search_service.py:33`) is English-only, so
nothing in the stack can currently match a Russian sentence to an English one.

## What we are building

Replace the boolean filter with real retrieval over the **raw** fact pool, on a
single multilingual embedding model, and give the explain prompt a positive
definition of what is worth saying.

Refined facts stay exactly where they are: a display surface. The backend reasons
only over raw sources, which are lossless and in the source's own words.

## Decisions

### One embedding model, forever: `Qwen/Qwen3-Embedding-0.6B`

`TEXT_MODELS` collapses to one entry. `SearchService._resolve_model_name`,
`EMBED_MODEL` (instance settings), `TEXT_MODEL` (env) and the wizard's tier slider
all go away.

The Qdrant vector is renamed from the model-derived
`text_{model_name.replace('/','_')}` (`model_registry.py:180`) to a flat `text`.
The name should not encode the model; a future model swap is a re-embed either way.

**Migration is a copy, not an in-place edit.** Verified against the client schema:
`UpdateCollection.vectors` takes `Dict[str, VectorParamsDiff]` and `VectorParamsDiff`
carries only `hnsw_config` / `quantization_config` / `on_disk` — Qdrant cannot add a
named vector to an existing collection, and 512-dim storage cannot be reused for
1024. So `scripts/migrate_dense.py` dumps CLAP vectors + payloads to disk, recreates
the collection with the new schema, re-embeds only the dense leg, and re-upserts.

CLAP is carried across as data — never recomputed. Payloads are additionally
mirrored in SQLite already, but CLAP vectors live only in Qdrant, which is why the
dump happens *before* the drop.

### Model runtime

Loaded in fp16 straight onto the GPU at startup and kept there (~1.2 GB vs 2.4 in
fp32). `max_seq_length = 2048` — set explicitly, because the model's own config
carries a 32768 window and a long lyric would otherwise be encoded in full.
Measured on prod (758 tracks): the longest deduped lyric is 6203 chars, ~1900
tokens, so 2048 covers the whole library; a Cyrillic-heavy outlier in the top 1%
may clip its tail, which is accepted.

Qwen3-Embedding is asymmetric, so `encode_text` grows an `is_query` flag that adds
the instruct prefix on the query side only.

The idle reaper is deleted outright — `_reaper_loop`, `_reap_once`,
`_ensure_reaper`, `TEXT_IDLE_TO_CPU_SEC`, `TEXT_IDLE_UNLOAD_SEC`,
`_REAPER_INTERVAL_SEC`, `_TextModelState.on_gpu`. A permanently resident model has
nothing to reap. `inflight`/`cond` stay as plain synchronisation; `begin_indexing`
/ `end_indexing` become no-ops kept for call-site stability. `FORCE_CPU` keeps
working.

### Paragraph dedup before embedding

`prepare_metadata` already computes `lyrics_chunked` (`qdrant_payload.py:92`) and
nothing reads it. Two defects, both confirmed on 758 prod tracks:

| | |
|---|---|
| `tuple(set(...))` reorders paragraphs | 711/758 tracks (94%) come out in a different order than the source |
| `split('\n\n')` misses CRLF and whitespace-only blank lines | 99/758 collapse to a single paragraph and dedup silently does nothing; a robust split leaves only 30 |

Fixed to: normalise `\r\n` → `\n`, split on `re.split(r"\n\s*\n")`, dedup with
`dict.fromkeys` on a folded key (collapsed whitespace + casefold) while keeping the
first occurrence's original text, and read lyrics with `.get("lyrics", "")` —
`prepare_metadata` has no lyrics gate, so the current bare `rec['lyrics']` is a
latent `KeyError`.

`build_text_for_embedding` then consumes the deduped paragraphs.

Savings measured at 13.5% of characters (472/758 tracks carry duplicate
paragraphs). The point is representational — a chorus repeated six times should not
carry six times the weight in the vector — not throughput.

### Facts retrieval

**Qdrant stores vectors only.** A `facts` collection holding the dense vector and
nothing else. Payload retrieval from Qdrant is slow enough that the codebase
already routes around it (`light_points`' 90 s memoised scroll,
`PAYLOAD_EXCLUDE_HEAVY`, credits read from the SQLite mirror), so text never goes
in. Search returns ids and scores; the texts are joined from SQLite.

Point ids are `uuid5(NAMESPACE, f"{kind}:{row_id}")`. Required, not cosmetic:
`song_facts.id` and `artist_facts.id` are independent autoincrement sequences and
would otherwise collide.

The collection is **shared, not per-account** — a deliberate exception to
`derive_collection_for_user`. `song_facts` / `artist_facts` are themselves a shared
pool keyed by slug, with per-account visibility in `fact_visibility`
(`metadata_db.py:90-98`); duplicating the same embedding into every account's
collection would buy nothing. Isolation is preserved by filtering *results* against
`fact_visibility` in SQLite — a few dozen rows per query.

Filled lazily in a background task on first use for a subject, so there is no new
indexing stage, no backfill gate, and nothing to keep in sync offline.

**BM25 and RRF run in Python**, over the retrieved candidate set, in
`app/services/facts_retrieval.py`. Not Qdrant's server-side BM25: that needs the
text in the collection, which is what we just decided against.
`catalog_search_service._bm25f` is not reused — it is BM25**F**, bound to
`FIELD_WEIGHTS[corpus.kind]` and the catalog's `_Corpus`/`_Doc` shapes. A
single-field BM25 over ~80 short documents is ~30 lines and borrows only
`tokenize` / `translit_variants` from `text_normalize`.

**Cross-language handling is pseudo-relevance feedback, not translation.** The
query is the tapped statement, which is Russian on a Russian UI. The dense leg
handles that natively — Qwen3-Embedding is multilingual. BM25 does not, so its
query is expanded with terms taken from the top dense hits' English text. Zero
extra LLM calls. An LLM translation step stays available as a fallback if this
proves weak, but is not built now.

### The explain branch

`_fact_evidence` is deleted. `_explain_fact` instead retrieves facts similar to the
tapped statement and places them alongside it — including facts belonging to
*neighbouring* entities, which is where the actual explanation often lives, and
which is the same retrieval the redesigned web search will use later.

The main fact stays citable. The consequence is accepted knowingly: a pure
paraphrase can still satisfy `_verify` by citing only `[1]`, so `_verify` gains a
one-line rule — `used` must contain at least one item that is not the main fact.

The web-search trigger moves off `len(evidence) <= 1` (`:1179`) onto retrieval
quality: too few candidates after filtering, or a top score below a threshold.
This is a proxy — a score does not know whether a fact is *explained* — so the
threshold is calibrated against the golden set from stage 0 rather than guessed.

Evidence cap rises from 10 to 20, and the lyrics blob, the catalog line
(`"{title} — {artist}: album …, released …, genre …"`) and gems are excluded from
explain mode. Those are the items that produce endings like «входит в альбом X,
жанр Pop».

### The prompt

`_EXPLAIN_SYSTEM`'s `STYLE` block is replaced by a positive specification: what
counts as an explanation (a cause or a consequence; a concrete detail that cannot
be guessed from the fact; a contradiction of the obvious reading; something that
changes how the song sounds next time), what does not (the fact in other words;
genre/charts/awards without a story; a retelling of the lyrics; general praise),
and a hard requirement that the answer carry at least one specific item absent
from the fact. No few-shot example — deliberately, at the owner's call.

The same criterion is aligned into `_SYSTEM`'s SELECT step (`:150`) so the two
branches do not drift apart in what they consider interesting.

## Order of work

Stages 1–4 ship as one unit: after the vector is renamed, an unmigrated collection
cannot be read.

| Stage | Work |
|---|---|
| 0 | `dry_run_assistant_facts.py` reads `(focus_fact, track_id)` pairs from a file and pins the subject; 15–20 real facts become the golden set |
| 1 | One model; vector renamed; model-choice config and UI removed |
| 2 | ModelRegistry: fp16, GPU-resident, `max_seq_length`, `is_query`, reaper deleted |
| 3 | Paragraph dedup fixed and consumed |
| 4 | `scripts/migrate_dense.py` |
| 5 | `facts` collection: vectors only, UUID5 ids, lazy fill |
| 6 | `facts_retrieval.py`: dense + BM25(PRF) + RRF, texts from SQLite, visibility filter |
| 7 | Explain branch reworked |
| 8 | Prompts |

## Testing

Unit tests run under `tests/conftest.py`'s stubs for `torch` /
`sentence_transformers`, so every new module must keep heavy imports inside
functions. New coverage:

- dedup: order preservation, CRLF input, whitespace-only blank lines, missing
  `lyrics` key
- BM25: scoring and the pseudo-relevance expansion, on a fixed toy corpus
- RRF fusion: rank ordering from two known input rankings
- `_verify`: rejects an answer citing only the main fact
- retrieval: `fact_visibility` filtering removes out-of-account rows

Frontend changes are verified by `npm --prefix frontend run build`.

The live-stack suite (`tests/docker/`) is where the model itself gets exercised;
the migration script is verified by point counts before and after, per collection.
