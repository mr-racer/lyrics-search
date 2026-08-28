# Model serving layer: one instance of the weights

Date: 2026-08-28 · Branch: `genius-addition`

Written in English on purpose: this layer is meant to be consumed by services
outside MusiX, so its contract needs to read the same to a reader who has never
opened this repository.

## Why

Two problems, unrelated in origin and solved by the same piece of work.

**1. Every model failure is silent, and the same failure means four different
things.** `ce_probabilities()` returns `None` when the cross-encoder is missing
or when scoring raised. Four call sites consume that `None`, and each invents
its own recovery:

| Call site | On `None` | Consequence |
|---|---|---|
| `bio_v2/article.py:82` | `probs = ce_probabilities(...) or [1.0] * len(pool)` | **every candidate scores 1.0 and clears the gate.** A gate that admits everyone when it breaks is not a gate |
| `bio_v2/sources.py:132` (`_gate_hits`) | `return hits[:1]` | conservative, has a comment, still unlogged |
| `bio_v2/sources.py:150` (`_gate_chunks`) | `return chunks` | **every chunk passes**, an ungated page reaches the biography |
| `bio_v2/retrieval.py:150` | `return [], None` | the facet is silently left unanswered |

The first row has already made a measurement lie: four artists "found" a
Wikipedia article that does not exist, and the run looked successful.

The same shape exists on the sparse leg. `encode_sparse()` returns `None` for
four distinct reasons; the retriever drops the signal and continues, which is
correct, but nothing says it happened. `diversity.is_duplicate` then judges
duplicates on one signal instead of two — that behaviour is deliberate and
documented ("Unanimity among the signals that were actually computed", with an
`if not scored: return False` guard), so it needs reporting, not fixing.

**2. The weights are locked inside the player.** `ModelRegistry`
(`app/resources/model_registry.py`) is a process-local singleton. Any other RAG
service on this machine would have to load its own copy: Octen 1.2 GB, MILCO
2.3 GB, bge-reranker-v2-m3 2.2 GB on disk, ~3.5 GB of VRAM resident, on a card
already shared with a separately-launched LLM. That is not affordable, and it is
the whole reason this document exists.

## Target shape

One process owns the weights. Everything else is a client — the player
included, eventually.

```
clients (MusiX, external RAG) ─HTTP─┐
                                    ▼
                       routes (OpenAI / Cohere shaped)
                                    │
                      cache ◀───────┤
                                    ▼
                   queue(leg, priority) ──▶ ONE worker loop
                                                 │
                                    single-thread GPU executor
                                                 │
                              ┌──────────────────┼──────────────────┐
                           Octen 1024        MILCO sparse      bge-reranker
                           (dense)           (280524-dim)      (cross-encoder)
```

The single worker is load-bearing. Three legs share one card, so their forwards
must not overlap; a single-threaded executor gives that for free and, unlike a
plain semaphore, converts the waiting into batching instead of into idle time.

## Part 1 — Error taxonomy

New package `app/resources/models/`:

```python
# errors.py
class ModelError(RuntimeError):
    leg: str    # "dense" | "sparse" | "cross_encoder"
    op: str     # "load" | "encode" | "score"

class ModelUnavailable(ModelError):    # weights never loaded, service unreachable
class ModelOverloaded(ModelError):     # queue full, wait timed out
class ModelOOM(ModelError):            # allocator refused even at one text
class ModelEncodeFailed(ModelError):   # anything else on the encode path
```

Rules:

- `ModelRegistry` and `ModelHub` **raise**. They never return `None` to signal
  failure. `None` (better: an empty result) survives only for "you passed no
  input".
- One logger, `musix.models`, with `extra={"leg", "op", "n", "ms"}`. `exc_info`
  only on `ModelEncodeFailed` — an unavailable leg is a state, not a stack
  trace.
- Counters (`_sparse_encode_failures`, `_ce_encode_failures`,
  `_sparse_oom_retries`) move into a `ModelStats` object. They are already
  surfaced by `GET /search/models/loaded`; that route becomes the one place that
  answers "is the stack actually working".
- `_failed` becomes a **circuit breaker with a TTL**. Today a leg that fails to
  load once is dead until the process restarts. That is wrong even now (an OOM
  at load time can clear when the LLM releases memory) and becomes indefensible
  once the weights live in a separate container that can restart on its own.

Every consumer then makes a written decision instead of an implied one:

| Consumer | Decision |
|---|---|
| `retrieval/hybrid.py` | catch, count, log, drop the signal — the one place where degrading is right. Records how many signals ranked the result so single-signal dedup is visible |
| `bio_v2/article.py` `gate` | **stops admitting everyone**: returns `(None, rejected + ["cross-encoder unavailable"])` |
| `bio_v2/sources.py` `_gate_chunks` | keeps admitting every chunk, logged and counted. Revised while implementing: this gate runs on pages `_gate_hits` has ALREADY judged to be about the artist, so what leaks through is a site's navigation, not somebody else's career — noise the retriever ranks down, against losing the web source entirely. The article gate faces the opposite trade and takes the opposite decision |
| `bio_v2/sources.py` `_gate_hits` | keeps `hits[:1]`, logged and counted |
| `bio_v2/retrieval.py` facets | keeps returning empty, but logs which facet and why |
| `indexing_service.py:721` | does not catch. The job fails loudly — already the stated intent of `encode_documents` ("raise and be re-run, not return less") |
| `facts_index.py:160`, `facts_retrieval.py:259`, `lyrics_search_engine.py:158` | 500 to the caller, not an empty result set |

Two things surfaced only once the legs actually raised, both of them latent
before:

- **Emptiness has to be checked before the load.** `encode_sparse` and
  `ce_probabilities` both called their loader first and only then looked at the
  input. That was invisible while a failed load returned `None` — the same value
  an empty input returns — and became "an empty batch raises when the leg is
  down" the moment it did not. Callers pass empty lists on entirely ordinary
  paths, so the checks moved above the load.
- **The breaker is process-global state that outlives a test.** Its TTL is five
  minutes; a single test that lets a load fail closed the leg for every test
  after it, and the victim then reported a failure it never caused, quoting an
  exception from a test it never ran. Reset in `tests/conftest.py`, autouse,
  next to the light-payload cache for the same reason.

Eight call sites currently reach `ModelRegistry` directly, bypassing
`ModelHub`; they move onto the hub as part of this work:
`indexing_service.py:721`, `facts_index.py:160`, `facts_retrieval.py:259`,
`lyrics_search_engine.py:158`, `bio_v2/article.py:82`,
`bio_v2/retrieval.py:150`, `bio_v2/sources.py:132,150`, and
`scripts/migrate_dense.py:138,284`.

## Part 2 — The batcher

`app/services/model_server/batcher.py`.

One queue per leg, one worker loop, one single-threaded GPU executor shared by
all legs.

```
wait for the first item
  → drain the queue until BATCH_WAIT_MS elapses or MAX_BATCH_TOKENS is reached
  → sort by length (reuse _length_sorted_order)
  → forward
  → scatter results back to each waiter's future
```

**Batch by token budget, not by item count.** A batch cannot be preempted, so
its duration is the head-of-line blocking every other client pays.
`MAX_BATCH_TOKENS` is chosen so no batch runs longer than ~200–300 ms. Item
count cannot express that: eight 2048-token lyrics and eight one-line queries
are the same count and two orders of magnitude apart in cost.

**Two priority lanes.** `interactive` (default) and `bulk`. An external RAG
indexing a corpus sets `bulk` and yields to the player's search. Bulk gets a
guaranteed share — every Nth batch is bulk unconditionally — or a large index
run never finishes while anyone is browsing.

Per-leg budgets stay separate, as they are today. They are not
interchangeable: MILCO's transient grows with the vocabulary (its `mlm_head`
projects every token into the 30522-entry SPLADE-v3 space and the mask multiply
keeps a second copy alive), which is why it is the leg that runs the card out of
memory and why it carries `SPARSE_MAX_LEN=512` / `SPARSE_BATCH=4` while dense
uses `MAX_SEQ_LENGTH=2048` / `ENCODE_BATCH=8` and the cross-encoder
`RERANK_MAX_LEN=512` / `RERANK_BATCH=8`.

The existing OOM behaviour is kept and now runs inside the batcher: an allocator
refusal halves what was actually attempted, keeps the work already done, and
only gives up at a single text.

## Part 3 — The cache

Key: `blake2b(model_name || revision || prompt_side || text)`. Value: fp16
vector. LRU with a cap in megabytes, in process.

- Dense and sparse: cached. For a RAG that re-indexes a corpus or asks the same
  question twice, this is the largest single win available here.
- Cross-encoder: **not** cached. Its inputs are (query, document) pairs; the hit
  rate is approximately zero and the memory is better spent elsewhere.

The prompt side and the model revision are part of the key, not decoration. Drop
the side and the query vector and the document vector of the same string
collide — a silent recall loss with no error anywhere. Drop the revision and a
future model swap is poisoned by its predecessor's cache.

## Part 4 — HTTP surface

`app/api/routes/models_public.py`. OpenAI shape for embeddings, Cohere shape for
reranking — both are what every RAG stack already speaks.

```
POST /v1/embeddings          dense (Octen)
POST /v1/embeddings/sparse   MILCO
POST /v1/rerank              bge-reranker-v2-m3
GET  /v1/models
GET  /health                 legs, counters, queue depth
```

### Two model names, one set of weights

```json
{ "model": "octen-query",
  "input": ["..."],
  "encoding_format": "base64",
  "priority": "interactive" }
```

`octen-query` and `octen-document` name the **same resident model** with its two
prompts. Octen is asymmetric and ships both sides in
`config_sentence_transformers.json` — `query` is an instruction, `document` is a
single space — and mixing them costs real recall.

Every OpenAI client can set `model`. None of them can set `prompt_name`. Routing
the asymmetry through `model` therefore solves it with a standard field, with no
second instance of the weights and no non-standard extension. `/v1/models` lists
both names with `dim: 1024`. `prompt_name` remains accepted as an explicit
override for callers that prefer it.

### `dimensions` is rejected

The OpenAI schema carries a `dimensions` field for Matryoshka models. Octen's
model card documents a fixed 1024 and says nothing about MRL. Any value other
than 1024 returns **400 with an explicit message**. Truncating a non-MRL
embedding degrades it invisibly, which is precisely the failure class this spec
exists to remove.

### Sparse

```json
{ "data": [{"indices": [...], "values": [...]}], "dim": 280524 }
```

280524 = 30522 (SPLADE-v3 English vocabulary, the `pivot_view`) + 250002
(bge-m3-unsupervised / XLM-R vocabulary, the `source_view` columns offset by
30522). This is also Qdrant's sparse-vector format, so a consumer can write it
straight into a collection.

There is no standard here and none is invented: the shape follows what the model
produces. MILCO's own `return_dict=True` term view is available behind
`?as=terms`, but never by default — `_sparse_to_dicts` walks every non-zero in
Python and rebuilds a 280k-entry vocabulary dict on **every call**.

### Errors on the wire

| Exception | HTTP |
|---|---|
| `ModelUnavailable` (cold start, leg never loaded) | 503 + `Retry-After` |
| `ModelOverloaded` (queue full, wait timed out) | 429 + `Retry-After` |
| `ModelOOM`, `ModelEncodeFailed` | 500, body `{"error": {"type": ..., "leg": ...}}` |

No `200` with an empty body, ever. This is Part 1's rule expressed in HTTP.

Truncation is reported explicitly (`"truncated": n` plus a log line). It is not
an exception — the request succeeded — but it is the same class of silent
degradation, and a client that sent 3000-token documents to a 2048-token model
has to be able to find that out.

### Auth

Static bearer token from `.env` (`MUSIX_MODELS_TOKEN`), not the account JWT: the
caller here is a service, not a person. The router registers before the SPA
catch-all, like every other router.

## Part 5 — Deployment

The router is written once and does not change between the two steps.

**Step 1 — mounted inside `musix`.** No new process, no extra VRAM. An external
RAG can already use it. MusiX itself keeps calling `ModelHub` **directly, in
process**; it must not reach its own HTTP endpoint, which on a single worker
loop is a deadlock.

**Step 2 — its own container.** Same image (`image: musix`, different
`command`), so no new layers and no second copy of the weights: the `hf-cache`
volume is shared, mounted read-only everywhere except the owner. MusiX switches
to `RemoteModelHub` via `MUSIX_MODELS_URL`, and `deploy.resources.devices` is
removed from the `musix` service.

Dropping the GPU from `musix` is safe and verified: outside `model_registry.py`
nothing in `app/` touches CUDA. CLAP is pinned to the CPU by design, GLiNER2
loads with `.cpu()`, MILCO already returns its sparse tensor on the CPU
(`reps.append(sparse_rep.cpu())`), so the only GPU tensor the player holds is
`HybridRetriever._dense` — a ~100×1024 matrix that becomes a CPU array when
dense moves behind the client.

Net cost of step 2: the CUDA context moves rather than duplicates, so ~0 VRAM
and ~0 SSD.

## Techniques considered

| Technique | Verdict |
|---|---|
| Dynamic micro-batching | **Taken.** The core of the design. It also removes GPU concurrency as a failure mode: one worker means forwards cannot overlap, so a separate semaphore is unnecessary |
| Token-budget batching | **Taken.** Bounds head-of-line blocking |
| Length-sorted batches | **Already present** (`_length_sorted_order`) |
| Embedding cache | **Taken.** Largest win for a re-indexing RAG |
| Priority lanes | **Taken.** Bulk indexing must not freeze the player |
| Admission control + 429 | **Taken.** Backpressure instead of OOM |
| base64 fp16 on the wire | **Taken.** ~8× smaller than JSON floats |
| `torch.compile` / CUDA graphs | Rejected. Encoder shapes vary continuously; recompilation would eat the gain |
| int8 / fp8 quantization | Rejected. It shifts the cross-encoder's sigmoid, and nine calibrated thresholds are expressed in that scale |
| ONNX Runtime | Rejected. Another export pipeline to keep in sync for a modest gain |
| Model replicas | Rejected. Directly contradicts "one instance of the weights" |
| Matryoshka `dimensions` | Rejected. Not supported by this model (see Part 4) |

## What must not move

- **`VECTOR_DIM = 1024` and the vector name `text`** are baked into every
  `acct_*` collection and into `facts_acct_*`. Changing the model is a re-embed
  (`scripts/migrate_dense.py`), not a config change. The service pins the model
  per route and never accepts a model identity from the client beyond the two
  prompt-side aliases.
- **The nine cross-encoder thresholds** stay on the current model's sigmoid:
  `CE_THRESHOLD_DOCS` 0.35, `CE_THRESHOLD_CHUNKS` 0.65, `CE_THRESHOLD_FACTS`
  0.25, `WEAK_CONTEXT_PROB` 0.45, `WEAK_LOCAL_PROB` 0.25, `LYRICS_MIN_PROB` 0.20
  (`assistant/config.py`), `CE_ARTICLE_GATE` 0.55, `CE_CHUNK_GATE` 0.65,
  `FACET_SENTENCE_FLOOR` 0.30 (`bio_v2/`). Nothing in this spec changes the
  weights, the dtype, or the pooling, so nothing here invalidates them — and
  that is a constraint on future work, not a happy accident.
- **`DEDUP_THRESHOLDS = {"dense": 0.95, "milco": 0.90}`** are cosines of these
  exact two models at these exact settings.
- **`SPARSE_MAX_LEN=512` must be passed explicitly.** MILCO's `encode_text`
  defaults `max_length` to the tokenizer's own ceiling; losing the argument
  quadruples the transient that already runs the card out of memory.

## Risks

1. **Double permutation — the most fragile point.** There is one permutation
   layer today (`_length_sorted_order` plus its inverse). The batcher adds a
   second: which text in a coalesced batch belongs to which waiter. A mistake
   here does not raise; it returns vectors attached to the wrong documents,
   exactly as `encode_sparse`'s docstring warns ("silently, and only visible as
   worse answers"). Permutation tests are mandatory and cheap.
2. **OOM shrink inside a coalesced batch.** A batch assembled from eight clients
   that halves on an allocator refusal must keep the mapping back to all eight.
   This is the seam between two non-trivial mechanisms.
3. **Cancellation.** A client that disconnects while queued must have its future
   dropped from the batch; a batch already in flight must not be discarded
   because one waiter left.
4. **Batching changes the numerics.** The same text in a batch of 4 and a batch
   of 64 differs in the low bits of fp16 (reduction order, padding). This is
   already true today and is not a regression, but it means a cached vector and
   a freshly computed one are not bit-identical, and nothing may depend on exact
   reproducibility.
5. **Bulk-lane starvation** if the priority scheme has no guaranteed share.
6. **Cold start.** Three models take a minute or two. Clients get 503 with
   `Retry-After`, not a timeout.
7. **MILCO relocates itself.** `encode_text` runs
   `self.to("cuda" if torch.cuda.is_available() else "cpu")` on every call,
   ignoring where the registry put it. Moving it to the CPU requires
   `CUDA_VISIBLE_DEVICES=`; `FORCE_CPU` will not do it.
8. **`hf-cache` needs a single writer** once step 2 lands.
9. **Branch drift.** `container-version` is `genius-addition` minus the Genius
   parsing. This work is ported by hand; the branches are not merged wholesale.

## Verification

- `pytest tests/unit tests/integration` — both suites are currently at zero
  failures, so any failure here is a regression, not the environment.
- New unit tests: permutation identity through the batcher (encode N texts of
  mixed length, assert every vector returns to its own input); OOM shrink
  preserves the waiter mapping; cancellation removes a waiter without dropping
  the batch; cache key separates `query` from `document` for the same string.
- Error-path tests: each of the nine consumers, with the leg forced to raise,
  asserted for the decision it is now documented to make — in particular that
  `bio_v2/article.py::gate` returns no article rather than admitting the pool.
- `GET /search/models/loaded` reports the new counters and queue depths.
- Live stack (`scripts/run_docker_tests.sh`) for the HTTP surface: a real
  request through `/v1/embeddings` with `octen-query` and with `octen-document`
  must produce different vectors, and the `octen-document` vector must match
  what the indexing path writes.

## Work breakdown

| # | Step | Estimate | Status |
|---|---|---|---|
| 1 | `errors.py`, `stats.py`; registry and hub raise instead of returning `None` | 0.5 d | **done** |
| 2 | Nine consumers make explicit decisions; `article.gate` stops admitting the pool; `_failed` becomes a TTL circuit breaker | 1 d | **done** |
| 5 | Router, schemas, error-to-HTTP mapping, static token | 1 d | **done** |
| 3 | Batcher: queues, priorities, token budget, cancellation, permutation tests | 1.5 d | not started |
| 4 | Cache | 0.5 d | not started |
| 6 | `RemoteModelHub`, `MUSIX_MODELS_URL`, second container, GPU removed from `musix` | 1 d | deferred |

Step 5 was taken before 3 and 4 by decision: the point of this work is to let
another RAG service reach these weights, and that is step 5. The order costs one
thing, recorded here so it is not discovered later — **there is no GPU
serialisation.** Two concurrent callers put two forwards on a card that is
already holding an LLM, and their transient peaks add up. What the change so far
buys is that the resulting failure is now typed, logged, counted and returned as
a status code rather than absorbed into a plausible-looking 200.

Until the batcher lands, a consumer indexing a corpus should keep its own
embedding concurrency at 1.

## Open question

Whether step 6 lands at all, or external consumers keep calling `musix`
directly. It affects only how `ModelHub` is factored, not the router or the wire
contract.
