# Stream exploration — letting the wave leave the genre

**Date:** 2026-09-06
**Status:** design approved, implementation in progress
**Scope:** backend (`app/services/stream/`, `stream_service.py`, `metadata_db.py`), one
line of frontend event plumbing. No UI surface, no new user-facing setting.

---

## 1. Context and goal

«Поток» matches well *inside* a sonic region and cannot leave it. The owner's
complaint, verbatim: falling into a musical direction produces good matches — better
than big-tech — but climbing out is hard, and most of the library is never offered.
Simply injecting unrelated tracks is not acceptable; the listener would reject it.

This spec does **not** rebuild the recommender. It removes four specific mechanics
that switch exploration off, and changes *what* fills an exploration slot from
"random track" to "a track from a controlled similarity band". The near-field
behaviour that already works is untouched.

## 2. Evidence from production

Measured 2026-09-06 against the live library
(`acct_c2b5b12d55d341feb0940929e5a12c0d`, 5959 tracks, 3428 playback events across
174 sessions and 55 active days).

**Logging hygiene is already good.** Zero events with a missing or zero `total_dur`,
zero with `played_sec <= 0`, zero clock anomalies. `interacted`, `influence` and
`skipped_early` are 100% populated. `sonic_axes` is present on all 5959 tracks.

**Exploration is off two thirds of the time.** 115 of 174 sessions reach the 8
signals that end the warmup ramp, and those post-warmup stretches carry **67.4%** of
all listening. In that state `explore_share_for_warmth` returns 0.12, `max`-ed with
`FRESH_STRATIFIED_SHARE = 0.25` to 0.25, and:

```
fresh_quota  = round(3 * (1 - 0.30)) = 2
n_stratified = int(round(2 * 0.25)) = int(round(0.5)) = 0      # banker's rounding
```

The filter-bubble insurance serves **zero** tracks. The only other trigger,
`skip_burst`, would fire in **5.7%** of chunk positions.

**The library is closing.** New tracks as a share of weekly plays: 88.7 → 70.7 →
58.8 → 73.4 → 85.8 → 56.2 → 37.5 → 55.9 → **26.4%**. **3855 of 5959** tracks have
never played. `STRATIFIED_SCROLL_CAP = 5000` against a 5959-track library leaves 959
tracks unreachable by the explore sampler entirely.

**The stickiness is sonic, not per-track or per-artist.** Only 47 tracks played 5+
times (9.4% of plays) while 1318 played exactly once; the median 20-track window
holds 12 distinct artists. Assembly rules are fine — candidate *generation* is the
problem, which is where CLAP kNN sits.

**New material is not rejected.** Skip rate by familiarity: first play **16.0%**,
2nd–5th 16.7%, 6th+ **25.0%**. Novelty is accepted as readily as the familiar, and
over-repetition is what actually gets skipped. Caveat: these are survivors of a
strict kNN filter, so this licenses *band* widening, not random injection.

**A ranker is premature.** 2476 positives / 565 negatives, and
`LONG_TERM_EVENT_CAP = 2000` already truncates the 3428-event history — the
long-term profile sees roughly one month. Enough for offline replay and constant
tuning; not enough for a GBDT.

**The one real logging gap.** `playback_events` records no provenance. `influence=0`
covers 48 of 3428 events, so 98.6% of listening cannot be attributed to the wave
versus a manual play, a playlist or search. In a radio surface every served track is
played, so plays *are* impressions and the target is computable — provenance is the
only missing field, and it blocks both per-generator measurement and edit 4 below.

## 3. Changes

### 3.1 Record the provenance of every listen

Add `playback_events.source TEXT` through the existing idempotent
`ALTER TABLE ADD COLUMN` list (no migration table in this project). Thread it:

- `PlaybackEventIn.source: str | None` — mirrors the existing `influence` field.
- `playback_service.record_event(..., source=...)` → `MetadataDB.record_playback_event`.
- `MetadataDB.get_playback_signals` selects it; `PlaybackSignal.source: str | None`.
- Frontend: `setSrc` stamps `el.dataset.playSource` from the track's `pool` — already
  delivered to the client on every stream track (`recommend.py`, `pool=c.pool`) —
  next to the existing `playNoInfluence` stamp; `flushAccumulatedListen` reads it.

Values are the existing pool labels (`fresh`, `familiar`, `liked`, `replay`, plus the
new `band`) and `manual` for anything started by hand. Legacy rows stay `NULL` and
are treated as `manual`, i.e. never as exploration.

This is invisible to the user. It exists to make edit 4 possible and to make
acceptance-rate-per-generator measurable.

### 3.2 An honest exploration quota

The share is not raised; the rounding that discards it is removed. The engine is
stateless, so the quota is derived deterministically from the session's position
rather than from a stored accumulator:

```
E = (1 - liked_share) * FRESH_STRATIFIED_SHARE          # default 0.7 * 0.25 = 0.175
k = len(session_events)
n_explore = floor((k + n) * E) - floor(k * E)
```

Every chunk boundary recomputes it from scratch, and the long-run share is exactly
`E` with no drift. On defaults this yields **one exploratory track roughly every
six** — deliberately modest, and infinitely more than the current zero.

The warmup ramp (`explore_share_for_warmth`) and the `skip_burst` widening keep
feeding `E` as they do today; only the final integer conversion changes.

### 3.3 Band candidates instead of random ones

An exploration slot is currently filled by `stratified_fresh` — a random library
track binned on two axes. That is noise, and noise is what makes widening feel
wrong.

New priority for the slot: candidates drawn from a **similarity band** around the
session's positive centroids — neighbours ranked roughly 150th to 400th, with a
floor on `sim_pct` so a small collection does not fall through into noise. The
random sampler remains as the fallback when the band is empty.

Cost: the positive-centroid search already runs; its `limit` goes from 150 to 400,
still `with_payload=False`, adding ~1250 ids+scores per chunk. Band candidates are
labelled `pool="band"` so 3.1 records them and 3.4 recognises them.

This is the missing middle: not what is already playing, not a stranger.

### 3.4 Exploration is not punished

With provenance available, a skip on a track whose `source` marks it exploratory
(`band`, `fresh` via the stratified sampler) contributes to the session's negative
clusters at **one quarter** weight and is excluded from `negative_track_ids`, the
hard filter that bans a track after ~2 decayed skips.

Rationale: today every failed escape narrows the region an escape could target —
the skip builds a negative cluster, which repels 200 neighbours and fences that area
off through `EXPLORE_NEG_PCT`. Exploration is self-extinguishing. Production data
puts the cost of this rule low: exploratory tracks are first plays, and first plays
skip at 16.0%.

### 3.5 Two one-line ceilings

- `LONG_TERM_EVENT_CAP` 2000 → 6000. At 3428 events the cap binds, so the long-term
  islands — the only force pulling *out* of the session — see about one month.
- `STRATIFIED_SCROLL_CAP` — sample across the whole collection instead of taking the
  first 5000 rows in insertion order, which currently hides 959 tracks.

## 4. Deliberately unchanged

Owner's explicit instruction, plus scope discipline:

- **The sonic axes, including `vocal_lead`.** `W_AXIS = 0.20`, `axis_match_score` and
  the axis profile stay exactly as they are.
- **No new user-facing setting.** No exploration-radius knob; the slider keeps its
  current single meaning.
- Score weights, cluster merge thresholds, fire/water clocks, the calibration table,
  the embedding model, the assembly rules.
- The ranker and the multi-generator architecture — revisited once `source` has
  accumulated history.

## 5. Measurement and rollback

`scripts/dry_run_stream.py` replays real production sessions prefix by prefix through
`next_chunk`, printing each served track with its pool and similarity percentile.
Before/after metrics, with today's baselines already captured in §2:

| metric | baseline 2026-09-06 |
|---|---|
| share of listening with zero explore slots | 67.4% |
| new tracks as share of weekly plays | 26.4% (last week) |
| library ever played | 35.3% (2104 / 5959) |
| skip rate, first play | 16.0% |
| acceptance rate per pool | not measurable — `source` missing |

Rollback: every new number is a module constant beside the existing ones. `E = 0`
restores today's behaviour exactly; the band fetch limit reverts to 150; the `source`
column is additive and inert if unread.

## 6. Testing

Unit tests under `tests/unit/` — all target pure functions in `pools.py`/`session.py`
and need no Qdrant:

- the stride quota: long-run share equals `E`, no drift across chunk boundaries,
  `E = 0` yields zero slots, quota survives varying `n`.
- band selection: rank window respected, `sim_pct` floor honoured, empty band falls
  back to the stratified sampler.
- skip forgiveness: an exploratory skip contributes quarter weight and never enters
  `negative_track_ids`; a non-exploratory skip is unaffected.
- provenance plumbing: `PlaybackSignal.source` round-trips, `NULL` reads as manual.

`npm --prefix frontend run build` is the gate for the one frontend edit.

## 7. Out of scope

Multi-generator candidate architecture (co-listen, artist/producer graph, lyrical
`text` vector), the learned ranker, embedding replacement or vocal-component
nulling. All remain open; none are prerequisites for this change.
