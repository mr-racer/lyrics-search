"""MusicBrainz verification running BESIDE the fact pipeline, not after it.

Verifying a sampling link costs one HTTP call to MusicBrainz, and MusicBrainz
answers about one request a second. Two placements were possible and only one
of them is free:

* a stage after the AI tasks — simple, but it adds its own minutes to the wall
  clock of an indexing run, and the user waits for them;
* a lane that starts with the extraction and drains while the LLM works — the
  network sits idle during the whole fact run anyway, so the verification is
  paid for out of time that was being spent regardless.

This is the second. :class:`VerifyLane` is fed the links of a song the moment
they are written, checks them on a worker thread (``mb.verify`` sleeps to pace
itself and blocks on urllib — it must never run on the event loop), and
rewrites that song's rows with the survivors. By the time the extraction ends
the queue is normally already empty; ``aclose`` waits for whatever is left.

Links are written unverified FIRST and pruned here, rather than held back
until they are confirmed. If MusicBrainz is unreachable — no network, no
``musicbrainzngs``, a proxy that is down — the lane degrades to doing nothing
and the user keeps the extracted links, which is the pre-verification
behaviour and strictly better than an empty pill.
"""

from __future__ import annotations

import asyncio
import logging

from app.resources.metadata_db import MetadataDB

from . import sample_links as sl

logger = logging.getLogger(__name__)

# Verdicts that already carry proof and need no external opinion: the user owns
# the recording, or several independent facts stated the same link.
_SETTLED = ("in_library", "corroborated")


def _needs_mb(link: dict) -> bool:
    return link.get("reason") not in _SETTLED


def _row_identity(links: list) -> set:
    rows = [sl.to_db_row(lk) for lk in links]
    return {(r["direction"], r["dst_key"], r.get("dst_slug")) for r in rows}


def clean_and_store(collection_name: str, src_slug: str, candidates: list, *,
                    resolve=None, lane=None) -> list:
    """Run the free cleaning tiers over one song's links, store, and queue.

    The single place a song's links are written. Both callers reach it: the
    extraction, with what the model just said, and :func:`seed_collection`,
    with what is already in the table.

    The store is skipped when nothing changed — a re-run over a clean library
    would otherwise be thousands of pointless delete-then-insert transactions.
    """
    cleaned = sl.clean(candidates, resolve=resolve, mb=None)
    keep = [lk for lk in cleaned if lk.get("verdict") != "reject"]
    stored = [lk for lk in candidates if lk.get("_stored")]
    if _row_identity(keep) != _row_identity(stored):
        MetadataDB.replace_sample_links(
            collection_name, src_slug, [sl.to_db_row(lk) for lk in keep],
        )
    if lane is not None:
        lane.submit(src_slug, keep)
    return keep


def seed_collection(collection_name: str, *, resolve=None) -> list:
    """Feed links ALREADY in the table through the same cleaning and checking.

    Without this the pipeline would only ever verify links it extracted in the
    same run, and a library indexed before verification existed could never be
    healed by re-running the task: ``refined_facts`` resumes from
    ``refined_fact_items`` and returns before the writer for every fact it has
    already processed, so the extraction path is never reached for them.

    Cheap to repeat. MusicBrainz verdicts are cached per (artist, title)
    across collections, so a second pass asks the network nothing and a song
    whose rows come back identical is not rewritten.

    Runs on a worker thread (it reads and writes SQLite), so it does NOT touch
    the lane itself: ``asyncio.Queue`` is not thread-safe, and feeding it from
    here would corrupt the loop's wakeup bookkeeping. It returns
    ``[(src_slug, links)]`` for the caller to submit from the event loop.
    """
    try:
        rows = MetadataDB.get_all_sample_links(collection_name)
    except Exception:                                       # noqa: BLE001
        logger.warning("[verify_lane] could not read stored links for %s",
                       collection_name, exc_info=True)
        return 0

    by_src: dict = {}
    for r in rows:
        by_src.setdefault(r["src_slug"], []).append({
            "artist": r.get("dst_artist") or "",
            "title": r.get("dst_title") or "",
            "direction": r.get("direction") or "source",
            "relation": r.get("relation") or "sample",
            "src_slug": r["src_slug"], "src_artist": "", "src_title": "",
            "fact": r.get("evidence") or "",
            "dst_slug": r.get("dst_slug"),
            "confidence": r.get("confidence"),
            "_stored": True,
        })

    out = []
    for src_slug, candidates in by_src.items():
        try:
            keep = clean_and_store(collection_name, src_slug, candidates,
                                   resolve=resolve, lane=None)
        except Exception:                                   # noqa: BLE001
            logger.warning("[verify_lane] seeding %s failed", src_slug,
                           exc_info=True)
            continue
        if keep:
            out.append((src_slug, keep))
    return out


def verify_song_links(collection_name: str, src_slug: str, links: list,
                      mb, resolve=None) -> dict:
    """Check one song's links and rewrite its rows. Runs on a worker thread.

    ``links`` are the cleaned dicts the writer already holds, so this costs no
    read. Returns counters for the log.
    """
    checked = kept = dropped = 0
    survivors = []
    for link in links:
        if not _needs_mb(link):
            survivors.append(link)
            kept += 1
            continue
        checked += 1
        try:
            res = mb.verify(link.get("artist") or "", link.get("title") or "")
        except Exception:                                   # noqa: BLE001
            logger.debug("[verify_lane] MB call raised for %s — keeping link",
                         src_slug, exc_info=True)
            survivors.append(link)
            kept += 1
            continue
        if res.get("verified"):
            link.update(reason="musicbrainz", confidence=0.9)
            survivors.append(link)
            kept += 1
            continue
        if not res.get("checked"):
            # Could not ask (offline, proxy down, no musicbrainzngs). Silence is
            # not a verdict — keep what the extraction produced.
            survivors.append(link)
            kept += 1
            continue
        dropped += 1

    if dropped:
        MetadataDB.replace_sample_links(
            collection_name, src_slug, [sl.to_db_row(lk) for lk in survivors],
        )
    return {"checked": checked, "kept": kept, "dropped": dropped}


class VerifyLane:
    """Per-run queue of songs whose links still need an external opinion."""

    def __init__(self, collection_name: str, *, enabled: bool = True,
                 interval: float = sl.MB_MIN_INTERVAL):
        self.collection_name = collection_name
        self.enabled = enabled
        self._interval = interval
        self._queue: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self._mb = None
        self.totals = {"songs": 0, "checked": 0, "kept": 0, "dropped": 0}

    def start(self) -> None:
        """Build the MusicBrainz client and launch the worker.

        A client that could not import ``musicbrainzngs`` reports
        ``enabled = False``; the lane then stays off rather than queueing work
        nobody will do.
        """
        if not self.enabled or self._worker is not None:
            return
        self._mb = sl.MusicBrainz(
            interval=self._interval,
            cache_get=MetadataDB.get_sample_link_verdict,
            cache_put=lambda a, t, res: MetadataDB.set_sample_link_verdict(
                a, t, verified=bool(res.get("verified")), score=res.get("score"),
                mb_artist=res.get("mb_artist"), mb_title=res.get("mb_title"),
                mbid=res.get("mbid")),
        )
        if not self._mb.enabled:
            self.enabled = False
            logger.info("[verify_lane] musicbrainzngs unavailable — lane off")
            return
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._run())

    def submit(self, src_slug: str, links: list) -> None:
        """Hand one song's freshly written links to the lane. Never blocks."""
        if not self.enabled or self._queue is None or not links:
            return
        self._queue.put_nowait((src_slug, list(links)))

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            if item is None:
                return
            src_slug, links = item
            try:
                got = await asyncio.to_thread(
                    verify_song_links, self.collection_name, src_slug, links,
                    self._mb,
                )
            except Exception:                               # noqa: BLE001
                logger.warning("[verify_lane] song %s failed", src_slug,
                               exc_info=True)
                continue
            self.totals["songs"] += 1
            for k in ("checked", "kept", "dropped"):
                self.totals[k] += got[k]

    async def aclose(self) -> None:
        """Drain what is left and stop. Safe to call on a lane never started."""
        if self._worker is None or self._queue is None:
            return
        self._queue.put_nowait(None)
        try:
            await self._worker
        except asyncio.CancelledError:                      # noqa: PERF203
            pass
        finally:
            self._worker = None
        logger.info(
            "[verify_lane] %s: %d songs, %d links checked, %d kept, %d dropped "
            "(%d MusicBrainz calls)",
            self.collection_name, self.totals["songs"], self.totals["checked"],
            self.totals["kept"], self.totals["dropped"],
            getattr(self._mb, "calls", 0),
        )
