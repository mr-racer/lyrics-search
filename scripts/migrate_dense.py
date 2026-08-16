"""Re-embed the dense leg of every collection onto the pinned text model.

Why a copy and not an edit: Qdrant cannot add a named vector to an existing
collection (``UpdateCollection.vectors`` only carries hnsw/quantization/on_disk
diffs), and 512-dim storage cannot be reused for 1024. So the collection is
rebuilt.

What is preserved, and must be:

* **point ids** — ``track_reactions``, ``playback_events``, ``track_gems``,
  ``playlist_tracks`` and the ``track_metadata`` mirror all key off them. A
  migration that minted fresh UUIDs would silently detach every like, every
  play count and every playlist from its track.
* **CLAP vectors** — carried across as data. They are the only thing in the
  collection that cannot be rebuilt from SQLite or the audio files without
  hours of re-analysis, which is why the dump is written and verified BEFORE
  anything is dropped.
* **payloads** — copied verbatim.

Usage::

    python -m scripts.migrate_dense --dry-run          # report, touch nothing
    python -m scripts.migrate_dense --yes              # migrate every acct_*
    python -m scripts.migrate_dense --collection acct_x --yes
    python -m scripts.migrate_dense --collection acct_x --collection acct_y --yes
    python -m scripts.migrate_dense --resume --yes     # reuse an existing dump

    # End-to-end check WITHOUT touching the real collection: copy N
    # lyric-bearing tracks into a temp octen_sample_* collection through the
    # exact write path above, then self-retrieve each track by a lyric
    # fragment. The temp collection is dropped afterwards (--keep-sample to
    # inspect it by hand).
    python -m scripts.migrate_dense --collection acct_x --sample 50

The dump lives in ``cache/migrate/<collection>.pkl`` and is kept after a
successful run — delete it by hand once the collection looks right.

Once every collection is through, the matching ``facts_acct_*`` collections are
dropped: their vectors came from the previous text model and would be compared
against queries encoded by the new one. They refill on demand.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.resources.qdrant_payload import (  # noqa: E402
    build_text_for_embedding,
    unique_paragraphs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("migrate_dense")

DUMP_DIR = Path(__file__).resolve().parent.parent / "cache" / "migrate"
SCROLL_BATCH = 256
SAMPLE_PREFIX = "octen_sample_"


def _client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=url, trust_env=False, timeout=120)


def _dump_path(collection: str) -> Path:
    return DUMP_DIR / f"{collection}.pkl"


def dump_collection(client: QdrantClient, collection: str) -> list[dict]:
    """Scroll every point out, keeping id + payload + the CLAP vector."""
    rows: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=SCROLL_BATCH, offset=offset,
            with_payload=True, with_vectors=True,
        )
        for p in points:
            vectors = p.vector if isinstance(p.vector, dict) else {}
            clap = vectors.get("clap")
            rows.append({
                "id": str(p.id),
                "payload": p.payload or {},
                "clap": list(clap) if clap is not None else None,
            })
        logger.info("[%s] scrolled %d points", collection, len(rows))
        if offset is None:
            break
    return rows


def recreate_collection(client: QdrantClient, collection: str, *,
                        vector_name: str, vector_dim: int, with_clap: bool) -> None:
    if client.collection_exists(collection):
        client.delete_collection(collection)

    vectors_config = {
        vector_name: models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
    }
    if with_clap:
        vectors_config["clap"] = models.VectorParams(size=512, distance=models.Distance.COSINE)

    client.create_collection(
        collection_name=collection,
        vectors_config=vectors_config,
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    try:
        client.create_payload_index(
            collection_name=collection, field_name="artist_slugs",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] artist_slugs index: %s", collection, e)


def reupload(client: QdrantClient, collection: str, rows: list[dict], *,
             vector_name: str, batch: int) -> None:
    from app.resources.model_registry import ModelRegistry

    texts = [build_text_for_embedding(r["payload"]) for r in rows]
    logger.info("[%s] encoding %d documents…", collection, len(texts))
    # Document side of the asymmetric pair — no instruction prefix, matching
    # what IndexingService.encode_dense writes at index time.
    vecs = ModelRegistry.encode_text(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True,
    )

    for i in range(0, len(rows), batch):
        chunk = rows[i: i + batch]
        points = []
        for row, vec in zip(chunk, vecs[i: i + batch]):
            vector = {
                vector_name: vec.tolist(),
                "bm25": models.Document(
                    text=build_text_for_embedding(row["payload"]), model="Qdrant/bm25",
                ),
            }
            if row["clap"] is not None:
                vector["clap"] = row["clap"]
            points.append(models.PointStruct(
                id=row["id"], vector=vector, payload=row["payload"],
            ))
        client.upsert(collection_name=collection, points=points)
        logger.info("[%s] upserted %d/%d", collection, min(i + batch, len(rows)), len(rows))


def migrate_one(client: QdrantClient, collection: str, *, resume: bool,
                dry_run: bool, batch: int) -> bool:
    from app.resources.model_registry import ModelRegistry

    before = client.get_collection(collection).points_count or 0
    path = _dump_path(collection)

    if resume and path.exists():
        rows = pickle.loads(path.read_bytes())
        logger.info("[%s] reusing dump: %d rows", collection, len(rows))
    else:
        rows = dump_collection(client, collection)
        if len(rows) != before:
            logger.error("[%s] dumped %d rows but the collection reports %d — "
                         "refusing to touch it", collection, len(rows), before)
            return False
        if dry_run:
            with_clap = sum(1 for r in rows if r["clap"] is not None)
            logger.info("[%s] DRY RUN: %d points, %d with CLAP — nothing written",
                        collection, len(rows), with_clap)
            return True
        DUMP_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(rows, protocol=5))
        logger.info("[%s] dump written: %s (%.1f MB)",
                    collection, path, path.stat().st_size / 1e6)

    if dry_run:
        logger.info("[%s] DRY RUN: would rebuild %d points", collection, len(rows))
        return True

    if not rows:
        logger.warning("[%s] empty collection — skipped", collection)
        return True

    with_clap = any(r["clap"] is not None for r in rows)
    recreate_collection(
        client, collection,
        vector_name=ModelRegistry.VECTOR_NAME, vector_dim=ModelRegistry.VECTOR_DIM,
        with_clap=with_clap,
    )
    reupload(client, collection, rows,
             vector_name=ModelRegistry.VECTOR_NAME, batch=batch)

    after = client.get_collection(collection).points_count or 0
    if after != before:
        logger.error("[%s] point count changed: %d → %d. The dump is still at %s",
                     collection, before, after, path)
        return False
    logger.info("[%s] done: %d points, CLAP carried over for %d",
                collection, after, sum(1 for r in rows if r["clap"] is not None))
    return True


def _query_fragment(payload: dict) -> str:
    """A realistic partial-lyrics query: the middle paragraph, capped.

    Deliberately NOT the full document the vector was built from — retrieving
    a track by its own full text would pass even with a broken document-side
    prompt. A middle fragment only ranks first when both sides of the
    asymmetric pair are wired correctly.
    """
    paragraphs = unique_paragraphs(payload.get("lyrics") or "")
    frag = paragraphs[len(paragraphs) // 2] if paragraphs else ""
    return frag[:400]


def sample_check(client: QdrantClient, collection: str, n: int, *,
                 keep: bool) -> bool:
    """End-to-end Octen check on ``n`` tracks, without touching ``collection``.

    Copies the first ``n`` lyric-bearing points into a temp collection through
    the SAME write path the real migration uses (recreate_collection +
    reupload), then self-retrieves every track by a lyric fragment encoded on
    the query side. Passing means: the model loads on this box, both prompt
    sides are wired, the vector name/dim match the schema, and search returns
    the right song.
    """
    from app.resources.model_registry import ModelRegistry

    rows: list[dict] = []
    offset = None
    while len(rows) < n:
        points, offset = client.scroll(
            collection_name=collection, limit=SCROLL_BATCH, offset=offset,
            with_payload=True, with_vectors=True,
        )
        for p in points:
            payload = p.payload or {}
            if len(payload.get("lyrics") or "") < 200:
                continue
            vectors = p.vector if isinstance(p.vector, dict) else {}
            clap = vectors.get("clap")
            rows.append({
                "id": str(p.id), "payload": payload,
                "clap": list(clap) if clap is not None else None,
            })
            if len(rows) >= n:
                break
        if offset is None:
            break

    if not rows:
        logger.error("[%s] no lyric-bearing tracks found — nothing to sample",
                     collection)
        return False
    logger.info("[%s] sampled %d lyric-bearing tracks", collection, len(rows))

    tmp = SAMPLE_PREFIX + collection
    recreate_collection(
        client, tmp,
        vector_name=ModelRegistry.VECTOR_NAME, vector_dim=ModelRegistry.VECTOR_DIM,
        with_clap=any(r["clap"] is not None for r in rows),
    )
    try:
        reupload(client, tmp, rows,
                 vector_name=ModelRegistry.VECTOR_NAME, batch=32)

        frags = [_query_fragment(r["payload"]) for r in rows]
        query_vecs = ModelRegistry.encode_text(
            frags, is_query=True, batch_size=16, convert_to_numpy=True,
        )
        hit1 = hit3 = 0
        for row, qvec in zip(rows, query_vecs):
            res = client.query_points(
                collection_name=tmp, query=qvec.tolist(),
                using=ModelRegistry.VECTOR_NAME, limit=3, with_payload=False,
            )
            ids = [str(pt.id) for pt in res.points]
            hit1 += ids[:1] == [row["id"]]
            hit3 += row["id"] in ids
        total = len(rows)
        logger.info("[%s] self-retrieval on %d tracks: hit@1 %d/%d (%.0f%%), "
                    "hit@3 %d/%d (%.0f%%)", collection, total,
                    hit1, total, 100 * hit1 / total,
                    hit3, total, 100 * hit3 / total)
        # Duplicate uploads of the same song legitimately steal rank 1 from
        # each other, so the pass bar sits on hit@3.
        ok = hit3 / total >= 0.8
        if not ok:
            logger.error("[%s] SAMPLE FAILED — hit@3 below 80%%. Do not run "
                         "the real migration until this is understood.",
                         collection)
        return ok
    finally:
        if keep:
            logger.info("[%s] temp collection kept for inspection: %s",
                        collection, tmp)
        else:
            client.delete_collection(tmp)
            logger.info("[%s] temp collection %s dropped", collection, tmp)


def drop_facts_collections(client: QdrantClient, targets: list[str], *,
                           dry_run: bool) -> None:
    """Drop the ``facts_acct_*`` companions of the migrated collections.

    Not optional housekeeping — correctness. Those collections hold fact
    vectors written by the PREVIOUS text model, and after a model swap the
    query is encoded by the new one. Comparing across two embedding spaces
    returns confident nonsense and logs nothing at all, which is the worst
    failure shape there is.

    Safe to drop: ``facts_index`` fills them on demand, so the first question
    about a subject rebuilds what that subject needs.
    """
    for collection in targets:
        facts = f"facts_{collection}"
        if not client.collection_exists(facts):
            continue
        if dry_run:
            logger.info("[%s] DRY RUN: would drop (stale vectors from the old "
                        "text model)", facts)
            continue
        client.delete_collection(facts)
        logger.info("[%s] dropped — refills on demand", facts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", action="append",
                    help="migrate just this one, repeatable (default: every acct_*)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse an existing dump instead of re-scrolling")
    ap.add_argument("--dry-run", action="store_true", help="report and exit")
    ap.add_argument("--batch", type=int, default=64, help="upsert batch size")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--sample", type=int, metavar="N",
                    help="end-to-end check on N tracks in a temp octen_sample_* "
                         "collection; the real collection is not touched")
    ap.add_argument("--keep-sample", action="store_true",
                    help="with --sample: keep the temp collection for inspection")
    args = ap.parse_args()

    client = _client()
    if args.collection:
        targets = list(args.collection)
    else:
        targets = sorted(c.name for c in client.get_collections().collections
                         if c.name.startswith("acct_"))

    if not targets:
        logger.error("no collections to migrate")
        return 1

    from app.resources.model_registry import ModelRegistry
    logger.info("target model: %s → vector %r (dim %d)",
                ModelRegistry.TEXT_MODEL_NAME, ModelRegistry.VECTOR_NAME,
                ModelRegistry.VECTOR_DIM)
    logger.info("collections: %s", ", ".join(targets))

    if args.sample:
        failed = [c for c in targets
                  if not sample_check(client, c, args.sample,
                                      keep=args.keep_sample)]
        if failed:
            logger.error("SAMPLE FAILED: %s", ", ".join(failed))
            return 1
        logger.info("sample check passed for every collection")
        return 0

    if not args.dry_run and not args.yes:
        print("\nThis DROPS and rebuilds each collection above. CLAP vectors and "
              "point ids are preserved via a dump written first.")
        print("The matching facts_acct_* collections are dropped too — they hold "
              "vectors from the old model and refill on demand.")
        if input("Type 'yes' to continue: ").strip().lower() != "yes":
            logger.info("aborted")
            return 1

    failed = [c for c in targets
              if not migrate_one(client, c, resume=args.resume,
                                 dry_run=args.dry_run, batch=args.batch)]
    if failed:
        logger.error("FAILED: %s", ", ".join(failed))
        return 1

    # Only after every track collection came through: a half-migrated instance
    # should keep whatever it still has.
    drop_facts_collections(client, targets, dry_run=args.dry_run)
    logger.info("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
