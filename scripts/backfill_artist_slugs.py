"""One-shot migration: derive artists/artist_slugs/primary_artist_slug from the
existing ``artist`` payload and push them onto every point — WITHOUT re-encoding
vectors. Also ensures the ``artist_slugs`` keyword index exists.

Writes one ``set_payload`` per point (consistent with ``backfill_sonic_payload.py``).
Fine for a one-shot manual run on libraries up to tens of thousands of points; for
much larger collections consider batching the updates.

Usage
-----
  python -m scripts.backfill_artist_slugs --collection New2
  python -m scripts.backfill_artist_slugs --collection New2 --dry-run
"""
from __future__ import annotations

import argparse
import logging
from typing import Iterable

from qdrant_client import QdrantClient, models

from app.services.artist_split import split_artists, artist_slugs

logger = logging.getLogger("backfill_artist_slugs")

BATCH = 256


def _iter_points(qdrant: QdrantClient, collection: str) -> Iterable[list]:
    offset = None
    while True:
        points, next_offset = qdrant.scroll(
            collection_name=collection,
            limit=BATCH,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            return
        yield points
        if next_offset is None:
            return
        offset = next_offset


def backfill_collection(
    qdrant: QdrantClient, collection: str, *, dry_run: bool = False
) -> int:
    """Return number of points whose payload received artist fields."""
    n_updated = 0
    for batch in _iter_points(qdrant, collection):
        for point in batch:
            raw = (point.payload or {}).get("artist") or ""
            if not raw.strip():
                continue
            # Skip points already backfilled — keeps re-runs O(unprocessed) and the
            # counter honest (also lets this double as a catch-up for new tracks).
            # A full re-backfill after rule changes is handled by re-indexing.
            if (point.payload or {}).get("artist_slugs"):
                continue
            names = split_artists(raw)
            slugs = artist_slugs(raw)
            if not slugs:
                continue
            n_updated += 1
            if not dry_run:
                qdrant.set_payload(
                    collection_name=collection,
                    points=[point.id],
                    payload={
                        "artists": names,
                        "artist_slugs": slugs,
                        "primary_artist_slug": slugs[0],
                    },
                )
    # Ensure the keyword index regardless of dry-run count (idempotent).
    if not dry_run:
        try:
            qdrant.create_payload_index(
                collection_name=collection,
                field_name="artist_slugs",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            logger.warning("artist_slugs index not created (%s): %s", type(e).__name__, e)
    return n_updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    qdrant = QdrantClient(url=args.qdrant_url)
    n = backfill_collection(qdrant, args.collection, dry_run=args.dry_run)
    verb = "would update" if args.dry_run else "updated"
    logger.info("%s %d point(s) in %r", verb, n, args.collection)


if __name__ == "__main__":
    main()
