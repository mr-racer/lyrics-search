"""One-shot migration: read sonic_tags from SQLite and push them into the
Qdrant payload for every point in a collection.

Usage
-----
  python -m scripts.backfill_sonic_payload --collection music_explorer
  python -m scripts.backfill_sonic_payload --collection music_explorer --dry-run
"""
from __future__ import annotations

import argparse
import logging
from typing import Iterable

from qdrant_client import QdrantClient

from app.resources.metadata_db import MetadataDB, _slugify

logger = logging.getLogger("backfill_sonic_payload")

BATCH = 256


def _slug_from_payload(payload: dict) -> str | None:
    artist = payload.get("artist")
    title = payload.get("title")
    if not (artist and title):
        return None
    try:
        return _slugify(artist) + "-" + _slugify(title)
    except Exception:
        return None


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
    """Return number of points whose payload received sonic_tags."""
    n_updated = 0
    for batch in _iter_points(qdrant, collection):
        for point in batch:
            slug = _slug_from_payload(point.payload or {})
            if not slug:
                continue
            desc = MetadataDB.get_sonic_descriptor(slug)
            if not desc:
                continue
            tags_obj = desc.get("tags") or []
            sonic_tags = [
                (t["tag"] if isinstance(t, dict) else t) for t in tags_obj
            ]
            if not sonic_tags:
                continue
            n_updated += 1
            if not dry_run:
                qdrant.set_payload(
                    collection_name=collection,
                    points=[point.id],
                    payload={"sonic_tags": sonic_tags},
                )
    return n_updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    MetadataDB.init()

    qdrant = QdrantClient(url=args.qdrant_url)
    n = backfill_collection(qdrant, args.collection, dry_run=args.dry_run)
    verb = "would update" if args.dry_run else "updated"
    logger.info("%s %d point(s) in %r", verb, n, args.collection)


if __name__ == "__main__":
    main()
