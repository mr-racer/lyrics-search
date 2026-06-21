#!/usr/bin/env python3
"""Backfill SQLite track_metadata from existing Qdrant collections.

Usage:
    python scripts/backfill_track_metadata.py [--collection COLLECTION_NAME]

If --collection is omitted, backfills ALL collections in Qdrant.

Reads all point payloads from Qdrant (lyrics excluded) and writes them to
the track_metadata + track_artist_slugs tables in SQLite.  Idempotent —
safe to run multiple times; existing rows are overwritten.

Requires Qdrant to be running and accessible via QDRANT_URL env var
(defaults to http://localhost:6333).
"""

import argparse
import logging
import os
import sys
import time

# Ensure the project root (parent of scripts/) is on sys.path so that
# `from app.*` imports work when run as `python scripts/backfill_track_metadata.py`
# inside the Docker container (WORKDIR /app, script lives at /app/scripts/).
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient

from app.resources.metadata_db import MetadataDB
from app.resources.qdrant_utils import PAYLOAD_EXCLUDE_LYRICS, scroll_all

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")


def backfill_collection(client: QdrantClient, collection_name: str, batch_size: int = 500) -> int:
    """Scroll a Qdrant collection and upsert every point to SQLite.

    Returns the number of tracks processed.
    """
    MetadataDB.init()

    # Clear existing track_metadata for this collection first
    MetadataDB.clear_track_metadata(collection_name)
    logger.info("[backfill] cleared existing track_metadata for '%s'", collection_name)

    count = 0
    start = time.monotonic()
    for point in scroll_all(
        client, collection_name, batch_size=batch_size,
        with_payload=PAYLOAD_EXCLUDE_LYRICS, with_vectors=False,
    ):
        payload = point.payload or {}
        try:
            MetadataDB.upsert_track_metadata(collection_name, str(point.id), payload)
        except Exception as e:
            logger.warning("[backfill] failed to upsert %s: %s", point.id, e)
            continue
        count += 1
        if count % 500 == 0:
            elapsed = time.monotonic() - start
            rate = count / elapsed if elapsed > 0 else float("inf")
            logger.info("[backfill] %s: %d tracks processed (%.1f/s)", collection_name, count, rate)

    elapsed = time.monotonic() - start
    rate = count / elapsed if elapsed > 0 else 0
    logger.info("[backfill] %s: done — %d tracks in %.1fs (%.1f/s)", collection_name, count, elapsed, rate)
    return count


def main():
    parser = argparse.ArgumentParser(description="Backfill SQLite track_metadata from Qdrant")
    parser.add_argument("--collection", default=None, help="Collection name (default: all)")
    parser.add_argument("--batch-size", type=int, default=500, help="Scroll batch size (default: 500)")
    args = parser.parse_args()

    qdrant_url = os.environ.get("QDRANT_URL")  # reads http://qdrant:6333 in Docker Compose
    try:
        client = QdrantClient(url=qdrant_url)
    except Exception as e:
        logger.error("[backfill] could not connect to Qdrant: %s", e)
        sys.exit(1)

    try:
        collections = client.get_collections().collections
    except Exception as e:
        logger.error("[backfill] could not list collections: %s", e)
        sys.exit(1)

    target_collections = []
    if args.collection:
        names = [c.name for c in collections]
        if args.collection in names:
            target_collections = [args.collection]
        else:
            logger.error("[backfill] collection '%s' not found. Available: %s",
                         args.collection, ", ".join(names))
            sys.exit(1)
    else:
        target_collections = [c.name for c in collections]

    if not target_collections:
        logger.info("[backfill] no collections found — nothing to do")
        sys.exit(0)

    total = 0
    for coll in target_collections:
        n = backfill_collection(client, coll, batch_size=args.batch_size)
        total += n

    logger.info("[backfill] grand total: %d tracks across %d collections", total, len(target_collections))


if __name__ == "__main__":
    main()
