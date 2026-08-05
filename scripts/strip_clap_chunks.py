"""Delete the dead ``clap_chunks`` payload field from existing collections.

The indexer used to attach every track's per-chunk CLAP vectors to its Qdrant
payload (``[[512 floats] × N]`` ≈ 100 KB of JSON per track). Nothing ever read
them back, but they rode along on every payload transfer: a 150-hit CLAP search
came back as a ~20 MB response, which is what made a «Поток» chunk take tens of
seconds. The indexer no longer writes the field and every read path excludes it
(``qdrant_utils.PAYLOAD_EXCLUDE_HEAVY``) — this script removes it from data that
was already indexed, so the collection stops carrying the weight on disk and in
RAM too.

Idempotent and safe to re-run: deleting an absent payload key is a no-op, and no
other field is touched.

Usage
-----
  # what would happen, no writes:
  python -m scripts.strip_clap_chunks --all --dry-run

  # one account, by owner email:
  python -m scripts.strip_clap_chunks --email vanya@sus.com

  # every collection on the instance:
  python -m scripts.strip_clap_chunks --all

Run from the repo root inside the container, so ``.env`` and
``cache/metadata.db`` resolve:

  docker compose exec musix python -m scripts.strip_clap_chunks --all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from qdrant_client import QdrantClient, models

from app.resources.metadata_db import MetadataDB

logger = logging.getLogger("strip_clap_chunks")

FIELD = "clap_chunks"
PROBE_LIMIT = 32   # points sampled to decide whether the field is present


def collections_for(args) -> list[str]:
    """Resolve the requested collections to ``acct_{user_id}`` names."""
    if args.collection:
        return [args.collection]
    users = MetadataDB.list_users_with_invite()
    if args.email:
        match = [u for u in users if (u.get("email") or "").lower() == args.email.lower()]
        if not match:
            raise SystemExit(f"no user with email {args.email!r}")
        return [f"acct_{match[0]['id']}"]
    return [f"acct_{u['id']}" for u in users]


def _has_field(qdrant: QdrantClient, collection: str) -> bool:
    """True when any of the first PROBE_LIMIT points still carries the field."""
    points, _ = qdrant.scroll(
        collection_name=collection, limit=PROBE_LIMIT,
        with_payload=[FIELD], with_vectors=False,
    )
    return any((p.payload or {}).get(FIELD) for p in points)


def run(qdrant: QdrantClient, collection: str, *, dry_run: bool) -> bool:
    """Strip one collection. Returns True when a delete was issued."""
    try:
        n_tracks = qdrant.count(collection_name=collection, exact=True).count
    except Exception as e:
        logger.warning("%s: not in Qdrant (%s) — skipping", collection, e)
        return False

    try:
        present = _has_field(qdrant, collection)
    except Exception as e:
        logger.warning("%s: probe failed (%s) — skipping", collection, e)
        return False
    if not present:
        logger.info("%s: %d tracks, no %s payload — nothing to do",
                    collection, n_tracks, FIELD)
        return False

    if dry_run:
        logger.info("%s: would strip %s from %d tracks", collection, FIELD, n_tracks)
        return False

    qdrant.delete_payload(
        collection_name=collection,
        keys=[FIELD],
        points=models.Filter(must=[]),   # empty filter = every point
        wait=True,
    )
    logger.info("%s: stripped %s from %d tracks", collection, FIELD, n_tracks)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="every account on the instance")
    target.add_argument("--email", help="the owner email of a single account")
    target.add_argument("--collection", help="an explicit acct_… collection name")
    parser.add_argument("--qdrant-url",
                        default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    MetadataDB.init()

    qdrant = QdrantClient(url=args.qdrant_url, timeout=600)
    names = collections_for(args)
    logger.info("%d collection(s) to consider", len(names))

    stripped = sum(run(qdrant, c, dry_run=args.dry_run) for c in names)
    verb = "would strip" if args.dry_run else "stripped"
    logger.info("%s %d collection(s)", verb, stripped)


if __name__ == "__main__":
    sys.exit(main())
