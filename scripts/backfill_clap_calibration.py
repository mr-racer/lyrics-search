"""Build the CLAP cosine → percentile calibration table for existing collections.

The table (design §5, ``app/services/stream/calibration.py``) is built lazily on
the first ``/stream/next`` after deploy, but that first request would then pay
for a scroll of ~512 vectors. This warms it up ahead of time — and, more
usefully, PRINTS the resulting quantiles so the percentile thresholds baked into
the recsys can be sanity-checked against a real library:

    CLUSTER_MERGE_PCT = 0.985   merge two tracks into one cluster
    FORGIVE_SIM_PCT   = 0.97    «this skip sounds like what you're enjoying»
    EXPLORE_NEG_PCT   = 0.95    a random pick sits inside a negative cluster

If p98.5 comes back at, say, 0.999 the merge threshold is too tight for that
library and the constant needs adjusting — that judgement is exactly what the
printed table is for.

Idempotent: collections with a fresh table are skipped unless ``--force``.

Usage
-----
  # what would happen, no writes:
  python -m scripts.backfill_clap_calibration --all --dry-run

  # one account, by owner email:
  python -m scripts.backfill_clap_calibration --email vanya@sus.com

  # every collection on the instance:
  python -m scripts.backfill_clap_calibration --all

  # rebuild even where the stored table is still fresh:
  python -m scripts.backfill_clap_calibration --all --force

Run from the repo root inside the container, so ``.env`` and
``cache/metadata.db`` resolve:

  docker compose exec musix python -m scripts.backfill_clap_calibration --all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

from qdrant_client import QdrantClient

from app.resources.metadata_db import MetadataDB
from app.services.stream import calibration as calib
from app.services.stream import pools as pools_mod
from app.services.stream import session as session_mod

logger = logging.getLogger("backfill_clap_calibration")

# Percentiles worth eyeballing: the three thresholds the recsys actually uses,
# plus p50 as a sense of where the bulk of the library sits.
REPORT_PCTS = (50, 90, 95, 97, 98, 99)


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


def describe(table: dict) -> str:
    q = table["quantiles"]
    parts = " ".join(f"p{p}={q[p]:.4f}" for p in REPORT_PCTS)
    return (f"n_tracks={table['n_tracks']} sample={table['n_sample']} "
            f"min={q[0]:.4f} {parts} max={q[-1]:.4f}")


def run(qdrant: QdrantClient, collection: str, *, force: bool, dry_run: bool) -> bool:
    """Build (or report on) one collection's table. Returns True when it was built."""
    now = datetime.utcnow()
    try:
        n_tracks = qdrant.count(collection_name=collection, exact=True).count
    except Exception as e:
        logger.warning("%s: not in Qdrant (%s) — skipping", collection, e)
        return False
    if n_tracks < calib.CALIB_MIN_TRACKS:
        logger.info("%s: %d tracks < %d — stays on raw cosine",
                    collection, n_tracks, calib.CALIB_MIN_TRACKS)
        return False

    stored = MetadataDB.get_clap_calibration(collection)
    if stored and not force and calib._is_fresh(stored, n_tracks, now):
        logger.info("%s: table already fresh — %s", collection, describe(stored))
        return False

    if dry_run:
        why = "missing" if not stored else ("forced" if force else "stale")
        logger.info("%s: would build (%s), %d tracks", collection, why, n_tracks)
        return False

    table = calib.build(qdrant, collection, now=now)
    if table is None:
        logger.warning("%s: build failed — leaving on raw cosine", collection)
        return False
    logger.info("%s: built — %s", collection, describe(table))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="every account on the instance")
    target.add_argument("--email", help="the owner email of a single account")
    target.add_argument("--collection", help="an explicit acct_… collection name")
    parser.add_argument("--qdrant-url",
                        default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--force", action="store_true",
                        help="rebuild even when the stored table is still fresh")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    MetadataDB.init()

    qdrant = QdrantClient(url=args.qdrant_url)
    names = collections_for(args)
    logger.info("%d collection(s) to consider", len(names))

    built = sum(run(qdrant, c, force=args.force, dry_run=args.dry_run) for c in names)
    verb = "would build" if args.dry_run else "built"
    logger.info("%s %d table(s)", verb, built)
    if not args.dry_run and built:
        logger.info("Compare the printed quantiles against the thresholds in use: "
                    "CLUSTER_MERGE_PCT=%.3f FORGIVE_SIM_PCT=%.3f EXPLORE_NEG_PCT=%.3f",
                    session_mod.CLUSTER_MERGE_PCT, session_mod.FORGIVE_SIM_PCT,
                    pools_mod.EXPLORE_NEG_PCT)


if __name__ == "__main__":
    sys.exit(main())
