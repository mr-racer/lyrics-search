"""Wipe stored sample relations so the fixed pipeline can re-extract them.

Run this ONLY after ``dry_run_relations.py`` looks right — it throws away the
current ``songs.samples_json`` for the collection and empties ``sample_links``.
Producer credits (``songs.producers`` / ``producers_genius``) are NOT touched:
they come from the same pipeline but are a separate feature, and clearing them
would blank the credits drawer for no reason.

After the wipe, the next ``fact_relations`` AI task re-processes every song
(the task picks up songs with no relations) and rebuilds the read cache with
both directions.

    docker exec musix python scripts/wipe_relations.py                # dry-run
    docker exec musix python scripts/wipe_relations.py --apply
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.resources.metadata_db import MetadataDB  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", help="acct_* collection (default: all)")
    ap.add_argument("--apply", action="store_true", help="actually write")
    args = ap.parse_args()

    MetadataDB.init()
    conn = MetadataDB._connect()

    if args.collection:
        collections = [args.collection]
    else:
        collections = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT collection_name FROM fact_visibility WHERE kind = 'song'",
            ).fetchall()
        ]
    if not collections:
        print("no collections found")
        return 1

    for collection in collections:
        n_links = conn.execute(
            "SELECT COUNT(*) FROM sample_links WHERE collection_name = ?", (collection,),
        ).fetchone()[0]
        n_cached = conn.execute(
            """SELECT COUNT(*) FROM songs s
               JOIN fact_visibility fv ON fv.kind = 'song' AND fv.slug = s.slug
               WHERE fv.collection_name = ? AND s.samples_json IS NOT NULL""",
            (collection,),
        ).fetchone()[0]
        print(f"{collection}: {n_links} sample_links, {n_cached} cached songs")

        if not args.apply:
            continue

        removed = MetadataDB.clear_sample_links(collection)
        conn.execute(
            """UPDATE songs SET samples_json = NULL
               WHERE slug IN (SELECT slug FROM fact_visibility
                              WHERE kind = 'song' AND collection_name = ?)""",
            (collection,),
        )
        conn.commit()
        print(f"  wiped: {removed} links, {n_cached} caches cleared")

    if not args.apply:
        print("\ndry-run — pass --apply to write")
    else:
        print("\nDone. Re-run the fact_relations AI task to re-extract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
