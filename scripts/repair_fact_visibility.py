"""One-off repair: restore per-account ``fact_visibility`` rows.

Background: the shared knowledge tables (artists / songs / *_facts) are keyed
by slug alone; per-account visibility lives in ``fact_visibility``. When that
table was introduced, its one-time backfill copied visibility from the legacy
``artists.collection_name`` / ``songs.collection_name`` columns — which hold
only the LAST WRITER. Any account that indexed an artist/song first and then
had another account re-index it lost its visibility row, so artist images and
facts silently vanished from its UI even though the data and files were intact.

This script re-derives visibility from each account's actual LIBRARY:

- kind='artist': every slug in ``track_artist_slugs`` for the collection;
- kind='song':   ``get_song_facts_key(artist, title)`` for every track in
  ``track_metadata`` (the same function every song-facts writer uses).

Purely additive (INSERT OR IGNORE) and idempotent — safe to re-run any time.

Usage (inside the app container / repo root, so ``app`` imports resolve):
    python scripts/repair_fact_visibility.py [--db cache/metadata.db] [--dry-run]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.song_facts_service import get_song_facts_key  # noqa: E402


def repair(db_path: str, dry_run: bool = False) -> None:
    conn = sqlite3.connect(db_path)
    collections = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT collection_name FROM track_metadata",
        )
    ]
    print(f"collections with tracks: {len(collections)}")

    total_a = total_s = 0
    for coll in collections:
        artist_slugs = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT artist_slug FROM track_artist_slugs "
                "WHERE collection_name = ? AND artist_slug IS NOT NULL AND artist_slug != ''",
                (coll,),
            )
        ]
        song_keys = {
            get_song_facts_key(artist, title)
            for artist, title in conn.execute(
                "SELECT artist, title FROM track_metadata WHERE collection_name = ?",
                (coll,),
            )
            if artist and title
        }
        ins_a = ins_s = 0
        for slug in artist_slugs:
            cur = conn.execute(
                "INSERT OR IGNORE INTO fact_visibility (kind, slug, collection_name) "
                "VALUES ('artist', ?, ?)",
                (slug, coll),
            )
            ins_a += cur.rowcount
        for key in song_keys:
            cur = conn.execute(
                "INSERT OR IGNORE INTO fact_visibility (kind, slug, collection_name) "
                "VALUES ('song', ?, ?)",
                (key, coll),
            )
            ins_s += cur.rowcount
        total_a += ins_a
        total_s += ins_s
        print(f"  {coll}: +{ins_a} artist, +{ins_s} song visibility rows "
              f"({len(artist_slugs)} artists, {len(song_keys)} songs in library)")

    if dry_run:
        conn.rollback()
        print(f"DRY RUN — rolled back (would insert {total_a} artist + {total_s} song rows)")
    else:
        conn.commit()
        print(f"committed: +{total_a} artist, +{total_s} song visibility rows")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="cache/metadata.db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    repair(args.db, dry_run=args.dry_run)
