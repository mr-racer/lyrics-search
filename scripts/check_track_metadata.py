#!/usr/bin/env python3
"""Quick diagnostic: check track_metadata and track_artist_slugs tables."""

import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from app.resources.metadata_db import MetadataDB

MetadataDB.init()
conn = MetadataDB._connect()

# List all collections in track_metadata
print("=== Collections in track_metadata ===")
rows = conn.execute(
    "SELECT collection_name, COUNT(*) as cnt FROM track_metadata GROUP BY collection_name ORDER BY cnt DESC"
).fetchall()
if not rows:
    print("(empty)")
else:
    for r in rows:
        print(f"  {r[0]}: {r[1]} tracks")

# List all collections in track_artist_slugs
print("\n=== Collections in track_artist_slugs ===")
rows = conn.execute(
    "SELECT collection_name, COUNT(DISTINCT artist_slug) as slug_count, COUNT(*) as rows "
    "FROM track_artist_slugs GROUP BY collection_name ORDER BY rows DESC"
).fetchall()
if not rows:
    print("(empty)")
else:
    for r in rows:
        print(f"  {r[0]}: {r[1]} unique slugs, {r[2]} total rows")

# Check Mattafix specifically
print("\n=== Mattafix check ===")
# Find the collection the user is in
collections = [r[0] for r in conn.execute("SELECT DISTINCT collection_name FROM track_metadata").fetchall()]
for coll in collections:
    # Check if mattafix slug exists in track_artist_slugs
    row = conn.execute(
        "SELECT COUNT(*) FROM track_artist_slugs WHERE artist_slug = ? AND collection_name = ?",
        ("mattafix", coll),
    ).fetchone()
    if row and row[0] > 0:
        print(f"  '{coll}': maffix slug found — {row[0]} tracks")

    # Also check what artist_slugs look like in the payload
    tm_row = conn.execute(
        "SELECT track_id, artist, artist_slugs FROM track_metadata WHERE collection_name = ? LIMIT 3",
        (coll,),
    ).fetchall()
    if tm_row:
        import json
        for tr in tm_row:
            slugs_raw = tr[2]
            try:
                slugs = json.loads(slugs_raw) if isinstance(slugs_raw, str) else slugs_raw
            except Exception:
                slugs = slugs_raw
            print(f"  '{coll}' sample: id={tr[0][:12]}... artist='{tr[1]}' slugs={slugs}")

# Check if artist_slugs are stored at all (non-null, non-empty)
print("\n=== artist_slugs field stats ===")
for coll in collections:
    total = conn.execute(
        "SELECT COUNT(*) FROM track_metadata WHERE collection_name = ?", (coll,)
    ).fetchone()[0]
    has_slugs = conn.execute(
        "SELECT COUNT(*) FROM track_metadata WHERE collection_name = ? AND artist_slugs NOT IN (NULL, '', '[]')",
        (coll,),
    ).fetchone()[0]
    print(f"  '{coll}': {has_slugs}/{total} tracks have artist_slugs")
