"""One-off repair: drop artist biographies that were researched under ANOTHER
artist's name, so the fixed ``artist_bio`` task regenerates them.

The bio task took its display name from
``MetadataDB.get_distinct_artist_slugs_from_sqlite``, which grouped by slug and
picked ``MAX(tm.artist)`` — the lexicographically largest RAW artist tag among
that slug's tracks. For anyone credited only in a TITLE («Kanye West — FML (ft.
The Weeknd)») no raw tag names them at all, so the slug inherited the headline
artist's name; ``name_for_slug`` then failed to resolve it and the code fell
back to that foreign tag. The task researched the wrong person and stored the
result under this slug — with the seed bio still coming from the slug's own
AudioDB row, which is why some of them open with the model apologising that the
supplied facts are about somebody else.

This script replays that exact name resolution against the current mirror. A
bio is deleted only when the name the old code would have used does NOT
canonicalize back to its own slug — i.e. the bio provably describes a different
artist. Slugs with no ``track_metadata`` rows are skipped (nothing to replay).

Idempotent; dry-run by default, pass ``--apply`` to delete.

Run inside the container (the metadata DB path resolves there):

    docker exec musix python scripts/repair_artist_bio_names.py           # dry-run
    docker exec musix python scripts/repair_artist_bio_names.py --apply

Then re-run the artist_bio AI-indexing task to refill the deleted rows.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.resources.metadata_db import MetadataDB  # noqa: E402
from app.services.artist_split import canonical_slug, name_for_slug  # noqa: E402


def old_names_by_slug(conn, collection: str) -> dict[str, str]:
    """Replay the pre-fix query: MAX(raw artist tag) per slug."""
    rows = conn.execute(
        """SELECT tas.artist_slug, MAX(tm.artist)
             FROM track_artist_slugs tas
             JOIN track_metadata tm
               ON tas.collection_name = tm.collection_name
              AND tas.track_id = tm.track_id
            WHERE tas.collection_name = ?
            GROUP BY tas.artist_slug""",
        (collection,),
    ).fetchall()
    return {r[0]: (r[1] or "") for r in rows if r[0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="actually delete rows")
    ap.add_argument("--collection", help="limit to one collection")
    args = ap.parse_args()

    MetadataDB.init()
    conn = MetadataDB.get()

    collections = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT collection_name FROM artist_bios ORDER BY 1",
        )
    ]
    if args.collection:
        collections = [c for c in collections if c == args.collection]

    total_bios = 0
    total_bad = 0
    total_skipped = 0
    for collection in collections:
        old_names = old_names_by_slug(conn, collection)
        # The name the FIXED resolver gives — i.e. what the regenerated bio
        # will actually be researched under.
        new_names = {
            r["slug"]: r["name"]
            for r in MetadataDB.get_distinct_artist_slugs_from_sqlite(collection)
        }
        bios = conn.execute(
            "SELECT artist_slug, lang FROM artist_bios WHERE collection_name = ?",
            (collection,),
        ).fetchall()
        bad: list[tuple[str, str, str, str]] = []
        skipped = 0
        for slug, lang in bios:
            raw = old_names.get(slug)
            if raw is None:
                skipped += 1
                continue
            used = name_for_slug(raw, slug) or raw or slug
            if canonical_slug(used) == slug:
                continue
            bad.append((slug, lang, used, new_names.get(slug, slug)))

        total_bios += len(bios)
        total_bad += len(bad)
        total_skipped += skipped
        print(f"[{collection}] bios={len(bios)} misattributed={len(bad)} "
              f"not-replayable={skipped}")
        for slug, lang, used, correct in sorted(bad):
            print(f"    {slug} ({lang}): researched as {used!r}, is {correct!r}")
        if args.apply and bad:
            conn.executemany(
                "DELETE FROM artist_bios "
                "WHERE artist_slug = ? AND collection_name = ? AND lang = ?",
                [(slug, collection, lang) for slug, lang, _u, _c in bad],
            )
            conn.commit()

    print(
        f"[repair] {total_bad} of {total_bios} biographies "
        f"{'deleted' if args.apply else 'to delete'} "
        f"({total_skipped} skipped: no mirror rows to replay)"
        + ("" if args.apply else "  — dry-run, pass --apply to write")
    )
    if args.apply and total_bad:
        print("[repair] re-run the artist_bio task to regenerate them")


if __name__ == "__main__":
    main()
