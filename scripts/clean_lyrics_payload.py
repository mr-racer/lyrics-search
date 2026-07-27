"""Backfill: run the lyrics sanitizer over already-indexed Qdrant payloads.

``app.indexing.lyrics_sanitizer`` cleans lyrics as they come in, but a library
indexed before it existed still carries the artifacts. This applies the same
function to the stored ``lyrics`` payload of every ``acct_*`` collection.

Measured on the 2026-07-26 production snapshot (main library, 5474 tracks with
lyrics): 1115 bare-CR texts, 112 CJK credit headers, 91 ``[mm:ss]`` timestamps.
The CR case is the visible one — the player does ``lyrics.split('\\n')``, so a
fifth of the library rendered as one unbroken paragraph.

Payload only: the dense text vectors are NOT re-encoded. Removing a handful of
metadata lines moves an embedding by a negligible amount, and re-encoding the
whole library is a far heavier operation than this fix warrants. Re-index if
you want the vectors to match exactly.

    docker exec musix python scripts/clean_lyrics_payload.py           # dry-run
    docker exec musix python scripts/clean_lyrics_payload.py --apply
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient  # noqa: E402

from app.indexing.lyrics_sanitizer import sanitize_lyrics  # noqa: E402

BATCH = 256


def _diff_kinds(before: str, after: str) -> list[str]:
    """Which artifact classes this track actually had."""
    kinds = []
    if "\r" in before:
        kinds.append("cr")
    n_lines_before = len(before.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    n_lines_after = len((after or "").split("\n"))
    if n_lines_after < n_lines_before:
        kinds.append("dropped-lines")
    if len(after or "") < len(before) and not kinds:
        kinds.append("timestamps")
    return kinds or ["whitespace"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--collection", help="single acct_* collection (default: all)")
    ap.add_argument("--apply", action="store_true", help="actually write payloads")
    ap.add_argument("--show", type=int, default=8, help="example changes to print per collection")
    args = ap.parse_args()

    client = QdrantClient(url=args.qdrant_url, timeout=120)
    names = (
        [args.collection] if args.collection
        else [c.name for c in client.get_collections().collections if c.name.startswith("acct_")]
    )
    if not names:
        print("no acct_* collections found")
        return 1

    grand_total = grand_changed = 0
    for name in names:
        total = changed = 0
        shown = 0
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=name, limit=BATCH, offset=offset,
                with_payload=["lyrics", "artist", "title"], with_vectors=False,
            )
            updates = []
            for pt in points:
                payload = pt.payload or {}
                before = payload.get("lyrics") or ""
                if not before:
                    continue
                total += 1
                after = sanitize_lyrics(before) or ""
                if after == before:
                    continue
                changed += 1
                updates.append((pt.id, after))
                if shown < args.show:
                    shown += 1
                    print(f"  [{payload.get('artist')} — {payload.get('title')}] "
                          f"{','.join(_diff_kinds(before, after))}: "
                          f"{len(before)} -> {len(after)} chars")
            if args.apply and updates:
                for pid, after in updates:
                    client.set_payload(
                        collection_name=name, payload={"lyrics": after}, points=[pid],
                    )
            if offset is None:
                break

        grand_total += total
        grand_changed += changed
        print(f"{name}: {changed} / {total} tracks would change"
              + (" — WRITTEN" if args.apply else ""))

    print(f"\ntotal: {grand_changed} / {grand_total}")
    if not args.apply:
        print("dry-run — pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
