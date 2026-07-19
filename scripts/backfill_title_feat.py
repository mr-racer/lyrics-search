"""Backfill feat-in-title credits into existing Qdrant payloads + SQLite mirror.

388 of 5630 tracks in the reference library carry the collaboration inside the
TITLE tag («Bangarang (ft. Sirah)», «Ride ft. MUTEMATH»), and in 381 of them
the featured artist is absent from the ``artist`` tag — so the track never
binds to the featured artist's page (binding is a filter over the
``artist_slugs`` payload field, computed at index time from the tag only).

For every ``acct_*`` collection this script re-runs
``artist_split.parse_title_feat`` over each point and, when the title carries
credits, writes:

- extended ``artists`` / ``artist_slugs`` («(with X)» co-leads right after the
  tag artists, feats last, deduped by slug);
- ``feat_artist_slugs`` — which of those slugs are feats (role marker);
- ``title_display`` — the cleaned title (only when it differs).

Payload-only ``set_payload`` — vectors are untouched, nothing is re-embedded.
The SQLite ``track_metadata`` mirror (what the artist page actually reads)
gets the same arrays via a narrow column update + join-table refresh.

Idempotent; dry-run by default, pass ``--apply`` to write.

Run inside the container (Qdrant + SQLite paths resolve there):

    docker exec musix python scripts/backfill_title_feat.py            # dry-run
    docker exec musix python scripts/backfill_title_feat.py --apply
    docker exec musix python scripts/backfill_title_feat.py --apply --collection acct_xyz
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient  # noqa: E402

from app.resources.metadata_db import MetadataDB  # noqa: E402
from app.services.artist_split import (  # noqa: E402
    parse_title_feat,
    tag_feat_slugs,
    _resolved_slug,
    artist_slugs as _artist_slugs,
    split_artists,
)


def compute_update(payload: dict) -> dict | None:
    """New payload fields for one point, or None when nothing to write.

    Mirrors ``indexing_service._build_payload_for_upsert``: base arrays from
    the payload (falling back to re-splitting the raw tag), then title credits
    appended — with-co-leads first, feats last, deduped by slug.
    """
    title = payload.get("title") or ""
    raw_artist = payload.get("artist") or ""
    parsed = parse_title_feat(title)

    artists = list(payload.get("artists") or split_artists(raw_artist))
    slugs = list(payload.get("artist_slugs") or _artist_slugs(raw_artist))

    prev_feat = payload.get("feat_artist_slugs") or []
    # Tag-level feats first («Drake feat. Future» in the artist tag) — mirrors
    # fresh-index behavior in _build_payload_for_upsert.
    _tag_feats = tag_feat_slugs(raw_artist)
    feat_artist_slugs: list[str] = [s for s in slugs if s in _tag_feats]
    for name, is_feat in (
        [(n, False) for n in parsed.with_names]
        + [(n, True) for n in parsed.feat_names]
    ):
        slug = _resolved_slug(name)
        if not slug or slug in slugs:
            # Second run: the feat is already merged into the arrays — keep its
            # existing marker so the recomputed list matches and nothing writes.
            if is_feat and slug in prev_feat and slug not in feat_artist_slugs:
                feat_artist_slugs.append(slug)
            continue
        artists.append(name)
        slugs.append(slug)
        if is_feat:
            feat_artist_slugs.append(slug)
    title_display = (
        parsed.clean_title
        if parsed.clean_title != " ".join(title.split())
        else None
    )

    update: dict = {}
    if artists != (payload.get("artists") or []):
        update["artists"] = artists
        update["artist_slugs"] = slugs
    if feat_artist_slugs != (payload.get("feat_artist_slugs") or []):
        update["feat_artist_slugs"] = feat_artist_slugs
    elif "feat_artist_slugs" not in payload and (update or title_display):
        # Stamp the marker even when empty so serializers know this payload
        # is authoritative (skip on-the-fly re-parsing).
        update["feat_artist_slugs"] = feat_artist_slugs
    if title_display and payload.get("title_display") != title_display:
        update["title_display"] = title_display
    return update or None


def process_collection(client: QdrantClient, name: str, apply: bool) -> tuple[int, int]:
    """Returns (points_seen, points_updated)."""
    seen = updated = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            offset=offset,
            limit=500,
            with_payload=["title", "title_display", "artist", "artists",
                          "artist_slugs", "feat_artist_slugs"],
            with_vectors=False,
        )
        for p in points or []:
            seen += 1
            payload = p.payload or {}
            update = compute_update(payload)
            if not update:
                continue
            updated += 1
            label = f"{payload.get('artist')} :: {payload.get('title')}"
            if not apply:
                shown = {k: v for k, v in update.items()
                         if k in ("feat_artist_slugs", "title_display")}
                print(f"  [dry] {label}  ->  {shown}")
                continue
            client.set_payload(collection_name=name, payload=update, points=[p.id])
            if "artists" in update:
                MetadataDB.update_track_artist_arrays(
                    name, str(p.id), update["artists"], update["artist_slugs"],
                )
        if offset is None or not points:
            break
    return seen, updated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    ap.add_argument("--collection", help="only this collection (default: every acct_*)")
    args = ap.parse_args()

    client = QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"), timeout=60,
    )
    if args.collection:
        names = [args.collection]
    else:
        names = sorted(
            c.name for c in client.get_collections().collections
            if c.name.startswith("acct_")
        )
    total_seen = total_updated = 0
    for name in names:
        print(f"[{name}]")
        seen, updated = process_collection(client, name, args.apply)
        total_seen += seen
        total_updated += updated
        print(f"  {seen} points, {updated} with title credits"
              f"{' (updated)' if args.apply else ' (dry-run)'}")
    print(f"TOTAL: {total_seen} points, {total_updated} "
          f"{'updated' if args.apply else 'would update'}")


if __name__ == "__main__":
    main()
