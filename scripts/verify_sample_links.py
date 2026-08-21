#!/usr/bin/env python
"""Verify and clean extracted sampling links for one collection.

Runs AFTER indexing, never inside it: MusicBrainz answers about one request a
second, so checking inline would make adding a song wait on the network once
per sampled track. Verdicts are cached in the database, so a second pass — or
an incremental one after new tracks arrive — costs only the new links.

    python scripts/verify_sample_links.py --collection acct_… [--dry-run]

What the three tiers can and cannot prove:
  * the shape checks reject the impossible (an empty side, an artist equal to
    the title, a TV show) and cost nothing;
  * a link resolved against the user's own library is PROVEN — they have the
    file;
  * for everything else MusicBrainz's `recording` entity is the only thing that
    separates a song from an album, and the only check that catches a
    misspelling: "Rogers and Hammerstein" comes back as "Richard Rodgers &
    Oscar Hammerstein II".
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.resources.metadata_db import MetadataDB          # noqa: E402
from app.services.facts_v2 import sample_links as sl      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("verify_sample_links")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="report without writing anything back")
    ap.add_argument("--interval", type=float, default=sl.MB_MIN_INTERVAL)
    ap.add_argument("--no-mb", action="store_true")
    args = ap.parse_args()

    rows = MetadataDB.get_sample_link_sources(args.collection)
    if not rows:
        log.info("no sampling links stored for %s", args.collection)
        return 0

    by_src: dict = {}
    for row in rows:
        by_src.setdefault(row["src_slug"], []).append({
            "src_slug": row["src_slug"], "direction": row["direction"],
            "artist": row.get("dst_artist") or "", "title": row.get("dst_title") or "",
            "relation": row.get("relation") or "sample",
            "fact": row.get("evidence") or "", "src_artist": "", "src_title": "",
        })

    resolve, n_index = sl.library_resolver_from_db(args.collection)
    mb = None if args.no_mb else sl.MusicBrainz(
        interval=args.interval,
        cache_get=MetadataDB.get_sample_link_verdict,
        cache_put=lambda a, t, res: MetadataDB.set_sample_link_verdict(
            a, t, verified=bool(res.get("verified")), score=res.get("score"),
            mb_artist=res.get("mb_artist"), mb_title=res.get("mb_title"),
            mbid=res.get("mbid")),
    )
    log.info("%d links across %d songs; library index: %d recordings",
             len(rows), len(by_src), n_index)

    totals = {"verified": 0, "unverified": 0, "reject": 0}
    for src_slug, links in by_src.items():
        cleaned = sl.clean(links, resolve=resolve, mb=mb)
        for link in cleaned:
            totals[link["verdict"]] = totals.get(link["verdict"], 0) + 1
        if args.dry_run:
            continue
        keep = [{
            "direction": lk["direction"],
            "dst_key": sl.norm(lk["artist"]) + "|" + sl.norm(lk["title"]),
            "dst_artist": lk["artist"], "dst_title": lk["title"],
            "dst_slug": lk.get("dst_slug"), "relation": lk["relation"],
            "src_year": None, "dst_year": None,
            "evidence": lk.get("fact"), "confidence": lk.get("confidence"),
        } for lk in cleaned if lk["verdict"] == "verified"]
        MetadataDB.replace_sample_links(args.collection, src_slug, keep)

    log.info("verified %d, unverified %d, rejected %d%s",
             totals.get("verified", 0), totals.get("unverified", 0),
             totals.get("reject", 0),
             f"; MusicBrainz calls {mb.calls}" if mb else "")
    if args.dry_run:
        log.info("dry run — nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
