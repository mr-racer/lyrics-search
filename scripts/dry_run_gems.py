"""Dry-run the gem gates over what is already stored — no model, no writes.

The gem pipeline's expensive half is GLiNER2 over every track's lyrics. The
part this pass changed is what happens to a span AFTER it is found: markup
detection, the namedrop self-check on slugs, and the songref citation gate.
All three can be replayed against the stored ``track_gems`` rows, so this
script answers "which of my current gems would survive?" in seconds and
without a GPU.

    docker exec musix python scripts/dry_run_gems.py
    docker exec musix python scripts/dry_run_gems.py --kind songref --verbose

What it CANNOT show: gems the fixed pipeline would newly find (a real quote
that the old markup filter hid). Those need a full re-run.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.resources.metadata_db import MetadataDB  # noqa: E402
from app.services.artist_facts_service import _slugify  # noqa: E402
from app.services.lyric_gems import matching, preprocess  # noqa: E402


def verdict(row: dict, track: dict, own_slugs: frozenset) -> str:
    """"keep" or the reason this stored gem would now be dropped."""
    quote = row.get("quote") or ""
    kind = row["kind"]

    if not quote:
        return "no-quote"
    if preprocess.is_structural_line(quote):
        return "markup-quote"
    if len(preprocess.strip_line_markup(quote).strip()) > preprocess.MAX_QUOTE_LEN:
        return "quote-too-long"
    # NB: the live pipeline also requires the found SPAN to appear in the quote
    # as a whole word. That check cannot be replayed here — ``track_gems``
    # stores the canonical name, not the surface form ("pager" for a lyric
    # saying "beeper", "Batman" for "Bruce Wayne"), so testing it against
    # ``canonical`` would report false drops. It only ever removes more.
    surface = row.get("canonical") or ""

    if kind == "namedrop":
        detail = _detail(row)
        if detail.get("artist_slug") in own_slugs:
            return "self-artist"
    elif kind == "songref":
        if len(matching.norm_name(surface).split()) < 2:
            return "one-word-title"
        own_title = matching.norm_name(matching.bare_title(track.get("title") or ""))
        key = matching.norm_name(surface)
        if key and own_title and (key == own_title or key in own_title):
            return "self-title"
        if (row.get("score") or 0) < matching.SONGREF_MIN_SCORE:
            return "low-score"
        if not matching.has_citation_marker(surface, quote):
            return "no-citation-marker"
    return "keep"


def _detail(row: dict) -> dict:
    try:
        return json.loads(row.get("detail") or "{}") or {}
    except (TypeError, ValueError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", help="acct_* collection (default: all)")
    ap.add_argument("--kind", help="capsule | namedrop | popculture | songref")
    ap.add_argument("--verbose", action="store_true", help="print every dropped gem")
    ap.add_argument("--show-kept", action="store_true", help="print every surviving gem")
    args = ap.parse_args()

    MetadataDB.init()
    conn = MetadataDB._connect()

    where, params = ["kind != 'none'"], []
    if args.collection:
        where.append("collection_name = ?")
        params.append(args.collection)
    if args.kind:
        where.append("kind = ?")
        params.append(args.kind)

    rows = conn.execute(
        "SELECT collection_name, track_id, kind, canonical, quote, detail, score "
        f"FROM track_gems WHERE {' AND '.join(where)}",
        tuple(params),
    ).fetchall()
    cols = ["collection_name", "track_id", "kind", "canonical", "quote", "detail", "score"]
    gems = [dict(zip(cols, r)) for r in rows]

    tracks = {}
    for r in conn.execute(
        "SELECT track_id, collection_name, title, artist, artists, artist_slugs FROM track_metadata",
    ).fetchall():
        tracks[(r[1], r[0])] = {
            "title": r[2], "artist": r[3], "artists": r[4], "artist_slugs": r[5],
        }

    outcome = collections.Counter()
    per_kind = collections.defaultdict(collections.Counter)
    dropped, kept_rows = [], []

    for gem in gems:
        track = tracks.get((gem["collection_name"], gem["track_id"]), {})
        own = _own_slugs(track)
        v = verdict(gem, track, own)
        outcome[v] += 1
        per_kind[gem["kind"]][v] += 1
        (kept_rows if v == "keep" else dropped).append((gem, track, v))

    total = len(gems)
    kept = outcome["keep"]
    print(f"gems: {total} -> {kept}  ({total - kept} dropped)\n")
    for kind in sorted(per_kind):
        c = per_kind[kind]
        print(f"  {kind:<12} {c['keep']:>4} / {sum(c.values()):<4}  "
              + ", ".join(f"{r}:{n}" for r, n in c.most_common() if r != "keep"))

    if args.show_kept:
        print("\n--- kept ---")
        for gem, track, _ in kept_rows:
            print(f"[{track.get('artist')} — {track.get('title')}] "
                  f"{gem['kind']} {gem['canonical']!r}")
            print(f"     quote: {(gem.get('quote') or '')[:110]!r}")

    if args.verbose:
        print("\n--- dropped ---")
        for gem, track, reason in dropped:
            print(f"[{track.get('artist')} — {track.get('title')}] "
                  f"{gem['kind']} {gem['canonical']!r} -> {reason}")
            print(f"     quote: {(gem.get('quote') or '')[:110]!r}")
    return 0


def _own_slugs(track: dict) -> frozenset:
    slugs = set()
    raw = track.get("artist_slugs")
    if raw:
        try:
            slugs |= {s for s in (json.loads(raw) if isinstance(raw, str) else raw) if s}
        except (TypeError, ValueError):
            pass
    names = [track.get("artist") or ""]
    raw_artists = track.get("artists")
    if raw_artists:
        try:
            names += list(json.loads(raw_artists) if isinstance(raw_artists, str) else raw_artists)
        except (TypeError, ValueError):
            pass
    for name in names:
        if str(name).strip():
            slugs.add(_slugify(str(name)))
    return frozenset(slugs)


if __name__ == "__main__":
    raise SystemExit(main())
