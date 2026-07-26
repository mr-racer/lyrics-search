"""Dry-run the sample pipeline over stored facts — no writes, full report.

This is what you look at BEFORE wiping and re-extracting. For every song it
prints the links currently stored, the links the new pipeline would keep, and
the reason each dropped one was dropped:

    kanye-west-bound-2  (Bound 2, 2013)
      было  11 : Bound / Ponderosa Twins Plus One, Juicy / Biggie, ...
      стало  3 : Bound / Ponderosa Twins Plus One [sample]
                 Aeroplane (Reprise) / Wee [interpolation]
                 Sweet Nothin's / Brenda Lee [sample]
      отказы : Juicy / Biggie -> relation:lyrical_reference
               MOR, R&B -> no-artist
               Illest Motherfucker Alive / Kanye West -> self-artist

The final summary counts survivors by relation and rejections by reason, so
you can see at a glance whether the gates are cutting where you expect.

Needs GLiNER2 and a reachable LLM: it runs the real pipeline, it just never
persists anything. Use ``--limit`` to sample a slice rather than the whole
library, and ``--slug`` to inspect specific songs.

    docker exec musix python scripts/dry_run_relations.py --limit 40
    docker exec musix python scripts/dry_run_relations.py --slug kanye-west-bound-2
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.resources.metadata_db import MetadataDB  # noqa: E402
from app.services.fact_relations.gates import SongContext, apply_gates  # noqa: E402
from app.services.fact_relations.library_index import LibraryIndex  # noqa: E402
from app.services.fact_relations.service import collect_claims  # noqa: E402


def _fmt(entry) -> str:
    song = entry.get("song") or entry.get("dst_title") or "—"
    artist = entry.get("artist") or entry.get("dst_artist")
    return f"{song} / {artist}" if artist else song


def _stored(slug: str) -> dict:
    conn = MetadataDB._connect()
    row = conn.execute("SELECT samples_json FROM songs WHERE slug = ?", (slug,)).fetchone()
    if not row or not row[0]:
        return {"samples": [], "sampled_by": []}
    try:
        rel = json.loads(row[0]) or {}
    except ValueError:
        return {"samples": [], "sampled_by": []}
    return {"samples": rel.get("samples") or [], "sampled_by": rel.get("sampled_by") or []}


def _artist_keys(artist_slug, aliases, members):
    from app.services.fact_relations.library_index import artist_key

    keys = {artist_key(artist_slug or "")}
    for name, slug in (*aliases.items(), *members.items()):
        if slug == artist_slug:
            keys.add(artist_key(name))
    return frozenset(k for k in keys if k)


def _ask_llm_sync(loop):
    """The same loop-bridging ask_llm the production pipeline uses."""
    from app.services.llm_client import ask_llm

    def _ask(messages):
        system = next((m["content"] for m in messages if m.get("role") == "system"), None)
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        return asyncio.run_coroutine_threadsafe(
            ask_llm(user, system_prompt=system, parse_json=True, temperature=0), loop,
        ).result()

    return _ask


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", help="acct_* collection (default: the only one with songs)")
    ap.add_argument("--limit", type=int, default=25, help="songs to process (default 25)")
    ap.add_argument("--slug", action="append", help="inspect these slugs only (repeatable)")
    ap.add_argument("--only-changed", action="store_true",
                    help="print a song only when the result differs from what is stored")
    args = ap.parse_args()

    MetadataDB.init()
    collection = args.collection or _guess_collection()
    if not collection:
        print("no collection found — pass --collection acct_<id>")
        return 1
    print(f"collection: {collection}\n")

    index = LibraryIndex(
        MetadataDB.get_tracks_slim(collection),
        MetadataDB.get_songs_for_collection(collection),
    )
    aliases = MetadataDB.get_artist_aliases_map()
    members = MetadataDB.get_artist_members_map()

    songs = _pick_songs(collection, args)
    ask = _ask_llm_sync(asyncio.get_running_loop())

    kept_by_relation = collections.Counter()
    reasons = collections.Counter()
    n_before = n_after = 0

    for slug, title, artist_slug in songs:
        facts = MetadataDB.get_song_facts(slug, collection)
        if not facts:
            continue
        own = index.resolve(title, (artist_slug or "").replace("-", " ")) or {}
        ctx = SongContext(
            slug=slug, title=title or "", artist=artist_slug or "",
            album=own.get("album") or "", year=own.get("year"),
            collection_name=collection,
            artist_keys=_artist_keys(artist_slug, aliases, members),
            library=index,
        )
        claims = await asyncio.to_thread(
            collect_claims, facts, title or "", artist_slug or "", ask,
        )
        links, rejected = apply_gates(claims["links"], ctx)

        stored = _stored(slug)
        before = stored["samples"] + stored["sampled_by"]
        n_before += len(before)
        n_after += len(links)
        for ln in links:
            kept_by_relation[ln["relation"]] += 1
        for _c, reason in rejected:
            reasons[reason] += 1

        if args.only_changed and len(before) == len(links):
            continue

        year = f", {ctx.year}" if ctx.year else ""
        print(f"{slug}  ({title}{year})")
        print(f"  было  {len(before):>2} : " + (", ".join(_fmt(e) for e in before) or "—"))
        shown = "\n              ".join(
            f"{_fmt(ln)} [{ln['relation']}, {ln['direction']}]" for ln in links
        )
        print(f"  стало {len(links):>2} : " + (shown or "—"))
        if rejected:
            print("  отказы : " + "\n           ".join(
                f"{_fmt(c)} -> {r}" for c, r in rejected))
        print()

    print("=" * 70)
    print(f"links: {n_before} -> {n_after}")
    print("kept by relation :", dict(kept_by_relation) or "—")
    print("rejections       :")
    for reason, n in reasons.most_common():
        print(f"    {n:>4}  {reason}")
    return 0


def _guess_collection() -> str:
    conn = MetadataDB._connect()
    row = conn.execute(
        "SELECT collection_name, COUNT(*) n FROM fact_visibility WHERE kind = 'song' "
        "GROUP BY collection_name ORDER BY n DESC LIMIT 1",
    ).fetchone()
    return row[0] if row else ""


def _pick_songs(collection: str, args) -> list:
    conn = MetadataDB._connect()
    if args.slug:
        marks = ",".join("?" * len(args.slug))
        rows = conn.execute(
            f"SELECT slug, title, artist_slug FROM songs WHERE slug IN ({marks})",
            tuple(args.slug),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]
    rows = conn.execute(
        """SELECT DISTINCT s.slug, s.title, s.artist_slug
           FROM songs s
           JOIN fact_visibility fv ON fv.kind = 'song' AND fv.slug = s.slug
           JOIN song_facts sf ON sf.song_slug = s.slug AND sf.lang = 'en'
           WHERE fv.collection_name = ?
           ORDER BY s.slug LIMIT ?""",
        (collection, args.limit),
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
