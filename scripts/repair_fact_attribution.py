"""One-off repair: restore real song titles / artist names garbled by the old
slug-derived placeholder writers in ``MetadataDB.add_song_fact(s_batch)``.

Those writers used ``ON CONFLICT DO UPDATE`` with placeholders computed from
the song slug — ``title = slug.replace('-', ' ')`` (the slug embeds the
artist, so "50-cent-in-da-club" became title "50 cent in da club") and
``artist_slug = slug.split('-', 1)[0]`` ("50") — clobbering the real rows
``ensure_song`` had written from the audio tags. The home random-facts strip
renders exactly these columns, so most fact-bearing songs showed a one-line
lowercase mess and no cover art.

The authoritative source of truth is the Qdrant payload (artist/artists/title
per track). For every ``acct_*`` collection this script:

1. rebuilds the expected songs row per track (slug via
   ``get_song_facts_key``, the same key every facts writer uses) and repairs
   rows whose title/artist_slug still carry the garbage markers;
2. repairs ``artists.name`` placeholders (``name == slug`` from the AudioDB
   fallback insert, or ``name == slug-with-spaces`` from the old fact
   writers) using display names seen in the payloads.

Only rows matching the garbage markers are touched — manually curated titles
and already-correct rows are left alone. Idempotent; dry-run by default,
pass ``--apply`` to write.

Run inside the container (Qdrant + cache paths resolve there):

    docker exec musix python scripts/repair_fact_attribution.py          # dry-run
    docker exec musix python scripts/repair_fact_attribution.py --apply
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient  # noqa: E402

from app.resources.metadata_db import MetadataDB  # noqa: E402
from app.services.artist_split import primary_artist  # noqa: E402
from app.services.song_facts_service import _slugify, get_song_facts_key  # noqa: E402
from app.services.artist_facts_service import _slugify as _slugify_artist  # noqa: E402


def collect_tracks(client: QdrantClient) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Scroll every acct_* collection; return (tracks, name_by_slug).

    tracks: unique (artist_tag, title) pairs.
    name_by_slug: display name for every artist-slug variant seen in payloads
    (both slugify flavours — the song-facts one and the artist-facts one).
    """
    tracks: dict[tuple[str, str], None] = {}
    name_by_slug: dict[str, str] = {}

    def note_name(name: str) -> None:
        name = (name or "").strip()
        if not name:
            return
        for slug in {_slugify(name), _slugify_artist(name)}:
            if slug and slug not in name_by_slug:
                name_by_slug[slug] = name

    for coll in client.get_collections().collections:
        if not coll.name.startswith("acct_"):
            continue
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=coll.name,
                offset=offset,
                limit=1000,
                with_payload=["artist", "artists", "title"],
                with_vectors=False,
            )
            for p in points or []:
                payload = p.payload or {}
                artist = (payload.get("artist") or "").strip()
                title = (payload.get("title") or "").strip()
                if artist and title:
                    tracks[(artist, title)] = None
                note_name(artist)
                note_name(primary_artist(artist) or "")
                for a in payload.get("artists") or []:
                    if isinstance(a, str):
                        note_name(a)
            if offset is None:
                break
    return list(tracks), name_by_slug


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = parser.parse_args()

    MetadataDB.init()
    conn = MetadataDB._connect()
    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"), timeout=60)

    tracks, name_by_slug = collect_tracks(client)
    print(f"[repair] {len(tracks)} unique tracks, {len(name_by_slug)} artist names from Qdrant")

    songs_fixed = 0
    for artist_tag, title in tracks:
        primary = primary_artist(artist_tag) or artist_tag
        slug = get_song_facts_key(artist_tag, title)
        a_slug = _slugify(primary)
        row = conn.execute(
            "SELECT title, artist_slug FROM songs WHERE slug = ?", (slug,),
        ).fetchone()
        if not row:
            continue
        cur_title, cur_artist = row
        garbage_title = cur_title == slug.replace("-", " ")
        garbage_artist = (
            cur_artist != a_slug
            and cur_artist == (slug.split("-", 1)[0] if "-" in slug else slug)
        )
        if not garbage_title and not garbage_artist:
            continue
        new_title = title if garbage_title else cur_title
        new_artist = a_slug if garbage_artist else cur_artist
        songs_fixed += 1
        if args.apply:
            conn.execute(
                """INSERT INTO artists (slug, name, collection_name)
                   VALUES (?, ?, '') ON CONFLICT(slug) DO NOTHING""",
                (new_artist, primary),
            )
            conn.execute(
                "UPDATE songs SET title = ?, artist_slug = ? WHERE slug = ?",
                (new_title, new_artist, slug),
            )
        else:
            print(f"  song {slug}: title {cur_title!r} -> {new_title!r}, "
                  f"artist_slug {cur_artist!r} -> {new_artist!r}")

    artists_fixed = 0
    rows = conn.execute(
        """SELECT slug, name FROM artists
           WHERE name = slug OR name = replace(slug, '-', ' ')""",
    ).fetchall()
    for slug, name in rows:
        real = name_by_slug.get(slug)
        if not real or real == name:
            continue
        artists_fixed += 1
        if args.apply:
            conn.execute("UPDATE artists SET name = ? WHERE slug = ?", (real, slug))
        else:
            print(f"  artist {slug}: name {name!r} -> {real!r}")

    if args.apply:
        conn.commit()
    print(f"[repair] songs {'fixed' if args.apply else 'to fix'}: {songs_fixed}, "
          f"artists {'fixed' if args.apply else 'to fix'}: {artists_fixed}"
          + ("" if args.apply else "  (dry-run — pass --apply to write)"))


if __name__ == "__main__":
    main()
