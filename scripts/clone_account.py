"""Clone an existing account — copy Qdrant collection + SQLite metadata to a new user.

If the target email already exists, the existing account is automatically deleted
(SQLite rows, Qdrant collection, cache files) before the clone proceeds.

Creates a brand-new user (member) whose library is an exact copy of an existing
account's collection. Playback history, recommendations, playlists, and pending
uploads are NOT copied — the new user starts fresh on those.

The source account is untouched.  The new account gets its own Qdrant collection
and its own rows in the SQLite tables that are scoped by collection_name.

Prerequisites
-------------
  - Qdrant must be running and reachable (QDRANT_URL env or http://localhost:6333)
  - The metadata DB must NOT be open by the server — stop the server first,
    or the SQLite operations will fail with a "database is locked" error.
  - MUSIX_JWT_SECRET must be set if you want --issue-token (otherwise optional).

Usage
-----
  python scripts/clone_account.py \\
      --source-email owner@example.com \\
      --target-email friend@example.com \\
      --target-password "strongpassword" \\
      --yes                          # omit for dry-run

  Optional:
      --issue-token    print a ready-to-use JWT for the new account
                        (requires MUSIX_JWT_SECRET env var)

What is copied
--------------
  Qdrant:
    - Every point (vectors + payload) from acct_<source_id> → acct_<target_id>
    - Collection config (vector dims, sparse BM25, CLAP if present)
    - Payload index on artist_slugs

  SQLite (metadata.db):
    - artists, artist_facts, songs, song_facts  (slug-scoped, shared)

What is NOT copied (starts empty)
----------------------------------
  - track_reactions   (liked/disliked songs)
  - sonic_vibes       (taste islands / острова вкуса)
  - collection_settings  (recommendation settings)
  - playback_events   (listening history)
  - recsys_llm_texts  (recommendation cache)
  - similarity cache  (top_pairs — reindexes on first search)
  - playlists / playlist_tracks
  - pending_uploads
  - ai_indexing_jobs
  - transcoded audio cache  (regenerated on first play)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

# ── Neutralize sys.argv for heavy imports (same pattern as create_owner) ──
_saved_argv = sys.argv
sys.argv = sys.argv[:1]

try:
    from argon2 import PasswordHasher

    import jwt as pyjwt
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    sys.argv = _saved_argv

# Re-resolve after argv reset (metadata_db reads env vars at import)
# NOTE: we do NOT import MetadataDB here — that would pull in torch/transformers.
# This script uses raw sqlite3 for DB operations.

logger = logging.getLogger("clone_account")


def _ts(msg: str) -> None:
    """Print a timestamped progress message — visible even when logging is suppressed."""
    import datetime as _dt
    t = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"  [{t}] {msg}", flush=True)


# ── Constants ──
DB_PATH = Path(
    os.environ.get("MUSIX_METADATA_DB", str(Path(__file__).resolve().parent.parent / "cache" / "metadata.db"))
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
TOKEN_TTL_SECONDS = 30 * 24 * 3600

# Tables scoped by collection_name — cleaned up when removing an existing
# target account so clone_account can proceed.
_DELETE_ACCOUNT_TABLES = [
    "track_reactions",
    "sonic_vibes",
    "collection_settings",
    "playback_events",
    "recsys_llm_texts",
    "ai_indexing_jobs",
]

# Tables keyed by slug — INSERT OR IGNORE (slug is unique PK, shared across collections).
# The artists/songs tables have a collection_name column but slug is the PK.
COPY_SLUG_SHARED = [
    "artists",
    "songs",
]

# Tables with no collection_name at all — shared across all collections via slug FK.
# artist_facts → artists(slug), song_facts → songs(slug), refined_facts (scope_key).
# These are naturally shared — no copy needed.


def _resolve_source_user(conn: sqlite3.Connection, email: str) -> tuple[str, str]:
    """Return (user_id, collection_name) for the source account."""
    row = conn.execute(
        "SELECT id FROM users WHERE email = ?", (email.lower().strip(),)
    ).fetchone()
    if row is None:
        print(f"[ERROR] No user found with email {email!r}", file=sys.stderr)
        sys.exit(1)
    user_id = row[0]
    return user_id, f"acct_{user_id}"


def _delete_existing_account(
    conn: sqlite3.Connection,
    client: QdrantClient,
    user_id: str,
    collection: str,
) -> None:
    """Fully remove an existing account (SQLite + Qdrant + cache)."""
    _ts(f"Deleting existing account {collection!r}...")

    # SQLite: all collection-scoped tables
    for table in _DELETE_ACCOUNT_TABLES:
        cur = conn.execute(
            f"DELETE FROM {table} WHERE collection_name = ?", (collection,)
        )
        if cur.rowcount:
            _ts(f"Deleted {cur.rowcount} rows from {table}")

    # Playlists (and playlist_tracks via CASCADE)
    cur = conn.execute(
        "DELETE FROM playlists WHERE collection_name = ?", (collection,)
    )
    if cur.rowcount:
        _ts(f"Deleted {cur.rowcount} playlists")

    # User row
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    _ts("Deleted user row")

    conn.commit()

    # Similarity cache
    cache_file = Path(__file__).resolve().parent.parent / "cache" / "top_pairs" / f"{collection}.json"
    if cache_file.exists():
        cache_file.unlink()
        _ts(f"Deleted {cache_file}")

    # Qdrant collection
    try:
        client.delete_collection(collection_name=collection)
        _ts(f"Deleted Qdrant collection {collection!r}")
    except Exception as e:
        _ts(f"Could not delete Qdrant collection {collection!r}: {e}")


def _create_target_user(
    conn: sqlite3.Connection, email: str, password: str, new_id: str
) -> None:
    ph = PasswordHasher()
    password_hash = ph.hash(password)
    now = time.time()
    conn.execute(
        """INSERT INTO users (id, email, password_hash, role, created_at)
           VALUES (?, ?, ?, 'member', ?)""",
        (new_id, email.lower().strip(), password_hash, now),
    )
    conn.commit()
    logger.info("Created user: id=%s email=%s", new_id, email)


def _copy_qdrant_collection(
    client: QdrantClient, source: str, target: str, dry_run: bool = False
) -> int:
    """Copy all points from source → target collection. Returns point count."""
    # Get source collection info to replicate config
    source_info = client.get_collection(source)
    source_config = source_info.config

    if dry_run:
        count = source_info.points_count or 0
        print(f"  Would copy {count} points from {source!r} → {target!r}")
        return count

    # Create target collection with same config
    _ts(f"Creating Qdrant collection {target!r} (config from {source!r})...")
    client.create_collection(
        collection_name=target,
        vectors_config=source_config.params.vectors,
        sparse_vectors_config=source_config.params.sparse_vectors,
    )
    _ts(f"Collection {target!r} created.")

    # Recreate the artist_slugs payload index
    _ts("Creating artist_slugs payload index...")
    try:
        client.create_payload_index(
            collection_name=target,
            field_name="artist_slugs",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
    except Exception as e:
        logger.warning("Could not create artist_slugs index: %s", e)

    # Scroll + upsert in batches
    # Qdrant has a 32MB request limit. With ~260KB/point, max ~120 points/batch.
    # Use 100 to stay safely under the limit.
    batch_size = 100
    total = 0
    offset = None

    # Check if target collection already exists (from a previous interrupted run)
    _ts(f"Checking if {target!r} already exists...")
    try:
        existing = client.get_collection(target)
        if existing.points_count and existing.points_count > 0:
            _ts(f"Target collection {target!r} already has {existing.points_count} points — deleting and recreating...")
            client.delete_collection(target)
    except Exception:
        pass  # Collection doesn't exist, which is fine

    # Suppress Qdrant client logging during batch copy
    import logging as _logging
    _logging.getLogger("qdrant_client").setLevel(_logging.WARNING)
    _logging.getLogger("httpx").setLevel(_logging.WARNING)
    _logging.getLogger("httpcore").setLevel(_logging.WARNING)

    # Get source point count for progress tracking
    source_info = client.get_collection(source)
    expected_count = source_info.points_count or 0
    _ts(f"Starting scroll+upsert loop (batch_size={batch_size}, expected={expected_count})...")
    import time as _time
    t0 = _time.monotonic()
    print(f"  Scrolling {source!r}...")

    # Track last point ID to detect infinite loop (Qdrant client bug: returns PointOffset(None) instead of None)
    last_id = None
    while True:
        points, offset = client.scroll(
            collection_name=source,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        # Detect loop: if first point ID matches last ID from previous batch, we're cycling
        if points[0].id == last_id and last_id is not None:
            _ts(f"Scroll loop detected (duplicate point {last_id}) — stopping at {total} points")
            break

        # Convert ScrollResponse points to PointStruct for upsert
        upsert_points = [
            qmodels.PointStruct(
                id=p.id,
                vector=dict(p.vector) if p.vector else None,  # type: ignore[arg-type]
                payload=p.payload,
            )
            for p in points
        ]

        client.upsert(collection_name=target, points=upsert_points)
        total += len(points)
        last_id = points[-1].id  # Track last point for loop detection

        elapsed = _time.monotonic() - t0
        rate = total / elapsed if elapsed > 0 else 0
        if total % 500 == 0:
            _ts(f"Copied {total} points ({rate:.0f} pts/s, {elapsed:.1f}s elapsed)")

        # Safety: stop if we've exceeded expected count
        if total >= expected_count and expected_count > 0:
            _ts(f"Reached expected count ({expected_count}) — stopping scroll")
            break

    elapsed = _time.monotonic() - t0
    _ts(f"Done: {total} points copied to {target!r} in {elapsed:.1f}s ({total/max(elapsed,1):.0f} pts/s)")
    return total


def _copy_sqlite_tables(
    conn: sqlite3.Connection,
    source_collection: str,
    target_collection: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """Copy collection-scoped tables from source → target.

    Returns dict of table_name → row_count.
    """
    counts: dict[str, int] = {}

    # ── Slug-shared tables (artists, songs) — INSERT OR IGNORE ──
    for table in COPY_SLUG_SHARED:
        _ts(f"Processing SQLite table {table!r} (slug-shared)...")
        source_cols = _get_table_columns(conn, table)
        if "collection_name" not in source_cols:
            # Table doesn't have collection_name — just copy all rows
            if dry_run:
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                counts[table] = n
                continue
            cols_str = ", ".join(source_cols)
            placeholders = ", ".join("?" for _ in source_cols)
            conn.executemany(
                f"INSERT OR IGNORE INTO {table} ({cols_str}) VALUES ({placeholders})",
                conn.execute(f"SELECT {cols_str} FROM {table}").fetchall(),
            )
            counts[table] = conn.total_changes
        else:
            # Has collection_name — filter by source, insert with target
            if dry_run:
                n = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE collection_name = ?",
                    (source_collection,),
                ).fetchone()[0]
                counts[table] = n
                continue

            # Read from source
            rows = conn.execute(
                f"SELECT * FROM {table} WHERE collection_name = ?",
                (source_collection,),
            ).fetchall()

            collection_name_index = source_cols.index("collection_name")
            _ts(f"Inserting {len(rows)} rows into {table!r} (replacing collection_name)...")
            for row in rows:
                new_row = list(row)
                new_row[collection_name_index] = target_collection
                placeholders = ", ".join("?" for _ in source_cols)
                conn.execute(
                    f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})",
                    new_row,
                )
            counts[table] = len(rows)

    # ── Collection-scoped tables — none copied (clean taste profile) ──
    # COPY_WITH_NEW_COLLECTION is intentionally empty; the loop below is kept
    # for extensibility but currently copies nothing.
    for table in []:
        _ts(f"Processing SQLite table {table!r} (collection-scoped)...")
        source_cols = _get_table_columns(conn, table)
        if dry_run:
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE collection_name = ?",
                (source_collection,),
            ).fetchone()[0]
            counts[table] = n
            continue

        rows = conn.execute(
            f"SELECT * FROM {table} WHERE collection_name = ?",
            (source_collection,),
        ).fetchall()

        collection_name_idx = source_cols.index("collection_name")
        for row in rows:
            new_row = list(row)
            new_row[collection_name_idx] = target_collection
            placeholders = ", ".join("?" for _ in source_cols)
            conn.execute(
                f"INSERT OR REPLACE INTO {table} VALUES ({placeholders})",
                new_row,
            )
        counts[table] = len(rows)

    conn.commit()
    return counts


def _get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Return column names for a table."""
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _copy_similarity_cache(
    source_collection: str,
    target_collection: str,
    dry_run: bool = False,
) -> bool:
    """Copy cache/top_pairs/<source>.json → <target>.json."""
    cache_dir = Path(__file__).resolve().parent.parent / "cache" / "top_pairs"
    src_file = cache_dir / f"{source_collection}.json"
    dst_file = cache_dir / f"{target_collection}.json"

    if not src_file.exists():
        print(f"  No similarity cache for {source_collection!r} — skipping")
        return False

    if dry_run:
        print(f"  Would copy {src_file} → {dst_file}")
        return True

    import shutil

    shutil.copy2(src_file, dst_file)
    print(f"  Copied similarity cache: {src_file.name} → {dst_file.name}")
    return True


def _issue_token(new_user_id: str, email: str) -> str | None:
    """Generate a JWT for the new user. Returns token or None."""
    secret = os.environ.get("MUSIX_JWT_SECRET", "")
    if not secret or len(secret) < 16:
        print(
            "[WARN] MUSIX_JWT_SECRET not set or too short — cannot issue token. "
            "Set the env var and re-run with --issue-token.",
            file=sys.stderr,
        )
        return None

    now = int(time.time())
    payload = {
        "sub": new_user_id,
        "email": email.lower().strip(),
        "role": "member",
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
    }
    return pyjwt.encode(payload, secret, algorithm="HS256")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="clone_account",
        description="Clone an existing account's library to a new user.",
    )
    ap.add_argument(
        "--source-email", required=True, help="Email of the account to clone from"
    )
    ap.add_argument(
        "--target-email", required=True, help="Email for the new account"
    )
    ap.add_argument(
        "--target-password", required=True, help="Password for the new account"
    )
    ap.add_argument(
        "--yes", action="store_true", help="Actually execute (without this = dry run)"
    )
    ap.add_argument(
        "--issue-token",
        action="store_true",
        help="Print a ready-to-use JWT for the new account",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    print("=" * 64)
    print("ACCOUNT CLONE")
    print(f"  source     : {args.source_email}")
    print(f"  target     : {args.target_email}")
    print(f"  DB         : {DB_PATH} ({'exists' if DB_PATH.exists() else 'MISSING'})")
    print(f"  Qdrant     : {QDRANT_URL}")
    print(f"  dry run    : {'no' if args.yes else 'YES — nothing will be changed'}")
    print("=" * 64)

    if not DB_PATH.exists():
        print(f"\n[ERROR] Metadata DB not found at {DB_PATH}", file=sys.stderr)
        print("Is the server configured correctly? Check MUSIX_METADATA_DB env.", file=sys.stderr)
        return 1

    # ── Connect to SQLite ──
    # timeout=30s so we don't hang forever if the DB is locked by another process
    conn = sqlite3.connect(f"file:{DB_PATH}?timeout=30", uri=True)
    conn.row_factory = sqlite3.Row
    
    # Verify we can actually write — fail fast instead of hanging later
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("COMMIT")
    except sqlite3.OperationalError as e:
        print(f"\n[ERROR] Cannot write to {DB_PATH}: {e}", file=sys.stderr)
        print("Is the musix container stopped? Another process may have the DB locked.", file=sys.stderr)
        conn.close()
        return 1
    
    _ts(f"Connected to {DB_PATH} (write OK)")
    qdrant: QdrantClient | None = None
    try:
        # ── Resolve source user ──
        source_id, source_col = _resolve_source_user(conn, args.source_email)
        print(f"\nSource: user_id={source_id}, collection={source_col}")

        # ── Connect to Qdrant ──
        try:
            qdrant = QdrantClient(url=QDRANT_URL, trust_env=False)
            # Verify source collection exists
            collections = qdrant.get_collections()
            col_names = [c.name for c in collections.collections]
            if source_col not in col_names:
                print(
                    f"\n[ERROR] Source collection {source_col!r} not found in Qdrant. "
                    f"Available: {col_names}",
                    file=sys.stderr,
                )
                return 1
            print(f"Qdrant: {len(col_names)} collections found, {source_col!r} OK")
        except Exception as e:
            print(f"\n[ERROR] Cannot connect to Qdrant at {QDRANT_URL}: {e}", file=sys.stderr)
            print("Make sure Qdrant is running.", file=sys.stderr)
            return 1

        # ── Ensure target slot is free (delete existing account if needed) ──
        # In dry-run mode, just check; in execute mode, actually delete.
        existing_row = conn.execute(
            "SELECT id FROM users WHERE email = ?", (args.target_email.lower().strip(),)
        ).fetchone()
        if existing_row:
            existing_id = existing_row[0]
            existing_col = f"acct_{existing_id}"
            print(f"\n[WARN] Target email {args.target_email!r} already exists (user_id={existing_id})")
            if args.yes:
                print("  Deleting existing account before cloning...\n")
                _delete_existing_account(conn, qdrant, existing_id, existing_col)
                print("  Existing account removed.\n")
            else:
                print("  (Will delete existing account when --yes is provided)\n")

        # ── Generate new user ID ──
        new_id = uuid.uuid4().hex
        target_col = f"acct_{new_id}"
        print(f"Target: user_id={new_id}, collection={target_col}")

        # ── DRY RUN SUMMARY ──
        if not args.yes:
            # Count source points
            try:
                info = qdrant.get_collection(source_col)
                point_count = info.points_count or 0
            except Exception:
                point_count = "?"

            # Count source SQLite rows
            sqlite_counts = _copy_sqlite_tables(
                conn, source_col, target_col, dry_run=True
            )

            print(f"\nDRY RUN — would copy:")
            print(f"  Qdrant points     : {point_count}")
            for table, count in sqlite_counts.items():
                print(f"  SQLite {table:<25s}: {count}")

            print(f"\n  New user: {args.target_email} (password: {'*' * len(args.target_password)})")
            print(f"  User ID : {new_id}")
            print(f"\nRe-run with --yes to execute.")
            return 0

        # ── EXECUTE ──
        _ts("=" * 50)
        print("\n[1/4] Creating new user in SQLite...", flush=True)
        _ts("Creating user in SQLite...")
        try:
            _create_target_user(conn, args.target_email, args.target_password, new_id)
        except sqlite3.OperationalError as e:
            _ts(f"FATAL: Cannot write to SQLite — {e}")
            _ts("The musix container may still be running. Stop it first: docker compose stop musix")
            raise
        _ts(f"User created: {new_id}")

        print("[2/4] Copying Qdrant collection...", flush=True)
        _ts("Starting Qdrant copy...")
        qdrant_count = _copy_qdrant_collection(qdrant, source_col, target_col, dry_run=False)
        _ts(f"Qdrant copy complete: {qdrant_count} points")

        print("[3/4] Copying SQLite metadata...", flush=True)
        _ts("Starting SQLite copy...")
        sqlite_counts = _copy_sqlite_tables(
            conn, source_col, target_col, dry_run=False
        )
        for table, count in sqlite_counts.items():
            print(f"  {table}: {count} rows", flush=True)
        _ts("SQLite copy complete")

        print("[4/4] Verifying...", flush=True)
        _ts("Verifying Qdrant...")
        # Verify Qdrant point count
        try:
            target_info = qdrant.get_collection(target_col)
            target_count = target_info.points_count or 0
            if target_count == qdrant_count:
                print(f"  Qdrant: {target_count} points in {target_col!r} ✓")
            else:
                print(
                    f"  Qdrant: expected {qdrant_count}, got {target_count} — may differ if source changed",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"  Qdrant verify: {e}", file=sys.stderr)

        # ── Issue optional token ──
        if args.issue_token:
            token = _issue_token(new_id, args.target_email)
            if token:
                print(f"\nJWT token for {args.target_email}:")
                print(f"  {token}")
                print("  (valid for 30 days)")

        print("\n" + "=" * 64)
        print("DONE — new account created:")
        print(f"  email       : {args.target_email}")
        print(f"  user_id     : {new_id}")
        print(f"  collection  : {target_col}")
        print(f"  tracks      : {qdrant_count}")
        print(f"  role        : member")
        print("=" * 64)
        print("\nThe new user can log in with the provided email and password.")
        print("Their library is already populated — no indexing required.")

    finally:
        conn.close()
        if qdrant is not None:
            qdrant.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
