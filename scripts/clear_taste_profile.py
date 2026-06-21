"""Clear taste profile or fully remove an existing account.

Two modes:
  1) --email xoma@sus.com          — clear taste profile only (reactions, vibes,
                                      settings, cache). Library stays intact.
  2) --email xoma@sus.com --delete-account — fully remove the user: SQLite rows,
                                              Qdrant collection, cache files.
                                              The source account for migration can
                                              then re-use this email.

Usage:
    python scripts/clear_taste_profile.py --email xoma@sus.com --yes
    python scripts/clear_taste_profile.py --email xoma@sus.com --delete-account --yes
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# Neutralize sys.argv for heavy imports
_saved_argv = sys.argv
sys.argv = sys.argv[:1]

try:
    from qdrant_client import QdrantClient
except ImportError:
    # Only needed for --delete-account
    QdrantClient = None  # type: ignore

sys.argv = _saved_argv

DB_PATH = Path(
    os.environ.get(
        "MUSIX_METADATA_DB",
        str(Path(__file__).resolve().parent.parent / "cache" / "metadata.db"),
    )
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")

# Tables scoped by collection_name — deleted when removing an account.
_COLLECTION_TABLES = [
    "track_reactions",
    "sonic_vibes",
    "collection_settings",
    "playback_events",
    "recsys_llm_texts",
    "ai_indexing_jobs",
]

# Tables where we only clear taste data (default mode).
_TASTE_TABLES = ["track_reactions", "sonic_vibes", "collection_settings"]


def _resolve_user(conn: sqlite3.Connection, email: str) -> tuple[str, str]:
    """Return (user_id, collection_name) or exit."""
    row = conn.execute(
        "SELECT id, email FROM users WHERE email = ?", (email.lower().strip(),)
    ).fetchone()
    if not row:
        print(f"[ERROR] No user found with email {email!r}", file=sys.stderr)
        sys.exit(1)
    uid = row[0]
    return uid, f"acct_{uid}"


def _count_rows(conn: sqlite3.Connection, tables: list[str], collection: str) -> dict[str, int]:
    counts = {}
    for table in tables:
        c = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE collection_name = ?", (collection,)
        ).fetchone()[0]
        counts[table] = c
    return counts


def _delete_from_tables(conn: sqlite3.Connection, tables: list[str], collection: str):
    for table in tables:
        cur = conn.execute(
            f"DELETE FROM {table} WHERE collection_name = ?", (collection,)
        )
        if cur.rowcount:
            print(f"  Deleted {cur.rowcount} rows from {table}")


def main():
    ap = argparse.ArgumentParser(description="Clear taste profile or remove an account")
    ap.add_argument("--email", required=True, help="Email of the target account")
    ap.add_argument(
        "--delete-account",
        action="store_true",
        help="Fully remove the account (user row, Qdrant collection, all data)",
    )
    ap.add_argument("--yes", action="store_true", help="Actually execute (without this = dry run)")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"[ERROR] DB not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(f"file:{DB_PATH}?timeout=30", uri=True)
    conn.row_factory = sqlite3.Row

    user_id, collection = _resolve_user(conn, args.email)
    print(f"User: {args.email}, ID: {user_id}, Collection: {collection}")

    # Similarity cache
    cache_file = (
        Path(__file__).resolve().parent.parent / "cache" / "top_pairs" / f"{collection}.json"
    )
    cache_exists = cache_file.exists()

    # ── Dry run ──
    if not args.yes:
        if args.delete_account:
            print("\nWould DELETE the entire account:")
            all_tables = _COLLECTION_TABLES
            counts = _count_rows(conn, all_tables, collection)
            for t, c in counts.items():
                print(f"  {t}: {c} rows")

            # Playlists (keyed by collection_name, not user_id)
            pl = conn.execute(
                "SELECT COUNT(*) FROM playlists WHERE collection_name = ?", (collection,)
            ).fetchone()[0]
            print(f"  playlists: {pl}")

            print(f"  similarity cache: {'WILL DELETE' if cache_exists else 'not found'}")
            print(f"  Qdrant collection {collection!r}: WILL DELETE")
            print(f"  users row: WILL DELETE")
        else:
            print("\nWould clear taste profile only:")
            counts = _count_rows(conn, _TASTE_TABLES, collection)
            for t, c in counts.items():
                print(f"  {t}: {c} rows")
            print(f"  similarity cache: {'WILL DELETE' if cache_exists else 'not found'}")

        print("\nRe-run with --yes to execute.")
        conn.close()
        return 0

    # ── Execute ──
    if args.delete_account:
        print("\nDeleting account...")

        # SQLite: all collection-scoped tables
        _delete_from_tables(conn, _COLLECTION_TABLES, collection)

        # Playlists (and playlist_tracks via CASCADE)
        cur = conn.execute("DELETE FROM playlists WHERE collection_name = ?", (collection,))
        if cur.rowcount:
            print(f"  Deleted {cur.rowcount} playlists")

        # User row
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        print(f"  Deleted user row")

        # Similarity cache
        if cache_exists:
            cache_file.unlink()
            print(f"  Deleted {cache_file}")

        conn.commit()
        conn.close()

        # Qdrant collection
        if QDRANT_URL:
            qdrant = None
            try:
                if QdrantClient is None:
                    print("  [WARN] qdrant_client not installed — skipping Qdrant collection deletion", file=sys.stderr)
                else:
                    qdrant = QdrantClient(url=QDRANT_URL, trust_env=False)
                    qdrant.delete_collection(collection_name=collection)
                    print(f"  Deleted Qdrant collection {collection!r}")
            except Exception as e:
                print(f"  [WARN] Could not delete Qdrant collection: {e}", file=sys.stderr)
            finally:
                if qdrant is not None:
                    qdrant.close()
        else:
            print("  [WARN] QDRANT_URL not set — skipping Qdrant collection deletion", file=sys.stderr)

        print("\nDone — account fully removed.")

    else:
        # Taste profile only
        print("\nClearing taste profile...")
        _delete_from_tables(conn, _TASTE_TABLES, collection)

        if cache_exists:
            cache_file.unlink()
            print(f"  Deleted {cache_file}")

        conn.commit()
        conn.close()
        print("\nDone — taste profile cleared. Library intact.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
