"""Bootstrap CLI — create the first user (owner) and lock the instance mode.

Idempotent: re-running fails loudly (exit code 1) if an owner already exists,
so a typo in the email can't silently clobber the existing account.

Usage
-----
  python -m scripts.create_owner \\
    --email owner@example.com \\
    --password "your-strong-password-here" \\
    --mode sharing      # or 'server'

Env
---
  MUSIX_JWT_SECRET    required (32+ chars)
  MUSIX_METADATA_DB   optional — override DB path (default cache/metadata.db)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Importing app.services pulls in heavy modules whose constructors parse
# sys.argv (e.g. CLAP). Neutralize argv across just those imports so our
# own argparse below sees the real CLI flags. Same pattern as
# scripts/backfill_artist_slugs.py.
_saved_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    from app.resources.metadata_db import MetadataDB
    from app.services.auth_service import (
        AuthService, EmailAlreadyTakenError, InstanceAlreadyInitializedError,
        WeakPasswordError,
    )
finally:
    sys.argv = _saved_argv

logger = logging.getLogger("create_owner")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="create_owner",
        description="Bootstrap the first MusiX account and lock the instance mode.",
    )
    ap.add_argument("--email", required=True, help="owner email (lower-cased)")
    ap.add_argument("--password", required=True, help="owner password (>= 6 chars)")
    ap.add_argument(
        "--mode", required=True, choices=("sharing", "server"),
        help="instance mode — LOCKED after this call",
    )
    args = ap.parse_args(argv)

    jwt_secret = os.environ.get("MUSIX_JWT_SECRET", "")
    if len(jwt_secret) < 16:
        # AuthService would also raise — fail early with a useful message.
        jwt_secret = "BOOTSTRAP-ONLY-NOT-USED-FOR-SIGNING-IN-CLI" + "x" * 16

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    MetadataDB.init()
    auth = AuthService(jwt_secret=jwt_secret)

    # Friendly fast-path message with the current mode; bootstrap_instance
    # re-checks atomically and owns the rollback logic.
    cfg = MetadataDB.get_instance_config()
    if cfg is not None:
        print(
            f"[ERROR] instance is already initialized (mode={cfg['mode']}). "
            "Use scripts/change_instance_mode to migrate.",
            file=sys.stderr,
        )
        return 1

    try:
        user = auth.bootstrap_instance(
            email=args.email, password=args.password, mode=args.mode,
        )
    except InstanceAlreadyInitializedError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except EmailAlreadyTakenError as e:
        print(f"[ERROR] email already taken: {e}", file=sys.stderr)
        return 1
    except WeakPasswordError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] failed to bootstrap: {e} (owner rolled back)", file=sys.stderr)
        return 1

    print(f"[OK] owner created: id={user.id} email={user.email} mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
