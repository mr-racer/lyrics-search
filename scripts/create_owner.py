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
import time

# Importing app.services pulls in heavy modules whose constructors parse
# sys.argv (e.g. CLAP). Neutralize argv across just those imports so our
# own argparse below sees the real CLI flags. Same pattern as
# scripts/backfill_artist_slugs.py.
_saved_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    from app.resources.metadata_db import MetadataDB
    from app.services.auth_service import (
        AuthService, EmailAlreadyTakenError, OwnerAlreadyExistsError,
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

    # Reject if instance_config OR an owner already exists — surface a single
    # error message that covers both conditions, since the recovery is the
    # same (use scripts/change_instance_mode if you really want to switch).
    cfg = MetadataDB.get_instance_config()
    if cfg is not None:
        print(
            f"[ERROR] instance is already initialized (mode={cfg['mode']}). "
            "Use scripts/change_instance_mode to migrate.",
            file=sys.stderr,
        )
        return 1

    try:
        uid = auth.create_owner(email=args.email, password=args.password)
    except OwnerAlreadyExistsError as e:
        print(f"[ERROR] owner already exists: {e}", file=sys.stderr)
        return 1
    except EmailAlreadyTakenError as e:
        print(f"[ERROR] email already taken: {e}", file=sys.stderr)
        return 1
    except WeakPasswordError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    # Atomicity: create_owner succeeded → write instance_config in the same
    # process. If this fails (which it shouldn't — schema constraints already
    # validated), roll back the user insert to keep the DB consistent.
    try:
        MetadataDB.set_instance_config(mode=args.mode, created_at=time.time())
    except Exception as e:
        MetadataDB.get().execute("DELETE FROM users WHERE id = ?", (uid,))
        MetadataDB.get().commit()
        print(f"[ERROR] failed to set instance_config: {e} (rolled back owner)",
              file=sys.stderr)
        return 1

    print(f"[OK] owner created: id={uid} email={args.email} mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
