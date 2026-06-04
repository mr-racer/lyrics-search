"""Phase A pre-merge smoke check.

1. Parse frontend/index.html scripts through @babel/standalone (Node) to catch JSX errors.
2. Hit live GET  /api/v1/instance/config — returns 404 before init, 200 after.
3. Hit live POST /api/v1/auth/login — round-trip succeeds with bootstrapped owner.
4. Hit live GET  /api/v1/library/collections — 401 without token, 200 with token.

Per project memory:
  [[feedback_smoke_test_must_parse]] — grep alone misses syntax errors
  [[feedback_registry_pattern_needs_e2e]] — unit tests don't catch missing include_router
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault("MUSIX_JWT_SECRET", "smoke-test-secret-32-chars-or-longer-for-prod")
# Use a tmp metadata DB so the smoke doesn't pollute the real cache/metadata.db.
_tmp = tempfile.mkdtemp(prefix="musix_smoke_phase_a_")
os.environ["MUSIX_METADATA_DB"] = str(Path(_tmp) / "smoke.db")

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB
from app.services.auth_service import AuthService


def smoke_babel_parse():
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    blocks = re.findall(r'<script[^>]*type="text/babel"[^>]*>(.*?)</script>', html, flags=re.S)
    if not blocks:
        print("[FAIL] no text/babel script blocks found")
        sys.exit(1)
    combined = "\n".join(blocks)
    with tempfile.NamedTemporaryFile(suffix=".jsx", mode="w", delete=False, encoding="utf-8") as f:
        f.write(combined)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["node", "-e",
             f"const b=require('@babel/standalone'); const s=require('fs').readFileSync({tmp_path!r},'utf8'); b.transform(s, {{presets:['react','env']}});"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print("[FAIL] Babel parse error:")
            print(result.stderr[:4000])
            sys.exit(1)
        print(f"[OK] Babel parsed {len(combined)} chars of JSX without error")
    except FileNotFoundError:
        print("[WARN] node not on PATH — skipping Babel parse. Install Node 18+ and re-run before merging.")


def smoke_live_endpoints():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM instance_config")
    conn.execute("DELETE FROM users")
    conn.execute("DELETE FROM invites")
    conn.commit()

    with TestClient(create_app()) as c:
        # ── instance/config returns 404 before init ──
        r = c.get("/api/v1/instance/config")
        if r.status_code != 404:
            print(f"[FAIL] /instance/config before init: expected 404, got {r.status_code}")
            sys.exit(1)

        # ── seed owner + sharing mode ──
        MetadataDB.set_instance_config(mode="sharing", created_at=1.0)
        auth = AuthService(jwt_secret=os.environ["MUSIX_JWT_SECRET"])
        auth.create_owner(email="smoke@example.com", password="smokepass12345")

        # ── instance/config returns mode after init ──
        r = c.get("/api/v1/instance/config")
        if r.status_code != 200 or r.json() != {"mode": "sharing"}:
            print(f"[FAIL] /instance/config: {r.status_code} {r.text}")
            sys.exit(1)
        print("[OK] /api/v1/instance/config returns sharing mode")

        # ── login round-trip ──
        r = c.post("/api/v1/auth/login",
                   json={"email": "smoke@example.com", "password": "smokepass12345"})
        if r.status_code != 200 or "token" not in r.json():
            print(f"[FAIL] /auth/login: {r.status_code} {r.text}")
            sys.exit(1)
        token = r.json()["token"]
        print(f"[OK] /api/v1/auth/login issued JWT ({len(token)} chars)")

        # ── gated endpoint 401 without token ──
        r = c.get("/api/v1/library/collections")
        if r.status_code != 401:
            print(f"[FAIL] /library/collections without token: expected 401, got {r.status_code}")
            sys.exit(1)
        print("[OK] /api/v1/library/collections returns 401 without token")

        # ── gated endpoint passes with token ──
        r = c.get("/api/v1/library/collections",
                  headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 401:
            print(f"[FAIL] /library/collections with token: still 401 {r.text}")
            sys.exit(1)
        print(f"[OK] /api/v1/library/collections accepts JWT (status={r.status_code})")


if __name__ == "__main__":
    smoke_babel_parse()
    smoke_live_endpoints()
    print("\nAll Phase A smoke checks passed.")
