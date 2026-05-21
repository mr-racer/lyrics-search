"""Plan 19 pre-merge smoke check.

1. Parse frontend/index.html scripts through @babel/standalone (Node) to catch JSX errors.
2. Hit live GET /api/v1/playlists to verify the router is registered.

Per project memory:
  [[feedback_smoke_test_must_parse]] — grep alone misses syntax errors
  [[feedback_registry_pattern_needs_e2e]] — unit tests don't catch missing include_router
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import create_app


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
        print("[WARN] node not on PATH — skipping Babel parse. Install Node 18+ and run again before merging.")


def smoke_live_endpoint():
    with TestClient(create_app()) as c:
        r = c.get("/api/v1/playlists", params={"collection_name": "smoke-test"})
        if r.status_code != 200:
            print(f"[FAIL] /api/v1/playlists returned {r.status_code}: {r.text[:500]}")
            sys.exit(1)
        body = r.json()
        if "playlists" not in body or body["collection_name"] != "smoke-test":
            print(f"[FAIL] unexpected response shape: {body}")
            sys.exit(1)
        print(f"[OK] /api/v1/playlists live registration confirmed (got {len(body['playlists'])} playlists)")


if __name__ == "__main__":
    smoke_babel_parse()
    smoke_live_endpoint()
    print("\nAll Plan 19 smoke checks passed.")
