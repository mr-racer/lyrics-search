"""JSX parse gate — runs frontend/index.html's text/babel blocks through
@babel/standalone via Node. Catches syntax errors grep can't.
Per memory: feedback_smoke_test_must_parse. Standalone on purpose: no app
imports, so it runs in <5s and can gate every frontend edit."""
import re
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    blocks = re.findall(
        r'<script[^>]*type="text/babel"[^>]*>(.*?)</script>', html, flags=re.S,
    )
    if not blocks:
        print("[FAIL] no text/babel script blocks found")
        return 1
    combined = "\n".join(blocks)
    with tempfile.NamedTemporaryFile(
        suffix=".jsx", mode="w", delete=False, encoding="utf-8",
    ) as f:
        f.write(combined)
        tmp = f.name
    try:
        result = subprocess.run(
            ["node", "-e",
             f"const b=require('@babel/standalone'); "
             f"const s=require('fs').readFileSync({tmp!r},'utf8'); "
             f"b.transform(s, {{presets:['react','env']}});"],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        print("[FAIL] node not on PATH — install Node 18+; this gate is mandatory")
        return 1
    if result.returncode != 0:
        print("[FAIL] Babel parse error:")
        print(result.stderr[:4000])
        return 1
    print(f"[OK] Babel parsed {len(combined)} chars of JSX without error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
