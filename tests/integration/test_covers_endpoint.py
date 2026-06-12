"""Covers endpoint — CORS-readability of cached responses.

Covers are loaded two ways by the SPA: as plain <img>/background-image
(no-cors, no Origin header) and via useCoverColor's crossOrigin='anonymous'
Image for canvas sampling. The response is cached for a year (immutable)
under ONE cache key, so the no-cors copy MUST already carry
Access-Control-Allow-Origin — otherwise the later CORS read is served the
ACAO-less cached response, the browser raises a CORS error, and the
equalizer falls back to the purple default.

CORSMiddleware only decorates requests that carry an Origin header, hence
serve_cover must set the header itself, unconditionally.
"""

import pytest


@pytest.fixture
def cover_file(tmp_path, monkeypatch):
    """Point COVERS_DIR at a temp dir holding one fake cover."""
    import app.api.main as main_mod

    monkeypatch.setattr(main_mod, "COVERS_DIR", tmp_path)
    name = "deadbeefcafebabe.jpg"
    (tmp_path / name).write_bytes(b"\xff\xd8\xff\xe0 fake jpeg body")
    return name


def test_cover_has_acao_without_origin_header(client, cover_file):
    """A no-cors fetch (no Origin header) must still get ACAO: *."""
    resp = client.get(f"/api/v1/covers/{cover_file}")
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"
    # Long cache must survive the change — it's why poisoning was sticky.
    assert "immutable" in resp.headers.get("cache-control", "")


def test_cover_has_acao_with_origin_header(client, cover_file):
    """CORS fetch path (middleware) keeps working — single ACAO value."""
    resp = client.get(
        f"/api/v1/covers/{cover_file}", headers={"Origin": "null"}
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"
