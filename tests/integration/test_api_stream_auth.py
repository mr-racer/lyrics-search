"""Audio stream endpoint auth: must accept ?st=<stream token> because
<audio> elements cannot send an Authorization header.

These tests assert gate behavior only (Qdrant may be down in CI — anything
that is not 401 means the request got PAST auth).
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB
from app.services.auth_service import AuthService


JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
    app = create_app()
    with TestClient(app) as c:
        conn = MetadataDB.get()
        conn.execute("DELETE FROM instance_config")
        conn.execute("DELETE FROM users")
        conn.commit()
        MetadataDB.set_instance_config(mode="sharing", created_at=1.0)
        auth = AuthService(jwt_secret=JWT_SECRET)
        auth.create_owner(email="owner@example.com", password="ownerpass12345")
        yield c


def _login_token(c) -> str:
    r = c.post("/api/v1/auth/login",
               json={"email": "owner@example.com", "password": "ownerpass12345"})
    return r.json()["token"]


def _stream_token(c) -> str:
    tok = _login_token(c)
    r = c.post("/api/v1/auth/stream-token",
               headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    return r.json()["token"]


STREAM_PATH = "/api/v1/search/tracks/00000000-0000-0000-0000-000000000000/stream"


def test_stream_401_without_any_credentials(client):
    assert client.get(STREAM_PATH).status_code == 401


def test_stream_passes_gate_with_st_query(client):
    st = _stream_token(client)
    r = client.get(f"{STREAM_PATH}?st={st}")
    assert r.status_code != 401, f"?st= rejected: {r.status_code} {r.text[:200]}"


def test_stream_passes_gate_with_bearer_header(client):
    tok = _login_token(client)
    r = client.get(STREAM_PATH, headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code != 401


def test_stream_401_with_login_token_in_st(client):
    """Full login tokens must not be usable in URLs."""
    tok = _login_token(client)
    assert client.get(f"{STREAM_PATH}?st={tok}").status_code == 401


def test_other_search_routes_still_gated(client):
    """Carving /stream out of the blanket gate must not open anything else."""
    r = client.post("/api/v1/search/", json={"query": "x", "mode": "text"})
    assert r.status_code == 401
