"""Phase A: every non-/auth, non-/instance endpoint must require a JWT."""
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB
from app.services.auth_service import AuthService


JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


@pytest.fixture
def gated_client(monkeypatch):
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


def _token(c) -> str:
    r = c.post("/api/v1/auth/login",
               json={"email": "owner@example.com", "password": "ownerpass12345"})
    return r.json()["token"]


GATED_PATHS = [
    ("GET",  "/api/v1/library/collections"),
    ("GET",  "/api/v1/library/browse?q=x&collection_name=any"),
    ("GET",  "/api/v1/playback/recent?collection_name=any"),
    ("GET",  "/api/v1/playlists?collection_name=any"),
    ("GET",  "/api/v1/artists/some-slug?collection=any"),
]


@pytest.mark.parametrize("method,path", GATED_PATHS)
def test_gated_path_returns_401_without_token(gated_client, method, path):
    r = gated_client.request(method, path)
    assert r.status_code == 401, f"{method} {path} expected 401, got {r.status_code}"


@pytest.mark.parametrize("method,path", GATED_PATHS)
def test_gated_path_accepts_valid_token(gated_client, method, path):
    tok = _token(gated_client)
    r = gated_client.request(method, path, headers={"Authorization": f"Bearer {tok}"})
    # We don't care about the body (Qdrant may be down — that returns 200 with
    # empty list or 503). We only assert the request got PAST the gate.
    assert r.status_code != 401, f"{method} {path}: expected !=401 with token, got {r.status_code}"


# ── Public endpoints stay open ──

def test_instance_config_no_auth_required(gated_client):
    r = gated_client.get("/api/v1/instance/config")
    assert r.status_code == 200


def test_login_no_auth_required(gated_client):
    r = gated_client.post(
        "/api/v1/auth/login",
        json={"email": "owner@example.com", "password": "ownerpass12345"},
    )
    assert r.status_code == 200


def test_health_no_auth_required(gated_client):
    r = gated_client.get("/health")
    assert r.status_code == 200


def test_root_no_auth_required(gated_client):
    r = gated_client.get("/")
    assert r.status_code == 200
