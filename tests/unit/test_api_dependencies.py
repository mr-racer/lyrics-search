"""Unit tests for FastAPI auth/mode dependencies."""
import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from app.resources.metadata_db import MetadataDB
from app.services.auth_service import AuthService
from app.api.dependencies import (
    get_current_user, get_owner, require_mode,
)


JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


@pytest.fixture(autouse=True)
def reset_db(tmp_path, monkeypatch):
    import app.resources.metadata_db as mod
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "deps_test.db")
    MetadataDB._instance = None
    MetadataDB.init()
    yield
    MetadataDB.close()


def _make_app(auth: AuthService):
    app = FastAPI()
    app.state.auth_service = auth

    @app.get("/protected")
    def protected(user=Depends(get_current_user)):
        return {"id": user.id, "role": user.role}

    @app.get("/owner-only")
    def owner_only(user=Depends(get_owner)):
        return {"id": user.id}

    return app


def test_get_current_user_401_without_header(monkeypatch):
    monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
    auth = AuthService(jwt_secret=JWT_SECRET)
    app = _make_app(auth)
    r = TestClient(app).get("/protected")
    assert r.status_code == 401
    assert r.json()["detail"] == "missing or invalid Authorization header"


def test_get_current_user_401_on_bad_token(monkeypatch):
    monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
    auth = AuthService(jwt_secret=JWT_SECRET)
    app = _make_app(auth)
    r = TestClient(app).get(
        "/protected", headers={"Authorization": "Bearer garbage.token.here"},
    )
    assert r.status_code == 401


def test_get_current_user_200_with_valid_token(monkeypatch):
    monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
    auth = AuthService(jwt_secret=JWT_SECRET)
    auth.create_owner(email="o@x.y", password="ownerpass1234")
    _, token = auth.login(email="o@x.y", password="ownerpass1234")
    app = _make_app(auth)
    r = TestClient(app).get(
        "/protected", headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert r.json()["role"] == "owner"


def test_get_owner_403_for_member(monkeypatch):
    monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
    auth = AuthService(jwt_secret=JWT_SECRET)
    oid = auth.create_owner(email="ownr@x.y", password="ownerpass1234")
    code = auth.create_invite(owner_id=oid).code
    auth.register_with_invite(
        email="m@x.y", password="memberpass12", invite_code=code,
    )
    _, member_token = auth.login(email="m@x.y", password="memberpass12")
    app = _make_app(auth)
    r = TestClient(app).get(
        "/owner-only", headers={"Authorization": f"Bearer {member_token}"},
    )
    assert r.status_code == 403


def test_require_mode_404_when_mode_mismatch(monkeypatch):
    monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
    MetadataDB.set_instance_config(mode="sharing", created_at=1.0)
    app = FastAPI()
    @app.get("/server-only")
    def server_only(_=Depends(require_mode("server"))):
        return {"ok": True}
    r = TestClient(app).get("/server-only")
    assert r.status_code == 404


def test_require_mode_passes_when_mode_matches(monkeypatch):
    monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
    MetadataDB.set_instance_config(mode="server", created_at=1.0)
    app = FastAPI()
    @app.get("/server-only")
    def server_only(_=Depends(require_mode("server"))):
        return {"ok": True}
    r = TestClient(app).get("/server-only")
    assert r.status_code == 200
