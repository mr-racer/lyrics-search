"""Stream-scoped tokens: short-lived JWTs that authenticate ONLY the audio
stream endpoint via a query parameter (?st=...), because <audio> elements
cannot send an Authorization header.

Security properties under test:
  - a stream token must NOT be accepted by the general API gate (it leaks
    into URLs / server logs, so its blast radius must stay "can fetch audio")
  - a regular login token must NOT be accepted as a stream token via ?st=
    (otherwise the full-power token would be encouraged to travel in URLs)
"""
import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from app.resources.metadata_db import MetadataDB
from app.services.auth_service import (
    AuthService, InvalidTokenError, TokenExpiredError,
    STREAM_TOKEN_TTL_SECONDS,
)
from app.api.dependencies import get_current_user, get_user_for_stream


JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


@pytest.fixture(autouse=True)
def reset_db(tmp_path, monkeypatch):
    import app.resources.metadata_db as mod
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "stream_token_test.db")
    MetadataDB._instance = None
    MetadataDB.init()
    yield
    MetadataDB.close()


@pytest.fixture
def auth():
    return AuthService(jwt_secret=JWT_SECRET)


@pytest.fixture
def owner(auth):
    auth.create_owner(email="o@x.y", password="ownerpass1234")
    user, _ = auth.login(email="o@x.y", password="ownerpass1234")
    return user


# ── AuthService.issue_stream_token / verify_stream_token ────────────────────

class TestStreamTokenService:
    def test_roundtrip_returns_same_user(self, auth, owner):
        st = auth.issue_stream_token(owner)
        verified = auth.verify_stream_token(st)
        assert verified.id == owner.id
        assert verified.role == owner.role

    def test_rejects_regular_login_token(self, auth, owner):
        login_token = auth.issue_token(owner)
        with pytest.raises(InvalidTokenError):
            auth.verify_stream_token(login_token)

    def test_main_verify_rejects_stream_token(self, auth, owner):
        st = auth.issue_stream_token(owner)
        with pytest.raises(InvalidTokenError):
            auth.verify_token(st)

    def test_expired_stream_token_raises(self, auth, owner, monkeypatch):
        st = auth.issue_stream_token(owner)
        import app.services.auth_service as mod
        real_now = mod._now()
        monkeypatch.setattr(
            mod, "_now", lambda: real_now + STREAM_TOKEN_TTL_SECONDS + 60,
        )
        with pytest.raises(TokenExpiredError):
            auth.verify_stream_token(st)

    def test_unknown_user_raises(self, auth, owner):
        st = auth.issue_stream_token(owner)
        MetadataDB.delete_user(owner.id)
        with pytest.raises(InvalidTokenError):
            auth.verify_stream_token(st)

    def test_garbage_raises(self, auth):
        with pytest.raises(InvalidTokenError):
            auth.verify_stream_token("garbage.token.here")


# ── get_user_for_stream dependency (header OR ?st=) ─────────────────────────

def _make_app(auth: AuthService):
    app = FastAPI()
    app.state.auth_service = auth

    @app.get("/audio")
    def audio(user=Depends(get_user_for_stream)):
        return {"id": user.id}

    return app


class TestStreamDependency:
    def test_401_without_header_or_st(self, auth):
        r = TestClient(_make_app(auth)).get("/audio")
        assert r.status_code == 401

    def test_200_with_valid_st_query(self, auth, owner):
        st = auth.issue_stream_token(owner)
        r = TestClient(_make_app(auth)).get(f"/audio?st={st}")
        assert r.status_code == 200
        assert r.json()["id"] == owner.id

    def test_200_with_bearer_header_still_works(self, auth, owner):
        token = auth.issue_token(owner)
        r = TestClient(_make_app(auth)).get(
            "/audio", headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["id"] == owner.id

    def test_401_with_login_token_in_st_query(self, auth, owner):
        login_token = auth.issue_token(owner)
        r = TestClient(_make_app(auth)).get(f"/audio?st={login_token}")
        assert r.status_code == 401

    def test_401_with_garbage_st(self, auth):
        r = TestClient(_make_app(auth)).get("/audio?st=garbage.token.here")
        assert r.status_code == 401


# ── POST /auth/stream-token route ────────────────────────────────────────────

def _make_auth_app(auth: AuthService):
    from app.api.routes.auth import router as auth_router
    app = FastAPI()
    app.state.auth_service = auth
    app.include_router(auth_router, prefix="/api/v1")
    return app


class TestStreamTokenRoute:
    def test_requires_auth(self, auth):
        r = TestClient(_make_auth_app(auth)).post("/api/v1/auth/stream-token")
        assert r.status_code == 401

    def test_returns_verifiable_stream_token(self, auth, owner):
        token = auth.issue_token(owner)
        r = TestClient(_make_auth_app(auth)).post(
            "/api/v1/auth/stream-token",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["expires_in"] == STREAM_TOKEN_TTL_SECONDS
        verified = auth.verify_stream_token(body["token"])
        assert verified.id == owner.id
