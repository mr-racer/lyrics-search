"""Consolidated unit tests for the auth stack.

Merged from:
  - test_auth_bootstrap.py          -> TestAuthBootstrap
  - test_auth_service.py            -> TestAuthService
  - test_api_dependencies.py        -> TestApiDependencies
  - test_derive_collection_for_user.py -> TestDeriveCollectionForUser
  - test_deprecated_collection_warning.py -> TestDeprecatedCollectionWarning

Each former module's `reset_db` autouse fixture monkeypatched the metadata-DB
singleton to a *different* path, so it is scoped into the class it came from
rather than made module-global.
"""
import logging
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from app.resources.metadata_db import MetadataDB
from app.services.auth_service import (
    AuthService,
    InstanceAlreadyInitializedError,
    WeakPasswordError,
    InvalidInviteError,
    EmailAlreadyTakenError,
    OwnerAlreadyExistsError,
    InvalidCredentialsError,
    TokenExpiredError,
    InvalidTokenError,
    OwnerOnlyError,
)
from app.api.dependencies import (
    get_current_user, get_owner, require_mode,
)
from app.api.helpers import (
    derive_collection_for_user, deprecated_collection_warning,
)


JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


# --- module-level helpers (from test_auth_service.py) ------------------------
def _seed_owner(auth: AuthService) -> str:
    """Bootstrap an owner directly (no invite required for owner)."""
    now = time.time()
    uid = auth.create_owner(email="owner@example.com", password="ownerpass12345")
    return uid


def _open_invite(auth: AuthService, owner_id: str) -> str:
    inv = auth.create_invite(owner_id=owner_id)
    return inv.code


# --- module-level helper (from test_api_dependencies.py) ---------------------
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


class TestAuthBootstrap:
    """AuthService.bootstrap_instance — atomic owner + mode lock.

    Shared by scripts/create_owner.py (CLI) and POST /instance/setup (web wizard).
    """

    @pytest.fixture(autouse=True)
    def reset_db(self, tmp_path, monkeypatch):
        import app.resources.metadata_db as mod
        monkeypatch.setattr(mod, "DB_PATH", tmp_path / "bootstrap_test.db")
        MetadataDB._instance = None
        MetadataDB.init()
        yield
        MetadataDB.close()

    @pytest.fixture
    def auth(self):
        return AuthService(db=MetadataDB, jwt_secret=JWT_SECRET)

    def test_bootstrap_creates_owner_and_locks_mode(self, auth):
        user = auth.bootstrap_instance(email="o@x.y", password="abc123", mode="server")
        assert user.role == "owner"
        assert user.email == "o@x.y"
        cfg = MetadataDB.get_instance_config()
        assert cfg["mode"] == "server"

    def test_bootstrap_rejects_weak_password_before_any_write(self, auth):
        with pytest.raises(WeakPasswordError):
            auth.bootstrap_instance(email="o@x.y", password="abc12", mode="sharing")
        assert MetadataDB.get_instance_config() is None
        assert not MetadataDB.has_owner()

    def test_bootstrap_second_call_raises(self, auth):
        auth.bootstrap_instance(email="o@x.y", password="abc123", mode="sharing")
        with pytest.raises(InstanceAlreadyInitializedError):
            auth.bootstrap_instance(email="p@x.y", password="abc123", mode="sharing")

    def test_bootstrap_refuses_when_config_preexists(self, auth):
        # CLI bootstrap or a concurrent setup already locked the mode.
        MetadataDB.set_instance_config(mode="sharing", created_at=1.0)
        with pytest.raises(InstanceAlreadyInitializedError):
            auth.bootstrap_instance(email="o@x.y", password="abc123", mode="sharing")
        assert not MetadataDB.has_owner()

    def test_bootstrap_rolls_back_owner_when_config_write_fails(self, auth, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("disk on fire")
        monkeypatch.setattr(MetadataDB, "set_instance_config", boom)
        with pytest.raises(RuntimeError):
            auth.bootstrap_instance(email="o@x.y", password="abc123", mode="sharing")
        assert not MetadataDB.has_owner()
        assert MetadataDB.get_user_by_email("o@x.y") is None

    def test_bootstrap_config_race_maps_to_already_initialized(self, auth, monkeypatch):
        """Two concurrent setups: the loser's set_instance_config hits the
        sqlite UNIQUE/PK constraint. The loser must roll back its owner row and
        surface InstanceAlreadyInitializedError, not a raw IntegrityError."""
        real_set = MetadataDB.set_instance_config.__func__

        def race(cls, *, mode, created_at):
            # Winner sneaks in between our pre-check and our write.
            real_set(cls, mode="server", created_at=0.5)
            return real_set(cls, mode=mode, created_at=created_at)  # → IntegrityError

        monkeypatch.setattr(MetadataDB, "set_instance_config", classmethod(race))
        with pytest.raises(InstanceAlreadyInitializedError):
            auth.bootstrap_instance(email="o@x.y", password="abc123", mode="sharing")
        assert not MetadataDB.has_owner()


class TestAuthService:
    """AuthService — register/login/JWT/invites."""

    @pytest.fixture(autouse=True)
    def reset_db(self, tmp_path, monkeypatch):
        import app.resources.metadata_db as mod
        monkeypatch.setattr(mod, "DB_PATH", tmp_path / "auth_test.db")
        MetadataDB._instance = None
        MetadataDB.init()
        yield
        MetadataDB.close()

    @pytest.fixture
    def auth(self):
        return AuthService(db=MetadataDB, jwt_secret=JWT_SECRET)

    def test_register_with_invite_creates_member(self, auth):
        owner_id = _seed_owner(auth)
        code = _open_invite(auth, owner_id)
        user = auth.register_with_invite(
            email="newbie@example.com", password="newbiepass123", invite_code=code,
        )
        assert user.role == "member"
        assert user.email == "newbie@example.com"
        # invite must now be consumed
        stored = MetadataDB.get_invite(code)
        assert stored["consumed_by"] == user.id

    def test_register_normalizes_email_lowercase(self, auth):
        owner_id = _seed_owner(auth)
        code = _open_invite(auth, owner_id)
        user = auth.register_with_invite(
            email="MixedCase@Example.COM", password="abcdefgh1234",
            invite_code=code,
        )
        assert user.email == "mixedcase@example.com"

    def test_register_rejects_unknown_invite(self, auth):
        with pytest.raises(InvalidInviteError):
            auth.register_with_invite(
                email="x@y.z", password="abcdefgh1234", invite_code="nope",
            )

    def test_register_rejects_consumed_invite(self, auth):
        owner_id = _seed_owner(auth)
        code = _open_invite(auth, owner_id)
        auth.register_with_invite(
            email="first@x.y", password="abcdefgh1234", invite_code=code,
        )
        with pytest.raises(InvalidInviteError):
            auth.register_with_invite(
                email="second@x.y", password="abcdefgh1234", invite_code=code,
            )

    def test_register_rejects_expired_invite(self, auth):
        owner_id = _seed_owner(auth)
        # Force invite to be 8 days old.
        MetadataDB.create_invite(
            code="oldcode12345",
            created_by=owner_id,
            created_at=time.time() - 9 * 86400,
            expires_at=time.time() - 2 * 86400,
        )
        with pytest.raises(InvalidInviteError):
            auth.register_with_invite(
                email="x@y.z", password="abcdefgh1234", invite_code="oldcode12345",
            )

    def test_register_rejects_duplicate_email(self, auth):
        owner_id = _seed_owner(auth)
        c1 = _open_invite(auth, owner_id)
        c2 = _open_invite(auth, owner_id)
        auth.register_with_invite(
            email="dup@x.y", password="abcdefgh1234", invite_code=c1,
        )
        with pytest.raises(EmailAlreadyTakenError):
            auth.register_with_invite(
                email="dup@x.y", password="abcdefgh1234", invite_code=c2,
            )

    def test_register_rejects_weak_password(self, auth):
        owner_id = _seed_owner(auth)
        code = _open_invite(auth, owner_id)
        with pytest.raises(WeakPasswordError):
            auth.register_with_invite(
                email="x@y.z", password="short", invite_code=code,
            )

    def test_register_rolls_back_user_when_invite_claim_lost(self, auth, monkeypatch):
        """Simulate the TOCTOU race: the invite passes the get_invite open-check but
        a concurrent registration claims it before our consume_invite runs (consume
        returns False). The just-created user must be rolled back and
        InvalidInviteError raised — no orphan account left behind."""
        owner_id = _seed_owner(auth)
        code = _open_invite(auth, owner_id)
        # Force the race-lost branch: consume_invite returns False as if another
        # request claimed the code in the get_invite→consume window.
        monkeypatch.setattr(auth.db, "consume_invite", lambda *a, **k: False)
        with pytest.raises(InvalidInviteError):
            auth.register_with_invite(
                email="raced@x.y", password="racedpass123", invite_code=code,
            )
        # The rolled-back user must NOT linger in the DB.
        assert MetadataDB.get_user_by_email("raced@x.y") is None

    def test_create_owner_rejects_second_owner(self, auth):
        """create_owner must refuse to clobber an existing owner row."""
        _seed_owner(auth)
        with pytest.raises(OwnerAlreadyExistsError):
            auth.create_owner(email="usurper@example.com", password="usurperpass12")

    def test_login_success_returns_user_and_token(self, auth):
        auth.create_owner(email="lx@example.com", password="strongpass12")
        user, token = auth.login(email="lx@example.com", password="strongpass12")
        assert user.email == "lx@example.com"
        assert token.count(".") == 2  # JWT has 3 segments

    def test_login_wrong_password_raises(self, auth):
        auth.create_owner(email="lx2@example.com", password="strongpass12")
        with pytest.raises(InvalidCredentialsError):
            auth.login(email="lx2@example.com", password="WRONGpass12")

    def test_login_unknown_email_raises(self, auth):
        with pytest.raises(InvalidCredentialsError):
            auth.login(email="nobody@example.com", password="anything12345")

    def test_login_updates_last_login(self, auth):
        auth.create_owner(email="lx3@example.com", password="strongpass12")
        before = MetadataDB.get_user_by_email("lx3@example.com")["last_login_at"]
        assert before is None
        _, _ = auth.login(email="lx3@example.com", password="strongpass12")
        after = MetadataDB.get_user_by_email("lx3@example.com")["last_login_at"]
        assert after is not None and after > 0

    def test_verify_token_round_trip(self, auth):
        auth.create_owner(email="vt@example.com", password="strongpass12")
        _, token = auth.login(email="vt@example.com", password="strongpass12")
        user = auth.verify_token(token)
        assert user.email == "vt@example.com"
        assert user.role == "owner"

    def test_verify_token_expired_raises(self, auth, monkeypatch):
        auth.create_owner(email="exp@example.com", password="strongpass12")
        _, token = auth.login(email="exp@example.com", password="strongpass12")
        # Move clock forward 31 days; our manual `_now() >= exp` check fires
        # (we disable PyJWT's built-in verify_exp to honor monkeypatched _now).
        import app.services.auth_service as mod
        monkeypatch.setattr(mod, "_now", lambda: time.time() + 31 * 86400)
        with pytest.raises(TokenExpiredError):
            auth.verify_token(token)

    def test_verify_token_bad_signature_raises(self, auth):
        auth.create_owner(email="bad@example.com", password="strongpass12")
        _, token = auth.login(email="bad@example.com", password="strongpass12")
        # Corrupt the signature segment.
        head, payload, _sig = token.split(".")
        corrupted = ".".join([head, payload, "AAAAAAAAAAAA"])
        with pytest.raises(InvalidTokenError):
            auth.verify_token(corrupted)

    def test_verify_token_unknown_user_raises(self, auth):
        auth.create_owner(email="del@example.com", password="strongpass12")
        _, token = auth.login(email="del@example.com", password="strongpass12")
        # Delete user out from under the token.
        MetadataDB.get()  # ensure connection
        MetadataDB.get().execute("DELETE FROM users")
        MetadataDB.get().commit()
        with pytest.raises(InvalidTokenError):
            auth.verify_token(token)

    def test_verify_token_wrong_algorithm_raises(self, auth):
        """Token forged with a different algorithm must yield 401, not 500.
        Defends against the alg-confusion attack class."""
        import jwt as pyjwt
        payload = {"sub": "x", "email": "x@y.z", "role": "owner",
                   "iat": int(time.time()), "exp": int(time.time()) + 60}
        # Sign with HS512 — our verify pins HS256 only.
        forged = pyjwt.encode(payload, "anything", algorithm="HS512")
        with pytest.raises(InvalidTokenError):
            auth.verify_token(forged)

    def test_create_invite_returns_12_char_code(self, auth):
        owner_id = auth.create_owner(email="iv@example.com", password="ownerpass1234")
        inv = auth.create_invite(owner_id=owner_id)
        assert len(inv.code) == 12
        assert inv.expires_at - inv.created_at == 7 * 86400

    def test_create_invite_rejects_non_owner(self, auth):
        owner_id = auth.create_owner(email="iv2@example.com", password="ownerpass1234")
        code = auth.create_invite(owner_id=owner_id).code
        member = auth.register_with_invite(
            email="mem@example.com", password="memberpass12", invite_code=code,
        )
        with pytest.raises(OwnerOnlyError):
            auth.create_invite(owner_id=member.id)

    def test_create_invite_rejects_unknown_user(self, auth):
        with pytest.raises(OwnerOnlyError):
            auth.create_invite(owner_id="ghost-uid")

    def test_list_invites_owner_only(self, auth):
        owner_id = auth.create_owner(email="li@example.com", password="ownerpass1234")
        auth.create_invite(owner_id=owner_id)
        auth.create_invite(owner_id=owner_id)
        invites = auth.list_invites(owner_id=owner_id, include_consumed=False)
        assert len(invites) == 2

        code = invites[0].code
        member = auth.register_with_invite(
            email="m2@example.com", password="memberpass12", invite_code=code,
        )
        with pytest.raises(OwnerOnlyError):
            auth.list_invites(owner_id=member.id, include_consumed=False)

    def test_list_invites_filters_consumed(self, auth):
        owner_id = auth.create_owner(email="li2@example.com", password="ownerpass1234")
        a = auth.create_invite(owner_id=owner_id).code
        b = auth.create_invite(owner_id=owner_id).code
        auth.register_with_invite(
            email="cn@example.com", password="memberpass12", invite_code=a,
        )
        open_only = auth.list_invites(owner_id=owner_id, include_consumed=False)
        assert {i.code for i in open_only} == {b}
        everything = auth.list_invites(owner_id=owner_id, include_consumed=True)
        assert {i.code for i in everything} == {a, b}

    def test_revoke_invite_removes_row(self, auth):
        owner_id = auth.create_owner(email="rv@example.com", password="ownerpass1234")
        code = auth.create_invite(owner_id=owner_id).code
        auth.revoke_invite(code=code, owner_id=owner_id)
        assert MetadataDB.get_invite(code) is None

    def test_revoke_invite_owner_only(self, auth):
        owner_id = auth.create_owner(email="rv2@example.com", password="ownerpass1234")
        code = auth.create_invite(owner_id=owner_id).code
        c2 = auth.create_invite(owner_id=owner_id).code
        member = auth.register_with_invite(
            email="rvm@example.com", password="memberpass12", invite_code=c2,
        )
        with pytest.raises(OwnerOnlyError):
            auth.revoke_invite(code=code, owner_id=member.id)

    def test_password_policy_accepts_6_chars(self, auth):
        """Spec 2026-06-10 first-run wizard: policy relaxed 10 → 6 chars."""
        uid = auth.create_owner(email="o6@x.y", password="abc123")
        assert uid

    def test_password_policy_rejects_5_chars(self, auth):
        with pytest.raises(WeakPasswordError):
            auth.create_owner(email="o5@x.y", password="abc12")


class TestApiDependencies:
    """FastAPI auth/mode dependencies."""

    @pytest.fixture(autouse=True)
    def reset_db(self, tmp_path, monkeypatch):
        import app.resources.metadata_db as mod
        monkeypatch.setattr(mod, "DB_PATH", tmp_path / "deps_test.db")
        MetadataDB._instance = None
        MetadataDB.init()
        yield
        MetadataDB.close()

    def test_get_current_user_401_without_header(self, monkeypatch):
        monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
        auth = AuthService(jwt_secret=JWT_SECRET)
        app = _make_app(auth)
        r = TestClient(app).get("/protected")
        assert r.status_code == 401
        assert r.json()["detail"] == "missing or invalid Authorization header"

    def test_get_current_user_401_on_bad_token(self, monkeypatch):
        monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
        auth = AuthService(jwt_secret=JWT_SECRET)
        app = _make_app(auth)
        r = TestClient(app).get(
            "/protected", headers={"Authorization": "Bearer garbage.token.here"},
        )
        assert r.status_code == 401

    def test_get_current_user_200_with_valid_token(self, monkeypatch):
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

    def test_get_owner_403_for_member(self, monkeypatch):
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

    def test_get_owner_200_for_owner(self, monkeypatch):
        monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
        auth = AuthService(jwt_secret=JWT_SECRET)
        auth.create_owner(email="owh@x.y", password="ownerpass1234")
        _, token = auth.login(email="owh@x.y", password="ownerpass1234")
        app = _make_app(auth)
        r = TestClient(app).get(
            "/owner-only", headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200

    def test_get_current_user_401_on_empty_bearer(self, monkeypatch):
        monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
        auth = AuthService(jwt_secret=JWT_SECRET)
        app = _make_app(auth)
        r = TestClient(app).get("/protected", headers={"Authorization": "Bearer "})
        assert r.status_code == 401
        assert r.json()["detail"] == "empty bearer token"

    def test_get_current_user_401_on_expired_token(self, monkeypatch):
        monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
        auth = AuthService(jwt_secret=JWT_SECRET)
        auth.create_owner(email="ex@x.y", password="ownerpass1234")
        _, token = auth.login(email="ex@x.y", password="ownerpass1234")
        import app.services.auth_service as mod
        monkeypatch.setattr(mod, "_now", lambda: __import__("time").time() + 31 * 86400)
        app = _make_app(auth)
        r = TestClient(app).get(
            "/protected", headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 401
        assert r.json()["detail"] == "token expired"

    def test_require_mode_404_when_mode_mismatch(self, monkeypatch):
        monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
        MetadataDB.set_instance_config(mode="sharing", created_at=1.0)
        app = FastAPI()
        @app.get("/server-only")
        def server_only(_=Depends(require_mode("server"))):
            return {"ok": True}
        r = TestClient(app).get("/server-only")
        assert r.status_code == 404

    def test_require_mode_passes_when_mode_matches(self, monkeypatch):
        monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
        MetadataDB.set_instance_config(mode="server", created_at=1.0)
        app = FastAPI()
        @app.get("/server-only")
        def server_only(_=Depends(require_mode("server"))):
            return {"ok": True}
        r = TestClient(app).get("/server-only")
        assert r.status_code == 200


class TestDeriveCollectionForUser:
    """derive_collection_for_user — single source of truth for mapping a
    JWT-identified user to their Qdrant collection name."""

    def test_returns_acct_prefix_plus_user_id(self):
        user = SimpleNamespace(id="abc-123", email="x@y.z")
        assert derive_collection_for_user(user) == "acct_abc-123"

    def test_uuid4_input(self):
        user = SimpleNamespace(id="550e8400-e29b-41d4-a716-446655440000")
        assert derive_collection_for_user(user) == "acct_550e8400-e29b-41d4-a716-446655440000"

    def test_rejects_user_without_id(self):
        user = SimpleNamespace(email="x@y.z")
        with pytest.raises(AttributeError):
            derive_collection_for_user(user)

    def test_rejects_empty_id(self):
        user = SimpleNamespace(id="")
        with pytest.raises(ValueError):
            derive_collection_for_user(user)

    def test_rejects_id_with_pathlike_chars(self):
        # Defense in depth: even though id comes from server-issued UUID, never
        # let it carry slashes/backslashes that could break Qdrant collection
        # naming or escape into a path. Belt-and-suspenders.
        user = SimpleNamespace(id="../etc/passwd")
        with pytest.raises(ValueError):
            derive_collection_for_user(user)


class TestDeprecatedCollectionWarning:
    """deprecated_collection_warning — verifies the right log level / message
    shape so operators can grep for migration progress."""

    def test_no_log_when_supplied_is_none(self, caplog):
        with caplog.at_level(logging.INFO, logger="app.api.helpers"):
            deprecated_collection_warning(None, "acct_X", "/foo")
        assert not caplog.records

    def test_info_when_supplied_matches_derived(self, caplog):
        with caplog.at_level(logging.INFO, logger="app.api.helpers"):
            deprecated_collection_warning("acct_X", "acct_X", "/foo")
        assert any(r.levelno == logging.INFO for r in caplog.records)
        assert "/foo" in caplog.text
        assert "acct_X" in caplog.text

    def test_warning_when_supplied_mismatches(self, caplog):
        with caplog.at_level(logging.WARNING, logger="app.api.helpers"):
            deprecated_collection_warning("acct_OTHER", "acct_ME", "/library/stats")
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert "acct_OTHER" in caplog.text
        assert "acct_ME" in caplog.text
        assert "/library/stats" in caplog.text
        assert "IGNORED" in caplog.text
