"""Unit tests for AuthService — register/login/JWT/invites."""
import time
import pytest

from app.resources.metadata_db import MetadataDB
from app.services.auth_service import (
    AuthService, InvalidInviteError, EmailAlreadyTakenError, WeakPasswordError,
)


JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


@pytest.fixture(autouse=True)
def reset_db(tmp_path, monkeypatch):
    import app.resources.metadata_db as mod
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "auth_test.db")
    MetadataDB._instance = None
    MetadataDB.init()
    yield
    MetadataDB.close()


@pytest.fixture
def auth():
    return AuthService(db=MetadataDB, jwt_secret=JWT_SECRET)


def _seed_owner(auth: AuthService) -> str:
    """Bootstrap an owner directly (no invite required for owner)."""
    now = time.time()
    uid = auth.create_owner(email="owner@example.com", password="ownerpass12345")
    return uid


def _open_invite(auth: AuthService, owner_id: str) -> str:
    inv = auth.create_invite(owner_id=owner_id)
    return inv.code


def test_register_with_invite_creates_member(auth):
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


def test_register_normalizes_email_lowercase(auth):
    owner_id = _seed_owner(auth)
    code = _open_invite(auth, owner_id)
    user = auth.register_with_invite(
        email="MixedCase@Example.COM", password="abcdefgh1234",
        invite_code=code,
    )
    assert user.email == "mixedcase@example.com"


def test_register_rejects_unknown_invite(auth):
    with pytest.raises(InvalidInviteError):
        auth.register_with_invite(
            email="x@y.z", password="abcdefgh1234", invite_code="nope",
        )


def test_register_rejects_consumed_invite(auth):
    owner_id = _seed_owner(auth)
    code = _open_invite(auth, owner_id)
    auth.register_with_invite(
        email="first@x.y", password="abcdefgh1234", invite_code=code,
    )
    with pytest.raises(InvalidInviteError):
        auth.register_with_invite(
            email="second@x.y", password="abcdefgh1234", invite_code=code,
        )


def test_register_rejects_expired_invite(auth, monkeypatch):
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


def test_register_rejects_duplicate_email(auth):
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


def test_register_rejects_weak_password(auth):
    owner_id = _seed_owner(auth)
    code = _open_invite(auth, owner_id)
    with pytest.raises(WeakPasswordError):
        auth.register_with_invite(
            email="x@y.z", password="short", invite_code=code,
        )
