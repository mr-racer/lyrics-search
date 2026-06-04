"""Unit tests for AuthService — register/login/JWT/invites."""
import time
import pytest

from app.resources.metadata_db import MetadataDB
from app.services.auth_service import (
    AuthService, InvalidInviteError, EmailAlreadyTakenError,
    WeakPasswordError, OwnerAlreadyExistsError,
    InvalidCredentialsError, TokenExpiredError, InvalidTokenError,
    OwnerOnlyError,
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


def test_register_rejects_expired_invite(auth):
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


def test_register_rolls_back_user_when_invite_claim_lost(auth, monkeypatch):
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


def test_create_owner_rejects_second_owner(auth):
    """create_owner must refuse to clobber an existing owner row."""
    _seed_owner(auth)
    with pytest.raises(OwnerAlreadyExistsError):
        auth.create_owner(email="usurper@example.com", password="usurperpass12")


def test_login_success_returns_user_and_token(auth):
    auth.create_owner(email="lx@example.com", password="strongpass12")
    user, token = auth.login(email="lx@example.com", password="strongpass12")
    assert user.email == "lx@example.com"
    assert token.count(".") == 2  # JWT has 3 segments


def test_login_wrong_password_raises(auth):
    auth.create_owner(email="lx2@example.com", password="strongpass12")
    with pytest.raises(InvalidCredentialsError):
        auth.login(email="lx2@example.com", password="WRONGpass12")


def test_login_unknown_email_raises(auth):
    with pytest.raises(InvalidCredentialsError):
        auth.login(email="nobody@example.com", password="anything12345")


def test_login_updates_last_login(auth):
    auth.create_owner(email="lx3@example.com", password="strongpass12")
    before = MetadataDB.get_user_by_email("lx3@example.com")["last_login_at"]
    assert before is None
    _, _ = auth.login(email="lx3@example.com", password="strongpass12")
    after = MetadataDB.get_user_by_email("lx3@example.com")["last_login_at"]
    assert after is not None and after > 0


def test_verify_token_round_trip(auth):
    auth.create_owner(email="vt@example.com", password="strongpass12")
    _, token = auth.login(email="vt@example.com", password="strongpass12")
    user = auth.verify_token(token)
    assert user.email == "vt@example.com"
    assert user.role == "owner"


def test_verify_token_expired_raises(auth, monkeypatch):
    auth.create_owner(email="exp@example.com", password="strongpass12")
    _, token = auth.login(email="exp@example.com", password="strongpass12")
    # Move clock forward 31 days; our manual `_now() >= exp` check fires
    # (we disable PyJWT's built-in verify_exp to honor monkeypatched _now).
    import app.services.auth_service as mod
    monkeypatch.setattr(mod, "_now", lambda: time.time() + 31 * 86400)
    with pytest.raises(TokenExpiredError):
        auth.verify_token(token)


def test_verify_token_bad_signature_raises(auth):
    auth.create_owner(email="bad@example.com", password="strongpass12")
    _, token = auth.login(email="bad@example.com", password="strongpass12")
    # Corrupt the signature segment.
    head, payload, _sig = token.split(".")
    corrupted = ".".join([head, payload, "AAAAAAAAAAAA"])
    with pytest.raises(InvalidTokenError):
        auth.verify_token(corrupted)


def test_verify_token_unknown_user_raises(auth):
    auth.create_owner(email="del@example.com", password="strongpass12")
    _, token = auth.login(email="del@example.com", password="strongpass12")
    # Delete user out from under the token.
    MetadataDB.get()  # ensure connection
    MetadataDB.get().execute("DELETE FROM users")
    MetadataDB.get().commit()
    with pytest.raises(InvalidTokenError):
        auth.verify_token(token)


def test_verify_token_wrong_algorithm_raises(auth):
    """Token forged with a different algorithm must yield 401, not 500.
    Defends against the alg-confusion attack class."""
    import jwt as pyjwt
    payload = {"sub": "x", "email": "x@y.z", "role": "owner",
               "iat": int(time.time()), "exp": int(time.time()) + 60}
    # Sign with HS512 — our verify pins HS256 only.
    forged = pyjwt.encode(payload, "anything", algorithm="HS512")
    with pytest.raises(InvalidTokenError):
        auth.verify_token(forged)


def test_create_invite_returns_12_char_code(auth):
    owner_id = auth.create_owner(email="iv@example.com", password="ownerpass1234")
    inv = auth.create_invite(owner_id=owner_id)
    assert len(inv.code) == 12
    assert inv.expires_at - inv.created_at == 7 * 86400


def test_create_invite_rejects_non_owner(auth):
    owner_id = auth.create_owner(email="iv2@example.com", password="ownerpass1234")
    code = auth.create_invite(owner_id=owner_id).code
    member = auth.register_with_invite(
        email="mem@example.com", password="memberpass12", invite_code=code,
    )
    with pytest.raises(OwnerOnlyError):
        auth.create_invite(owner_id=member.id)


def test_create_invite_rejects_unknown_user(auth):
    with pytest.raises(OwnerOnlyError):
        auth.create_invite(owner_id="ghost-uid")


def test_list_invites_owner_only(auth):
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


def test_list_invites_filters_consumed(auth):
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


def test_revoke_invite_removes_row(auth):
    owner_id = auth.create_owner(email="rv@example.com", password="ownerpass1234")
    code = auth.create_invite(owner_id=owner_id).code
    auth.revoke_invite(code=code, owner_id=owner_id)
    assert MetadataDB.get_invite(code) is None


def test_revoke_invite_owner_only(auth):
    owner_id = auth.create_owner(email="rv2@example.com", password="ownerpass1234")
    code = auth.create_invite(owner_id=owner_id).code
    c2 = auth.create_invite(owner_id=owner_id).code
    member = auth.register_with_invite(
        email="rvm@example.com", password="memberpass12", invite_code=c2,
    )
    with pytest.raises(OwnerOnlyError):
        auth.revoke_invite(code=code, owner_id=member.id)
