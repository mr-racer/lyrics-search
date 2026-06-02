"""Pydantic model contracts for Phase A auth."""
import pytest
from pydantic import ValidationError

from app.domain.models import (
    User, Invite, InstanceConfigResponse,
    LoginRequest, RegisterRequest, AuthResponse, InviteResponse,
)


def test_user_round_trip():
    u = User(id="uid-1", email="x@y.z", role="owner",
             created_at=1700000000.0, last_login_at=None)
    assert u.id == "uid-1"
    assert u.role == "owner"
    assert "password_hash" not in u.model_dump()  # never exposed


def test_user_role_must_be_owner_or_member():
    with pytest.raises(ValidationError):
        User(id="x", email="x@y.z", role="admin", created_at=1.0, last_login_at=None)


def test_invite_round_trip():
    inv = Invite(code="abcdefghij12", created_by="uid-1",
                 created_at=1.0, expires_at=2.0,
                 consumed_by=None, consumed_at=None)
    assert inv.code == "abcdefghij12"


def test_instance_config_response_modes():
    assert InstanceConfigResponse(mode="sharing").mode == "sharing"
    assert InstanceConfigResponse(mode="server").mode == "server"
    with pytest.raises(ValidationError):
        InstanceConfigResponse(mode="other")


def test_login_request_requires_both_fields():
    with pytest.raises(ValidationError):
        LoginRequest(email="x@y.z")


def test_register_request_requires_invite_code():
    with pytest.raises(ValidationError):
        RegisterRequest(email="x@y.z", password="pw12345678")


def test_auth_response_carries_token_and_user():
    ar = AuthResponse(
        token="jwt.token.here",
        user=User(id="u", email="x@y.z", role="owner",
                  created_at=1.0, last_login_at=None),
    )
    assert ar.token.startswith("jwt.")


def test_invite_response_omits_consumer_when_open():
    ir = InviteResponse(
        code="abcdefghij12", created_at=1.0, expires_at=2.0,
        consumed=False, consumed_at=None,
    )
    assert ir.consumed is False
