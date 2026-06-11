"""Unit tests for AuthService.bootstrap_instance — atomic owner + mode lock.

Shared by scripts/create_owner.py (CLI) and POST /instance/setup (web wizard).
"""
import pytest

from app.resources.metadata_db import MetadataDB
from app.services.auth_service import (
    AuthService, InstanceAlreadyInitializedError, WeakPasswordError,
)

JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


@pytest.fixture(autouse=True)
def reset_db(tmp_path, monkeypatch):
    import app.resources.metadata_db as mod
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "bootstrap_test.db")
    MetadataDB._instance = None
    MetadataDB.init()
    yield
    MetadataDB.close()


@pytest.fixture
def auth():
    return AuthService(db=MetadataDB, jwt_secret=JWT_SECRET)


def test_bootstrap_creates_owner_and_locks_mode(auth):
    user = auth.bootstrap_instance(email="o@x.y", password="abc123", mode="server")
    assert user.role == "owner"
    assert user.email == "o@x.y"
    cfg = MetadataDB.get_instance_config()
    assert cfg["mode"] == "server"


def test_bootstrap_rejects_weak_password_before_any_write(auth):
    with pytest.raises(WeakPasswordError):
        auth.bootstrap_instance(email="o@x.y", password="abc12", mode="sharing")
    assert MetadataDB.get_instance_config() is None
    assert not MetadataDB.has_owner()


def test_bootstrap_second_call_raises(auth):
    auth.bootstrap_instance(email="o@x.y", password="abc123", mode="sharing")
    with pytest.raises(InstanceAlreadyInitializedError):
        auth.bootstrap_instance(email="p@x.y", password="abc123", mode="sharing")


def test_bootstrap_refuses_when_config_preexists(auth):
    # CLI bootstrap or a concurrent setup already locked the mode.
    MetadataDB.set_instance_config(mode="sharing", created_at=1.0)
    with pytest.raises(InstanceAlreadyInitializedError):
        auth.bootstrap_instance(email="o@x.y", password="abc123", mode="sharing")
    assert not MetadataDB.has_owner()


def test_bootstrap_rolls_back_owner_when_config_write_fails(auth, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(MetadataDB, "set_instance_config", boom)
    with pytest.raises(RuntimeError):
        auth.bootstrap_instance(email="o@x.y", password="abc123", mode="sharing")
    assert not MetadataDB.has_owner()
    assert MetadataDB.get_user_by_email("o@x.y") is None


def test_bootstrap_config_race_maps_to_already_initialized(auth, monkeypatch):
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
