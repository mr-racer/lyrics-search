"""Per-user settings round-trip: text_model_name + clap_enabled.

ADAPTED for merged Phase A: the `users` table already exists with NOT NULL
`email`/`password_hash` (created by Phase A). Per-user settings are COLUMNS on
that table (added via _ensure_columns), keyed by the real `users.id`. Tests
seed a real user via MetadataDB.create_user; there is no bare settings-row
upsert (creating users is Phase A's job).
"""

import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    import app.resources.metadata_db as mod
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "settings.db")
    MetadataDB._instance = None
    MetadataDB.init()
    yield
    MetadataDB.close()


def _seed_user(user_id: str) -> None:
    """Create a real (Phase A) user row so settings columns get their defaults."""
    MetadataDB.create_user(
        user_id=user_id, email=f"{user_id}@x.y", password_hash="h",
        role="member", created_at=1700000000.0,
    )


def test_new_user_has_default_settings():
    _seed_user("acct-1")
    s = MetadataDB.get_user_settings("acct-1")
    assert s["text_model_name"] == "jinaai/jina-embeddings-v3"
    assert s["clap_enabled"] is True


def test_update_user_settings_persists_text_model_name():
    _seed_user("acct-1")
    MetadataDB.update_user_settings("acct-1", text_model_name="Qwen/Qwen3-Embedding-0.6B")
    s = MetadataDB.get_user_settings("acct-1")
    assert s["text_model_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert s["clap_enabled"] is True


def test_update_user_settings_persists_clap_enabled():
    _seed_user("acct-1")
    MetadataDB.update_user_settings("acct-1", clap_enabled=False)
    s = MetadataDB.get_user_settings("acct-1")
    assert s["clap_enabled"] is False
    # text_model_name untouched by a clap-only update
    assert s["text_model_name"] == "jinaai/jina-embeddings-v3"


def test_update_user_settings_noop_when_all_none():
    _seed_user("acct-1")
    MetadataDB.update_user_settings("acct-1")  # no fields → no-op, no error
    s = MetadataDB.get_user_settings("acct-1")
    assert s["text_model_name"] == "jinaai/jina-embeddings-v3"
    assert s["clap_enabled"] is True


def test_get_user_settings_unknown_user_returns_none():
    assert MetadataDB.get_user_settings("does-not-exist") is None


def test_init_is_idempotent():
    """Re-init must not raise on the ALTER TABLE column adds."""
    MetadataDB.init()
    MetadataDB.init()  # second call: _ensure_columns silently skips existing cols
