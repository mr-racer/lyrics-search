"""Unit tests for SettingsService — the DB > env > default resolver, secret
masking, and set_many normalization. Uses a FakeDB stub so the resolver logic
is tested in isolation; the real MetadataDB methods are covered by the
integration tests (real sqlite round-trip)."""
import pytest

from app.services.settings_service import (
    SettingsService, UNCHANGED_SENTINEL,
)


class FakeDB:
    """In-memory stand-in mirroring MetadataDB's delete-on-empty semantics."""

    def __init__(self, store=None):
        self.store = dict(store or {})
        self.set_calls = []

    def get_instance_setting(self, key):
        return self.store.get(key)

    def set_instance_settings(self, mapping, updated_at):
        self.set_calls.append((dict(mapping), updated_at))
        for k, v in mapping.items():
            if v is None or v == "":
                self.store.pop(k, None)
            else:
                self.store[k] = v


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Strip the mirrored env vars so 'default' is the baseline unless a test
    sets one explicitly."""
    for name in ("LLM_BASE_URL", "LLM_MODEL", "OPENAI_API_KEY",
                 "EMBED_MODEL", "CLAP_ENABLED", "AI_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    yield


# ── resolver precedence ──────────────────────────────────────────────────────

def test_default_when_nothing_set():
    s = SettingsService(db=FakeDB())
    assert s.llm_base_url() is None
    assert s.llm_api_key() == "lm-studio"   # built-in default
    assert s.embed_model() is None
    assert s.clap_enabled() is True
    assert s.ai_enabled() is False


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://env:8000/v1")
    s = SettingsService(db=FakeDB())
    assert s.llm_base_url() == "http://env:8000/v1"
    assert s.source("LLM_BASE_URL") == "env"


def test_db_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://env:8000/v1")
    s = SettingsService(db=FakeDB({"LLM_BASE_URL": "http://db:9000/v1"}))
    assert s.llm_base_url() == "http://db:9000/v1"
    assert s.source("LLM_BASE_URL") == "db"


def test_source_default_when_unset():
    s = SettingsService(db=FakeDB())
    assert s.source("LLM_MODEL") == "default"


def test_unknown_key_raises():
    s = SettingsService(db=FakeDB())
    with pytest.raises(KeyError):
        s.get("NOPE")


# ── api key env fallback is OPENAI_API_KEY ──────────────────────────────────

def test_api_key_env_fallback_uses_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    s = SettingsService(db=FakeDB())
    assert s.llm_api_key() == "sk-from-env"
    assert s.source("LLM_API_KEY") == "env"


# ── bool parsing + ai_available ─────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("false", False), ("", False), ("nope", False),
])
def test_bool_parsing(raw, expected):
    s = SettingsService(db=FakeDB({"AI_ENABLED": raw}))
    assert s.ai_enabled() is expected


def test_ai_available_requires_enabled_and_endpoint():
    # enabled but no endpoint → not available
    s = SettingsService(db=FakeDB({"AI_ENABLED": "1"}))
    assert s.ai_available() is False
    # endpoint but disabled → not available
    s = SettingsService(db=FakeDB({"LLM_BASE_URL": "http://x/v1"}))
    assert s.ai_available() is False
    # both → available
    s = SettingsService(db=FakeDB({"AI_ENABLED": "1", "LLM_BASE_URL": "http://x/v1"}))
    assert s.ai_available() is True


# ── set_many normalization ──────────────────────────────────────────────────

def test_set_many_bool_normalized_to_1_0():
    db = FakeDB()
    s = SettingsService(db=db)
    s.set_many({"CLAP_ENABLED": True, "AI_ENABLED": False}, updated_at=1.0)
    assert db.store["CLAP_ENABLED"] == "1"
    assert db.store["AI_ENABLED"] == "0"


def test_set_many_secret_sentinel_is_skipped():
    db = FakeDB({"LLM_API_KEY": "sk-existing"})
    s = SettingsService(db=db)
    s.set_many({"LLM_API_KEY": UNCHANGED_SENTINEL}, updated_at=1.0)
    assert db.store["LLM_API_KEY"] == "sk-existing"   # untouched


def test_set_many_secret_new_value_written():
    db = FakeDB({"LLM_API_KEY": "sk-old"})
    s = SettingsService(db=db)
    s.set_many({"LLM_API_KEY": "sk-new"}, updated_at=1.0)
    assert db.store["LLM_API_KEY"] == "sk-new"


def test_set_many_none_clears_key_falls_back():
    db = FakeDB({"LLM_BASE_URL": "http://db/v1"})
    s = SettingsService(db=db)
    s.set_many({"LLM_BASE_URL": None}, updated_at=1.0)
    assert "LLM_BASE_URL" not in db.store
    assert s.llm_base_url() is None   # back to default


def test_set_many_ignores_unknown_keys():
    db = FakeDB()
    s = SettingsService(db=db)
    s.set_many({"BOGUS": "x", "LLM_MODEL": "m"}, updated_at=1.0)
    assert "BOGUS" not in db.store
    assert db.store["LLM_MODEL"] == "m"


# ── public_view masks the secret ────────────────────────────────────────────

def test_public_view_masks_api_key():
    s = SettingsService(db=FakeDB({"LLM_API_KEY": "sk-secret", "LLM_MODEL": "m"}))
    view = s.public_view()
    assert view["LLM_API_KEY"]["value"] is None          # never leaked
    assert view["LLM_API_KEY"]["has_value"] is True
    assert view["LLM_API_KEY"]["source"] == "db"
    # non-secret values are returned plainly
    assert view["LLM_MODEL"]["value"] == "m"
    assert "has_value" not in view["LLM_MODEL"]
