"""SearchService._resolve_model_name precedence.

ADAPTED order (vs the A-independent plan): collection_settings wins over the
user setting, because collection_settings records the model the collection was
ACTUALLY indexed with (the Qdrant vectors are named after it). A user's
text_model_name defaults to a value that may not match an existing collection's
vectors, so trusting it over the indexed model would break search. Order:

    explicit text_model arg  →  collection_settings  →  user.text_model_name  →  default
"""

from unittest.mock import MagicMock

import pytest

from app.resources.metadata_db import MetadataDB
from app.services.search_service import SearchService


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    import app.resources.metadata_db as mod
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "rm.db")
    MetadataDB._instance = None
    MetadataDB.init()
    yield
    MetadataDB.close()


@pytest.fixture
def svc():
    return SearchService(lyrics_db=MagicMock())


def _seed_user(user_id: str, model: str) -> None:
    MetadataDB.create_user(
        user_id=user_id, email=f"{user_id}@x.y", password_hash="h",
        role="member", created_at=1700000000.0,
    )
    MetadataDB.update_user_settings(user_id, text_model_name=model)


def test_collection_setting_wins_over_user_default(svc):
    """A user whose text_model_name is the column default must NOT override a
    collection that records a different indexed model — searching with the
    user's default would query vectors that don't exist in that collection."""
    MetadataDB.create_user(
        user_id="acct-1", email="a@x.y", password_hash="h",
        role="member", created_at=1700000000.0,
    )  # text_model_name takes the column default
    MetadataDB.set_collection_text_model("col-X", "collection-indexed-model")
    assert svc._resolve_model_name("acct-1", "col-X") == "collection-indexed-model"


def test_resolves_from_user_when_no_collection_setting(svc):
    _seed_user("acct-1", "Qwen/Qwen3-Embedding-0.6B")
    assert svc._resolve_model_name("acct-1", "no-such-col") == "Qwen/Qwen3-Embedding-0.6B"


def test_resolves_from_collection_when_user_unset(svc):
    MetadataDB.set_collection_text_model("col-X", "collection-pinned-model")
    assert svc._resolve_model_name("no-such-user", "col-X") == "collection-pinned-model"


def test_falls_back_to_default_when_nothing_set(svc):
    assert svc._resolve_model_name("no-such-user", "no-such-col") == SearchService.DEFAULT_TEXT_MODEL


def test_none_account_id_uses_collection_then_default(svc):
    MetadataDB.set_collection_text_model("col-X", "collection-pinned-model")
    assert svc._resolve_model_name(None, "col-X") == "collection-pinned-model"
    assert svc._resolve_model_name(None, "no-such-col") == SearchService.DEFAULT_TEXT_MODEL
