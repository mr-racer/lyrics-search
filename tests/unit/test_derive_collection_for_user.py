"""Unit tests for derive_collection_for_user — single source of truth for
mapping a JWT-identified user to their Qdrant collection name."""

import pytest
from types import SimpleNamespace

from app.api.helpers import derive_collection_for_user


class TestDeriveCollectionForUser:
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
