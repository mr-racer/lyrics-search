"""Yandex account linking: capturing + returning the display login.

Covers two layers:
1. MetadataDB.save_yandex_account/get_yandex_account round-trip a
   ``yandex_login`` column (separate from the encrypted token blob — it's not
   a secret, just a display name).
2. GET /api/v1/import/yandex (link-status) surfaces it to the frontend.

Relies on the autouse ``clean_metadata_db`` fixture (tests/integration/conftest)
for an isolated, thread-safe tmp DB.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.api.main import create_app
from app.domain.models import User
from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _seed_user():
    # yandex_accounts.account_id is a FOREIGN KEY into users(id)
    # (PRAGMA foreign_keys=ON), so the referenced user must exist first.
    MetadataDB.create_user(
        user_id="acct1", email="acct1@x.y", password_hash="h",
        role="member", created_at=1700000000.0,
    )


@pytest.mark.integration
def test_yandex_account_stores_login():
    MetadataDB.save_yandex_account(
        account_id="acct1", enc_token="enc", yandex_uid="42",
        expires_at=None, yandex_login="ivan-petrov",
    )
    row = MetadataDB.get_yandex_account("acct1")
    assert row["yandex_login"] == "ivan-petrov"


@pytest.mark.integration
def test_yandex_account_login_defaults_to_none():
    MetadataDB.save_yandex_account(
        account_id="acct1", enc_token="enc", yandex_uid="42", expires_at=None,
    )
    row = MetadataDB.get_yandex_account("acct1")
    assert row["yandex_login"] is None


@pytest.mark.integration
def test_link_status_route_returns_login():
    MetadataDB.set_instance_config(mode="server", created_at=1700000000.0)
    MetadataDB.save_yandex_account(
        account_id="acct1", enc_token="enc", yandex_uid="42",
        expires_at=1234.0, yandex_login="ivan-petrov",
    )
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: User(
        id="acct1", email="acct1@x.y", role="member", created_at=0.0,
    )
    with TestClient(app) as c:
        resp = c.get("/api/v1/import/yandex")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["linked"] is True
    assert body["yandex_uid"] == "42"
    assert body["yandex_login"] == "ivan-petrov"


@pytest.mark.integration
def test_link_status_route_when_unlinked():
    MetadataDB.set_instance_config(mode="server", created_at=1700000000.0)
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: User(
        id="acct1", email="acct1@x.y", role="member", created_at=0.0,
    )
    with TestClient(app) as c:
        resp = c.get("/api/v1/import/yandex")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["linked"] is False
    assert body["yandex_login"] is None
