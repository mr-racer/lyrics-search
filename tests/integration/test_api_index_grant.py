"""Folder indexing is a per-account grant, not an instance-wide switch.

Regression cover for a real hole: with ``MEMBER_INDEX_ROOT`` set, ANY member
could POST /library/index for that root and clone the whole by-reference
library into their own collection, and the path itself was published through
the unauthenticated GET /config.

The gate deliberately runs before the service-availability check, so these
tests need no Qdrant — a refusal must hold even while the stack is degraded.
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.api.main import create_app
from app.resources.metadata_db import MetadataDB
from app.services.auth_service import AuthService

JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "grant.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    MetadataDB.set_instance_config(mode="server", created_at=1.0)
    yield
    MetadataDB._reset_for_tests()


def _as(user):
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: user
    return app


def _member(index_root=None, uid="member-1"):
    return SimpleNamespace(
        id=uid, email="m@x", role="member", created_at=0.0,
        last_login_at=None, premium=False, index_root=index_root,
    )


def _owner(uid="owner-1"):
    return SimpleNamespace(
        id=uid, email="o@x", role="owner", created_at=0.0,
        last_login_at=None, premium=False, index_root=None,
    )


def _index(client, path):
    return client.post("/api/v1/library/index", json={"folder_path": str(path)})


class TestIndexGate:
    def test_member_without_a_grant_is_refused(self, tmp_path):
        with TestClient(_as(_member())) as c:
            assert _index(c, tmp_path).status_code == 403

    def test_the_refusal_names_no_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMBER_INDEX_ROOT", "/music")
        with TestClient(_as(_member())) as c:
            body = _index(c, tmp_path).json()["detail"]
        assert "/music" not in body

    def test_member_with_a_grant_passes_the_gate(self, tmp_path):
        root = tmp_path / "music"
        root.mkdir()
        with TestClient(_as(_member(index_root=str(root)))) as c:
            assert _index(c, root).status_code != 403

    def test_member_with_a_grant_may_index_beneath_it(self, tmp_path):
        album = tmp_path / "music" / "Album"
        album.mkdir(parents=True)
        with TestClient(_as(_member(index_root=str(tmp_path / "music")))) as c:
            assert _index(c, album).status_code != 403

    def test_member_with_a_grant_cannot_index_outside_it(self, tmp_path):
        (tmp_path / "music").mkdir()
        other = tmp_path / "elsewhere"
        other.mkdir()
        with TestClient(_as(_member(index_root=str(tmp_path / "music")))) as c:
            assert _index(c, other).status_code == 403

    def test_owner_needs_no_grant(self, tmp_path):
        with TestClient(_as(_owner())) as c:
            assert _index(c, tmp_path).status_code != 403

    def test_sharing_mode_still_lets_any_account_index(self, tmp_path):
        conn = MetadataDB.get()
        conn.execute("DELETE FROM instance_config")
        conn.commit()
        MetadataDB.set_instance_config(mode="sharing", created_at=1.0)
        with TestClient(_as(_member())) as c:
            assert _index(c, tmp_path).status_code != 403


class TestConfigDoesNotLeak:
    def test_config_never_publishes_the_index_root(self, monkeypatch):
        monkeypatch.setenv("MEMBER_INDEX_ROOT", "/music")
        with TestClient(create_app()) as c:
            resp = c.get("/api/v1/instance/config")
        assert resp.status_code == 200
        assert "/music" not in resp.text


class TestMeCarriesTheGrant:
    def _bootstrap(self, c):
        AuthService(jwt_secret=JWT_SECRET).create_owner(
            email="owner@example.com", password="ownerpass12345",
        )
        token = c.post(
            "/api/v1/auth/login",
            json={"email": "owner@example.com", "password": "ownerpass12345"},
        ).json()["token"]
        return token

    def test_me_reports_no_grant_by_default(self):
        with TestClient(create_app()) as c:
            token = self._bootstrap(c)
            me = c.get("/api/v1/auth/me",
                       headers={"Authorization": f"Bearer {token}"}).json()
        assert me["index_root"] is None

    def test_me_reports_the_callers_own_grant(self):
        with TestClient(create_app()) as c:
            token = self._bootstrap(c)
            uid = MetadataDB.get_user_by_email("owner@example.com")["id"]
            MetadataDB.set_index_root(uid, "/music")
            me = c.get("/api/v1/auth/me",
                       headers={"Authorization": f"Bearer {token}"}).json()
        assert me["index_root"] == "/music"


class TestAdminGrantEndpoint:
    def _seed_member(self, uid="member-1"):
        MetadataDB.create_user(
            user_id=uid, email=f"{uid}@x", password_hash="h",
            role="member", created_at=0.0,
        )
        return uid

    def _patch(self, client, uid, root):
        return client.patch(
            f"/api/v1/admin/accounts/{uid}/index-root",
            json={"index_root": root},
        )

    def test_owner_can_grant_folder_indexing(self):
        uid = self._seed_member()
        with TestClient(_as(_owner())) as c:
            assert self._patch(c, uid, "/music").status_code == 200
        assert MetadataDB.get_user_by_id(uid)["index_root"] == "/music"

    def test_owner_can_revoke_folder_indexing(self):
        uid = self._seed_member()
        MetadataDB.set_index_root(uid, "/music")
        with TestClient(_as(_owner())) as c:
            assert self._patch(c, uid, None).status_code == 200
        assert MetadataDB.get_user_by_id(uid)["index_root"] is None

    def test_a_member_cannot_grant_themselves(self):
        uid = self._seed_member()
        with TestClient(_as(_member(uid=uid))) as c:
            assert self._patch(c, uid, "/music").status_code == 403
        assert MetadataDB.get_user_by_id(uid)["index_root"] is None

    def test_a_grant_outside_the_ceiling_is_refused(self, monkeypatch):
        monkeypatch.setenv("MEMBER_INDEX_ROOT", "/music")
        uid = self._seed_member()
        with TestClient(_as(_owner())) as c:
            assert self._patch(c, uid, "/etc").status_code == 400
        assert MetadataDB.get_user_by_id(uid)["index_root"] is None

    def test_granting_a_missing_account_is_a_404(self):
        with TestClient(_as(_owner())) as c:
            assert self._patch(c, "ghost", "/music").status_code == 404
