"""Unit tests for the Yandex token store + yandex_* MetadataDB CRUD."""

import pytest

from app.resources.metadata_db import MetadataDB
from app.services.yandex import token_store


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Tmp DB + a derivable Fernet key for every test in this module.

    yandex_accounts.account_id is a FOREIGN KEY into users(id) (PRAGMA
    foreign_keys=ON), so we seed the accounts the tests reference.
    """
    db_file = tmp_path / "metadata.db"
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", db_file)
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    # Token encryption key is derived from MUSIX_JWT_SECRET when no explicit
    # MUSIX_YM_TOKEN_KEY is set — exercise the derivation path.
    monkeypatch.delenv("MUSIX_YM_TOKEN_KEY", raising=False)
    monkeypatch.setenv("MUSIX_JWT_SECRET", "x" * 32)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    for uid in ("u1", "u2"):
        MetadataDB.create_user(
            user_id=uid, email=f"{uid}@x.y", password_hash="h",
            role="member", created_at=1700000000.0,
        )
    yield
    MetadataDB._reset_for_tests()


class TestSchema:
    def test_yandex_tables_exist(self):
        conn = MetadataDB.get()
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"yandex_accounts", "yandex_imports"} <= tables


class TestTokenStore:
    def test_save_then_load_roundtrip(self):
        token_store.save_token(
            "u1", access_token="AT", refresh_token="RT",
            expires_at=1234.0, yandex_uid="42",
        )
        blob = token_store.load_token("u1")
        assert blob["access_token"] == "AT"
        assert blob["refresh_token"] == "RT"
        assert blob["yandex_uid"] == "42"
        assert blob["expires_at"] == 1234.0

    def test_encrypted_at_rest(self):
        token_store.save_token("u1", access_token="super-secret-token")
        row = MetadataDB.get_yandex_account("u1")
        # The plaintext token must never appear in the stored blob.
        assert "super-secret-token" not in row["enc_token"]

    def test_load_missing_returns_none(self):
        assert token_store.load_token("u2") is None

    def test_is_linked(self):
        assert token_store.is_linked("u1") is False
        token_store.save_token("u1", access_token="AT")
        assert token_store.is_linked("u1") is True

    def test_delete_unlinks(self):
        token_store.save_token("u1", access_token="AT")
        assert token_store.delete_token("u1") is True
        assert token_store.load_token("u1") is None
        assert token_store.delete_token("u1") is False  # already gone

    def test_relink_overwrites(self):
        token_store.save_token("u1", access_token="OLD")
        token_store.save_token("u1", access_token="NEW")
        assert token_store.load_token("u1")["access_token"] == "NEW"

    def test_key_rotation_makes_token_unreadable(self, monkeypatch):
        token_store.save_token("u1", access_token="AT")
        # Rotate the derived key by changing the secret it derives from.
        monkeypatch.setenv("MUSIX_JWT_SECRET", "y" * 32)
        assert token_store.load_token("u1") is None  # decrypt fails → unlinked

    def test_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("MUSIX_YM_TOKEN_KEY", raising=False)
        monkeypatch.delenv("MUSIX_JWT_SECRET", raising=False)
        with pytest.raises(token_store.TokenStoreError):
            token_store.save_token("u1", access_token="AT")


class TestYandexImportsCrud:
    def test_upsert_and_dedup_set(self):
        MetadataDB.upsert_yandex_import(
            account_id="u1", yandex_track_id="t1", status="downloaded",
        )
        MetadataDB.upsert_yandex_import(
            account_id="u1", yandex_track_id="t2", status="indexed",
        )
        # 'skipped' must NOT count as already-imported (so it can be retried).
        MetadataDB.upsert_yandex_import(
            account_id="u1", yandex_track_id="t3", status="skipped",
            reason="no subscription",
        )
        done = MetadataDB.get_imported_yandex_track_ids("u1")
        assert done == {"t1", "t2"}

    def test_upsert_updates_status_and_preserves_ids(self):
        MetadataDB.upsert_yandex_import(
            account_id="u1", yandex_track_id="t1", status="downloaded",
            upload_id="up1",
        )
        MetadataDB.upsert_yandex_import(
            account_id="u1", yandex_track_id="t1", status="indexed",
            track_id="tr1",
        )
        conn = MetadataDB.get()
        row = conn.execute(
            "SELECT status, upload_id, track_id FROM yandex_imports "
            "WHERE account_id='u1' AND yandex_track_id='t1'",
        ).fetchone()
        assert row[0] == "indexed"
        assert row[1] == "up1"   # preserved via COALESCE
        assert row[2] == "tr1"

    def test_imports_scoped_per_account(self):
        MetadataDB.upsert_yandex_import(
            account_id="u1", yandex_track_id="t1", status="indexed",
        )
        assert MetadataDB.get_imported_yandex_track_ids("u2") == set()
