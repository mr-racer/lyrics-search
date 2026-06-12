"""MetadataDB must survive concurrent access from FastAPI's request threadpool.

Regression for the server-mode auth loop: every request runs get_current_user
→ get_user_by_id in a threadpool worker. With a single shared sqlite3
connection the concurrent statement use raced inside CPython's sqlite3 module
and produced sqlite3.InterfaceError ("bad parameter or other API misuse") →
500, or silently wrong/empty rows → "unknown user id" → 401 → the frontend
logged the user out in a loop.
"""
import threading

import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def reset_db(tmp_path, monkeypatch):
    import app.resources.metadata_db as mod
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "threads_test.db")
    MetadataDB._instance = None
    MetadataDB.close()
    MetadataDB.init()
    yield
    MetadataDB.close()


def test_concurrent_user_lookups_do_not_corrupt(tmp_path):
    MetadataDB.create_user(
        user_id="u-threads", email="threads@example.com",
        password_hash="x", role="owner", created_at=1.0,
    )

    n_threads = 8
    iterations = 400
    errors: list[BaseException] = []
    barrier = threading.Barrier(n_threads)

    def hammer(i: int):
        try:
            barrier.wait()
            for k in range(iterations):
                row = MetadataDB.get_user_by_id("u-threads")
                assert row is not None, "existing user resolved to None"
                assert row["id"] == "u-threads"
                row2 = MetadataDB.get_user_by_email("threads@example.com")
                assert row2 is not None and row2["id"] == "u-threads"
                # Mix in a write the way real traffic does (login timestamps).
                if k % 50 == i % 50:
                    MetadataDB.update_last_login("u-threads", float(k))
        except BaseException as e:  # noqa: BLE001 — collect everything for the report
            errors.append(e)

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, f"{len(errors)} thread(s) failed; first: {errors[0]!r}"
