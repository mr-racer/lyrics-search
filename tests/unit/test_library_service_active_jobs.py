"""Per-account indexing queue: two accounts ok in parallel, same account → 409."""

import threading
from unittest.mock import MagicMock

import pytest

from app.services.library_service import LibraryService


@pytest.fixture
def svc():
    return LibraryService(search_service=MagicMock(), db_client=MagicMock())


def test_first_start_succeeds(svc):
    assert svc.try_start_job(account_id="acct-A", job_id="j1") is True
    assert svc.is_account_indexing("acct-A") is True


def test_same_account_second_start_fails(svc):
    assert svc.try_start_job(account_id="acct-A", job_id="j1") is True
    assert svc.try_start_job(account_id="acct-A", job_id="j2") is False


def test_two_different_accounts_both_start(svc):
    assert svc.try_start_job(account_id="acct-A", job_id="j1") is True
    assert svc.try_start_job(account_id="acct-B", job_id="j2") is True
    assert svc.is_account_indexing("acct-A") is True
    assert svc.is_account_indexing("acct-B") is True


def test_finish_releases_slot(svc):
    svc.try_start_job(account_id="acct-A", job_id="j1")
    svc.finish_job(account_id="acct-A")
    assert svc.is_account_indexing("acct-A") is False
    assert svc.try_start_job(account_id="acct-A", job_id="j2") is True


def test_finish_is_idempotent(svc):
    svc.finish_job(account_id="never-started")  # no error
    svc.try_start_job(account_id="acct-A", job_id="j1")
    svc.finish_job(account_id="acct-A")
    svc.finish_job(account_id="acct-A")  # second release is a no-op


def test_get_account_job_id_returns_current(svc):
    svc.try_start_job(account_id="acct-A", job_id="abc123")
    assert svc.get_account_job_id("acct-A") == "abc123"
    assert svc.get_account_job_id("acct-B") is None


def test_concurrent_starts_for_same_account_only_one_wins():
    """50 threads race to start under the same account_id. Exactly one wins."""
    svc = LibraryService(search_service=MagicMock(), db_client=MagicMock())
    wins: list[bool] = []
    lock = threading.Lock()

    def _worker(i: int):
        ok = svc.try_start_job(account_id="acct-A", job_id=f"j{i}")
        with lock:
            wins.append(ok)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(wins) == 1, f"expected exactly one winner, got {sum(wins)}"
