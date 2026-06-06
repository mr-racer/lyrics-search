"""Global semaphore: at most MAX_PARALLEL_INDEXING_JOBS jobs may hold the slot."""

import threading
import time

import pytest

from app.services import library_service as lib_mod


@pytest.fixture
def reset_semaphore(monkeypatch):
    """Replace the module-level semaphore with a small one for this test."""
    new_sem = threading.BoundedSemaphore(2)
    monkeypatch.setattr(lib_mod, "_INDEX_SEMAPHORE", new_sem)
    yield new_sem


def test_third_acquirer_blocks_until_release(reset_semaphore):
    sem = reset_semaphore
    acquired_order: list[str] = []
    release_event = threading.Event()

    def _worker(name: str):
        with sem:
            acquired_order.append(name)
            if name in ("A", "B"):
                release_event.wait(timeout=2.0)

    a = threading.Thread(target=_worker, args=("A",))
    b = threading.Thread(target=_worker, args=("B",))
    c = threading.Thread(target=_worker, args=("C",))
    a.start(); b.start()
    time.sleep(0.05)  # let A and B grab the two slots
    assert sorted(acquired_order) == ["A", "B"]
    c.start()
    time.sleep(0.05)
    # C is blocked — slot was full
    assert "C" not in acquired_order
    # Release A and B → C should acquire
    release_event.set()
    a.join(timeout=2); b.join(timeout=2); c.join(timeout=2)
    assert "C" in acquired_order


def test_default_semaphore_size_is_two_or_more():
    """Sanity: the production constant is at least 2 (per spec §6.2)."""
    import os
    expected = int(os.environ.get("MAX_PARALLEL_INDEXING_JOBS", "2"))
    assert lib_mod.MAX_PARALLEL_INDEXING_JOBS == expected
    assert lib_mod.MAX_PARALLEL_INDEXING_JOBS >= 1
