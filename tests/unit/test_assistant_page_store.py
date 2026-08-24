"""Unit tests for the assistant's cross-turn caches (assistant/page_store.py).

Two layers with deliberately different rules, and the tests exist to pin the
difference: pages are public content shared by the whole instance, turn contexts
are private and bound to one account.
"""

import pytest

from app.services.assistant.contracts import Chunk, Page
from app.services.assistant.page_store import ContextStore, PageStore

pytestmark = pytest.mark.unit


class _Clock:
    """A hand-cranked monotonic clock, so TTL tests never sleep."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _page(url, title="T"):
    return Page(url=url, title=title, markdown="body", source="web")


def _chunk(cid, text="passage"):
    return Chunk(id=cid, path=["T"], body=text, url="https://e.org/a", title="T")


# ── page layer ───────────────────────────────────────────────────────────────


def test_page_round_trips_on_the_canonical_key():
    """Two spellings of one article are one entry — that is the whole point."""
    clock = _Clock()
    store = PageStore(ttl=60.0, max_pages=10, clock=clock)
    store.put(_page("https://EN.wikipedia.org/wiki/Kanye_West?utm_source=x"))

    got = store.get("https://en.wikipedia.org/wiki/Kanye_West")
    assert got is not None
    assert got.title == "T"


def test_page_expires_after_the_ttl():
    clock = _Clock()
    store = PageStore(ttl=60.0, max_pages=10, clock=clock)
    store.put(_page("https://e.org/a"))

    clock.advance(59.0)
    assert store.get("https://e.org/a") is not None
    clock.advance(2.0)
    assert store.get("https://e.org/a") is None


def test_failed_pages_are_not_cached():
    """A 403 is a property of the moment, not of the URL. Caching it would
    poison the next turn for a full minute over one bad fetch."""
    store = PageStore(ttl=60.0, max_pages=10, clock=_Clock())
    bad = Page(url="https://e.org/a", title="", markdown="", source="web",
               error="403")
    store.put(bad)
    assert store.get("https://e.org/a") is None


def test_lru_evicts_the_least_recently_used():
    clock = _Clock()
    store = PageStore(ttl=60.0, max_pages=2, clock=clock)
    store.put(_page("https://e.org/1"))
    store.put(_page("https://e.org/2"))
    store.get("https://e.org/1")          # 1 is now the fresher of the two
    store.put(_page("https://e.org/3"))

    assert store.get("https://e.org/2") is None
    assert store.get("https://e.org/1") is not None
    assert store.get("https://e.org/3") is not None


def test_page_layer_is_shared_across_accounts():
    """No account key anywhere: the article is the same article for everyone."""
    store = PageStore(ttl=60.0, max_pages=10, clock=_Clock())
    store.put(_page("https://e.org/a"))
    assert store.get("https://e.org/a") is not None


# ── turn contexts ────────────────────────────────────────────────────────────


def test_context_round_trips_for_its_owner():
    store = ContextStore(ttl=60.0, clock=_Clock())
    cid = store.save(user_id=7, chunks=[_chunk(0)], used_queries=["q"],
                     evidence=[], subject=None)

    ctx = store.load(cid, user_id=7)
    assert ctx is not None
    assert [c.body for c in ctx.chunks] == ["passage"]
    assert ctx.used_queries == ["q"]


def test_context_is_invisible_to_another_account():
    """The private thing is not the page — it is that THIS person searched for
    it. A context handed to the wrong account must read as nonexistent."""
    store = ContextStore(ttl=60.0, clock=_Clock())
    cid = store.save(user_id=7, chunks=[_chunk(0)], used_queries=[],
                     evidence=[], subject=None)

    assert store.load(cid, user_id=8) is None
    assert store.load(cid, user_id=7) is not None    # and the owner still has it


def test_context_expires_after_the_ttl():
    clock = _Clock()
    store = ContextStore(ttl=60.0, clock=clock)
    cid = store.save(user_id=7, chunks=[_chunk(0)], used_queries=[],
                     evidence=[], subject=None)

    clock.advance(61.0)
    assert store.load(cid, user_id=7) is None


def test_release_drops_it_immediately():
    store = ContextStore(ttl=60.0, clock=_Clock())
    cid = store.save(user_id=7, chunks=[], used_queries=[], evidence=[],
                     subject=None)

    assert store.release(cid, user_id=7) is True
    assert store.load(cid, user_id=7) is None


def test_release_by_a_stranger_does_nothing():
    store = ContextStore(ttl=60.0, clock=_Clock())
    cid = store.save(user_id=7, chunks=[], used_queries=[], evidence=[],
                     subject=None)

    assert store.release(cid, user_id=8) is False
    assert store.load(cid, user_id=7) is not None


def test_unknown_context_id_is_not_an_error():
    """An expired tab must degrade into a slow turn, never into an error frame."""
    store = ContextStore(ttl=60.0, clock=_Clock())
    assert store.load("nope", user_id=7) is None
    assert store.release("nope", user_id=7) is False
