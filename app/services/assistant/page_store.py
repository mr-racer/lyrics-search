"""What survives between two turns of the assistant.

Every message builds a fresh :class:`~app.services.assistant.agent.Assistant`,
and that is deliberate — the budgets live on the instance. The cost of it is
that a tapped follow-up re-downloads the pages the previous turn had already
read, and pays the SearXNG pacing again to find them.

Two layers, with deliberately different rules:

**Pages** are keyed by canonical URL and shared by the whole instance. A
Wikipedia article is the same article for every account; it is public content,
not library data, so there is nothing here to isolate. Two listeners asking
about the same artist inside a minute means the second one downloads nothing.

**Turn contexts** are keyed by an opaque id and bound to ONE account. What is
private is not the page but the link between a page and who searched for it, so
this layer is isolated where the page layer is not. A context presented by
another account reads as nonexistent — ignored rather than refused, so a stale
tab degrades into an ordinary slow turn instead of an error frame.

Embeddings are NOT stored. ``HybridRetriever`` is rebuilt from the same chunk
texts on the next turn: encoding a hundred short strings on the resident models
costs seconds, and downloading the pages again costs tens of seconds. Chunks are
held pre-embedding, as :class:`~app.services.assistant.contracts.Chunk` objects.

Single-process by construction — ``Dockerfile`` forbids uvicorn workers so the
models are not loaded onto the card twice — so an in-process store is the whole
mechanism. Writes arrive from ``asyncio.to_thread``, hence the lock; eviction is
lazy, because with a 60-second TTL a sweeper task would cost more than it saves.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.services.assistant.config import (CONTEXT_TTL, PAGE_CACHE_MAX,
                                           PAGE_CACHE_TTL)
from app.services.assistant.contracts import Page, Subject
from app.services.assistant.web_urls import canonical_url

logger = logging.getLogger(__name__)


class PageStore:
    """Downloaded pages, keyed by canonical URL, shared across accounts."""

    def __init__(self, *, ttl: float = PAGE_CACHE_TTL,
                 max_pages: int = PAGE_CACHE_MAX,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl
        self.max_pages = max(1, max_pages)
        self._clock = clock
        self._lock = threading.Lock()
        # Ordered by last use: the front is the eviction candidate.
        self._items: "OrderedDict[str, tuple[float, Page]]" = OrderedDict()

    def get(self, url: str) -> Optional[Page]:
        key = canonical_url(url)
        if not key:
            return None
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            stored_at, page = entry
            if self._clock() - stored_at > self.ttl:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return page

    def put(self, page: Optional[Page]) -> None:
        """Store one page. Failed fetches are dropped on the floor.

        A 403 or a challenge page is a property of the moment, not of the URL:
        caching it would poison every turn for a full minute over one unlucky
        request, which is the opposite of what this store is for.
        """
        if page is None or not getattr(page, "ok", False):
            return
        key = canonical_url(page.url)
        if not key:
            return
        with self._lock:
            self._items[key] = (self._clock(), page)
            self._items.move_to_end(key)
            while len(self._items) > self.max_pages:
                evicted, _ = self._items.popitem(last=False)
                logger.debug("[page_store] evicted %s", evicted)

    def put_many(self, pages) -> None:
        for page in pages or ():
            self.put(page)

    def known(self, urls) -> set:
        """Which of ``urls`` this store can serve right now."""
        return {u for u in (urls or ()) if self.get(u) is not None}

    def size(self) -> int:
        with self._lock:
            return len(self._items)


@dataclass(slots=True)
class TurnContext:
    """One finished turn's material, ready for the next question about it."""

    context_id: str
    user_id: object
    chunks: list = field(default_factory=list)
    used_queries: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    subject: Optional[Subject] = None
    stored_at: float = 0.0


class ContextStore:
    """Turn contexts, bound to the account that produced them."""

    def __init__(self, *, ttl: float = CONTEXT_TTL,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._items: dict = {}

    def save(self, *, user_id, chunks: list, used_queries: list,
             evidence: list, subject: Optional[Subject]) -> str:
        context_id = uuid.uuid4().hex
        with self._lock:
            self._sweep_locked()
            self._items[context_id] = TurnContext(
                context_id=context_id, user_id=user_id,
                chunks=list(chunks or []), used_queries=list(used_queries or []),
                evidence=list(evidence or []), subject=subject,
                stored_at=self._clock())
        logger.info("[context_store] saved %s: %d chunks, %d queries",
                    context_id[:8], len(chunks or []), len(used_queries or []))
        return context_id

    def load(self, context_id: Optional[str], user_id) -> Optional[TurnContext]:
        """The context, or None — expired, unknown and someone else's all alike.

        Deliberately one return value for all three. The caller's job is to
        answer the question either way; distinguishing them would only tempt it
        into telling the user which.
        """
        if not context_id:
            return None
        with self._lock:
            ctx = self._items.get(context_id)
            if ctx is None:
                return None
            if self._clock() - ctx.stored_at > self.ttl:
                del self._items[context_id]
                return None
            if ctx.user_id != user_id:
                logger.warning("[context_store] %s presented by another account",
                               context_id[:8])
                return None
            return ctx

    def release(self, context_id: Optional[str], user_id) -> bool:
        """Drop a context the listener is done with. True when one went away."""
        if not context_id:
            return False
        with self._lock:
            ctx = self._items.get(context_id)
            if ctx is None or ctx.user_id != user_id:
                return False
            del self._items[context_id]
            return True

    def _sweep_locked(self) -> None:
        now = self._clock()
        stale = [k for k, v in self._items.items() if now - v.stored_at > self.ttl]
        for key in stale:
            del self._items[key]

    def size(self) -> int:
        with self._lock:
            return len(self._items)


# The process-wide instances. One of each, for the same reason ModelRegistry is
# a singleton: a second store is a second cache that never hits.
PAGES = PageStore()
CONTEXTS = ContextStore()
