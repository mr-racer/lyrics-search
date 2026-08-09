"""Retrieval over the raw facts of ONE subject.

What changed against ``app/services/facts_retrieval.py``, and why:

* **No Qdrant.** The candidate pool is every raw fact of the subject — the song
  and its artist — read straight from SQLite. A vector index earns its keep
  when you search a pool you cannot hold in memory; eighty short texts is not
  that pool. Dropping it also drops a whole class of failure (a stale index, a
  cold index, an index built by a different embedding model).
* **Nothing is filtered before ranking.** No minimum fact length, no cropping
  to N characters. Short stubs cost one slot in a batch encode and the
  cross-encoder is perfectly able to score them low.
* **Selection is by threshold, not by count.** Everything above ``min_prob``
  goes to the model. That is a deliberate choice by the owner of this code: it
  makes the threshold the single thing to tune. The size of what passes is
  logged on every call so the effect is visible rather than assumed.
* **No pseudo-relevance feedback.** It existed to give BM25 something to match
  on when the query was Russian and the facts English. MILCO does that job now.

Storage is injected, not imported: ``FactSource`` is the whole contract, and
the app's ``MetadataDB`` satisfies it with a four-line adapter when this moves
into production.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

from lab.agent.retrieval import HybridRetriever
from lab.agent.retrieval.hub import ModelHub
from lab.agent.retrieval.types import Fact

logger = logging.getLogger(__name__)


class FactSource(Protocol):
    """Where raw facts come from. One method, so anything can supply them."""

    def facts_for(self, kind: str, slug: str) -> list[Fact]:
        ...


class SqliteFactSource:
    """Facts out of the app's ``cache/metadata.db``.

    Reads ``song_facts.song_slug`` / ``artist_facts.artist_slug`` — the two
    tables the indexing pipeline fills from songfacts.com and Genius. The rows
    are shared between accounts and separated by ``fact_visibility``; that
    table is not consulted here because the subject was resolved out of THIS
    library in the first place, so its own facts are by definition visible.
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)

    def _connect(self):
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def facts_for(self, kind: str, slug: str) -> list[Fact]:
        if kind not in ("song", "artist") or not slug:
            return []
        table = "song_facts" if kind == "song" else "artist_facts"
        column = "song_slug" if kind == "song" else "artist_slug"
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT id, fact, source, category FROM {table} "
                    f"WHERE {column} = ?", (slug,)).fetchall()
        except Exception:
            logger.warning("[facts] read failed for %s/%s", kind, slug,
                           exc_info=True)
            return []
        return [
            Fact(row_id=r["id"], kind=kind, slug=slug,
                 text=(r["fact"] or "").strip(),
                 source=r["source"] or "", category=r["category"] or "")
            for r in rows if (r["fact"] or "").strip()
        ]


class FactsRetriever:
    """Ranks a subject's own facts against a question."""

    def __init__(self, source: FactSource, *, hub: Optional[ModelHub] = None,
                 config=None):
        from lab.agent.config import AgentConfig

        self.source = source
        self.cfg = config or (hub.cfg if hub is not None else AgentConfig())
        self.hub = hub or ModelHub(self.cfg)

    def pool(self, *, song_slug: Optional[str] = None,
             artist_slug: Optional[str] = None) -> list[Fact]:
        """Every fact of the subject, deduplicated on text.

        Duplicates are common — the same story reaches songfacts.com and a
        Genius description in near-identical words — and a duplicate costs a
        slot in the answer's context for nothing.
        """
        facts: list[Fact] = []
        if song_slug:
            facts += self.source.facts_for("song", song_slug)
        if artist_slug:
            facts += self.source.facts_for("artist", artist_slug)

        seen: set[str] = set()
        out: list[Fact] = []
        for f in facts:
            key = " ".join(f.text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
        return out

    def retrieve(self, query: str, *, song_slug: Optional[str] = None,
                 artist_slug: Optional[str] = None,
                 min_prob: Optional[float] = None) -> list[Fact]:
        """Facts that plausibly answer ``query``, best first.

        Never raises: an unavailable model degrades the ranking, never the
        call. Returns ``[]`` only when the subject genuinely has no facts.
        """
        facts = self.pool(song_slug=song_slug, artist_slug=artist_slug)
        if not facts or not (query or "").strip():
            return facts

        threshold = self.cfg.ce_threshold if min_prob is None else min_prob
        retriever = HybridRetriever([f.text for f in facts], hub=self.hub,
                                    config=self.cfg)
        ranked = retriever.search(query, min_prob=threshold)

        out: list[Fact] = []
        for r in ranked:
            fact = facts[r.index]
            fact.ce_prob = r.ce_prob
            out.append(fact)

        chars = sum(len(f.text) for f in out)
        if any(f.ce_prob is None for f in out):
            # Say so plainly. "N above p>=0.20" when no cross-encoder ran is a
            # lie that reads as a working filter, and the symptom downstream is
            # an oversized prompt full of loosely related facts.
            logger.warning(
                "[facts] %s/%s: %d facts, NO cross-encoder — threshold not "
                "applied, all %d go to the prompt (%d chars). Check hub.status().",
                song_slug or "-", artist_slug or "-", len(facts), len(out), chars)
        else:
            logger.info(
                "[facts] %s/%s: %d facts -> %d above p>=%.2f (%d chars into the prompt)",
                song_slug or "-", artist_slug or "-", len(facts), len(out),
                threshold, chars)
        return out
