"""Retrieval over the raw facts of ONE subject.

What differs from ``app/services/facts_retrieval.py``, and why:

* **No Qdrant.** The candidate pool is every raw fact of the subject — the song
  and its artist — read straight from SQLite. A vector index earns its keep when
  you search a pool you cannot hold in memory; eighty short texts is not that
  pool. Dropping it also drops a whole class of failure (a stale index, a cold
  index, an index built by a different embedding model).
* **Nothing is filtered before ranking.** No minimum fact length, no cropping to
  N characters. Short stubs cost one slot in a batch encode and the cross-encoder
  is perfectly able to score them low.
* **Selection is by threshold, not by count.** Everything above ``min_prob`` goes
  to the model. That makes the threshold the single thing to tune, and the size
  of what passes is logged on every call so the effect is visible rather than
  assumed.
* **No pseudo-relevance feedback.** It existed to give BM25 something to match on
  when the query was Russian and the facts English. The learned-sparse leg does
  that job now.

``artist_facts`` and ``song_facts`` are a SHARED pool keyed by slug — the same
real-world artist has the same facts for everyone — and ``fact_visibility`` is
what makes them per-account. The join is applied here even though the subject was
already resolved out of this library: it costs one index lookup, and per-account
isolation is the invariant this codebase treats as load-bearing.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Protocol

from app.services.assistant.config import AgentConfig
from app.services.assistant.contracts import Fact
from app.services.retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class FactSource(Protocol):
    """Where raw facts come from. One method, so anything can supply them."""

    def facts_for(self, kind: str, slug: str) -> list:
        ...


class MetadataFactSource:
    """Facts out of ``cache/metadata.db``, gated on this account's visibility."""

    def __init__(self, collection_name: str, *, use_refined: bool = False,
                 lang: str = "en"):
        self.collection_name = collection_name
        self.use_refined = use_refined
        self.lang = lang

    def facts_for(self, kind: str, slug: str) -> list:
        if kind not in ("song", "artist") or not slug:
            return []
        if self.use_refined:
            refined = self._refined(kind, slug)
            if refined:
                return refined
        return self._raw(kind, slug)

    def _rows(self, sql: str, params: tuple) -> list:
        from app.resources.metadata_db import MetadataDB

        MetadataDB.init()
        return MetadataDB.get().execute(sql, params).fetchall()

    def _raw(self, kind: str, slug: str) -> list:
        table = "song_facts" if kind == "song" else "artist_facts"
        column = "song_slug" if kind == "song" else "artist_slug"
        try:
            rows = self._rows(
                f"SELECT f.id, f.fact, f.source, f.category FROM {table} f "
                f"JOIN fact_visibility v ON v.kind = ? AND v.slug = f.{column} "
                f" AND v.collection_name = ? "
                f"WHERE f.{column} = ?",
                (kind, self.collection_name, slug))
        except Exception:  # noqa: BLE001
            logger.warning("[facts] read failed for %s/%s", kind, slug,
                           exc_info=True)
            return []
        return [
            Fact(row_id=r[0], kind=kind, slug=slug, text=(r[1] or "").strip(),
                 source=r[2] or "", category=r[3] or "")
            for r in rows if (r[1] or "").strip()
        ]

    def _refined(self, kind: str, slug: str) -> list:
        """The AI-rewritten facts, when the config asks for them.

        Off by default: ``CE_THRESHOLD_FACTS`` was measured against raw text, and
        refined facts are shorter and more uniform, so they need their own number.
        Visibility is checked the same way — ``refined_facts`` carries a
        ``collection_name`` column but its own comment marks it as provenance
        only, NOT part of the key, so it cannot be used as a gate.
        """
        try:
            rows = self._rows(
                "SELECT r.refined_json FROM refined_facts r "
                "JOIN fact_visibility v ON v.kind = ? AND v.slug = r.scope_key "
                " AND v.collection_name = ? "
                "WHERE r.scope = ? AND r.scope_key = ? AND r.lang = ?",
                (kind, self.collection_name, kind, slug, self.lang))
        except Exception:  # noqa: BLE001
            logger.warning("[facts] refined read failed for %s/%s", kind, slug,
                           exc_info=True)
            return []
        out: list = []
        for (payload,) in rows:
            try:
                items = json.loads(payload or "[]")
            except (TypeError, ValueError):
                continue
            for item in items if isinstance(items, list) else []:
                text = (item.get("text") if isinstance(item, dict) else item) or ""
                text = str(text).strip()
                if text:
                    out.append(Fact(row_id=len(out), kind=kind, slug=slug,
                                    text=text, source="refined"))
        return out


class FactsRetriever:
    """Ranks a subject's own facts against a question."""

    def __init__(self, source: FactSource, *, hub=None,
                 config: Optional[AgentConfig] = None):
        self.source = source
        self.cfg = config or AgentConfig()
        self.hub = hub

    def pool(self, *, song_slug: Optional[str] = None,
             artist_slug: Optional[str] = None) -> list:
        """Every fact of the subject, deduplicated on text.

        Duplicates are common — the same story reaches songfacts.com and a Genius
        description in near-identical words — and a duplicate costs a slot in the
        answer's context for nothing.
        """
        facts: list = []
        if song_slug:
            facts += self.source.facts_for("song", song_slug)
        if artist_slug:
            facts += self.source.facts_for("artist", artist_slug)

        seen: set = set()
        out: list = []
        for f in facts:
            key = " ".join(f.text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
        return out

    def retrieve(self, query: str, *, song_slug: Optional[str] = None,
                 artist_slug: Optional[str] = None,
                 min_prob: Optional[float] = None) -> list:
        """Facts that plausibly answer ``query``, best first.

        Never raises: an unavailable model degrades the ranking, never the call.
        Returns ``[]`` only when the subject genuinely has no facts.
        """
        facts = self.pool(song_slug=song_slug, artist_slug=artist_slug)
        if not facts or not (query or "").strip():
            return facts

        threshold = self.cfg.ce_threshold_facts if min_prob is None else min_prob
        retriever = HybridRetriever([f.text for f in facts], hub=self.hub)
        ranked = retriever.search(query, min_prob=threshold,
                                  alpha=self.cfg.ce_alpha,
                                  weights=self.cfg.fusion_weights)

        out: list = []
        for r in ranked:
            fact = facts[r.index]
            fact.ce_prob = r.ce_prob
            out.append(fact)

        chars = sum(len(f.text) for f in out)
        if any(f.ce_prob is None for f in out):
            # Say so plainly. "N above p>=0.25" when no cross-encoder ran is a lie
            # that reads as a working filter, and the symptom downstream is an
            # oversized prompt full of loosely related facts.
            logger.warning(
                "[facts] %s/%s: %d facts, NO cross-encoder — threshold not "
                "applied, all %d go to the prompt (%d chars).",
                song_slug or "-", artist_slug or "-", len(facts), len(out), chars)
        else:
            logger.info(
                "[facts] %s/%s: %d facts -> %d above p>=%.2f (%d chars into the prompt)",
                song_slug or "-", artist_slug or "-", len(facts), len(out),
                threshold, chars)
        return out
