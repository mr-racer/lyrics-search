"""One article, one index, many questions.

The network is slower than generation, so the article is fetched once, chunked
once and indexed once; every later question is a new QUERY against that index
rather than a new round trip. On a GPU the whole selection stage costs about a
second per artist — against 100 seconds when the same code ran on CPU.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.resources.model_registry import ModelRegistry
from app.resources.models import STATS, ModelError
from app.services.assistant.chunking import MarkdownChunker
from app.services.assistant.config import AgentConfig
from app.services.retrieval import HybridRetriever
from app.services.retrieval.hub import DEFAULT_HUB
from app.services.retrieval.hybrid import rrf

logger = logging.getLogger(__name__)

# The chunk gate for the BIO — the same number the assistant uses for prose it
# shows a reader (AgentConfig.ce_threshold_chunks).
CE_CHUNK_GATE = 0.65
CHUNKS_IN_BIO = 10
PER_QUERY_POOL = 8
# A floor for "nothing at all", not a quality bar. Facet sentences are ranked
# only to point at their parent chunk, and the 0.65 gate stays where it belongs.
FACET_SENTENCE_FLOOR = 0.30
TOP_SENTENCES = 6

# The facet's own retrieval: a shallower pool than the bio's, because a facet is
# one fact and the prompt that reads it works better on four focused passages
# than on ten that bury the sentence carrying the answer.
FACET_POOL = 6
FACET_CHUNKS = 4

# Five facets of one life, hardcoded on purpose: the model does not get to
# decide what a biography is made of. It writes the one these queries found.
BIO_QUERIES = [
    "early life, origins and how the artist or band started out",
    "musical style, genre, influences and signature sound",
    "breakthrough, best known albums and commercial success",
    "awards, critical recognition and cultural impact",
    "line-up changes, break-up, hiatus and what the artist is doing now",
]

_SUP = re.compile(r"<sup>\[[^\]]*\]</sup>|\[edit\]|\[\d+\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ\"'*])")


def chunk_page(page, config: Optional[AgentConfig] = None) -> list:
    return MarkdownChunker(config or AgentConfig()).split_page(page)


def build_index(chunks: list) -> Optional[HybridRetriever]:
    if not chunks:
        return None
    return HybridRetriever([c.text for c in chunks], hub=DEFAULT_HUB)


def bio_chunks(retriever, artist: str, *, limit: int = CHUNKS_IN_BIO) -> list:
    """Indices of the chunks the biography is written from.

    Each of the five queries retrieves its own ranked list above the gate, and
    the five lists are fused by RRF over the RANKS — not merged on score.
    Cross-encoder probabilities are not comparable ACROSS queries: a style query
    can top out at 0.91 while an awards query peaks at 0.68, so a score merge
    lets one facet own every slot and the bio comes out one-note. RRF reads
    positions, so each facet gets a fair shot at the top.
    """
    if retriever is None:
        return []
    rankings = {}
    for i, question in enumerate(BIO_QUERIES):
        ranked = retriever.search(f"{artist}: {question}",
                                  min_prob=CE_CHUNK_GATE, limit=PER_QUERY_POOL)
        rankings[f"q{i}"] = [r.index for r in ranked]
    if not any(rankings.values()):
        return []
    fused = rrf(rankings)
    return sorted(fused, key=lambda idx: -fused[idx])[:limit]


def facet_chunks_hybrid(retriever, artist: str, queries: list, *,
                        limit: int = FACET_CHUNKS) -> list:
    """Indices of the chunks a facet is read from. A QUERY, never a new search.

    The facets used to reach for the open web whenever the article did not
    answer them — one search per empty facet, up to four per artist, each one
    fetching two more pages. That is a lot of network to ask a question the
    corpus on disk may already answer under different words, and it is what put
    brave into a rate-limit suspension and duckduckgo behind a CAPTCHA.

    So the corpus is asked again instead, with words the model chose: every
    paraphrase is a full hybrid pass — dense and learned-sparse and BM25, fused
    by RRF and reranked by the cross-encoder — and the paraphrases are fused
    with each other by RRF over the RANKS. Not over the scores: cross-encoder
    probabilities are not comparable ACROSS queries, so a merge on score lets
    the sharpest phrasing own every slot, which is the failure the bio's five
    facets already ran into.
    """
    if retriever is None or not queries:
        return []
    rankings = {}
    for i, question in enumerate(queries):
        if not (question or "").strip():
            continue
        ranked = retriever.search(f"{artist}: {question}",
                                  min_prob=CE_CHUNK_GATE, limit=FACET_POOL)
        rankings[f"q{i}"] = [r.index for r in ranked]
    if not any(rankings.values()):
        return []
    fused = rrf(rankings)
    return sorted(fused, key=lambda idx: -fused[idx])[:limit]


def sentence_index(chunks: list) -> tuple:
    """(texts, parent_chunk_ids) — every sentence, carrying its heading path."""
    texts, parent = [], []
    for ci, chunk in enumerate(chunks):
        head = " › ".join(chunk.path)
        for sentence in _SENT.split(_SUP.sub("", chunk.body)):
            sentence = " ".join(sentence.split())
            if 40 <= len(sentence) <= 400:
                texts.append(f"{head}\n{sentence}" if head else sentence)
                parent.append(ci)
    return texts, parent


def facet_chunks(artist: str, question: str, sents: tuple, *,
                 top: int = TOP_SENTENCES) -> tuple:
    """Rank SENTENCES, read their PARENT chunks. Returns (chunk_indices, best).

    A facet answer is one sentence inside a chunk about something else, and the
    cross-encoder scores a chunk on its dominant topic — at chunk granularity
    M83's "They decided to name their band M83, after the galaxy of that name"
    never surfaced at all.

    The parents are what get read, because a sentence can answer the question
    without naming the artist: Daft Punk's origin sentence opens "Their name was
    taken from…" and scores 0.14 on its own, while three of the top five
    sentences sit in its chunk and carry it into the pack anyway.
    """
    texts, parent = sents
    if not texts:
        return [], None
    try:
        probs = ModelRegistry.ce_probabilities(f"{artist}: {question}", texts)
    except ModelError as e:
        # Nothing rather than everything: a facet is a CLAIM about the artist,
        # and the score is the only thing between a sentence and a sentence
        # presented as fact. An unscored facet is not a weaker facet, it is an
        # unsourced one.
        STATS.degraded("cross_encoder", "bio.facet_sentences")
        logger.warning("[bio.retrieval] no cross-encoder for facet %s (%s) — "
                       "leaving it unanswered: %s", question, artist, e)
        return [], None
    order = sorted(range(len(texts)), key=lambda i: -probs[i])[:top]
    best = probs[order[0]] if order else None
    picked, seen = [], set()
    for i in order:
        if probs[i] < FACET_SENTENCE_FLOOR:
            continue
        if parent[i] not in seen:
            seen.add(parent[i])
            picked.append(parent[i])
    return picked, best


def passages(chunks: list, indices: list) -> str:
    out = []
    for n, idx in enumerate(indices, 1):
        chunk = chunks[idx]
        head = " › ".join(chunk.path) if chunk.path else chunk.title
        out.append(f"[{n}] {head}\n{chunk.body}")
    return "\n\n".join(out)
