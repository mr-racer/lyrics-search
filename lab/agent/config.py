"""Every knob in one place.

The lab runs on a laptop and talks to a GPU box over the network: SearXNG and
the LLM live on the server, the models and the library dump live locally. So
nothing here is read from the environment at import time — you build an
:class:`AgentConfig` in the notebook and hand it to the assistant.

Thresholds start where the notebook left them. ``ce_threshold`` is 0.2 for
every cross-encoder call in the pipeline; it is deliberately ONE number so it
can be tuned as one number before anyone starts splitting it per stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

ClarifyPolicy = Literal["auto", "ask", "wiki"]


@dataclass
class AgentConfig:
    # ── where things are ──────────────────────────────────────────────────
    searxng_url: str = "http://localhost:8088"
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "openai/gpt-oss-20b"
    llm_api_key: str = "lm-studio"
    # Dump of the prod cache/metadata.db. Empty means "no library": name
    # resolution and playlist assembly are unavailable, the general branch
    # still works off the web.
    db_path: Optional[str] = None
    # None → the collection with the most tracks in that dump.
    collection_name: Optional[str] = None

    # ── models ────────────────────────────────────────────────────────────
    dense_model: str = "Octen/Octen-Embedding-0.6B"
    sparse_model: str = "omai-research/milco-650m"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    device: Optional[str] = None            # None → cuda if available
    # The cross-encoder is the biggest of the three and the easiest to do
    # without on a tight card: False keeps it off the GPU until first use and
    # releases it after each call.
    keep_reranker_resident: bool = True
    rerank_max_len: int = 512
    encode_batch: int = 8
    rerank_batch: int = 16
    # LexEcho source view — matters for proper nouns in non-English text.
    milco_source_view: bool = True

    # ── retrieval ─────────────────────────────────────────────────────────
    ce_threshold: float = 0.2
    # 1.0 puts the order entirely in the cross-encoder's hands; 0.8 keeps a
    # fifth of the first-stage signal, which stops a single confident CE
    # mistake from burying a chunk every other signal liked.
    ce_alpha: float = 0.8
    fusion_weights: dict = field(
        default_factory=lambda: {"dense": 1.0, "milco": 1.0, "bm25": 0.3})
    max_chunks_in_context: int = 10

    # ── chunking ──────────────────────────────────────────────────────────
    chunk_max_chars: int = 1200
    chunk_min_chars: int = 200
    chunk_overlap_blocks: int = 1

    # ── search budgets ────────────────────────────────────────────────────
    searx_pool: int = 20            # results pulled before the CE pass
    max_pages_per_iteration: int = 5
    general_max_iterations: int = 2
    playlist_max_iterations: int = 3
    max_web_searches: int = 8
    fetch_timeout: float = 15.0
    fetch_concurrency: int = 4

    # ── playlist assembly ─────────────────────────────────────────────────
    default_target_count: int = 15
    # Wikipedia/Apple titles are more often real than a listicle's.
    source_weights: dict = field(
        default_factory=lambda: {"wikipedia": 2.0, "apple": 2.0,
                                 "fandom": 2.0, "web": 1.0})
    # Below this share of the target, the constraint comes out of the query
    # text and becomes a post-hoc filter instead.
    min_yield_ratio: float = 0.6
    max_relaxations: int = 2
    fuzzy_title_threshold: float = 0.75

    # ── behaviour ─────────────────────────────────────────────────────────
    # What to do with an abbreviation the model expanded on its own. "auto"
    # accepts it (the notebook default), "ask" goes through on_clarify, "wiki"
    # skips the model's guess and asks Wikipedia straight away.
    clarify_policy: ClarifyPolicy = "auto"
    lang: str = "ru"
    llm_timeout: float = 180.0
    llm_max_tokens: int = 2000
    llm_temperature: float = 0.2

    def __post_init__(self) -> None:
        self.searxng_url = (self.searxng_url or "").rstrip("/")
        self.llm_base_url = (self.llm_base_url or "").rstrip("/")
