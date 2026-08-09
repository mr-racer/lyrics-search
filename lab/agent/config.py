"""Every knob in one place.

The lab runs on a laptop and talks to a GPU box over the network: SearXNG and
the LLM live on the server, the models and the library dump live locally. So
nothing here is read from the environment at import time — you build an
:class:`AgentConfig` in the notebook and hand it to the assistant.

The cross-encoder threshold is split three ways — documents, chunks, facts —
because they are three different questions asked of three different corpora.
See the fields for what each one is defending against.
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
    # Three thresholds, because they are three different decisions over three
    # different score distributions — one number cannot serve all of them.
    #
    # Documents: "is this page worth downloading?", judged on a title and a
    # two-line snippet. Cheap to be wrong in either direction — a bad page
    # costs one fetch, a missed page costs the answer — so the bar is low.
    ce_threshold_docs: float = 0.3
    # Chunks: "does this passage go in front of the model?". Expensive to be
    # wrong: a loose passage takes a slot in a small context window and pulls
    # the answer towards whatever it is about. The bar is high.
    ce_threshold_chunks: float = 0.75
    # Facts: the library's own material about the subject. Left where it was
    # while the threshold is being calibrated by hand — the corpus is tens of
    # short texts rather than page passages, and its scores do not sit on the
    # same scale as either of the above.
    ce_threshold_facts: float = 0.2
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

    # ── search ────────────────────────────────────────────────────────────
    searx_pool: int = 20            # results pulled before the CE pass
    # The SearXNG UI defaults to `default_lang: en` from its settings, so an
    # API call sending "all" is not the search you ran by hand in the browser —
    # it also pulls de/fr/es mirrors of the same article. English matches both
    # the UI and this pipeline's own "search in English" rule.
    searx_language: str = "en"
    # An explicit whitelist, because the server default is SearXNG's stock set
    # of ~70 engines in the general category and it shows: one run of
    # "Kanye West G.O.A.T. and Yeezus era hit songs" came back with Indonesian
    # journal PDFs, MusicBrainz release UUIDs and the front page of a Czech
    # web portal. Pinning the engines removes that whole class of result
    # regardless of which engine misbehaved, and it does it per call, without
    # touching what the SearXNG UI does for a human.
    #
    # reddit is in on purpose — threads there are often the only place a
    # niche soundtrack is discussed. An `engines=` parameter activates an
    # engine even when settings.yml has it `disabled: true` (same mechanism as
    # a `!re` bang), so no server change is needed to try it.
    #
    # startpage is deliberately absent: it answers a CAPTCHA under any burst,
    # and settings.yml suspends a CAPTCHA'd engine for 3600 seconds — one
    # tripped run costs it for the rest of the hour.
    #
    # Set to None to fall back to the server's default set.
    searx_engines: Optional[str] = "google,duckduckgo,brave,bing,reddit"
    # Seconds to leave between SearXNG calls. Not politeness — self-defence.
    # An iteration fires up to eight searches, each fanning out to every engine
    # in the whitelist, so a burst is ~50 outbound requests from one IP in a
    # couple of seconds. DuckDuckGo and Brave rate-limit that immediately;
    # SearXNG reads the timeout as a failure and suspends the engine for up to
    # `max_ban_time_on_fail` (120s), and the rest of the run gets served by
    # whatever fringe engines are left. That is the mechanism behind a Kanye
    # query coming back with Indonesian journal PDFs.
    searx_min_interval: float = 1.5
    # At most this many results from one host. A real answer is spread across
    # hosts; a broken scraper returns one site's navigation menu, ten links
    # deep. Observed as a Polish TV guide's channel list interleaved
    # one-for-one with genuine music results.
    max_results_per_host: int = 3
    # ...and when one host is this much of the whole result set, it is not a
    # popular source, it is an engine dumping a page it landed on by mistake.
    # Dropped entirely, with the engine named in the log. Host-pinned queries
    # (site:music.apple.com, engines=wikipedia) are exempt — they are supposed
    # to come back from one host.
    host_takeover_share: float = 0.4
    host_takeover_min: int = 4
    # Host-pinned sources run on the FIRST query only. A rephrasing rarely
    # surfaces a different Apple playlist or a different Fandom wiki, and each
    # extra call is another turn of the burst that costs the good engines.
    structured_first_query_only: bool = True
    # Strip the References / External links tail off MediaWiki pages before
    # anything is chunked or embedded.
    strip_appendix: bool = True
    # Wikis to read through api.php FIRST. These answer a Cloudflare challenge
    # to every HTTP fetcher, so scraping simply does not work on them. Every
    # other MediaWiki host is scraped first and only falls back to the API —
    # on Wikipedia the ordinary path is both faster and cleaner.
    mediawiki_api_first: tuple = ("fandom.com", "wikia.org", "wiki.gg")

    # ── budgets ───────────────────────────────────────────────────────────
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
    # Below this many library tracks, the run reads the artist's Wikipedia
    # discography as a rescue. "Хиты Канье после 2020" has no page of its own —
    # nobody writes a listicle per artist per era — but the singles table
    # exists, and the era filter can do the narrowing itself.
    discography_min_tracks: int = 10
    # How many title spellings to try before giving up. Wikipedia has no single
    # convention: Kanye's songs are under "singles discography" (the plain
    # "discography" title is a disambiguation stub), JAY-Z's under "albums
    # discography", Sade's under the plain one. Each try costs one search and
    # one fetch, and stops at the first page that actually has rows.
    discography_max_queries: int = 3
    fuzzy_title_threshold: float = 0.75
    # Final pass: the model sees where each confirmed track was found (page,
    # section, row) and returns the ids that actually answer the request.
    # Nothing else in the pipeline can remove a track once it matched the
    # library, and matching only proves the library HAS it — not that the page
    # was offering it as an answer.
    llm_triage: bool = True
    # Below this there is nothing to triage; the whole list is the answer.
    triage_min_candidates: int = 8

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
