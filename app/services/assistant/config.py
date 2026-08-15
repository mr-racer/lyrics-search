"""Every knob of the assistant, in one place, at the top of one file.

These are the numbers to turn. They were set against real runs rather than
reasoned out, and the comment on each says what it is defending against — read
it before moving one, because most of them are guarding a specific failure that
is invisible once it comes back.

Endpoints are NOT here. The LLM address, model and key are resolved per request
through ``llm_client`` (instance settings > request body > env) and SearXNG's
address through ``resources.searxng_client``; duplicating them would create a
second, quieter source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
#  THE KNOBS
# ═══════════════════════════════════════════════════════════════════════════

# ── cross-encoder thresholds ────────────────────────────────────────────────
# Three of them, because they are three different decisions over three different
# score distributions, and one number cannot serve all of them.
#
# DOCUMENTS: "is this page worth downloading?", judged on a title and a two-line
# snippet. Cheap to be wrong in either direction — a bad page costs one fetch, a
# missed page costs the answer — so the bar is low.
CE_THRESHOLD_DOCS = 0.35
# CHUNKS: "does this passage go in front of the model?". Expensive to be wrong: a
# loose passage takes a slot in a small context window and pulls the answer
# towards whatever it is about. The bar is high.
CE_THRESHOLD_CHUNKS = 0.65
# FACTS: the library's own material about the subject. Tens of short texts rather
# than page passages, and their scores do not sit on the same scale as either of
# the above.
CE_THRESHOLD_FACTS = 0.25
# 1.0 puts the order entirely in the cross-encoder's hands; 0.8 keeps a fifth of
# the first-stage signal, which stops a single confident CE mistake from burying
# a chunk every other signal liked.
CE_ALPHA = 0.8

# ── near-duplicate passages ─────────────────────────────────────────────────
# The top hits for "who is X" are routinely the same paragraph on five hosts: an
# artist bio gets syndicated, and every copy scores alike because every copy IS
# alike. Without this the pack spends five of its ten slots saying one thing.
DEDUP_CHUNKS = True
# Ranked candidates pulled per context slot. The freed slots are refilled from
# further down, so removing a copy costs no material — it buys a different
# passage. 1 turns the refill off and keeps only the saving.
DEDUP_POOL_FACTOR = 3
# A duplicate needs BOTH signals to agree, and they fail differently: dense reads
# meaning and blurs two paragraphs about one event into near-identity, sparse
# reads (expanded) terms and separates them again. Deliberately conservative — a
# missed duplicate wastes a slot, a false one loses a fact.
DEDUP_THRESHOLDS = {"dense": 0.95, "milco": 0.90}
# When the later copy is this much longer, it takes the earlier one's slot
# instead of being dropped: same content, more of it.
DEDUP_PREFER_LONGER = 1.2

# ── context pack ────────────────────────────────────────────────────────────
MAX_CHUNKS_IN_CONTEXT = 10
FUSION_WEIGHTS = {"dense": 1.0, "milco": 1.0, "bm25": 0.3}

# ── chunking ────────────────────────────────────────────────────────────────
CHUNK_MAX_CHARS = 1200
CHUNK_MIN_CHARS = 200
CHUNK_OVERLAP_BLOCKS = 1

# ── library facts ───────────────────────────────────────────────────────────
# Raw ``song_facts`` / ``artist_facts``, not the AI-rewritten ones. The threshold
# above was measured on raw text; refined facts are shorter and more uniform and
# would need their own number. Flip this and re-measure, not one without the
# other.
FACTS_USE_REFINED = False

# ── search ──────────────────────────────────────────────────────────────────
SEARX_POOL = 20            # results pulled before the CE pass
# SearXNG's own default is `default_lang: en`, so an API call sending "all" is
# not the search you ran by hand in the browser — it also pulls de/fr/es mirrors
# of the same article.
SEARX_LANGUAGE = "en"
# An explicit whitelist, because the server default is SearXNG's stock set of ~70
# engines in the general category and it shows: one run of "Kanye West G.O.A.T.
# and Yeezus era hit songs" came back with Indonesian journal PDFs, MusicBrainz
# release UUIDs and the front page of a Czech web portal.
#
# reddit is deliberately NOT here: SearXNG's reddit engine queries Reddit
# unauthenticated from the server's IP and Reddit blocks datacenter ranges
# outright (searxng#3444). It is reached through `site:reddit.com` instead.
# startpage is absent: it answers a CAPTCHA under any burst, and a CAPTCHA'd
# engine is suspended for 3600 seconds — one tripped run costs it for the hour.
# google is absent for a duller reason: it answers, and it answers with nothing —
# it has been serving SearXNG a 403 since early 2026, reported as an empty result
# set rather than a failure.
#
# None falls back to the server's default set.
SEARX_ENGINES = "duckduckgo,brave,bing"
# Seconds between SearXNG calls. Not politeness — self-defence. See
# ``resources/searxng_client``.
SEARX_MIN_INTERVAL = 1.5
# At most this many results from one host. A real answer is spread across hosts;
# a broken scraper returns one site's navigation menu, ten links deep.
MAX_RESULTS_PER_HOST = 3
# ...and when one host is this much of the whole result set, it is not a popular
# source, it is an engine dumping a page it landed on by mistake.
HOST_TAKEOVER_SHARE = 0.4
HOST_TAKEOVER_MIN = 4
# Host-pinned sources run on the FIRST query only. A rephrasing rarely surfaces a
# different Apple playlist or a different Fandom wiki, and each extra call is
# another turn of the burst that costs the good engines.
STRUCTURED_FIRST_QUERY_ONLY = True

# ── Reddit ──────────────────────────────────────────────────────────────────
# A PARACHUTE on the general branch, not a fourth search tier: it deploys only
# when nothing else cleared the chunk threshold. A listicle or a wiki beats a
# comment thread whenever both exist, and threads win exactly where neither does.
SEARCH_REDDIT = True
REDDIT_MAX_PAGES = 2
# Seconds between requests to reddit.com. This is a GATE, not a wait: if the
# cooldown has not elapsed the read is skipped and the run continues. Reddit
# answers about once a minute from a blocked IP, and making a user wait that long
# on the chance a thread helps is not a trade worth making.
REDDIT_COOLDOWN = 60.0

# ── fetching ────────────────────────────────────────────────────────────────
# Per HTTP attempt. Low on purpose: a host that has not answered in eight seconds
# usually will not, and there are two more fetchers behind this one.
FETCH_TIMEOUT = 8.0
# Ceiling over the whole cascade for one page — three fetchers plus extraction.
# Bounds the damage from a library that ignores its timeout or grows a retry loop.
FETCH_DEADLINE = 25.0
FETCH_CONCURRENCY = 4
# Extra pages a single batch may try after failures, taken from the ranked
# candidates the batch did not reach. One or two hosts refusing every fetcher is
# the normal case, not the exception, and without this a page the cross-encoder
# scored 0.9 is simply lost while a 0.4 one sits unread.
FETCH_REFILL_ATTEMPTS = 4
# Strip the References / External links tail off MediaWiki pages before anything
# is chunked or embedded.
STRIP_APPENDIX = True

# ── budgets ─────────────────────────────────────────────────────────────────
MAX_PAGES_PER_ITERATION = 5
GENERAL_MAX_ITERATIONS = 2
PLAYLIST_MAX_ITERATIONS = 3
MAX_WEB_SEARCHES = 8
# Below this, whatever the retriever returned is not worth calling context. Used
# only for the iterate/stop veto, never to drop material.
WEAK_CONTEXT_PROB = 0.45

# ── playlist assembly ───────────────────────────────────────────────────────
DEFAULT_TARGET_COUNT = 15
# How many tracks a soundtrack request may RETURN. Not a second target: the run
# still stops searching at DEFAULT_TARGET_COUNT, because a soundtrack arrives in
# one piece — one wiki table hands over forty rows in a single parse. This only
# changes the truncation at the end; Non-Stop-Pop FM alone is 42 rows.
WORK_TARGET_COUNT = 45
# Votes. Every source that names a track adds its weight, so a track found by
# Apple and Wikipedia scores 3.5 and outranks one found by Apple alone. Apple
# leads because its lists are the product itself; Wikipedia's tables are
# exhaustive but they are discographies, not selections. Reddit sits at the
# baseline with the listicles and deliberately not below it: a thread is people
# remembering, which is the only source that exists for "the song in the third
# mission".
SOURCE_WEIGHTS = {"apple": 2.0, "wikipedia": 1.5, "fandom": 1.5,
                  "web": 1.0, "reddit": 1.0}
# Below this share of the target, the constraint comes out of the query text and
# becomes a post-hoc filter instead.
MIN_YIELD_RATIO = 0.6
MAX_RELAXATIONS = 2
# Below this many library tracks, the run reads the artist's Wikipedia
# discography as a rescue. "Хиты Канье после 2020" has no page of its own, but
# the singles table exists and the era filter can do the narrowing itself.
DISCOGRAPHY_MIN_TRACKS = 10
# How many title spellings to try before giving up. Wikipedia has no single
# convention: Kanye's songs are under "singles discography", JAY-Z's under
# "albums discography", Sade's under the plain one.
DISCOGRAPHY_MAX_QUERIES = 3
# Final pass: the model sees where each confirmed track was found (page, section,
# row) and returns the ids that actually answer the request. Matching only proves
# the library HAS a track — not that the page was offering it as an answer.
LLM_TRIAGE = True
TRIAGE_MIN_CANDIDATES = 8
# Whether the vote outranks the model's sequencing in the FINAL order. A track
# named by Apple and Wikipedia sitting eleventh behind six single-source album
# cuts is not a flow decision, it is the evidence being ignored. Stable sort:
# equal weights keep the order the model chose.
CURATE_RESPECTS_WEIGHT = True

# ── lyrics search ───────────────────────────────────────────────────────────
# Qdrant candidates pulled before the cross-encoder sees any of them.
LYRICS_POOL = 40
# Tracks whose lyrics go into the answer prompt. Carried over from the previous
# engine, where it was MAX_CTX_HITS.
LYRICS_CTX_HITS = 12
# The reranker reads (question, window) pairs rather than whole lyrics: a 3000-
# character text truncated at 512 tokens loses the second half of the song, and
# the line being asked about is as likely to be there as anywhere. The window
# that scores best is also the line to highlight.
LYRICS_WINDOW_WORDS = 24
LYRICS_WINDOW_STRIDE = 12
# Below this the top window is not evidence that the track is the answer.
LYRICS_MIN_PROB = 0.20

# ── audio search (CLAP) ─────────────────────────────────────────────────────
# Rephrasings per request. Each is the same intent seen from a different acoustic
# angle, and RRF over four ranked lists is what turns "one lucky prompt" into a
# result you can defend.
CLAP_QUERIES = 4
CLAP_LIMIT_PER_QUERY = 10
CLAP_RESULT_COUNT = 15
CLAP_RRF_K = 60

# ── behaviour ───────────────────────────────────────────────────────────────
LLM_TIMEOUT = 180.0
# Output ceiling for every call in the package. Sized for the two that actually
# generate at length, because a cap that truncates does not save time — it costs
# a whole second generation through the repair round:
# * extraction returns one JSON object per track, ~30 tokens each, and the code
#   downstream accepts up to 120 of them;
# * a Russian answer of the allowed 4000 characters is ~1500 tokens on its own,
#   before `used`, `missing` and `follow_ups`.
# A reasoning preamble is spent out of the same budget, so leave headroom.
LLM_MAX_TOKENS = 4000
LLM_TEMPERATURE = 0.2
# What to do with an abbreviation the model expanded on its own. "auto" accepts
# it, "ask" goes through the clarify frame, "wiki" skips the model's guess and
# asks Wikipedia straight away.
CLARIFY_POLICY = "auto"

# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AgentConfig:
    """The knobs above, per run, so a caller can override one without a global.

    Defaults are the module constants — change them THERE. A field here that
    disagrees with its constant is a bug waiting to be discovered by someone
    tuning the wrong number.
    """

    # thresholds
    ce_threshold_docs: float = CE_THRESHOLD_DOCS
    ce_threshold_chunks: float = CE_THRESHOLD_CHUNKS
    ce_threshold_facts: float = CE_THRESHOLD_FACTS
    ce_alpha: float = CE_ALPHA
    fusion_weights: dict = field(default_factory=lambda: dict(FUSION_WEIGHTS))
    max_chunks_in_context: int = MAX_CHUNKS_IN_CONTEXT

    # duplicates
    dedup_chunks: bool = DEDUP_CHUNKS
    dedup_pool_factor: int = DEDUP_POOL_FACTOR
    dedup_thresholds: dict = field(
        default_factory=lambda: dict(DEDUP_THRESHOLDS))
    dedup_prefer_longer: float = DEDUP_PREFER_LONGER

    # chunking
    chunk_max_chars: int = CHUNK_MAX_CHARS
    chunk_min_chars: int = CHUNK_MIN_CHARS
    chunk_overlap_blocks: int = CHUNK_OVERLAP_BLOCKS

    # facts
    facts_use_refined: bool = FACTS_USE_REFINED

    # search
    searx_pool: int = SEARX_POOL
    searx_language: str = SEARX_LANGUAGE
    searx_engines: Optional[str] = SEARX_ENGINES
    searx_min_interval: float = SEARX_MIN_INTERVAL
    max_results_per_host: int = MAX_RESULTS_PER_HOST
    host_takeover_share: float = HOST_TAKEOVER_SHARE
    host_takeover_min: int = HOST_TAKEOVER_MIN
    structured_first_query_only: bool = STRUCTURED_FIRST_QUERY_ONLY
    search_reddit: bool = SEARCH_REDDIT
    reddit_max_pages: int = REDDIT_MAX_PAGES
    reddit_cooldown: float = REDDIT_COOLDOWN

    # fetching
    fetch_timeout: float = FETCH_TIMEOUT
    fetch_deadline: float = FETCH_DEADLINE
    fetch_concurrency: int = FETCH_CONCURRENCY
    fetch_refill_attempts: int = FETCH_REFILL_ATTEMPTS
    strip_appendix: bool = STRIP_APPENDIX

    # budgets
    max_pages_per_iteration: int = MAX_PAGES_PER_ITERATION
    general_max_iterations: int = GENERAL_MAX_ITERATIONS
    playlist_max_iterations: int = PLAYLIST_MAX_ITERATIONS
    max_web_searches: int = MAX_WEB_SEARCHES
    weak_context_prob: float = WEAK_CONTEXT_PROB

    # playlist
    default_target_count: int = DEFAULT_TARGET_COUNT
    work_target_count: int = WORK_TARGET_COUNT
    source_weights: dict = field(default_factory=lambda: dict(SOURCE_WEIGHTS))
    min_yield_ratio: float = MIN_YIELD_RATIO
    max_relaxations: int = MAX_RELAXATIONS
    discography_min_tracks: int = DISCOGRAPHY_MIN_TRACKS
    discography_max_queries: int = DISCOGRAPHY_MAX_QUERIES
    llm_triage: bool = LLM_TRIAGE
    triage_min_candidates: int = TRIAGE_MIN_CANDIDATES
    curate_respects_weight: bool = CURATE_RESPECTS_WEIGHT

    # lyrics
    lyrics_pool: int = LYRICS_POOL
    lyrics_ctx_hits: int = LYRICS_CTX_HITS
    lyrics_window_words: int = LYRICS_WINDOW_WORDS
    lyrics_window_stride: int = LYRICS_WINDOW_STRIDE
    lyrics_min_prob: float = LYRICS_MIN_PROB

    # audio
    clap_queries: int = CLAP_QUERIES
    clap_limit_per_query: int = CLAP_LIMIT_PER_QUERY
    clap_result_count: int = CLAP_RESULT_COUNT
    clap_rrf_k: int = CLAP_RRF_K

    # behaviour
    clarify_policy: str = CLARIFY_POLICY
    lang: str = "en"
    llm_timeout: float = LLM_TIMEOUT
    llm_max_tokens: int = LLM_MAX_TOKENS
    llm_temperature: float = LLM_TEMPERATURE

    # per-request LLM endpoint overrides, passed through to ``llm_client``
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
