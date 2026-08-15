"""Contracts every stage of the assistant passes to the next.

Plain dataclasses on purpose: they cross a thread boundary (model calls run in
``asyncio.to_thread``), and none of them is validated at an API edge — the
route maps them onto the Pydantic models in ``app/domain/models.py`` on the way
out. Pydantic here would buy nothing and would put validation in the middle of
a pipeline instead of at its edge.

Everything the LLM produces lands in one of these only AFTER the validators in
``planner`` and ``tracklists`` have been over it. A field on a dataclass here
means "code has checked this", not "the model said so".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Intent = Literal["general", "playlist", "lyrics_search", "audio_search"]
SourceKind = Literal["web", "wikipedia", "apple", "fandom", "reddit"]
MatchMode = Literal["exact", "fuzzy", "none"]


@dataclass(slots=True)
class Fact:
    """A raw source fact about a song or an artist, as stored."""

    row_id: int
    kind: Literal["song", "artist"]
    slug: str
    text: str
    source: str = ""
    category: str = ""
    ce_prob: Optional[float] = None


@dataclass(slots=True)
class SearchHit:
    """One SearXNG result, before anything has been downloaded."""

    url: str
    title: str
    snippet: str
    source: SourceKind
    rank: int
    # Filled by the cross-encoder pass over title+snippet. None means the pass
    # did not run (structured sources skip it — see web_sources.py).
    ce_prob: Optional[float] = None


@dataclass(slots=True)
class Page:
    """A fetched page. ``error`` set means the fetch failed and the rest is empty."""

    url: str
    title: str
    markdown: str
    source: SourceKind
    meta: dict = field(default_factory=dict)
    fetcher: Optional[str] = None
    error: Optional[str] = None
    # Raw body, kept only where a structured parser needs it. Apple Music puts
    # its track list in embedded JSON and renders almost nothing as text, so
    # markdown extraction throws away the only part worth having.
    html: str = ""

    @property
    def ok(self) -> bool:
        """Usable by SOMETHING — not necessarily by the chunker.

        An Apple page can be perfectly good with no extractable prose at all:
        its value is in the JSON. Judging it by markdown alone dropped it before
        the parser ever saw it.
        """
        return not self.error and bool(self.markdown.strip() or self.html)


@dataclass(slots=True)
class Chunk:
    """A context-aware slice of one page.

    ``text`` is what gets embedded: the heading path followed by the body. A body
    lifted out of "Kanye West > Controversies > 2009 VMA incident" means
    something the bare paragraph does not, and the retriever only ever sees
    ``text``.
    """

    id: int
    path: list
    body: str
    url: str = ""
    title: str = ""
    source: SourceKind = "web"
    text: str = ""

    def __post_init__(self) -> None:
        if not self.text:
            self.text = (" > ".join(self.path) + "\n\n" + self.body).strip()


@dataclass(slots=True)
class TrackRef:
    """A track a PAGE claims exists. Not yet anything in the library.

    The provenance fields are not decoration. A title lifted from a table under
    "Other appearances" and one from "Soundtrack" are worth different things, and
    the difference is invisible once both are just a string — which is how a
    listicle's "you might also like" sidebar ends up in a playlist.
    """

    title: str
    artist: Optional[str] = None
    year: Optional[int] = None
    source: SourceKind = "web"
    source_url: str = ""
    # Heading path the claim was found under, e.g. "Discography > Singles".
    section: str = ""
    # The page's own title.
    page_title: str = ""
    # A few words either side, for a human or a model to judge by.
    context: str = ""


@dataclass(slots=True)
class ResolvedTrack:
    """A :class:`TrackRef` matched against the user's library."""

    track_id: str
    title: str
    artist: str
    year: Optional[int]
    match: MatchMode
    # Sum of the per-source weights that produced this track. Wikipedia and Apple
    # Music count double: a title from a curated tracklist is simply more often
    # real than one from a content-farm listicle.
    weight: float = 0.0
    sources: list = field(default_factory=list)
    reason: Optional[str] = None
    # Carried over from the claim that produced it, so the final triage pass can
    # see WHERE a track came from and not just that it exists.
    section: str = ""
    page_title: str = ""
    context: str = ""
    # Carried from track_metadata by the matcher, so the payload can be built
    # without a second read per row.
    cover_art_path: Optional[str] = None
    album: Optional[str] = None
    duration_sec: float = 0.0
    file_path: str = ""


@dataclass(slots=True)
class Filters:
    """The constraints code will enforce, extracted from the user's sentence.

    Every one of these is checked by ``planner._validate`` before it lands here;
    the LLM cannot put anything in that the user did not say.
    """

    era: Optional[tuple] = None
    # Kept AS the user wrote it. The audio branch is what consumes this: it
    # rewrites the phrase into CLAP prompts, and interpreting it earlier would
    # mean interpreting it twice.
    style: Optional[str] = None
    # A film or game the request is about, already expanded and confirmed.
    work: Optional[str] = None
    artist: Optional[str] = None
    song: Optional[str] = None
    count: Optional[int] = None


@dataclass(slots=True)
class Abbreviation:
    """An abbreviation the planner wants to expand before searching."""

    raw: str
    expansion: str
    confidence: float
    # How the final expansion was reached, for the log and the UI.
    resolved_by: Literal["llm", "user", "wikipedia", "unresolved"] = "llm"


@dataclass(slots=True)
class Subject:
    """Who and what a question is about, once the library has had its say.

    ``how`` records which tier answered, because the tiers differ in kind and not
    just in confidence:

    ``pinned``       the caller supplied a track id or an artist slug — no name
                     matching happened and none may
    ``song-row``     the named song is in the library and its row carries the
                     artist slug
    ``exact-name``   the artist name matched a library name exactly (folded)
    ``participant``  the query is the leading participant of a collab tag
                     ("Amerie" of "Amerie feat. Nas") — structure, not similarity
    ``shortlist``    genuinely ambiguous; ``candidates`` is for a judgement call
    ``none``         nothing plausible. No facts, and that is the right answer.
    """

    song_slug: Optional[str] = None
    artist_slug: Optional[str] = None
    artist_name: Optional[str] = None
    song_title: Optional[str] = None
    track_id: Optional[str] = None
    how: str = "none"
    candidates: list = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return bool(self.song_slug or self.artist_slug)


@dataclass(slots=True)
class Plan:
    intent: Intent
    filters: Filters
    web_queries: list
    ce_query: str
    abbreviation: Optional[Abbreviation] = None
    # The model's own one-line reading of the request. Display only.
    rationale: str = ""
    # lyrics_search only: what goes to the embedding, and what the cross-encoder
    # is asked. They differ on purpose — see planner.
    lyrics_query: str = ""


@dataclass(slots=True)
class Evidence:
    """One numbered item of the grounding pack.

    The number is the whole anti-hallucination mechanism: the model answers with
    ``used: [n, ...]`` and code throws the answer away if the citations do not
    check out.
    """

    n: int
    text: str
    kind: Literal["fact", "chunk"]
    source: str = ""
    url: str = ""
    ce_prob: Optional[float] = None


@dataclass(slots=True)
class ClarifyRequest:
    """Handed to the caller when the agent needs a human to settle something."""

    kind: Literal["abbreviation", "subject"]
    question: str
    suggestion: Optional[str] = None
    options: list = field(default_factory=list)


@dataclass(slots=True)
class GeneralResult:
    answer: str
    evidence: list
    used: list
    grounded: bool
    iterations: int
    subject: Optional[Subject] = None
    # Set when the turn was "explain THIS statement" rather than "tell me about
    # this subject": the statement itself, and whether anything explained it.
    focus_fact: Optional[str] = None
    explained: Optional[bool] = None
    follow_ups: list = field(default_factory=list)
    clarify: Optional[ClarifyRequest] = None
    notes: list = field(default_factory=list)


@dataclass(slots=True)
class PlaylistResult:
    title: str
    comment: str
    tracks: list
    # Titles found on the web that the library does not have. Useful for judging
    # whether the search worked but the library is thin.
    missing: list = field(default_factory=list)
    iterations: int = 0
    relaxations: int = 0
    clarify: Optional[ClarifyRequest] = None
    notes: list = field(default_factory=list)


@dataclass(slots=True)
class LyricsResult:
    """Shaped to match what the search card already renders."""

    message: str
    song: Optional[str] = None
    artist: Optional[str] = None
    confidence: str = "low"
    # ``TrackHit`` instances from SearchService.
    best_hit: Optional[Any] = None
    hits: list = field(default_factory=list)
    clarify: Optional[ClarifyRequest] = None
    notes: list = field(default_factory=list)


@dataclass(slots=True)
class AudioResult:
    """A sound-alike list. Rendered by the playlist card."""

    title: str
    comment: str
    # ``TrackHit`` instances, RRF-merged across the CLAP rephrasings.
    tracks: list = field(default_factory=list)
    # The rephrasings that were actually searched, for the log and the UI.
    queries: list = field(default_factory=list)
    clarify: Optional[ClarifyRequest] = None
    notes: list = field(default_factory=list)
