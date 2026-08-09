"""Contracts every stage of the agent passes to the next.

Plain dataclasses on purpose: they cross a thread boundary (model calls run in
``asyncio.to_thread``), they are printed in a notebook, and none of them is
validated at an API edge. Pydantic would buy nothing here and would drag the
lab package towards the app's dependency set.

Everything the LLM produces lands in one of these only AFTER the validators in
``planner`` and ``extraction`` have been over it. A field on a dataclass here
means "code has checked this", not "the model said so".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# Owned by the retrieval subpackage (which must not import from here),
# re-exported so the rest of the package has one place to import contracts from.
from lab.agent.retrieval.types import Fact, Ranked  # noqa: F401

Intent = Literal["general", "playlist"]
SourceKind = Literal["web", "wikipedia", "apple", "fandom"]
MatchMode = Literal["exact", "fuzzy", "none"]


@dataclass(slots=True)
class SearchHit:
    """One SearXNG result, before anything has been downloaded."""

    url: str
    title: str
    snippet: str
    source: SourceKind
    rank: int
    # Filled by the cross-encoder pass over title+snippet. None means the pass
    # did not run (structured sources skip it — see sources.py).
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
        its value is in the JSON. Judging it by markdown alone dropped it
        before the parser ever saw it.
        """
        return not self.error and bool(self.markdown.strip() or self.html)


@dataclass(slots=True)
class Chunk:
    """A context-aware slice of one page.

    ``text`` is what gets embedded: the heading path followed by the body. A
    body lifted out of "Kanye West > Controversies > 2009 VMA incident" means
    something the bare paragraph does not, and the retriever only ever sees
    ``text``.
    """

    id: int
    path: list[str]
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
    "Other appearances" and one from "Soundtrack" are worth different things,
    and the difference is invisible once both are just a string — which is how
    a listicle's "you might also like" sidebar ends up in a playlist.
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
    # Sum of the per-source weights that produced this track. Wikipedia and
    # Apple Music count double: a title from a curated tracklist is simply more
    # often real than one from a content-farm listicle.
    weight: float = 0.0
    sources: list[str] = field(default_factory=list)
    reason: Optional[str] = None
    # Carried over from the claim that produced it, so the final triage pass
    # can see WHERE a track came from and not just that it exists.
    section: str = ""
    page_title: str = ""
    context: str = ""


@dataclass(slots=True)
class Filters:
    """The constraints code will enforce, extracted from the user's sentence.

    Every one of these is checked by ``planner._validate`` before it lands
    here; the LLM cannot put anything in that the user did not say.
    """

    era: Optional[tuple[int, int]] = None
    # Kept AS the user wrote it — the CLAP branch that will consume it is not
    # built yet, so nothing normalises or interprets this.
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
    # How the final expansion was reached, for the notebook and the UI.
    resolved_by: Literal["llm", "user", "wikipedia", "unresolved"] = "llm"


@dataclass(slots=True)
class Plan:
    intent: Intent
    filters: Filters
    web_queries: list[str]
    ce_query: str
    allowed_tools: list[str]
    abbreviation: Optional[Abbreviation] = None
    # The model's own one-line reading of the request. Display only.
    rationale: str = ""


@dataclass(slots=True)
class Evidence:
    """One numbered item of the grounding pack.

    The number is the whole anti-hallucination mechanism: the model answers
    with ``used: [n, ...]`` and code throws the answer away if the citations do
    not check out.
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
    options: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class GeneralResult:
    answer: str
    evidence: list[Evidence]
    used: list[int]
    grounded: bool
    iterations: int
    # Set when the run ended without an answer and needs the user.
    clarify: Optional[ClarifyRequest] = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PlaylistResult:
    title: str
    comment: str
    tracks: list[ResolvedTrack]
    # Titles found on the web that the library does not have. Useful in the
    # notebook for judging whether the search worked but the library is thin.
    missing: list[TrackRef] = field(default_factory=list)
    iterations: int = 0
    relaxations: int = 0
    clarify: Optional[ClarifyRequest] = None
    notes: list[str] = field(default_factory=list)
