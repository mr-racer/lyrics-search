"""Both branches end to end, with the network, the LLM and the models faked.

This exists because the wiring is the part that cannot be checked by reading:
which stage feeds which, whether a structured page skips the reranker, whether
a hallucinated title really does vanish, whether an ungrounded answer really is
thrown away. Every external edge is stubbed, so the test asserts the pipeline's
own decisions and nothing else.
"""

import sqlite3

import pytest

from lab.agent import AgentConfig, Assistant, LibraryCatalog
from lab.agent import pipeline as P
from lab.agent.models import Page, SearchHit

COLLECTION = "acct_test"

WIKI_MD = """# Test Drive Unlimited 2

## Soundtrack

| Title | Artist |
| --- | --- |
| Kids | MGMT |
| Runaway | Kanye West |
| Not In The Library | Someone Else |
"""

PROSE_MD = """# Best driving songs

## The list

Everybody agrees that Bohemian Rhapsody by Queen belongs on any driving
playlist, and it has done since 1975. It is the single most requested song on
British radio and it never leaves a rotation once it enters one.
"""

ARTIST_MD = """# Eminem

## Stage name

Marshall Mathers took the name from his initials, M and M, which he wrote out
as Eminem. He has said the spelling came later, once the nickname stuck among
friends in Detroit who had been calling him that for years.
"""


# ── fakes ────────────────────────────────────────────────────────────────────


class FakeLLM:
    """Answers by looking at which system prompt it was handed."""

    def __init__(self, responses):
        self.responses = responses
        self.calls: list[str] = []

    async def ask_json(self, messages, *, required=(), **kw):
        system = messages[0]["content"]
        for marker, payload in self.responses.items():
            if marker in system:
                self.calls.append(marker)
                return payload() if callable(payload) else payload
        self.calls.append("UNMATCHED")
        return None


class FakeSources:
    def __init__(self, cfg=None, sink=None, hits=None):
        self.cfg = cfg
        self.sink = sink
        self.hits = hits or {}
        self.searches = 0
        self.queries: list[str] = []

    def _get(self, kind, query):
        self.queries.append(f"{kind}:{query}")
        return list(self.hits.get(kind, []))

    def web(self, query):
        return self._get("web", query)

    def wikipedia(self, query, limit=2):
        return self._get("wikipedia", query)

    def apple_music(self, query, limit=3):
        return self._get("apple", query)

    def fandom(self, query, limit=2):
        return self._get("fandom", query)

    def wikipedia_title(self, term):
        return self.hits.get("title")


class FakeFetcher:
    def __init__(self, cfg=None, sink=None, pages=None):
        self.cfg = cfg
        self.sink = sink
        self.pages = pages or {}
        self.fetched: list[str] = []

    async def fetch_many(self, hits, *, limit=None):
        out = []
        for hit in hits:
            if hit.url in self.fetched:
                continue
            page = self.pages.get(hit.url)
            if page is None:
                continue
            self.fetched.append(hit.url)
            out.append(page)
            if limit and len(out) >= limit:
                break
        return out


class FakeHub:
    """No embeddings; a cross-encoder that scores by word overlap."""

    def __init__(self, cfg=None):
        self.cfg = cfg or AgentConfig()

    def encode_dense(self, texts, *, is_query=False):
        return None

    def encode_sparse(self, texts, *, is_query=False):
        return None

    def ce_probabilities(self, query, docs):
        import re

        words = set(re.findall(r"\w+", query.lower()))
        out = []
        for doc in docs:
            found = set(re.findall(r"\w+", doc.lower()))
            out.append(min(1.0, len(words & found) / (len(words) or 1) + 0.25))
        return out


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "metadata.db"
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE track_metadata (
        collection_name TEXT, track_id TEXT, title TEXT, artist TEXT,
        artists TEXT, artist_slugs TEXT, primary_artist_slug TEXT,
        album TEXT, year INTEGER)""")
    conn.execute("CREATE TABLE artists (slug TEXT, name TEXT, collection_name TEXT)")
    conn.execute("CREATE TABLE songs (slug TEXT, title TEXT, artist_slug TEXT, "
                 "collection_name TEXT)")
    conn.execute("""CREATE TABLE artist_facts (
        id INTEGER, artist_slug TEXT, lang TEXT, fact TEXT, category TEXT,
        source TEXT)""")
    conn.execute("""CREATE TABLE song_facts (
        id INTEGER, song_slug TEXT, lang TEXT, fact TEXT, category TEXT,
        source TEXT)""")
    rows = [("t1", "Kids", "MGMT", 2007),
            ("t2", "Runaway", "Kanye West", 2010),
            ("t3", "Bohemian Rhapsody", "Queen", 1975)]
    for track_id, title, artist, year in rows:
        slug = artist.lower().replace(" ", "-")
        conn.execute("INSERT INTO track_metadata VALUES (?,?,?,?,?,?,?,?,?)",
                     (COLLECTION, track_id, title, artist, None, None, slug,
                      "", year))
        conn.execute("INSERT INTO songs VALUES (?,?,?,?)",
                     (f"{slug}-{title.lower().replace(' ', '-')}", title, slug,
                      COLLECTION))
        conn.execute("INSERT INTO artists VALUES (?,?,?)", (slug, artist, COLLECTION))
    conn.execute("INSERT INTO artists VALUES ('eminem','Eminem',?)", (COLLECTION,))
    conn.execute("INSERT INTO artist_facts VALUES (1,'eminem','en',"
                 "'Eminem is a stage name built from the initials M and M.',"
                 "'name','songfacts.com')")
    conn.execute("INSERT INTO artist_facts VALUES (2,'eminem','en',"
                 "'He grew up in Detroit and started rapping at fourteen.',"
                 "'bio','songfacts.com')")
    conn.commit()
    conn.close()
    return str(path)


def _assistant(monkeypatch, db, *, llm, sources, pages, **cfg_kwargs):
    cfg = AgentConfig(db_path=db, collection_name=COLLECTION, lang="ru",
                      **cfg_kwargs)
    monkeypatch.setattr(P, "SearchSources",
                        lambda c, s: FakeSources(c, s, hits=sources))
    monkeypatch.setattr(P, "PageFetcher",
                        lambda c, s: FakeFetcher(c, s, pages=pages))
    agent = Assistant(cfg, hub=FakeHub(cfg),
                      catalog=LibraryCatalog(db, COLLECTION))
    agent.llm = llm
    agent.planner.llm = llm
    return agent


# ── the general branch ───────────────────────────────────────────────────────


class TestGeneralBranch:
    @pytest.fixture
    def agent(self, monkeypatch, db):
        llm = FakeLLM({
            "You plan how to answer": {
                "intent": "general", "artist": "Эминем",
                "web_queries": ["Eminem stage name origin"],
                "ce_query": "the origin of Eminem's stage name",
                "rationale": "Про псевдоним."},
            "You answer a music question": {
                "answer": "Псевдоним вырос из инициалов M и M.",
                "used": [1], "sufficient": True, "missing": ""},
        })
        return _assistant(
            monkeypatch, db, llm=llm,
            sources={"web": [SearchHit(url="https://ex/eminem", title="Eminem",
                                       snippet="stage name origin",
                                       source="web", rank=0)],
                     "wikipedia": []},
            pages={"https://ex/eminem": Page(url="https://ex/eminem",
                                             title="Eminem", markdown=ARTIST_MD,
                                             source="web")})

    async def test_it_answers_and_cites(self, agent):
        result = await agent.run("Почему Эминем взял такой псевдоним?")
        assert result.grounded is True
        assert result.used == [1]
        assert "инициал" in result.answer.lower()

    async def test_library_facts_lead_the_pack(self, agent):
        """The subject's own facts are cheaper and more trustworthy than the
        web, so they are numbered first and the model sees them first."""
        result = await agent.run("Почему Эминем взял такой псевдоним?")
        assert result.evidence[0].kind == "fact"
        assert any(e.kind == "chunk" for e in result.evidence)

    async def test_one_round_is_enough_when_the_model_is_confident(self, agent):
        result = await agent.run("Почему Эминем взял такой псевдоним?")
        assert result.iterations == 1


class TestGroundingGate:
    async def test_an_answer_with_no_citations_is_thrown_away(self, monkeypatch, db):
        """The whole anti-hallucination mechanism: an untraceable paragraph
        never reaches the caller, however plausible it reads."""
        llm = FakeLLM({
            "You plan how to answer": {
                "intent": "general", "artist": "Эминем",
                "web_queries": ["Eminem"], "ce_query": "Eminem"},
            "You answer a music question": {
                "answer": "Он назвался так в честь конфет M&M's.",
                "used": [], "sufficient": True},
            "You already searched": {"web_queries": []},
        })
        agent = _assistant(monkeypatch, db, llm=llm, sources={}, pages={})
        result = await agent.run("Почему Эминем взял такой псевдоним?")
        assert result.grounded is False
        # The invented claim is gone; what replaces it is the raw material,
        # verbatim, so the user can read the sources themselves.
        assert "конфет" not in result.answer
        assert "initials M and M" in result.answer
        assert any("discarded" in n for n in result.notes)

    async def test_a_citation_out_of_range_counts_as_no_citation(self, monkeypatch, db):
        llm = FakeLLM({
            "You plan how to answer": {
                "intent": "general", "artist": "Эминем",
                "web_queries": ["Eminem"], "ce_query": "Eminem"},
            "You answer a music question": {
                "answer": "Согласно источнику.", "used": [99], "sufficient": True},
            "You already searched": {"web_queries": []},
        })
        agent = _assistant(monkeypatch, db, llm=llm, sources={}, pages={})
        result = await agent.run("Почему Эминем взял такой псевдоним?")
        assert result.grounded is False


class TestVetoInPractice:
    async def test_a_weak_context_forces_a_second_round(self, monkeypatch, db):
        """The model says it is done; the cross-encoder says the material is
        unrelated. Code sends it back out."""
        llm = FakeLLM({
            "You plan how to answer": {
                "intent": "general", "web_queries": ["zzz unrelated"],
                "ce_query": "zzz"},
            "You answer a music question": {
                "answer": "Ответ.", "used": [1], "sufficient": True},
            "You already searched": {
                "web_queries": ["another angle"], "ce_query": "zzz"},
        })
        page = Page(url="https://ex/w", title="W", markdown=PROSE_MD, source="web")
        agent = _assistant(
            monkeypatch, db, llm=llm,
            sources={"web": [SearchHit(url="https://ex/w", title="W",
                                       snippet="driving songs", source="web",
                                       rank=0)]},
            pages={"https://ex/w": page})
        result = await agent.run("вопрос ни о чём")
        assert result.iterations == 2
        assert any("searching again" in n for n in result.notes)


# ── the playlist branch ──────────────────────────────────────────────────────


class TestPlaylistBranch:
    def _llm(self, **extra):
        responses = {
            "You plan how to answer": {
                "intent": "playlist", "work": "Test Drive Unlimited 2",
                "web_queries": ["Test Drive Unlimited 2 soundtrack"],
                "ce_query": "songs in the game"},
            "Pull song titles": {"tracks": []},
            "You are finishing a playlist": {
                "title": "Из TDU2", "comment": "Саундтрек.",
                "order": [{"id": "T1", "reason": "Открывает подборку задорно"},
                          {"id": "T2", "reason": "Отличный трек"}]},
            "You already searched": {"web_queries": []},
        }
        responses.update(extra)
        return FakeLLM(responses)

    @pytest.fixture
    def agent(self, monkeypatch, db):
        wiki = Page(url="https://wiki/tdu2", title="TDU2", markdown=WIKI_MD,
                    source="wikipedia")
        return _assistant(
            monkeypatch, db, llm=self._llm(),
            sources={"wikipedia": [SearchHit(url="https://wiki/tdu2",
                                             title="TDU2", snippet="",
                                             source="wikipedia", rank=0)]},
            pages={"https://wiki/tdu2": wiki})

    async def test_a_wiki_table_becomes_library_tracks(self, agent):
        result = await agent.run("Песни из Test Drive Unlimited 2")
        titles = {t.title for t in result.tracks}
        assert titles == {"Kids", "Runaway"}

    async def test_a_title_the_library_lacks_is_reported_not_invented(self, agent):
        result = await agent.run("Песни из Test Drive Unlimited 2")
        assert any(m.title == "Not In The Library" for m in result.missing)

    async def test_wiki_sourced_tracks_carry_the_double_weight(self, agent):
        result = await agent.run("Песни из Test Drive Unlimited 2")
        assert all(t.weight == 2.0 for t in result.tracks)

    async def test_filler_reasons_are_stripped_but_the_track_stays(self, agent):
        result = await agent.run("Песни из Test Drive Unlimited 2")
        by_title = {t.title: t for t in result.tracks}
        assert by_title["Kids"].reason == "Открывает подборку задорно"
        assert by_title["Runaway"].reason is None

    async def test_a_hallucinated_track_never_reaches_the_playlist(self, monkeypatch, db):
        """The model is asked for titles from prose and invents one. It has to
        die at the library boundary."""
        llm = self._llm(**{"Pull song titles": {
            "tracks": [{"title": "Interstellar Dogfight", "artist": "Kanye West"},
                       {"title": "Bohemian Rhapsody", "artist": "Queen"}]}})
        page = Page(url="https://ex/list", title="Best driving songs",
                    markdown=PROSE_MD, source="web")
        agent = _assistant(
            monkeypatch, db, llm=llm,
            sources={"web": [SearchHit(url="https://ex/list",
                                       title="Best driving songs",
                                       snippet="Bohemian Rhapsody Queen",
                                       source="web", rank=0)]},
            pages={"https://ex/list": page})
        result = await agent.run("Песни для поездки")
        titles = {t.title for t in result.tracks}
        assert "Bohemian Rhapsody" in titles
        assert "Interstellar Dogfight" not in titles

    async def test_the_era_filter_runs_in_code(self, monkeypatch, db):
        llm = self._llm(**{"You plan how to answer": {
            "intent": "playlist", "era": "2008-2015",
            "work": "Test Drive Unlimited 2",
            "web_queries": ["TDU2 soundtrack"], "ce_query": "songs"}})
        wiki = Page(url="https://wiki/tdu2", title="TDU2", markdown=WIKI_MD,
                    source="wikipedia")
        agent = _assistant(
            monkeypatch, db, llm=llm,
            sources={"wikipedia": [SearchHit(url="https://wiki/tdu2",
                                             title="TDU2", snippet="",
                                             source="wikipedia", rank=0)]},
            pages={"https://wiki/tdu2": wiki})
        result = await agent.run("Песни из TDU2 после 2008")
        # Kids is 2007 in the library and must be filtered out; Runaway is 2010.
        assert {t.title for t in result.tracks} == {"Runaway"}


class TestClarify:
    async def test_a_confident_expansion_is_taken_and_pinned_into_the_query(
            self, monkeypatch, db):
        llm = FakeLLM({
            "You plan how to answer": {
                "intent": "playlist", "work": "Grand Theft Auto V",
                "abbreviation": {"raw": "гта 5",
                                 "expansion": "Grand Theft Auto V",
                                 "confidence": 0.95},
                "web_queries": ["гта 5 soundtrack"], "ce_query": "songs"},
            "Pull song titles": {"tracks": []},
            "You are finishing a playlist": {"title": "GTA V", "comment": "",
                                             "order": []},
            "You already searched": {"web_queries": []},
        })
        agent = _assistant(monkeypatch, db, llm=llm, sources={}, pages={})
        await agent.run("музыка из гта 5")
        plan_event = agent.sink.of("plan")[0]
        assert plan_event["work"] == "Grand Theft Auto V"
        assert all('"Grand Theft Auto V"' in q for q in plan_event["queries"])

    async def test_an_unsure_expansion_falls_back_to_wikipedia(self, monkeypatch, db):
        llm = FakeLLM({
            "You plan how to answer": {
                "intent": "playlist", "work": "Touring Drive Unlimited",
                "abbreviation": {"raw": "TDU 2",
                                 "expansion": "Touring Drive Unlimited",
                                 "confidence": 0.3},
                "web_queries": ["TDU 2 soundtrack"], "ce_query": "songs"},
            "Pull song titles": {"tracks": []},
            "You are finishing a playlist": {"title": "x", "comment": "",
                                             "order": []},
            "You already searched": {"web_queries": []},
        })
        agent = _assistant(monkeypatch, db, llm=llm,
                           sources={"title": "Test Drive Unlimited 2"},
                           pages={})
        await agent.run("песни из TDU 2")
        done = agent.sink.of("clarify_done")[0]
        assert done["by"] == "wikipedia"
        assert done["expansion"] == "Test Drive Unlimited 2"


class TestPlannerFailure:
    async def test_an_unusable_plan_ends_the_run_honestly(self, monkeypatch, db):
        agent = _assistant(monkeypatch, db, llm=FakeLLM({}), sources={}, pages={})
        result = await agent.run("что-нибудь")
        assert result.grounded is False
        assert result.answer == ""
        assert result.notes
