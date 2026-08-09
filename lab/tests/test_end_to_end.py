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
        """Hits for a kind. A dict value keys them by query substring, so a
        test can make a page reachable ONLY by a particular query."""
        self.queries.append(f"{kind}:{query}")
        value = self.hits.get(kind, [])
        if isinstance(value, dict):
            low = (query or "").lower()
            for marker, hits in value.items():
                if marker.lower() in low:
                    return list(hits)
            return []
        return list(value)

    def web(self, query):
        return self._get("web", query)

    def wikipedia(self, query, limit=2, force=False):
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
    def _weak_agent(self, monkeypatch, db, **cfg):
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
        return _assistant(
            monkeypatch, db, llm=llm,
            sources={"web": [SearchHit(url="https://ex/w", title="W",
                                       snippet="driving songs", source="web",
                                       rank=0)]},
            pages={"https://ex/w": page}, **cfg)

    async def test_a_weak_context_forces_a_second_round(self, monkeypatch, db):
        """The model says it is done; the cross-encoder says the material is
        unrelated. Code sends it back out.

        The chunk threshold is lowered here on purpose: at the default 0.75 a
        weak passage never reaches the pack at all, and the case under test is
        the one where it does — weak enough to distrust, strong enough to read.
        """
        agent = self._weak_agent(monkeypatch, db, ce_threshold_docs=0.1,
                                 ce_threshold_chunks=0.2)
        result = await agent.run("вопрос ни о чём")
        assert result.iterations == 2
        assert any("searching again" in n for n in result.notes)

    async def test_nothing_clearing_the_threshold_is_a_reason_to_search_again(
            self, monkeypatch, db):
        """Not a reason to stop. The pages were wrong; the question was not.
        Treating the empty case as an answer is what a high chunk threshold
        turns into otherwise — one unlucky pair of queries ends the run."""
        agent = self._weak_agent(monkeypatch, db, ce_threshold_docs=0.1,
                                 ce_threshold_chunks=0.99)
        result = await agent.run("вопрос ни о чём")
        assert result.iterations == 2
        assert any("nothing cleared" in n for n in result.notes)


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


class TestSubjectDisambiguation:
    """Layer 4: code builds the shortlist, the model judges, code checks the
    judgement against the shortlist before acting on it."""

    def _db_with_two_hurts(self, tmp_path):
        path = tmp_path / "two.db"
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
        for slug, name in (("nine-inch-nails", "Nine Inch Nails"),
                           ("johnny-cash", "Johnny Cash")):
            conn.execute("INSERT INTO artists VALUES (?,?,?)", (slug, name, COLLECTION))
            conn.execute("INSERT INTO songs VALUES (?,?,?,?)",
                         (f"{slug}-hurt", "Hurt", slug, COLLECTION))
            conn.execute("INSERT INTO track_metadata VALUES (?,?,?,?,?,?,?,?,?)",
                         (COLLECTION, slug, "Hurt", name, None, None, slug, "", 1994))
        conn.execute("INSERT INTO song_facts VALUES (1,'johnny-cash-hurt','en',"
                     "'Johnny Cash recorded Hurt in 2002 as a Nine Inch Nails cover.',"
                     "'cover','songfacts.com')")
        conn.execute("INSERT INTO song_facts VALUES (2,'nine-inch-nails-hurt','en',"
                     "'Trent Reznor wrote Hurt for The Downward Spiral in 1994.',"
                     "'origin','songfacts.com')")
        conn.commit()
        conn.close()
        return str(path)

    def _llm(self, pick):
        return FakeLLM({
            "You plan how to answer": {
                "intent": "general", "song": "Hurt",
                "web_queries": ["Hurt song"], "ce_query": "the song Hurt"},
            "The listener named an artist": pick,
            "You answer a music question": {
                "answer": "Ответ.", "used": [1], "sufficient": True},
            "You already searched": {"web_queries": []},
        })

    async def test_the_models_pick_selects_whose_facts_are_loaded(
            self, monkeypatch, tmp_path):
        db = self._db_with_two_hurts(tmp_path)
        agent = _assistant(monkeypatch, db,
                           llm=self._llm({"artist": "Johnny Cash",
                                          "why": "the 2002 cover"}),
                           sources={}, pages={})
        result = await agent.run("расскажи про Hurt")
        assert agent.sink.of("subject")[0]["how"] == "model-pick"
        assert "Johnny Cash recorded Hurt" in result.evidence[0].text

    async def test_a_pick_outside_the_shortlist_is_refused(
            self, monkeypatch, tmp_path):
        """The model can only choose from what code offered it. An invented
        name means "unresolved", not "load whatever that is"."""
        db = self._db_with_two_hurts(tmp_path)
        agent = _assistant(monkeypatch, db,
                           llm=self._llm({"artist": "Fergie", "why": "vibes"}),
                           sources={}, pages={})
        await agent.run("расскажи про Hurt")
        assert agent.sink.of("subject")[0]["how"] == "none"

    async def test_declining_costs_the_facts_and_nothing_else(
            self, monkeypatch, tmp_path):
        """Saying null loses a few library facts. Picking wrong would put a
        stranger's song into the answer — so null is the cheaper mistake."""
        db = self._db_with_two_hurts(tmp_path)
        agent = _assistant(monkeypatch, db,
                           llm=self._llm({"artist": None, "why": "not sure"}),
                           sources={}, pages={})
        result = await agent.run("расскажи про Hurt")
        assert agent.sink.of("subject")[0]["how"] == "none"
        assert not [e for e in result.evidence if e.kind == "fact"]

    async def test_a_structural_match_never_reaches_the_model(
            self, monkeypatch, db):
        """One LLM call saved on every unambiguous subject, which is most of
        them. "Эминем" is "Eminem" across alphabets — nothing to judge."""
        llm = FakeLLM({
            "You plan how to answer": {
                "intent": "general", "artist": "Эминем",
                "web_queries": ["Eminem"], "ce_query": "Eminem"},
            "You answer a music question": {
                "answer": "Ответ.", "used": [1], "sufficient": True},
            "You already searched": {"web_queries": []},
        })
        agent = _assistant(monkeypatch, db, llm=llm, sources={}, pages={})
        await agent.run("расскажи про Эминема")
        assert "The listener named an artist" not in llm.calls
        assert agent.sink.of("subject")[0]["artist"] == "eminem"


class TestPlannerFailure:
    """"No plan" has two causes with two different fixes, so the run must say
    which one happened. They used to produce the same opaque note."""

    async def test_an_unusable_plan_ends_the_run_honestly(self, monkeypatch, db):
        agent = _assistant(monkeypatch, db, llm=FakeLLM({}), sources={}, pages={})
        result = await agent.run("что-нибудь")
        assert result.grounded is False
        assert result.answer == ""
        assert result.notes

    async def test_an_unreachable_llm_says_so_with_the_endpoint(
            self, monkeypatch, db):
        """The failure the owner actually hit. "Nothing usable" sent them
        looking at the planner; the endpoint is what they needed to see."""
        class _Dead:
            last_error = ("ConnectError: [Errno 111] Connection refused "
                          "(POST http://192.168.0.168:8082/v1/chat/completions, "
                          "model='gemma-4-12b')")
            last_raw = ""

            async def ask_json(self, messages, *, required=(), **kw):
                return None

        agent = _assistant(monkeypatch, db, llm=_Dead(), sources={}, pages={})
        result = await agent.run("Песни из Test Drive Unlimited 2")
        note = result.notes[0]
        assert "did not answer" in note
        assert "8082" in note and "Connection refused" in note
        assert agent.sink.of("plan_failed")

    async def test_a_talkative_llm_is_reported_differently(self, monkeypatch, db):
        """Reachable, but answering in prose. Same empty result, opposite fix —
        so the note quotes what it actually said."""
        class _Chatty:
            last_error = None
            last_raw = "Sure! I can help you build that playlist."

            async def ask_json(self, messages, *, required=(), **kw):
                return None

        agent = _assistant(monkeypatch, db, llm=_Chatty(), sources={}, pages={})
        result = await agent.run("Песни из Test Drive Unlimited 2")
        note = result.notes[0]
        assert "not with a JSON object" in note
        assert "Sure!" in note

    async def test_an_out_of_list_intent_names_what_the_model_said(
            self, monkeypatch, db):
        llm = FakeLLM({"You plan how to answer": {"intent": "vibes"}})
        agent = _assistant(monkeypatch, db, llm=llm, sources={}, pages={})
        result = await agent.run("что-нибудь")
        assert "vibes" in result.notes[0]


class TestTriage:
    """The last gate: matching the library proves the library HAS a track, not
    that the page was offering it as an answer."""

    WIKI = """# Kanye West discography

## Singles

| Title | Year |
| --- | --- |
| Runaway | 2010 |
| Power | 2010 |

## Other appearances

| Title | Artist | Year |
| --- | --- | --- |
| Kids | MGMT | 2007 |
"""

    def _llm(self, triage):
        return FakeLLM({
            "You plan how to answer": {
                "intent": "playlist", "artist": "Kanye West",
                "web_queries": ["Kanye West singles"], "ce_query": "singles"},
            "Pull song titles": {"tracks": []},
            "Below are songs found on web pages": triage,
            "You are finishing a playlist": {"title": "K", "comment": "",
                                             "order": []},
            "You already searched": {"web_queries": []},
        })

    def _agent(self, monkeypatch, db, triage, **cfg):
        page = Page(url="https://wiki/kw", title="Kanye West discography",
                    markdown=self.WIKI, source="wikipedia")
        return _assistant(
            monkeypatch, db, llm=self._llm(triage),
            sources={"wikipedia": [SearchHit(url="https://wiki/kw", title="d",
                                             snippet="", source="wikipedia",
                                             rank=0)]},
            pages={"https://wiki/kw": page}, **cfg)

    async def test_the_model_can_drop_a_track_that_only_shared_the_page(
            self, monkeypatch, db):
        # Candidates are ordered by weight, then match, then title — so T1 is
        # "Kids" (from Other appearances) and T2 is "Runaway" (from Singles).
        agent = self._agent(monkeypatch, db,
                            {"keep": ["T2"],
                             "dropped_because": "Kids is an MGMT track listed "
                                                "under Other appearances"},
                            triage_min_candidates=1)
        result = await agent.run("хиты Канье")
        assert {t.title for t in result.tracks} == {"Runaway"}
        assert agent.sink.of("triage_done")[0]["dropped"] == 1

    async def test_an_id_the_model_invented_is_ignored(self, monkeypatch, db):
        agent = self._agent(monkeypatch, db,
                            {"keep": ["T1", "T99"]}, triage_min_candidates=1)
        result = await agent.run("хиты Канье")
        assert len(result.tracks) == 1

    async def test_keeping_nothing_is_treated_as_a_model_failure(
            self, monkeypatch, db):
        """Far likelier than a genuine verdict that the whole page was junk —
        and emptying the playlist on it would be the worst possible response."""
        agent = self._agent(monkeypatch, db, {"keep": []},
                            triage_min_candidates=1)
        result = await agent.run("хиты Канье")
        assert len(result.tracks) >= 2

    async def test_a_short_list_skips_triage_entirely(self, monkeypatch, db):
        """Nothing to triage when the whole list is the answer — and it saves
        an LLM call on most runs."""
        agent = self._agent(monkeypatch, db, {"keep": ["T1"]},
                            triage_min_candidates=50)
        result = await agent.run("хиты Канье")
        assert not agent.sink.of("triage")
        assert len(result.tracks) >= 2

    async def test_provenance_reaches_the_resolved_track(self, monkeypatch, db):
        agent = self._agent(monkeypatch, db, {"keep": ["T1", "T2", "T3"]},
                            triage_min_candidates=1)
        result = await agent.run("хиты Канье")
        runaway = next(t for t in result.tracks if t.title == "Runaway")
        assert runaway.section.endswith("Singles")
        assert runaway.page_title == "Kanye West discography"


class TestDiscographyRescue:
    """"Хиты Канье после 2020" has no page. The singles table does."""

    # Only Kanye's own singles: a discography page listing MGMT's "Kids" would
    # be wrong, and the artist check would (correctly) refuse it.
    DISCOGRAPHY = """# Kanye West discography

## Singles

| Title | Year |
| --- | --- |
| Runaway | 2010 |
| Power | 2010 |
"""

    THIN = "# News\n\nKanye West gave an interview about his plans.\n"

    def _llm(self):
        return FakeLLM({
            "You plan how to answer": {
                "intent": "playlist", "artist": "Канье",
                "web_queries": ["Kanye West hits"], "ce_query": "hits"},
            "Pull song titles": {"tracks": []},
            "Below are songs found on web pages": {"keep": ["T1", "T2", "T3"]},
            "You are finishing a playlist": {"title": "K", "comment": "",
                                             "order": []},
            "You already searched": {"web_queries": []},
        })

    def _agent(self, monkeypatch, db, *, wiki_pages, **cfg):
        pages = {"https://news/x": Page(url="https://news/x", title="News",
                                        markdown=self.THIN, source="web")}
        pages.update(wiki_pages)
        # The discography page is reachable ONLY by the rescue's own query, so
        # the test proves the rescue found it and not the main loop.
        hits = {"web": [SearchHit(url="https://news/x", title="News",
                                  snippet="Kanye interview", source="web",
                                  rank=0)],
                "wikipedia": {"discography": [
                    SearchHit(url="https://wiki/disco",
                              title="Kanye West discography",
                              snippet="", source="wikipedia", rank=0)]}}
        return _assistant(monkeypatch, db, llm=self._llm(), sources=hits,
                          pages=pages, **cfg)

    async def test_a_thin_result_pulls_in_the_discography(self, monkeypatch, db):
        wiki = {"https://wiki/disco": Page(url="https://wiki/disco",
                                           title="Kanye West discography",
                                           markdown=self.DISCOGRAPHY,
                                           source="wikipedia")}
        agent = self._agent(monkeypatch, db, wiki_pages=wiki,
                            triage_min_candidates=99)
        result = await agent.run("хиты Канье")
        assert agent.sink.of("discography")
        # Nothing matched before the rescue; "Runaway" only exists in the
        # discography table, which only the rescue's query reaches.
        assert [t.title for t in result.tracks] == ["Runaway"]
        assert any("discography rescue" in n for n in result.notes)

    async def test_the_query_uses_the_librarys_spelling(self, monkeypatch, db):
        """The user typed «Канье»; a Cyrillic query to the English Wikipedia
        finds nothing, and the library already knows the real name."""
        wiki = {"https://wiki/disco": Page(url="https://wiki/disco",
                                           title="d", markdown=self.DISCOGRAPHY,
                                           source="wikipedia")}
        agent = self._agent(monkeypatch, db, wiki_pages=wiki,
                            triage_min_candidates=99)
        await agent.run("хиты Канье")
        assert agent.sink.of("discography")[0]["artist"] == "Kanye West"

    async def test_a_full_playlist_skips_the_rescue(self, monkeypatch, db):
        """It costs a search and two fetches — only worth it when thin."""
        wiki = {"https://wiki/disco": Page(url="https://wiki/disco",
                                           title="d", markdown=self.DISCOGRAPHY,
                                           source="wikipedia")}
        agent = self._agent(monkeypatch, db, wiki_pages=wiki,
                            discography_min_tracks=0, triage_min_candidates=99)
        await agent.run("хиты Канье")
        assert not agent.sink.of("discography")

    async def test_no_artist_means_no_rescue(self, monkeypatch, db):
        """There is nothing to look up for "спокойные хиты 80х"."""
        llm = FakeLLM({
            "You plan how to answer": {
                "intent": "playlist", "web_queries": ["calm 80s hits"],
                "ce_query": "calm hits"},
            "Pull song titles": {"tracks": []},
            "You are finishing a playlist": {"title": "x", "comment": "",
                                             "order": []},
            "You already searched": {"web_queries": []},
        })
        agent = _assistant(monkeypatch, db, llm=llm, sources={}, pages={})
        await agent.run("спокойные хиты 80х")
        assert not agent.sink.of("discography")
