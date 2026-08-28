"""Собрать корпус за три запроса — и не пустить в него чужое.

Веб перестал быть отдельной веткой с собственным поисковым клиентом и стал
источником внутри того же конвейера. Значит, всё, чем вики-путь защищался, обязано
работать и здесь: гейт кросс-энкодера до скачивания, он же по телу страницы, и
общий бюджет в один поиск на артиста.
"""

import pytest

from app.resources.mediawiki import _follow, _title_hops, probe_titles_batch
from app.resources.model_registry import ModelRegistry
from app.resources.models import ModelUnavailable
from app.services.assistant.config import AgentConfig
from app.services.assistant.contracts import Page, SearchHit
from app.services.bio_v2 import article as art
from app.services.bio_v2 import pipeline as bio2
from app.services.bio_v2 import retrieval as R
from app.services.bio_v2 import sources

pytestmark = pytest.mark.unit


def _hit(url, title, snippet):
    return SearchHit(url=url, title=title, snippet=snippet, source="web", rank=0)


class _Searcher:
    """Ровно то, что from_web зовёт у поисковика, плюс счётчик запросов."""

    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def web(self, query):
        self.queries.append(query)
        return list(self.hits)


class _Fetcher:
    def __init__(self, bodies=None):
        self.bodies = bodies or {}
        self.fetched = []

    async def fetch(self, url, *, source="web", title=""):
        self.fetched.append(url)
        return Page(url=url, title=title, source=source,
                    markdown=self.bodies.get(url, ""))


# ── батч-проба заголовков ────────────────────────────────────────────────────

class TestProbeTitlesBatch:
    def test_one_request_for_every_disambiguator(self, monkeypatch):
        """Ступень стоила 9-10 запросов на артиста. Теперь один."""
        calls = []

        class _Resp:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"query": {"pages": [
                    {"title": "Merk (musician)", "extract": "A producer."},
                ]}}

        def _get(url, **kw):
            calls.append(kw.get("params", {}))
            return _Resp()

        import httpx
        monkeypatch.setattr(httpx, "get", _get)

        out = probe_titles_batch("Merk", "en")

        assert len(calls) == 1, "проба снова ходит по одному запросу на суффикс"
        assert calls[0]["titles"].count("|") == 7, "не все суффиксы в запросе"
        assert calls[0]["exlimit"] == 20, "без exlimit придёт один extract"
        assert [c["title"] for c in out] == ["Merk (musician)"]

    def test_a_disambiguation_page_is_not_an_article(self, monkeypatch):
        class _Resp:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"query": {"pages": [
                    {"title": "Bullet (band)",
                     "pageprops": {"disambiguation": ""}},
                ]}}

        import httpx
        monkeypatch.setattr(httpx, "get", lambda url, **kw: _Resp())

        assert probe_titles_batch("Bullet", "en") == []

    def test_an_answer_under_a_redirected_title_is_still_found(self):
        """Батч отвечает под КАНОНИЧЕСКИМ заголовком, и без разбора хопов
        ответ не сопоставить с суффиксом, который его запросил."""
        hops = _title_hops({
            "normalized": [{"from": "aria (band)", "to": "Aria (band)"}],
            "redirects": [{"from": "Aria (band)", "to": "Aria (Russian band)"}],
        })

        assert _follow("aria (band)", hops) == "Aria (Russian band)"

    def test_a_hop_cycle_terminates(self):
        assert _follow("A", {"A": "B", "B": "A"}) in {"A", "B"}


# ── веб как источник ─────────────────────────────────────────────────────────

def _cross_encoder_down(query, docs):
    """The leg is down. It RAISES now — returning ``None`` was the
    contract that made a dead cross-encoder indistinguishable from an
    empty batch at every call site."""
    raise ModelUnavailable("cross_encoder", "load", "no reranker in this test")


class TestFromWeb:
    @pytest.fixture
    def cfg(self):
        return AgentConfig()

    async def test_the_gate_runs_before_anything_is_downloaded(self, cfg,
                                                               monkeypatch):
        """Читать дорого, судить дёшево: страница не о том артисте не должна
        стоить ни одного скачивания."""
        monkeypatch.setattr(
            ModelRegistry, "ce_probabilities",
            staticmethod(lambda q, docs: [0.9 if "band" in d else 0.02
                                          for d in docs]))
        fetcher = _Fetcher({"https://ok/": "# Sade\n\nEnglish band."})
        searcher = _Searcher([
            # Тот же артист в заголовке и совершенно не тот предмет — ровно то,
            # что фильтр по строке пропустил бы, а гейт отсекает.
            _hit("https://junk/", "Sade knives", "cutlery reviews"),
            _hit("https://ok/", "Sade", "English band formed in 1982."),
        ])

        await sources.from_web("Sade", cfg=cfg, fetcher=fetcher,
                               searcher=searcher, seed_bio=None)

        assert fetcher.fetched == ["https://ok/"], "скачали отсеянное гейтом"

    async def test_exactly_one_search_per_artist(self, cfg, monkeypatch):
        monkeypatch.setattr(ModelRegistry, "ce_probabilities",
                            staticmethod(lambda q, docs: [0.0] * len(docs)))
        searcher = _Searcher([_hit("https://x/", "x", "x")])

        await sources.from_web("Sade", cfg=cfg, fetcher=_Fetcher(),
                               searcher=searcher, seed_bio=None)

        assert len(searcher.queries) == 1

    async def test_without_a_cross_encoder_only_the_top_result_is_read(
            self, cfg, monkeypatch):
        """Гейта нет — значит нет и суждения. Непрогейченная выдача уже дарила
        новозеландскому музыканту четыре чужих «Грэмми»."""
        monkeypatch.setattr(ModelRegistry, "ce_probabilities",
                            staticmethod(_cross_encoder_down))
        fetcher = _Fetcher()
        searcher = _Searcher([_hit(f"https://{i}/", "t", "s") for i in range(4)])

        await sources.from_web("Sade", cfg=cfg, fetcher=fetcher,
                               searcher=searcher, seed_bio=None)

        assert fetcher.fetched == ["https://0/"]

    async def test_the_seed_alone_is_a_corpus(self, cfg, monkeypatch):
        """Ни статьи, ни выдачи — абзац из AudioDB это разница между
        биографией и пустой страницей."""
        monkeypatch.setattr(ModelRegistry, "ce_probabilities",
                            staticmethod(_cross_encoder_down))
        chunks, meta = await sources.from_web(
            "Sade", cfg=cfg, fetcher=_Fetcher(), searcher=_Searcher([]),
            seed_bio="Sade are an English band formed in London in 1982.")

        assert chunks, "сид не доехал до корпуса"
        assert meta["source_kind"] == "audiodb"

    async def test_a_search_that_raises_is_a_miss_not_a_crash(self, cfg):
        class _Broken:
            def web(self, query):
                raise RuntimeError("searxng down")

        chunks, meta = await sources.from_web(
            "Sade", cfg=cfg, fetcher=_Fetcher(), searcher=_Broken(),
            seed_bio=None)

        assert chunks == [] and meta.get("web_hits") == 0


# ── фасеты: запросы к корпусу вместо поисков ─────────────────────────────────

class TestFacetQueries:
    async def test_the_hardcoded_question_stays_first(self):
        async def ask(prompt, temperature=0.3):
            return '{"grammy": ["awards won"], "formed": ["formed in"]}'

        out = await bio2.facet_queries(ask, "Sade", "English")

        for name, question in bio2.FACETS.items():
            assert out[name][0] == question
        assert "awards won" in out["grammy"]

    @pytest.mark.parametrize("answer", ["not json at all", "{}", '{"grammy": 5}',
                                        '["a", "b"]'])
    async def test_garbage_falls_back_to_the_defaults(self, answer):
        async def ask(prompt, temperature=0.3):
            return answer

        out = await bio2.facet_queries(ask, "Sade", "English")

        assert out == {n: [q] for n, q in bio2.FACETS.items()}

    async def test_a_model_that_raises_costs_the_facets_nothing(self):
        async def ask(prompt, temperature=0.3):
            raise RuntimeError("llm down")

        out = await bio2.facet_queries(ask, "Sade", "English")

        assert out == {n: [q] for n, q in bio2.FACETS.items()}


class _Ranked:
    def __init__(self, index):
        self.index = index


class _FakeRetriever:
    """Отдаёт заранее заданный порядок на каждый запрос."""

    def __init__(self, by_query):
        self.by_query = by_query
        self.asked = []

    def search(self, query, *, min_prob=None, limit=None, **kw):
        self.asked.append((query, min_prob))
        for needle, order in self.by_query.items():
            if needle in query:
                return [_Ranked(i) for i in order]
        return []


class TestFacetChunksHybrid:
    def test_paraphrases_are_fused_not_concatenated(self):
        """RRF по РАНГАМ: чанк, который обе формулировки подняли высоко, обязан
        обойти чужого лидера одной из них."""
        retriever = _FakeRetriever({"first": [7, 3], "second": [3, 9]})

        out = R.facet_chunks_hybrid(retriever, "Sade", ["first", "second"])

        assert out[0] == 3

    def test_the_chunk_gate_is_applied(self):
        retriever = _FakeRetriever({"q": [1]})

        R.facet_chunks_hybrid(retriever, "Sade", ["q"])

        assert all(prob == R.CE_CHUNK_GATE for _, prob in retriever.asked)

    def test_nothing_above_the_gate_is_an_empty_answer(self):
        assert R.facet_chunks_hybrid(_FakeRetriever({}), "Sade", ["q"]) == []

    def test_no_retriever_is_not_a_crash(self):
        assert R.facet_chunks_hybrid(None, "Sade", ["q"]) == []


class TestFacetSource:
    """Откуда взят факт, помечается ПО ФАСЕТУ: один корпус держит и статью, и
    страницы, которыми закрыли её пробелы."""

    class _C:
        def __init__(self, source):
            self.source = source

    def test_only_the_article_is_wiki(self):
        chunks = [self._C("wikipedia"), self._C("wikipedia")]
        assert bio2._facet_source(chunks, [0, 1]) == "wiki"

    def test_one_web_passage_makes_it_web(self):
        chunks = [self._C("wikipedia"), self._C("web")]
        assert bio2._facet_source(chunks, [0, 1]) == "web"


class TestCorpusLanguage:
    """Запрос на языке корпуса, а не биографии: BM25 и sparse-нога лексические,
    и русский запрос по английской статье не совпадёт ни с чем."""

    def test_the_article_subdomain_decides(self):
        assert bio2._corpus_lang(
            "Ария", {"source_url": "https://en.wikipedia.org/wiki/Aria"}) == "English"

    def test_without_an_article_the_name_decides(self):
        assert bio2._corpus_lang("Ария", {}) == "Russian"
        assert bio2._corpus_lang("Sade", {}) == "English"


class TestArticleGateReuse:
    def test_sources_and_the_article_gate_agree_on_the_threshold(self):
        """Одно суждение об одном артисте не должно ехать от того, откуда текст."""
        assert "{artist}" in sources.RELEVANCE
        assert art.CE_ARTICLE_GATE == 0.55
