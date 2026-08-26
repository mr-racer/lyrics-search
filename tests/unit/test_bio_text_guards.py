"""Что делает система, когда писать биографию не из чего.

Двадцать одна биография в проде была моделью, объясняющей, почему она ничего не
нашла — и все двадцать одна пришли из веб-фоллбэка, которого больше нет.
Проверяется не текст ответа, а два факта, которыми владеет сам код: попросили
сентинел — он пришёл; в корпусе нет ни одного пассажа — писать не из чего.
Списка фраз здесь нет намеренно: он требует новой строчки на каждую модель,
язык и переформулировку, а выглядит как проверка.
"""

import pytest

from app.services import llm_web_search as lws
from app.services import text_quality as tq
from app.services.bio_v2 import pipeline as bio2


# ── сентинел ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "NO_DATA",
    "  NO_DATA  ",
    "NO_DATA.",
    '"NO_DATA"',
    "**NO_DATA**",
    "",
    "   ",
])
def test_sentinel_and_emptiness_are_a_refusal(text):
    assert tq.is_refusal(text) is True


@pytest.mark.parametrize("text", [
    "The Prodigy — английская группа электронной музыки, основанная в 1990 году.",
    # Строки, о которые спотыкался прежний список словоформ: настоящая
    # биография вправе сказать, что у группы что-то не вышло.
    "Группе не удалось повторить успех дебюта, и второй альбом провалился.",
    "Записи ранних концертов не сохранились: их никто не вёл.",
    "NO_DATA было рабочим названием их четвёртого альбома.",
])
def test_a_real_biography_is_not_a_refusal(text):
    assert tq.is_refusal(text) is False


# ── структурный гейт: источника нет → биографии нет ──────────────────────────
#
# Гейт переехал вместе с веткой. Раньше он стоял в агенте и считал, вернул ли
# хоть один поиск хоть что-нибудь; теперь считать нечего — либо в корпусе есть
# пассажи, либо биографию писать не из чего, и это же условие решает, тратить
# ли единственный веб-поиск.


class _Chunk:
    """Ровно то, что build трогает у чанка: текст и вид источника."""

    def __init__(self, text, source="wikipedia"):
        self.text = text
        self.source = source


class _Retriever:
    """Индекс, который умеет ровно одно — принять дошитые документы."""

    def __init__(self, docs):
        self.docs = list(docs)

    def extend(self, docs):
        self.docs += list(docs)
        return len(docs)


def _stub_sources(monkeypatch, *, wiki=(), web=(), bio_when):
    """Конвейер над подменёнными источниками: без сети, без моделей, без LLM.

    ``bio_when(chunks) -> str`` — что напишет write_bio на данном корпусе.
    Возвращает журнал вызовов, чтобы тест мог спросить, ходили ли в веб.
    """
    log = {"web_calls": 0, "write_calls": []}

    async def _from_wikipedia(artist, **kw):
        if wiki:
            return list(wiki), {"source_kind": "wikipedia",
                                "source_url": "https://en.wikipedia.org/wiki/X"}
        return [], {"error": "no wikipedia article passed the gate"}

    async def _from_web(artist, **kw):
        log["web_calls"] += 1
        if web:
            return list(web), {"source_kind": "web", "source_url": "https://x/"}
        return [], {"web_hits": 0}

    async def _write_bio(ask, artist, chunks, retriever, **kw):
        log["write_calls"].append(len(chunks))
        return bio_when(chunks), {}

    async def _read_facets(*a, **kw):
        return {}

    monkeypatch.setattr(bio2.sources, "from_wikipedia", _from_wikipedia)
    monkeypatch.setattr(bio2.sources, "from_web", _from_web)
    monkeypatch.setattr(bio2, "write_bio", _write_bio)
    monkeypatch.setattr(bio2, "read_facets", _read_facets)
    monkeypatch.setattr(bio2.R, "build_index",
                        lambda chunks: _Retriever(c.text for c in chunks))
    monkeypatch.setattr(bio2.R, "sentence_index", lambda chunks: ([], []))
    return log


async def _build(**kw):
    async def ask(prompt, temperature=0.3):
        return "{}"
    return await bio2.build(ask, "Some Artist", searcher=object(), **kw)


@pytest.mark.asyncio
async def test_no_bio_when_no_source_yielded_a_passage(monkeypatch):
    """Ни статьи, ни веба — правдоподобный абзац писать не из чего."""
    log = _stub_sources(monkeypatch, bio_when=lambda chunks: "Правдоподобно.")

    got = await _build()

    assert got.get("bio", "") == ""
    assert got["error"]
    assert log["write_calls"] == [], "писали при пустом корпусе"


@pytest.mark.asyncio
async def test_web_pages_are_a_source_like_the_article(monkeypatch):
    """Артист без статьи обслуживается ТЕМ ЖЕ конвейером, а не вторым."""
    log = _stub_sources(monkeypatch, web=[_Chunk("web body", "web")],
                        bio_when=lambda chunks: "Настоящая биография.")

    got = await _build()

    assert got["bio"] == "Настоящая биография."
    assert got["facets"]["source_kind"] == "web"
    assert log["web_calls"] == 1


@pytest.mark.asyncio
async def test_web_search_is_not_spent_when_the_article_answered(monkeypatch):
    """Бюджет ленивый: статья написалась — в сеть больше не ходим."""
    log = _stub_sources(monkeypatch, wiki=[_Chunk("article body")],
                        bio_when=lambda chunks: "Из статьи.")

    got = await _build()

    assert got["bio"] == "Из статьи."
    assert got["facets"]["source_kind"] == "wikipedia"
    assert log["web_calls"] == 0, "потратили веб-поиск впустую"


@pytest.mark.asyncio
async def test_article_that_clears_no_gate_still_reaches_the_web(monkeypatch):
    """Статья есть, но писать из неё нечего — это второй повод потратить веб."""
    log = _stub_sources(
        monkeypatch, wiki=[_Chunk("article body")],
        web=[_Chunk("web body", "web")],
        bio_when=lambda chunks: "Дописано." if len(chunks) > 1 else "")

    got = await _build()

    assert got["bio"] == "Дописано."
    assert log["web_calls"] == 1
    assert log["write_calls"] == [1, 2], "второй заход шёл не по общему корпусу"


@pytest.mark.asyncio
async def test_an_audiodb_seed_is_a_source_of_its_own(monkeypatch):
    """Сид AudioDB приходит из from_web и один держит корпус."""
    log = _stub_sources(monkeypatch, web=[_Chunk("seed text", "web")],
                        bio_when=lambda chunks: "Переписанная биография.")

    got = await _build(seed_bio="Sade are an English band formed in 1982.")

    assert got["bio"] == "Переписанная биография."
    assert log["web_calls"] == 1


def test_the_no_results_marker_is_one_literal():
    """Гейт сравнивает строки, поэтому маркер должен быть один на модуль."""
    import inspect
    src = inspect.getsource(lws)
    assert src.count('"No results found"') == 1
