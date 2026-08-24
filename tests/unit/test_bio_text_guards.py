"""Что делает система, когда писать биографию не из чего.

Двадцать одна биография в проде была моделью, объясняющей, почему она ничего не
нашла — и все двадцать одна пришли из веб-фоллбэка. Проверяется не текст ответа,
а два факта, которыми владеет сам код: попросили сентинел — он пришёл; поиск
ничего не вернул — писать не из чего. Списка фраз здесь нет намеренно: он
требует новой строчки на каждую модель, язык и переформулировку, а выглядит как
проверка.
"""

from unittest.mock import patch

import pytest

from app.services import llm_web_search as lws
from app.services import text_quality as tq


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


# ── структурный гейт: искали и не нашли → биографии нет ──────────────────────

class _FakeRun:
    def __init__(self, output):
        self.output = output


class _FakeAgent:
    def __init__(self, output):
        self._output = output

    async def run(self, *a, **kw):
        return _FakeRun(self._output)


def _patched(output, stats):
    """Подменить фабрику агента: свой ответ и своя статистика поисков."""
    return patch.object(lws, "_create_agent",
                        return_value=(_FakeAgent(output), stats))


@pytest.mark.asyncio
async def test_no_bio_when_every_search_came_back_empty():
    """Ответ модели даже не читается: писать было не из чего."""
    plausible = ("DJ Zeph — американский продюсер из Лос-Анджелеса, "
                 "работающий в жанрах хип-хоп и электроника.")
    with _patched(plausible, {"searches": 3, "hits": 0}):
        got = await lws.web_research_bio(artist_name="DJ Zeph", lang="Russian")
    assert got == ""


@pytest.mark.asyncio
async def test_no_bio_when_the_agent_never_searched_at_all():
    """Ответ «из памяти» — это тот же вымысел, только без следов поиска."""
    with _patched("Некий артист играет инди-поп.", {"searches": 0, "hits": 0}):
        got = await lws.web_research_bio(artist_name="Al Be Back", lang="Russian")
    assert got == ""


@pytest.mark.asyncio
async def test_bio_survives_when_a_search_returned_something():
    with _patched("Настоящая биография.", {"searches": 2, "hits": 1}):
        got = await lws.web_research_bio(artist_name="Rooga", lang="Russian")
    assert got == "Настоящая биография."


@pytest.mark.asyncio
async def test_an_audiodb_seed_is_a_source_of_its_own():
    """С сидом агент вправе не искать вовсе — гейт не должен его убивать."""
    with _patched("Переписанная биография.", {"searches": 0, "hits": 0}):
        got = await lws.web_research_bio(
            artist_name="Sade", lang="Russian",
            seed_bio="Sade are an English band formed in London in 1982.")
    assert got == "Переписанная биография."


def test_the_no_results_marker_is_one_literal():
    """Гейт сравнивает строки, поэтому маркер должен быть один на модуль."""
    import inspect
    src = inspect.getsource(lws)
    assert src.count('"No results found"') == 1
