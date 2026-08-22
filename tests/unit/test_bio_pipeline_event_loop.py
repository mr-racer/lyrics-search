"""Писать биографию нельзя ценой всего сервиса.

Задача биографий живёт в `asyncio.create_task`, то есть на главном event loop.
Всё, что она делает синхронно — поход в Википедию, кросс-энкодер, сборка
dense+sparse индекса по кускам статьи, — останавливает uvicorn целиком: пока
корутина не отдаёт управление, не обслуживается НИ ОДИН запрос.

Прод 2026-08-22: в логе паузы по 18-19 секунд без единой строки, обрывающиеся
на `[retrieval] indexed 53 docs`, и следом залп из накопившихся ответов.
Со стороны выглядит как «подвис Qdrant» — при том что Qdrant в этот момент
на 0,1% CPU.
"""
import asyncio
import time
from types import SimpleNamespace

import pytest

from app.services.bio_v2 import pipeline

pytestmark = pytest.mark.unit

SLOW = 0.3          # столько «считает» подменённая тяжёлая работа
TICK = 0.01         # шаг пульса


class _Page:
    ok = True
    markdown = "статья"
    error = None


class _Fetcher:
    async def fetch(self, url, **kw):
        return _Page()


async def _ask(prompt, temperature=0.0):
    return "текст"


@pytest.fixture
def stubbed(monkeypatch):
    """Весь конвейер — заглушки; тяжёлой остаётся ровно одна ступень."""
    monkeypatch.setattr(pipeline.art, "find",
                        lambda artist, **kw: ({"url": "https://en.wikipedia.org/wiki/X",
                                               "title": "X"}, []))
    monkeypatch.setattr(pipeline.R, "chunk_page",
                        lambda page, cfg=None: [SimpleNamespace(text="кусок")])
    monkeypatch.setattr(pipeline.R, "sentence_index", lambda chunks: ())

    async def _bio(*a, **kw):
        return "биография", {}

    async def _facets(*a, **kw):
        return {}

    monkeypatch.setattr(pipeline, "write_bio", _bio)
    monkeypatch.setattr(pipeline, "read_facets", _facets)


async def _pulse_while(coro):
    """Запустить coro, считая тики параллельного пульса."""
    ticks = 0

    async def beat():
        nonlocal ticks
        while True:
            await asyncio.sleep(TICK)
            ticks += 1

    hb = asyncio.create_task(beat())
    try:
        result = await coro
    finally:
        hb.cancel()
    return result, ticks


async def test_index_build_does_not_freeze_the_loop(stubbed, monkeypatch):
    """Сборка индекса по статье — самая долгая синхронная ступень."""
    def slow_build_index(chunks):
        time.sleep(SLOW)
        return object()

    monkeypatch.setattr(pipeline.R, "build_index", slow_build_index)

    result, ticks = await _pulse_while(pipeline.build(_ask, "X", fetcher=_Fetcher()))

    assert result.get("bio") == "биография"
    assert ticks >= SLOW / TICK / 2, (
        f"пульс тикнул всего {ticks} раз за {SLOW} c — event loop стоял")


async def test_article_search_does_not_freeze_the_loop(stubbed, monkeypatch):
    """Поиск статьи ходит в сеть и гоняет кросс-энкодер — тоже синхронно."""
    monkeypatch.setattr(pipeline.R, "build_index", lambda chunks: object())

    def slow_find(artist, **kw):
        time.sleep(SLOW)
        return {"url": "https://en.wikipedia.org/wiki/X", "title": "X"}, []

    monkeypatch.setattr(pipeline.art, "find", slow_find)

    _, ticks = await _pulse_while(pipeline.build(_ask, "X", fetcher=_Fetcher()))

    assert ticks >= SLOW / TICK / 2, (
        f"пульс тикнул всего {ticks} раз за {SLOW} c — event loop стоял")
