"""EventSink — the seam where three producers with two event vocabularies meet.

This is the one piece of the assistant that every branch flows through, and the
integration tests stub the executors out, so it needs direct coverage: a facts
event carrying BOTH ``type`` and ``stage`` reached production and crashed with
"human() got multiple values for argument 'stage'".
"""
from __future__ import annotations

import asyncio

from app.services.assistant.service import EventSink


def drain(sink: EventSink) -> list[dict]:
    out = []
    while not sink.queue.empty():
        out.append(sink.queue.get_nowait())
    return out


async def test_chat_style_event_names_its_stage_in_type():
    """``chat_search_service`` emits ``{"type": <stage>, …}``."""
    sink = EventSink("ru")
    await sink.emit({"type": "search", "found": 3, "queries": ["rain"]})
    (frame,) = drain(sink)
    assert frame["type"] == "status"
    assert frame["stage"] == "search"
    assert frame["found"] == 3
    assert "3" in frame["human"]


async def test_facts_style_event_carries_both_type_and_stage():
    """The regression: ``facts_executor`` emits type AND stage AND human.

    Both keys must be stripped before the rest is forwarded to ``human()``.
    """
    sink = EventSink("ru")
    await sink.emit({"type": "status", "stage": "collecting",
                     "human": "Нашёл 5 фактов в библиотеке", "found": 5})
    (frame,) = drain(sink)
    assert frame == {"type": "status", "stage": "collecting",
                     "human": "Нашёл 5 фактов в библиотеке", "found": 5}


async def test_a_supplied_caption_is_never_overwritten():
    sink = EventSink("ru")
    await sink.emit({"type": "status", "stage": "web_search",
                     "human": "custom caption", "query": "q"})
    (frame,) = drain(sink)
    assert frame["human"] == "custom caption"


async def test_playlist_style_event_names_its_stage_in_stage():
    """``ai_playlist`` emits ``{"stage": …}`` with no ``type`` and no caption."""
    sink = EventSink("ru")
    sink.on_status({"stage": "web_search", "query": "Kanye greatest hits"})
    (frame,) = drain(sink)
    assert frame["stage"] == "web_search"
    assert "Kanye greatest hits" in frame["human"]


async def test_unknown_stage_still_streams_without_a_caption():
    """A new producer stage must never break the stream — it just renders bare."""
    sink = EventSink("en")
    sink.on_status({"stage": "brand_new_stage", "whatever": 1})
    (frame,) = drain(sink)
    assert frame["stage"] == "brand_new_stage"
    assert frame["human"] == ""
    assert frame["whatever"] == 1


async def test_event_with_no_stage_at_all_degrades():
    sink = EventSink("en")
    sink.on_status({"query": "x"})
    (frame,) = drain(sink)
    assert frame["stage"] == "status"


async def test_worker_thread_events_reach_the_queue_in_order():
    """``playlist_agent`` tools run inside ``asyncio.to_thread`` and call
    ``on_status`` from there; without the loop hop those events are lost."""
    sink = EventSink("ru")

    def in_thread():
        sink.on_status({"stage": "web_search", "query": "a"})
        sink.on_status({"stage": "auto_matched", "found": 4})

    await asyncio.to_thread(in_thread)
    await asyncio.sleep(0)  # let call_soon_threadsafe callbacks run
    frames = drain(sink)
    assert [f["stage"] for f in frames] == ["web_search", "auto_matched"]
