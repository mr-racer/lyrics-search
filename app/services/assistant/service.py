"""Assistant orchestrator — routes one message and normalises the three
executors into a single NDJSON envelope.

Owns no business logic: ``search`` delegates to ``chat_search_service``,
``playlist`` to ``recsys_ai_service.ai_playlist``, ``facts`` to
``facts_executor``. What it *does* own is the plumbing those three don't share.

**The callback problem.** The two existing stacks report progress differently:
``chat_search_service`` takes an *async* ``emit``; ``ai_playlist`` takes a
*sync* ``on_status`` that its playlist-agent tools may call from a worker thread
(they run inside ``asyncio.to_thread``). Feeding both into one stream naively
either drops the thread-borne events or lets them overtake the terminal
``result``. :class:`EventSink` solves it the way ``routes/recommend.py`` already
does — one ``asyncio.Queue`` plus ``loop.call_soon_threadsafe`` for anything
arriving off-loop.
"""

from __future__ import annotations

import asyncio
import logging

from app.services.assistant import facts_executor, reason_gate, router as intent_router
from app.services.assistant.humanize import clarify_labels, human

logger = logging.getLogger(__name__)

# How long the stream may stay silent before a keepalive line goes out. The SSE
# chat had a heartbeat; the NDJSON playlist route did not, and behind the VPS
# nginx a long quiet stretch during an LLM call gets the connection torn down.
HEARTBEAT_SEC = 15.0

_CLARIFY_ORDER = ("search", "playlist", "facts")


class EventSink:
    """One queue fed by both callback styles, drained by the streaming route.

    ``emit`` is the async form the search engine expects; ``on_status`` is the
    sync form the playlist pipeline expects and is safe to call from any thread.
    """

    def __init__(self, lang: str | None = None):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.lang = lang
        self._loop = asyncio.get_running_loop()

    def put(self, item: dict) -> None:
        """Enqueue from the event loop thread."""
        self.queue.put_nowait(item)

    def _frame(self, event: dict) -> dict:
        """Normalise any producer's event into one ``status`` frame.

        Two vocabularies arrive here: ``chat_search_service`` names the stage in
        ``type`` (``{"type": "search", "found": 3}``), while the playlist and
        facts branches name it in ``stage``. Either may already carry a
        ``human`` caption. Both keys are stripped before the remaining fields
        are handed to :func:`human` — passing them through is what made a facts
        event blow up with "human() got multiple values for argument 'stage'".
        """
        stage = event.get("stage") or event.get("type") or "status"
        fields = {k: v for k, v in event.items()
                  if k not in ("type", "stage", "human")}
        caption = event.get("human") or human(stage, self.lang, **fields)
        return {"type": "status", "stage": stage, "human": caption, **fields}

    async def emit(self, event: dict) -> None:
        """Async callback for ``chat_search_service`` and ``facts_executor``."""
        self.put(self._frame(event))

    def on_status(self, event: dict) -> None:
        """Sync callback for ``ai_playlist`` / ``playlist_agent``.

        May be invoked from a worker thread — anything not on the loop is
        hopped over with ``call_soon_threadsafe`` so ordering against the final
        ``result`` frame is preserved.
        """
        item = self._frame(event)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self.queue.put_nowait(item)
        else:
            self._loop.call_soon_threadsafe(self.queue.put_nowait, item)


def _clarify_options(lang: str | None) -> list[dict]:
    labels = clarify_labels(lang)
    return [{"intent": i, "label": labels[i]} for i in _CLARIFY_ORDER]


async def run_assistant(req, *, search_service, qdrant, collection_name: str,
                        current_user, sink: EventSink | None = None) -> dict:
    """Route ``req`` and run the matching executor to completion.

    Returns the terminal payload — the same dict the non-streaming endpoint
    serves and the streaming one ships as its final ``result`` frame. Progress
    reaches the client through ``sink`` while this runs.
    """
    from app.domain.models import AssistantSlots

    lang = req.lang
    slots = req.slots or AssistantSlots()

    def _say(item: dict) -> None:
        if sink is not None:
            sink.put(item)

    # ── route (LLM reads the sentence, GLiNER pulls the spans) ──
    # The last user turn goes along so a modifier («а побыстрее?») is recognised
    # as one; the structural state the executors need lives in `slots`.
    last_message = next(
        (m.content for m in reversed(req.history or []) if getattr(m, "role", "") == "user"),
        None,
    )
    route = await intent_router.route(
        req.message, slots, explicit_intent=req.intent, last_message=last_message,
        llm_base_url=req.llm_base_url, llm_model=req.llm_model,
    )

    if route.intent is None:
        # Refusing to guess is a feature: a wrong branch costs the user a whole
        # wasted pipeline run, one tap costs a second.
        options = _clarify_options(lang)
        _say({"type": "clarify", "human": human("clarify", lang), "options": options})
        return {
            "intent": None,
            "human": human("clarify", lang),
            "slots": intent_router.merge_slots(slots, None).model_dump(),
            "clarify": options,
        }

    _say({"type": "route", "intent": route.intent,
          "confidence": route.confidence,
          "human": human("route", lang, intent=route.intent)})

    merged = intent_router.merge_slots(slots, route)

    if route.intent == "search":
        return await _run_search(req, route, merged, search_service, current_user, sink, lang)
    if route.intent == "playlist":
        return await _run_playlist(req, route, merged, search_service, qdrant,
                                   collection_name, sink, lang)
    return await _run_facts(req, route, merged, qdrant,
                            collection_name, sink, lang)


async def _run_search(req, route, slots, search_service, current_user, sink, lang) -> dict:
    """Delegate to the agentic lyrics/audio search engine.

    ``planner_enabled`` is forced on: the intent is already known, so the old
    ``CLASSIFICATION_SYSTEM_PROMPT`` round-trip would be a wasted LLM call, and
    the planner is what produces filters and CLAP-ready queries.
    """
    from app.domain.models import ChatRequest
    from app.services.chat_search_service import run_chat_search

    chat_req = ChatRequest(
        message=req.message,
        history=req.history,
        auto_mode=True,
        planner_enabled=True,
        llm_base_url=req.llm_base_url,
        llm_model=req.llm_model,
        lang=req.lang,
    )
    result = await run_chat_search(
        chat_req, search_service, current_user,
        emit=(sink.emit if sink is not None else None),
    )
    best = result.get("best_hit") or {}
    track = (best.get("track") or {}) if isinstance(best, dict) else {}
    slots = intent_router.merge_slots(
        slots, None,
        last_track_id=track.get("track_id"),
        last_song=result.get("song"),
        last_artist=result.get("artist"),
    )
    return {
        "intent": "search",
        "human": human("answer", lang),
        "slots": slots.model_dump(),
        "search": result,
    }


async def _run_playlist(req, route, slots, search_service, qdrant,
                        collection_name, sink, lang) -> dict:
    """Delegate to the plan→execute→select playlist pipeline (unchanged)."""
    from app.services import recsys_ai_service

    result = await recsys_ai_service.ai_playlist(
        search_service=search_service,
        qdrant_client=qdrant,
        collection_name=collection_name,
        prompt=req.message,
        lang=req.lang,
        # An explicit "на 20 треков" is the user's own cap and beats the default.
        limit=route.count if (route.count and 1 <= route.count <= 40) else req.limit,
        llm_base_url=req.llm_base_url,
        llm_model=req.llm_model,
        on_status=(sink.on_status if sink is not None else None),
    )
    payload = recsys_ai_service.build_playlist_response(result)
    ids = [t.track_id for t in payload.tracks]
    slots = intent_router.merge_slots(slots, None, last_playlist_ids=ids or None)
    return {
        "intent": "playlist",
        "human": human("select_done", lang, picked=len(ids)),
        "slots": slots.model_dump(),
        # Filler reasons never reach the card — see reason_gate for why the
        # check lives here and not in the (frozen) playlist pipeline.
        "playlist": reason_gate.gate_playlist(payload.model_dump(mode="json")),
    }


async def _run_facts(req, route, slots, qdrant,
                     collection_name, sink, lang) -> dict:
    """Run the grounded facts branch; may end in a ``disambiguate`` frame."""
    payload, options = await facts_executor.run(
        qdrant=qdrant,
        collection_name=collection_name,
        message=req.message,
        route=route,
        slots=slots,
        lang=req.lang,
        subject_track_id=req.subject_track_id,
        subject_artist_slug=req.subject_artist_slug,
        now_playing_track_id=req.now_playing_track_id,
        focus_fact=req.focus_fact,
        llm_base_url=req.llm_base_url,
        llm_model=req.llm_model,
        emit=(sink.emit if sink is not None else None),
    )

    if options:
        subject_options = [
            {
                "kind": o["kind"],
                "title": o["title"],
                "subtitle": o.get("subtitle"),
                "track_id": o.get("track_id"),
                "artist_slug": o.get("artist_slug"),
                "cover_art_path": o.get("image_path"),
            }
            for o in options
        ]
        if sink is not None:
            sink.put({"type": "disambiguate", "human": human("disambiguate", lang),
                      "subject_options": subject_options})
        return {
            "intent": "facts",
            "human": human("disambiguate", lang),
            "slots": slots.model_dump(),
            "disambiguate": subject_options,
        }

    if payload is None:
        ru = (lang or "en").lower().startswith("ru")
        return {
            "intent": "facts",
            "human": human("answer", lang),
            "slots": slots.model_dump(),
            "facts": {
                "subject_kind": "song",
                "subject_title": req.message[:80],
                "answer": ("Не нашёл в твоей библиотеке, о чём речь. Уточни название "
                           "трека или имя артиста."
                           if ru else
                           "I couldn't find what you mean in your library. Try naming "
                           "the track or the artist."),
                "grounded": False,
                "focus_fact": req.focus_fact,
                "explained": False if req.focus_fact else None,
                "items": [],
                "related_tracks": [],
            },
        }

    slots = intent_router.merge_slots(
        slots, None,
        last_track_id=payload.get("track_id"),
        last_artist=(payload.get("subject_title")
                     if payload.get("subject_kind") == "artist" else None),
    )
    return {
        "intent": "facts",
        "human": human("answer", lang),
        "slots": slots.model_dump(),
        "facts": payload,
    }
