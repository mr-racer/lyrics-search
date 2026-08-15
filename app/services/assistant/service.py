"""Assistant orchestrator — one entry point over the deterministic web agent.

Owns no business logic. It builds the run's config out of the request, hands the
message to :class:`agent.Assistant`, and turns whichever of the four result
contracts comes back into the payload the route serves.

**The callback problem.** The pipeline reports progress from wherever it happens
to be: the LLM calls are on the event loop, but the whole search and fetch phase
runs inside ``asyncio.to_thread``. Feeding both into one stream naively either
drops the thread-borne events or lets them overtake the terminal ``result``.
:class:`EventSink` solves it the way ``routes/recommend.py`` already does — one
``asyncio.Queue`` plus ``loop.call_soon_threadsafe`` for anything arriving
off-loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.services.assistant.humanize import clarify_labels, human

logger = logging.getLogger(__name__)

# How long the stream may stay silent before a keepalive line goes out. Behind
# the VPS nginx a long quiet stretch during an LLM call gets the connection torn
# down, and an LLM call here can be silent for a minute.
HEARTBEAT_SEC = 15.0

_CLARIFY_ORDER = ("lyrics_search", "audio_search", "playlist", "general")


class EventSink:
    """One queue fed from any thread, drained by the streaming route."""

    def __init__(self, lang: str | None = None):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.lang = lang
        self._loop = asyncio.get_running_loop()

    def put(self, item: dict) -> None:
        """Enqueue from the event loop thread."""
        self.queue.put_nowait(item)

    def _frame(self, event: dict) -> dict:
        """Normalise a pipeline event into one ``status`` frame.

        ``stage`` and ``human`` are stripped before the remaining fields reach
        :func:`human` — passing them through is what once made an event blow up
        with "human() got multiple values for argument 'stage'".
        """
        stage = event.get("stage") or event.get("type") or "status"
        fields = {k: v for k, v in event.items()
                  if k not in ("type", "stage", "human")}
        caption = event.get("human") or human(stage, self.lang, **fields)
        return {"type": "status", "stage": stage, "human": caption, **fields}

    async def emit(self, event: dict) -> None:
        """Async callback, for anything already on the loop."""
        self.put(self._frame(event))

    def on_status(self, event: dict) -> None:
        """Sync callback, safe from a worker thread.

        Anything not on the loop is hopped over with ``call_soon_threadsafe`` so
        ordering against the final ``result`` frame is preserved.
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


def _clarify_options(lang: str | None) -> list:
    labels = clarify_labels(lang)
    return [{"intent": i, "label": labels[i]} for i in _CLARIFY_ORDER]


async def run_assistant(req, *, search_service, qdrant, collection_name: str,
                        current_user, sink: EventSink | None = None) -> dict:
    """Run one message to completion and return the terminal payload.

    ``qdrant`` and ``current_user`` are part of the route's contract and are not
    used here: the agent reaches the vector store through ``search_service`` and
    the library through ``collection_name``, which is the only thing that decides
    what this run may see.
    """
    from app.services.assistant.agent import Assistant
    from app.services.assistant.config import AgentConfig
    from app.services.assistant.contracts import (AudioResult, GeneralResult,
                                                  LyricsResult, PlaylistResult)
    from app.services.assistant.events import AgentSink

    lang = req.lang
    cfg = AgentConfig(lang=lang, llm_base_url=req.llm_base_url,
                      llm_model=req.llm_model)
    # An explicit "на 20 треков" is the user's own cap and beats the default.
    if req.limit:
        cfg.default_target_count = req.limit
        cfg.clap_result_count = req.limit

    agent_sink = AgentSink(sink.on_status if sink is not None else None)
    agent = Assistant(collection_name, config=cfg, sink=agent_sink,
                      search_service=search_service)

    result = await agent.run(req.message, focus_fact=req.focus_fact,
                             subject_track_id=req.subject_track_id,
                             subject_artist_slug=req.subject_artist_slug,
                             forced_intent=req.intent)

    slots = _merge_slots(req.slots)
    if isinstance(result, LyricsResult):
        return _lyrics_payload(result, slots, lang)
    if isinstance(result, PlaylistResult):
        return _playlist_payload(result, slots, lang)
    if isinstance(result, AudioResult):
        return _audio_payload(result, slots, lang)
    if isinstance(result, GeneralResult):
        return _answer_payload(result, slots, lang, sink=sink,
                               collection_name=collection_name,
                               catalog=agent.catalog)
    logger.error("[assistant] unknown result type %s", type(result))
    return {"intent": None, "human": human("error", lang),
            "slots": slots.model_dump(), "clarify": _clarify_options(lang)}


# ── payload builders ────────────────────────────────────────────────────────


def _merge_slots(slots, **updates):
    """Carry the client's slots forward, overwriting only what this turn learned.

    The merge rule is unconditional on purpose: slots always carry, freshly
    resolved entities overwrite. That removes any need to decide "is this a
    follow-up?" — «ещё у этого артиста» simply finds no artist and falls back to
    whatever the last turn stored.
    """
    from app.domain.models import AssistantSlots

    base = (slots or AssistantSlots()).model_dump()
    for key, value in updates.items():
        if value is not None:
            base[key] = value
    return AssistantSlots(**base)


def _lyrics_payload(result, slots, lang: str) -> dict:
    best = result.best_hit
    track = best.track if best is not None else None
    slots = _merge_slots(slots, last_intent="lyrics_search",
                         last_track_id=(track.track_id if track else None),
                         last_song=result.song, last_artist=result.artist)
    return {
        "intent": "lyrics_search",
        "human": human("answer", lang),
        "slots": slots.model_dump(),
        "search": {
            "message": result.message,
            "song": result.song,
            "artist": result.artist,
            "confidence": result.confidence,
            "best_hit": best.model_dump() if best is not None else None,
            "hits": [h.model_dump() for h in result.hits],
            "attempts": 1,
            "classification": "text",
        },
    }


def _playlist_payload(result, slots, lang: str) -> dict:
    tracks = [_playlist_track(t) for t in result.tracks]
    slots = _merge_slots(slots, last_intent="playlist",
                         last_playlist_ids=[t["track_id"] for t in tracks] or None)
    return {
        "intent": "playlist",
        "human": human("select_done", lang, picked=len(tracks)),
        "slots": slots.model_dump(),
        "playlist": {"title": result.title, "steps": [], "tracks": tracks},
    }


def _audio_payload(result, slots, lang: str) -> dict:
    """The sound-alike list, rendered by the playlist card.

    It is a list to play and to save, which is what that card is for. The tracks
    carry no ``reason``: nothing wrote one, and having a model invent one is
    exactly what this branch exists to avoid.
    """
    tracks = []
    for hit in result.tracks:
        row = hit.track.model_dump()
        row["reason"] = None
        row["source_tool"] = "clap_search"
        row["score"] = float(getattr(hit, "score", 0.0) or 0.0)
        tracks.append(row)
    slots = _merge_slots(slots, last_intent="audio_search",
                         last_playlist_ids=[t["track_id"] for t in tracks] or None)
    return {
        "intent": "audio_search",
        "human": human("select_done", lang, picked=len(tracks)),
        "slots": slots.model_dump(),
        "playlist": {"title": result.title, "steps": [], "tracks": tracks},
    }


def _playlist_track(track) -> dict:
    """A matched library track as the playlist card expects it."""
    return {
        "track_id": track.track_id,
        "title": track.title,
        "artist": track.artist,
        "album": track.album,
        "year": track.year,
        "duration_sec": track.duration_sec,
        "file_path": track.file_path,
        "cover_art_path": track.cover_art_path,
        "reason": track.reason,
        "source_tool": "+".join(track.sources) or "web",
        # The vote, surfaced as the card's score: a track three pages named is a
        # different thing from one a single listicle did, and the number is the
        # only place that difference is visible.
        "score": round(track.weight, 2),
    }


def _answer_payload(result, slots, lang: str, *, sink=None,
                    collection_name: str = "", catalog=None) -> dict:
    if result.clarify is not None:
        # The only clarify the new pipeline raises is an abbreviation it could not
        # expand. There is nothing to route between, so it is surfaced as the
        # question it is rather than as a list of branches.
        question = result.clarify.question
        if sink is not None:
            sink.put({"type": "clarify", "human": question, "options": []})
        return {"intent": "general", "human": question,
                "slots": slots.model_dump(),
                "answer": {"answer": question, "grounded": False,
                           "iterations": 0, "evidence": [],
                           "notes": result.notes}}

    used = set(result.used)
    evidence = [{
        "n": e.n, "kind": e.kind, "text": e.text, "source": e.source,
        "url": e.url or None, "ce_prob": e.ce_prob, "used": e.n in used,
    } for e in result.evidence]

    subject = result.subject
    slots = _merge_slots(
        slots, last_intent="general",
        last_track_id=(subject.track_id if subject else None),
        last_artist=(subject.artist_name if subject else None),
        last_song=(subject.song_title if subject else None))

    return {
        "intent": "general",
        "human": human("answer", lang),
        "slots": slots.model_dump(),
        "answer": {
            "answer": result.answer,
            "grounded": result.grounded,
            "iterations": result.iterations,
            "evidence": evidence,
            "subject": _subject_ref(subject, collection_name, catalog),
            "focus_fact": result.focus_fact,
            "explained": result.explained,
            "follow_ups": result.follow_ups,
            "notes": result.notes,
        },
    }


def _subject_ref(subject, collection_name: str = "",
                 catalog=None) -> Optional[dict]:
    """The card header, or None when the answer had no library subject.

    None is the honest answer for a purely web-sourced reply, and the card is
    built to render without a header rather than to invent one.
    """
    if subject is None or not subject.resolved:
        return None
    if subject.song_title:
        cover = None
        if catalog is not None and subject.track_id:
            row = catalog.track(subject.track_id)
            cover = (row or {}).get("cover_art_path")
        return {"kind": "song", "title": subject.song_title,
                "subtitle": subject.artist_name,
                "artist_slug": subject.artist_slug,
                "track_id": subject.track_id, "image_path": cover}
    return {"kind": "artist", "title": subject.artist_name or "",
            "subtitle": None, "artist_slug": subject.artist_slug,
            "track_id": None,
            "image_path": _artist_image(subject.artist_slug, collection_name)}


def _artist_image(slug: Optional[str], collection_name: str) -> Optional[str]:
    """The artist's cached AudioDB photo, gated by this account's visibility."""
    if not slug or not collection_name:
        return None
    try:
        from app.resources.metadata_db import MetadataDB

        MetadataDB.init()
        row = MetadataDB.get_artist_audiodb(slug, collection_name) or {}
    except Exception:  # noqa: BLE001 — a missing photo is not an error
        return None
    return row.get("thumb_path") or row.get("cutout_path")
