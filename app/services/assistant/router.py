"""Intent routing for the unified assistant — GLiNER2, never the LLM.

One forward pass over the already-loaded GLiNER2 singleton
(``fact_relations.extractor.get_model()``, ``fastino/gliner2-multi-v1``)
returns three things at once: the intent label, the artist/song spans, and a
small structure holding the requested track count and era. ~200 ms on CPU and
zero LLM tokens, replacing the old ``CLASSIFICATION_SYSTEM_PROMPT`` round-trip.

Why not regexes: they would need per-language phrase lists ("собери", "подборка",
"hits", "best of", …) maintained forever. GLiNER classifies cross-lingually out
of the box and — crucially — *cannot* return a label outside the list it is
given, so there is nothing to hallucinate.

Why not the LLM: a 12b model confuses «найди песню про дождь» with «собери
плейлист про дождь», and those are different pipelines with different UX.

Everything uncertain is resolved by CODE, not by prompting: see
:func:`route` for the threshold / stickiness / override table. When the model is
genuinely unsure the router returns ``intent=None`` and the caller asks the user
— guessing badly is worse than one extra tap.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading

logger = logging.getLogger(__name__)

# ── Tuning knobs ─────────────────────────────────────────────────────────────
# Below MIN_CONFIDENCE, or with less than MIN_MARGIN between the top two labels,
# we refuse to guess and ask. Calibrated against tests/docker/test_assistant_routing.py.
MIN_CONFIDENCE = 0.45
MIN_MARGIN = 0.15
# A short message with no clear intent is almost always a follow-up
# («а побыстрее?», «ещё такого же») — reuse the previous intent instead of asking.
STICKY_MAX_WORDS = 5
# cls_threshold is deliberately near-zero: we want ALL label scores back so the
# margin can be computed. multi_label=True is what makes GLiNER2 return the full
# scored list rather than just its argmax.
CLS_THRESHOLD = 0.05

# The closed label set. Descriptive sentences beat one-word labels for zero-shot
# classification, so the model sees these and the code maps them back.
_LABEL_SEARCH = "find one specific song by its lyrics, words or sound"
_LABEL_PLAYLIST = "build a playlist or a collection of many songs"
_LABEL_FACTS = "learn facts, history or biography about an artist or a song"

_LABELS = [_LABEL_SEARCH, _LABEL_PLAYLIST, _LABEL_FACTS]
_LABEL_TO_INTENT = {
    _LABEL_SEARCH: "search",
    _LABEL_PLAYLIST: "playlist",
    _LABEL_FACTS: "facts",
}

_lock = threading.Lock()
_schema = None


def _get_schema():
    """Build (once) the combined classification + entities + structure schema.

    Cached at module level: the schema object is tied to the process-wide model
    and rebuilding it per request is pure waste.
    """
    global _schema
    if _schema is not None:
        return _schema
    with _lock:
        if _schema is None:
            from app.services.fact_relations.extractor import get_model

            model = get_model()
            schema = model.create_schema()
            schema.classification(
                "intent", _LABELS, multi_label=True, cls_threshold=CLS_THRESHOLD,
            )
            schema.entities({
                "artist": "name of a music artist, band or performer",
                "song": "title of a specific song or album",
            })
            req = schema.structure("request")
            req.field("count", dtype="str",
                      description="how many songs the user asks for")
            req.field("era", dtype="str",
                      description="decade, year or year range mentioned")
            _schema = (model, schema)
    return _schema


# ── Output parsing (defensive: GLiNER2 shapes vary with include_confidence) ──


def _scored_labels(raw: object) -> list[tuple[str, float]]:
    """Normalise the ``intent`` value into ``[(label, score), …]``, best first.

    Handles every shape the library can return: a bare string, a
    ``{"label", "confidence"}`` dict, or a list of either.
    """
    def one(item: object) -> tuple[str, float] | None:
        if isinstance(item, str):
            return (item, 1.0)
        if isinstance(item, dict):
            label = item.get("label") or item.get("text")
            if not label:
                return None
            score = item.get("confidence", item.get("score", 0.0))
            try:
                return (str(label), float(score))
            except (TypeError, ValueError):
                return (str(label), 0.0)
        return None

    items = raw if isinstance(raw, list) else [raw]
    out = [p for p in (one(i) for i in items) if p is not None]
    out.sort(key=lambda p: p[1], reverse=True)
    return out


def _first_entity(raw: object, label: str) -> str | None:
    """Highest-scoring span for ``label`` from the entities section, or None."""
    ent_map = raw if isinstance(raw, dict) else {}
    items = ent_map.get(label) or []
    best: tuple[float, str] | None = None
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict):
            text = (it.get("text") or "").strip()
            try:
                score = float(it.get("confidence", it.get("score", 0.0)) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
        else:
            text, score = str(it).strip(), 1.0
        if text and (best is None or score > best[0]):
            best = (score, text)
    return best[1] if best else None


def _structure_field(raw: object, field: str) -> str | None:
    """Read one field out of the ``request`` structure section."""
    node = raw
    if isinstance(node, list):
        node = node[0] if node else None
    if not isinstance(node, dict):
        return None
    val = node.get(field)
    if isinstance(val, dict):
        val = val.get("text") or val.get("value")
    if isinstance(val, list):
        val = val[0] if val else None
        if isinstance(val, dict):
            val = val.get("text") or val.get("value")
    text = str(val).strip() if val is not None else ""
    return text or None


def _parse_count(text: str | None) -> int | None:
    """Digits out of a count span ("на 20 треков" → 20). Not a language rule —
    a digit is a digit in every language we serve, so nothing to maintain."""
    if not text:
        return None
    m = re.search(r"\d+", text)
    if not m:
        return None
    try:
        n = int(m.group())
    except ValueError:
        return None
    return n if 1 <= n <= 200 else None


def _extract_sync(message: str) -> dict:
    """The blocking GLiNER2 call. Always runs inside ``asyncio.to_thread``."""
    model, schema = _get_schema()
    try:
        return model.extract(message, schema, include_confidence=True)
    except TypeError:
        # Older signature without include_confidence — labels still come back,
        # just unscored; _scored_labels handles that shape.
        return model.extract(message, schema)


# ── Public API ───────────────────────────────────────────────────────────────


async def route(
    message: str,
    slots=None,
    *,
    explicit_intent: str | None = None,
):
    """Decide which executor handles ``message``.

    Returns an :class:`~app.domain.models.AssistantRoute`. ``intent=None`` means
    "ask the user" — the caller emits a ``clarify`` frame.

    Decision table (all of it code, none of it prompting):

    ================================  ==========================================
    condition                          outcome
    ================================  ==========================================
    ``explicit_intent`` set            used verbatim, GLiNER not called at all
                                       (this is the reply to a ``clarify`` frame)
    count >= 2 recognised              hard override → ``playlist``
    conf >= MIN_CONFIDENCE and
    margin >= MIN_MARGIN               the model's pick
    short message + known last_intent  sticky: reuse it (follow-ups)
    otherwise                          ``None`` → clarify
    ================================  ==========================================
    """
    from app.domain.models import AssistantRoute

    last_intent = getattr(slots, "last_intent", None) if slots else None

    if explicit_intent in ("search", "playlist", "facts"):
        return AssistantRoute(intent=explicit_intent, confidence=1.0, margin=1.0,
                              source="explicit")

    text = (message or "").strip()
    if not text:
        return AssistantRoute(intent=last_intent, source="sticky" if last_intent else "unclear")

    try:
        raw = await asyncio.to_thread(_extract_sync, text)
    except Exception as exc:
        # GLiNER unavailable (model download failed, OOM…) — degrade to the
        # previous intent, else to search, which is the historical default.
        logger.warning("[assistant/router] GLiNER2 failed (%s) — falling back", exc)
        return AssistantRoute(intent=last_intent or "search",
                              source="sticky" if last_intent else "unclear")

    scored = _scored_labels((raw or {}).get("intent"))
    artist = _first_entity((raw or {}).get("entities"), "artist")
    song = _first_entity((raw or {}).get("entities"), "song")
    struct = (raw or {}).get("request")
    count = _parse_count(_structure_field(struct, "count"))
    era = _structure_field(struct, "era")

    top_intent = _LABEL_TO_INTENT.get(scored[0][0]) if scored else None
    confidence = scored[0][1] if scored else 0.0
    margin = (scored[0][1] - scored[1][1]) if len(scored) >= 2 else confidence

    base = dict(confidence=round(confidence, 3), margin=round(margin, 3),
                artist=artist, song=song, count=count, era=era)

    logger.info(
        "[assistant/router] msg=%r → %s conf=%.2f margin=%.2f artist=%r song=%r count=%s",
        text[:60], top_intent, confidence, margin, artist, song, count,
    )

    # A concrete number of tracks is a request for many songs, whatever the
    # model thought. Digit-based, so it holds across languages.
    if count is not None and count >= 2:
        return AssistantRoute(intent="playlist", source="count_override", **base)

    if top_intent and confidence >= MIN_CONFIDENCE and margin >= MIN_MARGIN:
        return AssistantRoute(intent=top_intent, source="gliner", **base)

    # Unsure, but the message is a short follow-up and we know what came before.
    if last_intent and len(text.split()) <= STICKY_MAX_WORDS:
        return AssistantRoute(intent=last_intent, source="sticky", **base)

    return AssistantRoute(intent=None, source="unclear", **base)


def merge_slots(slots, route, **updates):
    """Carry slots forward, overwriting with anything freshly extracted.

    Unconditional by design: there is no "is this a follow-up?" decision to get
    wrong. «ещё у этого артиста» simply extracts no artist, so ``last_artist``
    survives and the executor uses it.
    """
    from app.domain.models import AssistantSlots

    merged = (slots.model_dump() if slots is not None else AssistantSlots().model_dump())
    if route is not None:
        if route.intent:
            merged["last_intent"] = route.intent
        if route.artist:
            merged["last_artist"] = route.artist
        if route.song:
            merged["last_song"] = route.song
    for key, value in updates.items():
        # None means "nothing new this turn" — never wipe a good slot with it.
        if value is not None:
            merged[key] = value
    return AssistantSlots(**merged)
