"""Intent routing for the unified assistant — GLiNER2, never the LLM.

Runs the already-loaded GLiNER2 singleton
(``fact_relations.extractor.get_model()``, ``fastino/gliner2-multi-v1``) over
the user's message: two classification passes whose scores are ensembled, plus
one pass for the artist/song spans and the requested count/era. ~0.5 s on CPU
and zero LLM tokens, replacing the old ``CLASSIFICATION_SYSTEM_PROMPT``
round-trip.

Why not regexes: they would need per-language phrase lists ("собери",
"подборка", "hits", "best of", …) maintained forever. GLiNER classifies
cross-lingually out of the box and — crucially — *cannot* return a label
outside the list it is given, so there is nothing to hallucinate.

Why not the LLM: a 12b model confuses «найди песню про дождь» with «собери
плейлист про дождь», and those are different pipelines with different UX.

Three design choices here were measured, not guessed — see
``tests/docker/test_assistant_routing.py`` for the fixture they were measured on:

1. **Classification gets its own schema.** Adding the entity and structure
   sections to the same schema dropped argmax accuracy from 81% to 70% and
   destroyed the score separation the gate depends on. Two passes cost ~200 ms
   and buy back both.
2. **Two label wordings, ensembled.** The terse wording (``_LABELS_A``) and the
   "the user is trying to…" wording (``_LABELS_B``) fail on different phrasings;
   normalised and summed they reach 89% argmax where A alone reaches 81%.
3. **The gate is a relative SHARE, not an absolute confidence.** With
   ``multi_label=True`` the scores are independent sigmoids and land anywhere in
   0.02–0.96 depending on phrasing, so no absolute threshold separates the
   classes. The winner's share of the total does: at ``MIN_SHARE`` the fixture
   routes 22/27 correctly with **zero** wrong branches, the other 5 asking.

Everything uncertain is resolved by CODE, not by prompting — see :func:`route`.
When the model is genuinely unsure the router returns ``intent=None`` and the
caller asks the user; guessing badly costs a whole wasted pipeline run, a tap
costs a second.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading

logger = logging.getLogger(__name__)

# ── Tuning knobs (calibrated on tests/docker/test_assistant_routing.py) ──────
# Share = top score / sum of all label scores. 0.45 was the knee of the curve:
# it keeps every correct route the ensemble can make while admitting no wrong
# ones. Raising it only converts correct routes into questions.
MIN_SHARE = 0.45
# Guards against dividing noise by noise on empty or nonsense input.
MIN_ABS_SCORE = 0.01
# A short message with no clear intent is almost always a follow-up
# («а побыстрее?», «ещё такого же») — reuse the previous intent instead of asking.
STICKY_MAX_WORDS = 5
# Near-zero on purpose: we want ALL label scores back so the share can be
# computed. multi_label=True is what makes GLiNER2 return the full scored list
# rather than just its argmax.
CLS_THRESHOLD = 0.01

# The closed label sets. Descriptive sentences beat one-word labels for
# zero-shot classification; the code maps them back to intents.
_LABELS_A = {
    "search": "find one specific song by its lyrics, words or sound",
    "playlist": "build a playlist or a collection of many songs",
    "facts": "learn facts, history or biography about an artist or a song",
}
_LABELS_B = {
    "search": "the user is trying to identify one particular track they have in mind",
    "playlist": "the user wants a set of several tracks assembled for them",
    "facts": "the user is asking a question and wants an explanation or information",
}

_lock = threading.Lock()
_schemas = None


def _get_schemas():
    """Build (once) the three schemas and cache them with their model.

    Cached at module level: the schema objects are tied to the process-wide
    model and rebuilding them per request is pure waste.
    """
    global _schemas
    if _schemas is not None:
        return _schemas
    with _lock:
        if _schemas is None:
            from app.services.fact_relations.extractor import get_model

            model = get_model()

            def _cls(labels):
                s = model.create_schema()
                s.classification("intent", list(labels.values()),
                                 multi_label=True, cls_threshold=CLS_THRESHOLD)
                return s

            ents = model.create_schema()
            ents.entities({
                "artist": "name of a music artist, band or performer",
                "song": "title of a specific song or album",
            })
            req = ents.structure("request")
            req.field("count", dtype="str",
                      description="how many songs the user asks for")
            req.field("era", dtype="str",
                      description="decade, year or year range mentioned")

            _schemas = (model, _cls(_LABELS_A), _cls(_LABELS_B), ents)
    return _schemas


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


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in scores.items()}


def _classify_sync(message: str) -> dict[str, float]:
    """Ensembled intent scores as shares summing to 1. Blocking — use to_thread.

    Each label set is normalised on its own before the sum, so a set that
    happens to fire hot on a phrasing cannot outvote the other purely on scale.
    """
    model, schema_a, schema_b, _ = _get_schemas()

    def one(schema, labels) -> dict[str, float]:
        inv = {v: k for k, v in labels.items()}
        try:
            raw = model.extract(message, schema, include_confidence=True)
        except TypeError:
            # Older signature without include_confidence — labels still come
            # back, just unscored; _scored_labels handles that shape.
            raw = model.extract(message, schema)
        scored = _scored_labels((raw or {}).get("intent"))
        return {inv[label]: score for label, score in scored if label in inv}

    raw_a = one(schema_a, _LABELS_A)
    raw_b = one(schema_b, _LABELS_B)
    if max([*raw_a.values(), *raw_b.values()], default=0.0) < MIN_ABS_SCORE:
        return {}
    norm_a, norm_b = _normalise(raw_a), _normalise(raw_b)
    combined = {
        intent: norm_a.get(intent, 0.0) + norm_b.get(intent, 0.0)
        for intent in set(norm_a) | set(norm_b)
    }
    return _normalise(combined)


def _entities_sync(message: str) -> dict:
    """Artist/song spans plus the count/era structure. Blocking — use to_thread."""
    model, _, _, schema = _get_schemas()
    try:
        return model.extract(message, schema, include_confidence=True)
    except TypeError:
        return model.extract(message, schema)


def _extract_sync(message: str) -> tuple[dict[str, float], dict]:
    """Both passes, off the event loop in one hop."""
    return _classify_sync(message), _entities_sync(message)


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
    winner's share >= MIN_SHARE        the ensemble's pick
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
        scores, ents = await asyncio.to_thread(_extract_sync, text)
    except Exception as exc:
        # GLiNER unavailable (model download failed, OOM…) — degrade to the
        # previous intent, else to search, which is the historical default.
        logger.warning("[assistant/router] GLiNER2 failed (%s) — falling back", exc)
        return AssistantRoute(intent=last_intent or "search",
                              source="sticky" if last_intent else "unclear")

    artist = _first_entity((ents or {}).get("entities"), "artist")
    song = _first_entity((ents or {}).get("entities"), "song")
    struct = (ents or {}).get("request")
    count = _parse_count(_structure_field(struct, "count"))
    era = _structure_field(struct, "era")

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_intent = ranked[0][0] if ranked else None
    share = ranked[0][1] if ranked else 0.0
    margin = (ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else share

    base = dict(confidence=round(share, 3), margin=round(margin, 3),
                artist=artist, song=song, count=count, era=era)

    logger.info(
        "[assistant/router] msg=%r → %s share=%.2f margin=%.2f artist=%r song=%r count=%s",
        text[:60], top_intent, share, margin, artist, song, count,
    )

    # A concrete number of tracks is a request for many songs, whatever the
    # model thought. Digit-based, so it holds across languages.
    if count is not None and count >= 2:
        return AssistantRoute(intent="playlist", source="count_override", **base)

    if top_intent and share >= MIN_SHARE:
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
