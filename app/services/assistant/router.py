"""Intent routing for the unified assistant — an LLM decision, GLiNER spans.

Two models, each doing what it is actually good at:

* **The LLM picks the intent** (:mod:`app.services.assistant.intent_llm`). It
  reads the sentence, so it separates the pairs that are only separable by
  meaning: «поставь Creep» (search) from «поставь что-нибудь из Radiohead»
  (playlist), «найди Runaway» (search) from «кто спродюсировал Runaway» (facts).
* **GLiNER2 pulls the spans** — artist, song, requested count, era — from the
  already-loaded singleton (``fact_relations.extractor.get_model()``,
  ``fastino/gliner2-multi-v1``). ~200 ms on CPU, zero tokens, cross-lingual, and
  physically unable to return anything outside the schema it is given.

Both run concurrently, so routing costs one LLM round-trip, not two steps.

**GLiNER is still the classifier of record when the LLM is not.** The ensemble
below, its share gate and its calibration are untouched and run whenever the LLM
is unreachable, slow, disabled (``ASSISTANT_LLM_ROUTING=0``) or answers off-list.
That is not a degraded mode bolted on — it is the same 89%-argmax router this
page shipped with, kept as the floor under a better ceiling.

Why not regexes anywhere in here: they would need per-language phrase lists
("собери", "подборка", "hits", "best of", …) maintained forever.

Three design choices in the GLiNER ensemble were measured, not guessed — see
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
When the models are genuinely unsure the router returns ``intent=None`` and the
caller asks the user; guessing badly costs a whole wasted pipeline run, a tap
costs a second. The LLM's own ``followup`` and ``unclear`` labels are answers to
the router, never destinations: neither can reach an executor without the code
below resolving it against the previous turn.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading

from app.services.assistant import intent_llm

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

# A follow-up modifier is detected in CODE, not by the model. Both model-side
# attempts were measured and rejected on 2026-07-27: a 4th "refine the previous
# request" label diluted the 3-way ensemble into two new confident wrong routes,
# and a dedicated binary "standalone vs modifier" pass showed no usable
# separation at any threshold (min follow-up score 0.68 vs max standalone 0.86).
# GLiNER simply has no concept of conversational reference.
#
# What it CAN see is that a message names nothing at all. «а побыстрее?»,
# «ещё такого же», «покороче давай» carry no artist, no song and no count —
# there is nothing in them to search FOR, so they can only mean "the last thing,
# but different". Every genuinely short request in the fixture does name
# something («собери хиты Канье», «лучшее у Queen», «биография Radiohead»), so
# the entity check is what keeps them out.
FOLLOWUP_MAX_WORDS = 3

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


async def _gliner_sync_or_empty(text: str) -> tuple[dict[str, float], dict, bool]:
    """``(scores, entities, ok)`` — never raises.

    ``ok=False`` means GLiNER itself is unavailable (model download failed, OOM),
    which is a different situation from "GLiNER ran and was unsure": with no
    spans there is nothing for the count override or the follow-up rule to read.
    """
    try:
        scores, ents = await asyncio.to_thread(_extract_sync, text)
        return scores, ents, True
    except Exception as exc:
        logger.warning("[assistant/router] GLiNER2 failed (%s) — spans unavailable", exc)
        return {}, {}, False


async def route(
    message: str,
    slots=None,
    *,
    explicit_intent: str | None = None,
    last_message: str | None = None,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
):
    """Decide which executor handles ``message``.

    Returns an :class:`~app.domain.models.AssistantRoute`. ``intent=None`` means
    "ask the user" — the caller emits a ``clarify`` frame.

    Decision table (all of it code, none of it prompting):

    ==================================  ========================================
    condition                            outcome
    ==================================  ========================================
    ``explicit_intent`` set              used verbatim, no model is called at all
                                         (this is the reply to a ``clarify`` frame)
    LLM label is one of the three        that executor  → ``source="llm"``
    LLM says ``followup`` + last_intent  sticky: reuse the previous intent
    count >= 2 recognised                hard override → ``playlist``
    winner's share >= MIN_SHARE          the GLiNER ensemble's pick
    short message + known last_intent    sticky: reuse it (follow-ups)
    otherwise                            ``None`` → clarify
    ==================================  ========================================

    The GLiNER rows are reached whenever the LLM gave no usable label — it is
    unreachable, timed out, disabled, or answered ``unclear``.
    """
    from app.domain.models import AssistantRoute

    last_intent = getattr(slots, "last_intent", None) if slots else None

    if explicit_intent in ("search", "playlist", "facts"):
        return AssistantRoute(intent=explicit_intent, confidence=1.0, margin=1.0,
                              source="explicit")

    text = (message or "").strip()
    if not text:
        return AssistantRoute(intent=last_intent, source="sticky" if last_intent else "unclear")

    # Concurrent on purpose: the spans are needed on every path (slots, the
    # count override, the follow-up rule), so waiting for the classification
    # first would add GLiNER's ~200 ms to the LLM's round-trip for nothing.
    label, (scores, ents, gliner_ok) = await asyncio.gather(
        intent_llm.classify(text, last_intent=last_intent, last_message=last_message,
                            base_url=llm_base_url, model=llm_model),
        _gliner_sync_or_empty(text),
    )

    if not gliner_ok and label is None:
        # Neither model answered — degrade to the previous intent, else to
        # search, which is the historical default. Never a 500.
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
        "[assistant/router] msg=%r → llm=%s gliner=%s share=%.2f margin=%.2f "
        "artist=%r song=%r count=%s",
        text[:60], label, top_intent, share, margin, artist, song, count,
    )

    # ── the LLM's answer, once code has made sense of it ──
    if label in intent_llm.INTENTS:
        # It read the sentence; a digit it decided to ignore ("расскажи 3 факта
        # про Radiohead") is a decision, not an oversight, so no count override
        # here. The override still guards the GLiNER path below.
        return AssistantRoute(intent=label, source="llm",
                              **{**base, "confidence": 1.0, "margin": 1.0})
    if label == "followup" and last_intent:
        return AssistantRoute(intent=last_intent, source="sticky", **base)
    # "followup" with nothing to refer back to, and "unclear", both fall
    # through: GLiNER may still have a confident, usable read of the message.

    # A concrete number of tracks is a request for many songs, whatever the
    # model thought. Digit-based, so it holds across languages.
    if count is not None and count >= 2:
        return AssistantRoute(intent="playlist", source="count_override", **base)

    # A very short message that names nothing can only be a tweak to the previous
    # turn. This is checked BEFORE the share gate on purpose: the classifier is
    # confidently WRONG on these (it read «а побыстрее?» as a lyric search for
    # "fast" at share 0.57), so leaving it as a below-threshold fallback never
    # fired. Without a previous turn there is nothing to refine — fall through.
    names_nothing = not (artist or song or count)
    if (last_intent and names_nothing
            and len(text.split()) <= FOLLOWUP_MAX_WORDS):
        return AssistantRoute(intent=last_intent, source="sticky", **base)

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
        # A "song" span is only a title when the turn is about one track. On a
        # playlist turn GLiNER labels the wish itself ("спокойного джаза на
        # вечер") as a song, and that would then answer a later "расскажи про
        # эту песню" with nonsense.
        if route.song and route.intent in ("search", "facts"):
            merged["last_song"] = route.song
    for key, value in updates.items():
        # None means "nothing new this turn" — never wipe a good slot with it.
        if value is not None:
            merged[key] = value
    return AssistantSlots(**merged)
