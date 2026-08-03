"""LLM intent classification for the assistant router.

The router used to be GLiNER2 alone. Zero-shot classification over three
descriptive labels reached ~89% argmax on the measured fixture, which sounds
fine until you notice *which* 11% it misses: the cases where the words of the
message point one way and its meaning another — «поставь мне что-нибудь
из Radiohead» (playlist) versus «поставь Creep» (search), «кто спродюсировал
Runaway» (facts) versus «найди Runaway» (search). GLiNER scores surface
similarity; those pairs are only separable by reading the sentence.

So the LLM decides the intent and GLiNER keeps the job it is genuinely better
at: pulling the artist / song / count / era spans out of the message, in any
language, at a cost of ~200 ms and zero tokens.

Nothing about the safety properties changes, because the decision is still
CODE-verified:

* The label set is closed and checked against it here — a model that answers
  "recommendation" or writes a paragraph is discarded, not interpreted.
* An unreachable / slow / broken LLM is not an error path but *the* fallback:
  :func:`classify` returns ``None`` and the router runs the old GLiNER ensemble
  with the old share gate. The assistant never 500s because a local LM Studio
  went away mid-session.
* ``followup`` and ``unclear`` never route anywhere on their own; the router
  resolves them against the previous turn, in code.

The prompt below is written for a small local model: every label is defined by
what the listener WANTS rather than by a word list, the close calls are spelled
out as explicit contrasts, and both languages the app serves appear in the
examples.
"""

from __future__ import annotations

import asyncio
import logging
import os

from app.services.assistant.llm_json import parse_json_object

logger = logging.getLogger(__name__)

# The closed label set. "search"/"playlist"/"facts" are the executors;
# "followup" and "unclear" are answers the ROUTER resolves, not destinations.
INTENTS = ("search", "playlist", "facts")
LABELS = (*INTENTS, "followup", "unclear")

# A local 12b answering a one-token classification should take ~1 s. Past this
# the GLiNER ensemble is simply the faster correct answer, so we stop waiting.
TIMEOUT_SEC = float(os.getenv("ASSISTANT_ROUTER_LLM_TIMEOUT", "12"))

# Escape hatch: ASSISTANT_LLM_ROUTING=0 puts routing back on GLiNER alone with
# no redeploy — the same rollback the rest of this page is designed around.
ENABLED = os.getenv("ASSISTANT_LLM_ROUTING", "1").strip().lower() not in ("0", "false", "no")

# How much of the previous turn the model gets. Enough to recognise a modifier,
# short enough that it cannot become the thing being classified.
MAX_PREV_CHARS = 120
MAX_MESSAGE_CHARS = 600


SYSTEM_PROMPT = """You are the INTENT ROUTER of a local music player. You read ONE message from a listener and decide which engine must handle it. You never answer the message, never search, never explain — you only route.

Output ONLY minified JSON, no prose, no code fence:
{"intent":"<label>"}

## The labels — nothing outside this list is valid

"search" — the listener wants ONE particular track: identified, found, or put on.
    They half-remember a lyric, quote a line, describe what the song sounds like
    or what happens in it, or they simply name a track and want it.

"playlist" — the listener wants a SET of tracks assembled for them: a mix, a
    selection, a "best of", music for an activity, a mood, a place, an evening,
    a decade; everything by an artist; the songs from a film or a game.

"facts" — the listener wants to KNOW something, not to hear something: an
    explanation, a meaning, a story, a biography, credits, who produced or
    sampled what, what a song is about, why something is the way it is,
    what a statement the app just showed them actually means.

"followup" — the message only modifies the PREVIOUS request and names nothing of
    its own: «а побыстрее?», «ещё такого же», «покороче давай», "and shorter",
    "more like that", "in English please". If there is no previous request,
    this label is invalid — use "unclear" instead.

"unclear" — genuinely impossible to place: the message would fit two engines
    equally well, or it carries no request at all. Prefer a real label when one
    fits; "unclear" costs the listener an extra tap.

## How to decide when it is close

- ONE track or MANY tracks separates "search" from "playlist".
  «найди песню про дождь» → search.  «накидай песен про дождь» → playlist.
  Any number greater than one — «20 треков», "top 10", «десяток» — → playlist.

- WANTING TO HEAR IT or WANTING TO KNOW ABOUT IT separates "search" from "facts".
  «поставь Runaway» → search.  «о чём Runaway» → facts.
  «найди трек, где поют про бетон» → search.  «что за история у этого трека» → facts.

- A bare artist name settles nothing; the verb around it does.
  «Radiohead, хиты» → playlist.  «расскажи про Radiohead» → facts.
  «Radiohead — Creep, включи» → search.

- Quoted or recited lyrics are a "search" even with no verb at all.

- An instruction to explain a statement the app showed — «объясни: X сэмплирует
  Y», "explain this fact", «а что это значит» right after a fact — is "facts".

- Route the request, not the politeness or the emotion wrapped around it.
  «слушай, а можешь мне собрать что-нибудь под вечер, а то грустно» → playlist.

## Examples

PREVIOUS: none
MESSAGE: где-то там про дождь и такси
{"intent":"search"}

PREVIOUS: none
MESSAGE: what's that song that goes "hello darkness my old friend"
{"intent":"search"}

PREVIOUS: none
MESSAGE: собери что-нибудь спокойное на вечер
{"intent":"playlist"}

PREVIOUS: none
MESSAGE: 20 tracks for a long drive
{"intent":"playlist"}

PREVIOUS: none
MESSAGE: расскажи про Radiohead
{"intent":"facts"}

PREVIOUS: none
MESSAGE: кто спродюсировал Runaway и почему там такое длинное соло
{"intent":"facts"}

PREVIOUS: playlist — «хиты Kanye West»
MESSAGE: а побыстрее?
{"intent":"followup"}

PREVIOUS: none
MESSAGE: музыка
{"intent":"unclear"}"""


def available(base_url: str | None = None) -> bool:
    """Is there an LLM worth calling for a ~1 s routing decision?

    Guards the unit-test / unconfigured case: with no base URL resolved and no
    OpenAI key in the environment, ``ask_llm`` would spend seconds failing
    against api.openai.com before the router could fall back to GLiNER.
    """
    if not ENABLED:
        return False
    try:
        from app.services.llm_client import resolve_base_url

        if resolve_base_url(base_url):
            return True
    except Exception:
        logger.debug("[assistant/router] base URL resolution failed", exc_info=True)
    return bool(os.getenv("OPENAI_API_KEY"))


def _user_prompt(message: str, *, last_intent: str | None, last_message: str | None) -> str:
    """The two lines the model classifies: what came before, and what to route."""
    if last_intent:
        previous = last_intent
        if last_message:
            previous += f" — «{last_message.strip()[:MAX_PREV_CHARS]}»"
    else:
        previous = "none"
    return f"PREVIOUS: {previous}\nMESSAGE: {(message or '').strip()[:MAX_MESSAGE_CHARS]}"


def parse_label(raw: object) -> str | None:
    """The label out of a model reply, or None if it is not one of ours.

    Deliberately strict: a model that returns "recommendation", a sentence, or
    an object with the wrong key is treated as a failed call, and the router
    falls back to GLiNER rather than guessing what it meant.
    """
    obj = parse_json_object(raw)
    if not isinstance(obj, dict):
        return None
    value = obj.get("intent")
    if not isinstance(value, str):
        return None
    label = value.strip().strip('"').lower()
    return label if label in LABELS else None


async def classify(message: str, *, last_intent: str | None = None,
                   last_message: str | None = None,
                   base_url: str | None = None, model: str | None = None) -> str | None:
    """One classification call. Returns a label from :data:`LABELS`, or None.

    None means "the LLM did not give a usable answer" — unreachable, slow,
    off-list, unparseable. Every one of those is the same thing to the caller:
    use GLiNER instead.
    """
    if not (message or "").strip() or not available(base_url):
        return None

    from app.services.llm_client import ask_llm

    try:
        raw = await asyncio.wait_for(
            ask_llm(
                _user_prompt(message, last_intent=last_intent, last_message=last_message),
                system_prompt=SYSTEM_PROMPT,
                parse_json=False,
                temperature=0.0,
                base_url=base_url,
                model=model,
                extra_body={"enable_thinking": False},
            ),
            timeout=TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.info("[assistant/router] LLM routing timed out after %.0fs — using GLiNER",
                    TIMEOUT_SEC)
        return None
    except Exception as exc:
        logger.info("[assistant/router] LLM routing unavailable (%s) — using GLiNER", exc)
        return None

    label = parse_label(raw)
    if label is None:
        logger.info("[assistant/router] LLM returned no usable label (%.80r)", raw)
    return label
