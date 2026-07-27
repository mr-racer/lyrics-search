"""Live-stack test: does the REAL GLiNER2 actually route these phrasings?

This is the test that justifies the design decision. Everything else about the
router is unit-tested against a stub; the open question — can a multilingual
zero-shot classifier separate "find me this song", "build me a playlist" and
"tell me about this artist" across Russian and English without a single
maintained keyword list — can only be answered by the real model.

It asserts an accuracy FLOOR, not per-phrase correctness: a zero-shot classifier
is allowed to miss individual phrasings, and the router's clarify fallback
covers those. What must not happen is broad confusion between the branches.

Run: ``scripts/run_docker_tests.sh -k assistant_routing``
"""
from __future__ import annotations

import pytest

from app.domain.models import AssistantSlots
from app.services.assistant import router as R

# (message, expected intent). Deliberately mixes languages, phrasings and
# lengths — including the shapes the two production stacks were built for.
CASES = [
    # ── search: one specific song, by lyrics or by sound ──
    ("найди песню где поётся про дождь и одиночество", "search"),
    ("что за трек со строчкой about a broken heart", "search"),
    ("помоги вспомнить песню про город и неоновые огни", "search"),
    ("find the song with the line i woke up this morning", "search"),
    ("что за песня такая грустная с женским вокалом и пианино", "search"),
    ("какой трек начинается со слов hello darkness", "search"),
    ("song about driving at night alone", "search"),
    ("ищу трек где припев про свободу", "search"),

    # ── playlist: many songs ──
    ("собери хиты Канье", "playlist"),
    ("сделай подборку под пробежку", "playlist"),
    ("плейлист для вечерней работы", "playlist"),
    ("build me a playlist of 90s eurodance", "playlist"),
    ("саундтрек к Интерстеллару", "playlist"),
    ("лучшее у Queen", "playlist"),
    ("накидай что-нибудь энергичное на утро", "playlist"),
    ("best of Radiohead", "playlist"),
    ("песни из GTA Vice City", "playlist"),
    ("подборка спокойного джаза на вечер", "playlist"),

    # ── facts: tell me about ──
    ("расскажи про Kanye West", "facts"),
    ("кто продюсировал этот трек", "facts"),
    ("о чём эта песня", "facts"),
    ("tell me about Björk", "facts"),
    ("какая история у этой песни", "facts"),
    ("биография Radiohead", "facts"),
    ("что известно про альбом OK Computer", "facts"),
    ("who wrote this song", "facts"),
    ("расскажи историю создания Bohemian Rhapsody", "facts"),
]

# Short remarks that only mean anything as a tweak to the previous turn. The
# router must resolve them against ``last_intent`` — routing them on their own
# content sends «а побыстрее?» into a lyric search for the word "fast".
FOLLOWUP_CASES = [
    "а побыстрее?",
    "ещё такого же",
    "а погромче",
    "покороче давай",
    "more like that",
    "а что-нибудь поспокойнее",
]

# Measured on this exact fixture (2026-07-27, fastino/gliner2-multi-v1): the
# ensemble routes 22/27 correctly with ZERO wrong branches, the remaining 5
# asking. The floor sits just under that so normal model/threshold drift shows
# up as a failure rather than silently degrading the UX.
MIN_ACCURACY = 0.75
# The failure that actually costs the user is a CONFIDENT wrong branch — a whole
# wasted pipeline run. An unclear result just prompts a clarify tap, so it is
# tolerated far more. The measured value is 0; 1 leaves room for drift.
MAX_CONFIDENT_WRONG = 1
# How often the router is allowed to fall back to asking. Above this the labels
# or MIN_SHARE need work — the assistant stops feeling like it understands.
MAX_ASK_RATE = 0.30


@pytest.fixture(scope="module")
def routed():
    """Route every case once; the model loads on the first call."""
    import asyncio

    async def run():
        out = []
        for message, expected in CASES:
            route = await R.route(message, AssistantSlots())
            out.append((message, expected, route))
        return out

    return asyncio.run(run())


def test_routing_accuracy_meets_the_floor(routed):
    hits = [(m, e, r) for m, e, r in routed if r.intent == e]
    accuracy = len(hits) / len(routed)
    misses = [
        f"{m!r}: expected {e}, got {r.intent} (conf={r.confidence}, margin={r.margin})"
        for m, e, r in routed if r.intent != e
    ]
    assert accuracy >= MIN_ACCURACY, (
        f"routing accuracy {accuracy:.0%} < {MIN_ACCURACY:.0%}\n" + "\n".join(misses)
    )


def test_asking_stays_the_exception(routed):
    """Clarify is the safety net, not the default. If the router asks on a third
    of ordinary phrasings, the "it just understands me" premise is gone."""
    asked = [m for m, _, r in routed if r.intent is None]
    rate = len(asked) / len(routed)
    assert rate <= MAX_ASK_RATE, (
        f"router asked on {rate:.0%} of cases (max {MAX_ASK_RATE:.0%}):\n"
        + "\n".join(repr(m) for m in asked)
    )


def test_confident_wrong_answers_are_rare(routed):
    """A wrong branch chosen CONFIDENTLY is the failure that costs the user a
    whole wasted pipeline run — unclear results just prompt a clarify tap."""
    confident_wrong = [
        f"{m!r}: expected {e}, got {r.intent} at conf={r.confidence}"
        for m, e, r in routed
        if r.intent is not None and r.intent != e and r.source == "gliner"
    ]
    assert len(confident_wrong) <= MAX_CONFIDENT_WRONG, (
        f"{len(confident_wrong)} confidently wrong routes:\n" + "\n".join(confident_wrong)
    )


def test_playlist_and_facts_are_not_confused(routed):
    """The two branches with the most divergent UX must not bleed into each
    other: a 15-track playlist in answer to «расскажи про артиста» is the worst
    possible outcome, and vice versa."""
    bad = [
        f"{m!r}: expected {e}, got {r.intent}"
        for m, e, r in routed
        if {e, r.intent} == {"playlist", "facts"}
    ]
    assert not bad, "playlist/facts confusion:\n" + "\n".join(bad)


def test_artist_spans_are_extracted(routed):
    """Slot carry-over depends on the entity pass, not just the label."""
    by_message = {m: r for m, _, r in routed}
    assert (by_message["расскажи про Kanye West"].artist or "").lower().startswith("kanye")
    assert (by_message["tell me about Björk"].artist or "") != ""


def test_count_override_survives_the_real_model():
    import asyncio

    route = asyncio.run(R.route("найди 20 треков под бег", AssistantSlots()))
    assert route.intent == "playlist"
    assert route.count == 20


def test_followups_resolve_against_the_previous_turn():
    """The multi-turn promise: «а побыстрее?» after a playlist must stay a
    playlist. Before the ``followup`` label these were classified as CONFIDENT
    searches and ran a nonsense lyric lookup."""
    import asyncio

    prev = AssistantSlots(last_intent="playlist")
    routed = [(m, asyncio.run(R.route(m, prev))) for m in FOLLOWUP_CASES]
    wrong = [f"{m!r}: got {r.intent} (source={r.source}, share={r.confidence})"
             for m, r in routed if r.intent != "playlist"]
    assert not wrong, ("follow-ups that did not stick to the previous intent:\n"
                       + "\n".join(wrong))


def test_a_followup_with_no_previous_turn_asks():
    """Nothing to refine on the first message — asking beats guessing."""
    import asyncio

    route = asyncio.run(R.route("а побыстрее?", AssistantSlots()))
    assert route.intent is None


def test_a_short_real_request_is_not_swallowed_by_stickiness():
    """«собери хиты Канье» is as short as a follow-up but is a real playlist
    request — the followup label must not eat it."""
    import asyncio

    prev = AssistantSlots(last_intent="facts")
    route = asyncio.run(R.route("собери хиты Канье", prev))
    assert route.intent == "playlist"
