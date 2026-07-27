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

# Zero-shot floor. Below this the design premise (GLiNER instead of an LLM
# classifier) does not hold and the thresholds in router.py need retuning.
MIN_ACCURACY = 0.75
# Sending a request down the wrong pipeline is the expensive failure; asking is
# cheap. So a miss is tolerated much more than a confident wrong branch.
MAX_CONFIDENT_WRONG = 3


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
