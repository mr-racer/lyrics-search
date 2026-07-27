"""Unit tests for the GLiNER2 intent router.

GLiNER itself is stubbed here — these tests pin the CODE around it: the
threshold/margin gate, the sticky follow-up rule, the count override, the
explicit-intent bypass, and the defensive parsing of every output shape the
library can return. Whether the real model actually classifies Russian phrasing
correctly is a live-model question, answered by
``tests/docker/test_assistant_routing.py``.
"""
from __future__ import annotations

import pytest

from app.domain.models import AssistantSlots
from app.services.assistant import router as R


def _raw(scores, *, artist=None, song=None, count=None, era=None):
    """Build a GLiNER2-shaped output dict from {label: confidence}."""
    out = {
        "intent": [{"label": label, "confidence": conf} for label, conf in scores.items()],
        "entities": {},
    }
    if artist:
        out["entities"]["artist"] = [{"text": artist, "confidence": 0.9}]
    if song:
        out["entities"]["song"] = [{"text": song, "confidence": 0.9}]
    if count or era:
        out["request"] = {"count": count or "", "era": era or ""}
    return out


@pytest.fixture
def fake_gliner(monkeypatch):
    """Patch the blocking extract call; tests set ``holder['raw']``."""
    holder = {"raw": _raw({R._LABEL_SEARCH: 0.9, R._LABEL_PLAYLIST: 0.1})}
    monkeypatch.setattr(R, "_extract_sync", lambda message: holder["raw"])
    return holder


# ── the confident path ───────────────────────────────────────────────────────


async def test_confident_label_routes(fake_gliner):
    fake_gliner["raw"] = _raw({R._LABEL_PLAYLIST: 0.88, R._LABEL_SEARCH: 0.2})
    route = await R.route("собери подборку под вечер", AssistantSlots())
    assert route.intent == "playlist"
    assert route.source == "gliner"


async def test_entities_are_extracted(fake_gliner):
    fake_gliner["raw"] = _raw(
        {R._LABEL_FACTS: 0.8, R._LABEL_SEARCH: 0.1},
        artist="Kanye West", song="Runaway",
    )
    route = await R.route("расскажи про Runaway", AssistantSlots())
    assert (route.intent, route.artist, route.song) == ("facts", "Kanye West", "Runaway")


# ── refusing to guess ────────────────────────────────────────────────────────


async def test_low_confidence_asks_instead_of_guessing(fake_gliner):
    fake_gliner["raw"] = _raw({R._LABEL_SEARCH: 0.3, R._LABEL_PLAYLIST: 0.1})
    route = await R.route("что-нибудь такое эдакое сегодня вечером", AssistantSlots())
    assert route.intent is None
    assert route.source == "unclear"


async def test_thin_margin_asks_instead_of_guessing(fake_gliner):
    # Both labels confident, neither clearly ahead — exactly the case where a
    # guess sends the user down a whole wrong pipeline.
    fake_gliner["raw"] = _raw({R._LABEL_SEARCH: 0.61, R._LABEL_PLAYLIST: 0.58})
    route = await R.route("что-нибудь про дождь и осень в городе", AssistantSlots())
    assert route.intent is None


# ── code overrides ───────────────────────────────────────────────────────────


async def test_count_forces_playlist(fake_gliner):
    """«найди 20 треков» is a playlist however the model scored it — a digit is
    a language-independent signal."""
    fake_gliner["raw"] = _raw({R._LABEL_SEARCH: 0.95, R._LABEL_PLAYLIST: 0.05},
                              count="20 треков")
    route = await R.route("найди 20 треков под бег", AssistantSlots())
    assert route.intent == "playlist"
    assert route.source == "count_override"
    assert route.count == 20


async def test_count_of_one_does_not_override(fake_gliner):
    fake_gliner["raw"] = _raw({R._LABEL_SEARCH: 0.95, R._LABEL_PLAYLIST: 0.05},
                              count="1")
    route = await R.route("найди 1 трек про дождь", AssistantSlots())
    assert route.intent == "search"


async def test_explicit_intent_skips_the_model(monkeypatch):
    """A reply to a clarify frame must not re-run GLiNER at all."""
    called = []
    monkeypatch.setattr(R, "_extract_sync", lambda m: called.append(m) or {})
    route = await R.route("та же фраза", AssistantSlots(), explicit_intent="facts")
    assert route.intent == "facts"
    assert route.source == "explicit"
    assert called == []


# ── stickiness ───────────────────────────────────────────────────────────────


async def test_short_followup_reuses_last_intent(fake_gliner):
    fake_gliner["raw"] = _raw({R._LABEL_SEARCH: 0.2, R._LABEL_PLAYLIST: 0.18})
    route = await R.route("а побыстрее?", AssistantSlots(last_intent="playlist"))
    assert route.intent == "playlist"
    assert route.source == "sticky"


async def test_long_unclear_message_does_not_stick(fake_gliner):
    fake_gliner["raw"] = _raw({R._LABEL_SEARCH: 0.2, R._LABEL_PLAYLIST: 0.18})
    slots = AssistantSlots(last_intent="playlist")
    route = await R.route(
        "слушай а вот эта штука которую я вчера слышал где-то в кафе", slots,
    )
    assert route.intent is None


async def test_gliner_failure_degrades_instead_of_raising(monkeypatch):
    def boom(_message):
        raise RuntimeError("model not loaded")

    monkeypatch.setattr(R, "_extract_sync", boom)
    route = await R.route("что угодно", AssistantSlots())
    assert route.intent == "search"  # historical default, never a 500


# ── defensive parsing of GLiNER2 output shapes ───────────────────────────────


def test_scored_labels_accepts_bare_string():
    assert R._scored_labels(R._LABEL_FACTS) == [(R._LABEL_FACTS, 1.0)]


def test_scored_labels_sorts_desc():
    raw = [{"label": "a", "confidence": 0.2}, {"label": "b", "confidence": 0.7}]
    assert [lbl for lbl, _ in R._scored_labels(raw)] == ["b", "a"]


def test_scored_labels_survives_garbage():
    assert R._scored_labels([None, 42, {"nope": 1}]) == []


@pytest.mark.parametrize("text,expected", [
    ("20 треков", 20), ("на 15 песен", 15), ("twenty", None),
    ("", None), (None, None), ("999", None),  # out of the sane 1..200 band
])
def test_parse_count(text, expected):
    assert R._parse_count(text) == expected


def test_first_entity_picks_highest_score():
    ents = {"artist": [{"text": "A", "confidence": 0.4},
                       {"text": "B", "confidence": 0.9}]}
    assert R._first_entity(ents, "artist") == "B"


def test_structure_field_unwraps_nested_shapes():
    assert R._structure_field({"count": {"text": "5"}}, "count") == "5"
    assert R._structure_field([{"count": "7"}], "count") == "7"
    assert R._structure_field({"count": ""}, "count") is None
