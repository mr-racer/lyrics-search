"""Unit tests for the intent router.

Both models are stubbed here — these tests pin the CODE around them: which
model's answer wins, the share gate, the sticky follow-up rule, the count
override, the explicit-intent bypass, the ensembling arithmetic, and the
defensive parsing of every output shape GLiNER2 can return.

The LLM is switched OFF for most of the file (see the ``no_llm`` autouse
fixture) so the GLiNER rows of the decision table are tested as the fallback
they are, on any machine, whether or not an LLM endpoint happens to be
configured. The LLM rows have their own section at the bottom.

Whether the real models actually classify Russian phrasing correctly is a
live-model question, answered by ``tests/docker/test_assistant_routing.py`` —
which is also where MIN_SHARE was calibrated.
"""
from __future__ import annotations

import asyncio

import pytest

from app.domain.models import AssistantSlots
from app.services.assistant import router as R


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Default every test to the GLiNER path — a configured LLM_BASE_URL in the
    developer's environment must not change what these tests exercise."""
    async def _off(*a, **kw):
        return None

    monkeypatch.setattr(R.intent_llm, "classify", _off)


@pytest.fixture
def fake_llm(monkeypatch):
    """Opt back in: tests set ``holder['label']`` to the model's answer."""
    holder = {"label": None, "seen": []}

    async def _classify(message, **kw):
        holder["seen"].append((message, kw))
        return holder["label"]

    monkeypatch.setattr(R.intent_llm, "classify", _classify)
    return holder


def _ents(artist=None, song=None, count=None, era=None):
    """Build a GLiNER2-shaped entity/structure output."""
    out: dict = {"entities": {}}
    if artist:
        out["entities"]["artist"] = [{"text": artist, "confidence": 0.9}]
    if song:
        out["entities"]["song"] = [{"text": song, "confidence": 0.9}]
    if count or era:
        out["request"] = {"count": count or "", "era": era or ""}
    return out


@pytest.fixture
def fake_gliner(monkeypatch):
    """Patch both blocking passes; tests set ``holder['scores']`` / ``['ents']``.

    ``scores`` are already-ensembled shares, i.e. what ``_classify_sync``
    returns — the arithmetic that produces them is tested separately.
    """
    holder = {"scores": {"search": 0.9, "playlist": 0.1}, "ents": _ents()}
    monkeypatch.setattr(R, "_extract_sync",
                        lambda message: (holder["scores"], holder["ents"]))
    return holder


# ── the confident path ───────────────────────────────────────────────────────


async def test_confident_share_routes(fake_gliner):
    fake_gliner["scores"] = {"playlist": 0.8, "search": 0.15, "facts": 0.05}
    route = await R.route("собери подборку под вечер", AssistantSlots())
    assert route.intent == "playlist"
    assert route.source == "gliner"
    assert route.confidence == 0.8


async def test_entities_are_extracted(fake_gliner):
    fake_gliner["scores"] = {"facts": 0.7, "search": 0.3}
    fake_gliner["ents"] = _ents(artist="Kanye West", song="Runaway")
    route = await R.route("расскажи про Runaway", AssistantSlots())
    assert (route.intent, route.artist, route.song) == ("facts", "Kanye West", "Runaway")


# ── refusing to guess ────────────────────────────────────────────────────────


async def test_split_share_asks_instead_of_guessing(fake_gliner):
    """Three-way split: no winner owns enough of the mass to act on."""
    fake_gliner["scores"] = {"search": 0.4, "playlist": 0.35, "facts": 0.25}
    route = await R.route("что-нибудь такое эдакое сегодня вечером", AssistantSlots())
    assert route.intent is None
    assert route.source == "unclear"


async def test_empty_scores_ask(fake_gliner):
    """All-noise input: _classify_sync returns {} below the absolute floor."""
    fake_gliner["scores"] = {}
    route = await R.route("...", AssistantSlots())
    assert route.intent is None


async def test_share_exactly_at_the_gate_routes(fake_gliner):
    """The gate is inclusive — a winner sitting exactly on MIN_SHARE acts."""
    rest = (1 - R.MIN_SHARE) / 2
    fake_gliner["scores"] = {"facts": R.MIN_SHARE, "search": rest, "playlist": rest}
    route = await R.route("о чём эта песня вообще", AssistantSlots())
    assert route.intent == "facts"
    assert route.confidence == round(R.MIN_SHARE, 3)


async def test_just_below_the_gate_asks(fake_gliner):
    fake_gliner["scores"] = {"facts": R.MIN_SHARE - 0.02,
                             "search": 0.30, "playlist": 0.27}
    route = await R.route("о чём эта песня вообще и что там дальше", AssistantSlots())
    assert route.intent is None


# ── code overrides ───────────────────────────────────────────────────────────


async def test_count_forces_playlist(fake_gliner):
    """«найди 20 треков» is a playlist however the model scored it — a digit is
    a language-independent signal."""
    fake_gliner["scores"] = {"search": 0.95, "playlist": 0.05}
    fake_gliner["ents"] = _ents(count="20 треков")
    route = await R.route("найди 20 треков под бег", AssistantSlots())
    assert route.intent == "playlist"
    assert route.source == "count_override"
    assert route.count == 20


async def test_count_of_one_does_not_override(fake_gliner):
    fake_gliner["scores"] = {"search": 0.95, "playlist": 0.05}
    fake_gliner["ents"] = _ents(count="1")
    route = await R.route("найди 1 трек про дождь", AssistantSlots())
    assert route.intent == "search"


async def test_explicit_intent_skips_the_model(monkeypatch):
    """A reply to a clarify frame must not re-run GLiNER at all."""
    called = []
    monkeypatch.setattr(R, "_extract_sync", lambda m: called.append(m) or ({}, {}))
    route = await R.route("та же фраза", AssistantSlots(), explicit_intent="facts")
    assert route.intent == "facts"
    assert route.source == "explicit"
    assert called == []


# ── stickiness ───────────────────────────────────────────────────────────────


async def test_short_followup_reuses_last_intent(fake_gliner):
    fake_gliner["scores"] = {"search": 0.4, "playlist": 0.35, "facts": 0.25}
    route = await R.route("а побыстрее?", AssistantSlots(last_intent="playlist"))
    assert route.intent == "playlist"
    assert route.source == "sticky"


async def test_long_unclear_message_does_not_stick(fake_gliner):
    fake_gliner["scores"] = {"search": 0.4, "playlist": 0.35, "facts": 0.25}
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


# ── the LLM path ─────────────────────────────────────────────────────────────
# The LLM reads the sentence and GLiNER supplies the spans. Everything the model
# says is still resolved in code: an off-list label cannot reach an executor,
# and neither "followup" nor "unclear" is a destination.


async def test_llm_label_beats_a_confident_gliner(fake_gliner, fake_llm):
    """The whole point of the change: GLiNER scores surface similarity, the LLM
    reads the sentence, and where they disagree the reading wins."""
    fake_gliner["scores"] = {"search": 0.9, "playlist": 0.05, "facts": 0.05}
    fake_llm["label"] = "facts"
    route = await R.route("кто спродюсировал Runaway", AssistantSlots())
    assert (route.intent, route.source) == ("facts", "llm")


async def test_llm_still_gets_the_spans_from_gliner(fake_gliner, fake_llm):
    """Slots and the executors need artist/song whichever model routed."""
    fake_gliner["ents"] = _ents(artist="Kanye West", song="Runaway")
    fake_llm["label"] = "facts"
    route = await R.route("расскажи про Runaway", AssistantSlots())
    assert (route.artist, route.song) == ("Kanye West", "Runaway")


async def test_llm_sees_the_previous_turn(fake_gliner, fake_llm):
    fake_llm["label"] = "playlist"
    await R.route("а побыстрее?", AssistantSlots(last_intent="playlist"),
                  last_message="хиты Kanye West")
    _, kw = fake_llm["seen"][0]
    assert kw["last_intent"] == "playlist"
    assert kw["last_message"] == "хиты Kanye West"


async def test_llm_followup_resolves_to_the_previous_intent(fake_gliner, fake_llm):
    fake_gliner["scores"] = {"search": 0.95, "playlist": 0.05}
    fake_llm["label"] = "followup"
    route = await R.route("а побыстрее?", AssistantSlots(last_intent="playlist"))
    assert (route.intent, route.source) == ("playlist", "sticky")


async def test_llm_followup_without_a_previous_turn_falls_through(fake_gliner, fake_llm):
    """Nothing to refine — GLiNER's own read is the next thing to try, not a 500
    and not a silent "search"."""
    fake_gliner["scores"] = {"playlist": 0.8, "search": 0.15, "facts": 0.05}
    fake_llm["label"] = "followup"
    route = await R.route("ещё такого же", AssistantSlots())
    assert (route.intent, route.source) == ("playlist", "gliner")


async def test_llm_unclear_hands_over_to_gliner(fake_gliner, fake_llm):
    fake_gliner["scores"] = {"facts": 0.8, "search": 0.15, "playlist": 0.05}
    fake_llm["label"] = "unclear"
    route = await R.route("что там у Radiohead", AssistantSlots())
    assert (route.intent, route.source) == ("facts", "gliner")


async def test_llm_unclear_and_gliner_unsure_still_asks(fake_gliner, fake_llm):
    fake_gliner["scores"] = {"search": 0.4, "playlist": 0.35, "facts": 0.25}
    fake_llm["label"] = "unclear"
    route = await R.route("ну это, как его, ты понял", AssistantSlots())
    assert route.intent is None


async def test_a_digit_does_not_override_the_llm(fake_gliner, fake_llm):
    """«расскажи 3 факта про Radiohead» is a facts turn. The count override
    exists to rescue a GLiNER misread; a model that read the sentence and still
    said "facts" made a decision, not a mistake."""
    fake_gliner["scores"] = {"search": 0.9, "playlist": 0.1}
    fake_gliner["ents"] = _ents(count="3 факта")
    fake_llm["label"] = "facts"
    route = await R.route("расскажи 3 факта про Radiohead", AssistantSlots())
    assert (route.intent, route.count) == ("facts", 3)


async def test_a_digit_still_overrides_on_the_gliner_path(fake_gliner, fake_llm):
    fake_gliner["scores"] = {"search": 0.95, "playlist": 0.05}
    fake_gliner["ents"] = _ents(count="20 треков")
    fake_llm["label"] = None                      # LLM unavailable
    route = await R.route("найди 20 треков под бег", AssistantSlots())
    assert route.source == "count_override"


async def test_explicit_intent_skips_the_llm_too(fake_gliner, fake_llm):
    route = await R.route("та же фраза", AssistantSlots(), explicit_intent="playlist")
    assert (route.intent, route.source) == ("playlist", "explicit")
    assert fake_llm["seen"] == []


async def test_both_models_down_never_raises(monkeypatch, fake_llm):
    def boom(_message):
        raise RuntimeError("model not loaded")

    monkeypatch.setattr(R, "_extract_sync", boom)
    fake_llm["label"] = None
    route = await R.route("что угодно", AssistantSlots(last_intent="facts"))
    assert (route.intent, route.source) == ("facts", "sticky")


async def test_the_two_models_run_concurrently(fake_gliner, fake_llm, monkeypatch):
    """Routing must cost one round-trip, not GLiNER's ~200 ms plus the LLM's."""
    async def _slow(message, **kw):
        await asyncio.sleep(0.05)
        return "facts"

    def _also_slow(_message):
        import time
        time.sleep(0.05)
        return fake_gliner["scores"], fake_gliner["ents"]

    monkeypatch.setattr(R.intent_llm, "classify", _slow)
    monkeypatch.setattr(R, "_extract_sync", _also_slow)
    loop = asyncio.get_running_loop()
    started = loop.time()
    await R.route("расскажи про Radiohead", AssistantSlots())
    assert loop.time() - started < 0.09  # sequential would be ≥ 0.10


# ── the ensemble arithmetic ──────────────────────────────────────────────────


def test_each_label_set_is_normalised_before_summing(monkeypatch):
    """A set that fires hot on a phrasing must not outvote the other on scale
    alone — that is the whole reason each side is normalised first."""
    calls = []

    class FakeModel:
        def create_schema(self):
            return object()

        def extract(self, message, schema, include_confidence=False):
            calls.append(schema)
            # First set screams "search" at high absolute scores; the second
            # quietly but decisively says "playlist".
            if len(calls) == 1:
                return {"intent": [
                    {"label": R._LABELS_A["search"], "confidence": 0.90},
                    {"label": R._LABELS_A["playlist"], "confidence": 0.60},
                ]}
            return {"intent": [
                {"label": R._LABELS_B["playlist"], "confidence": 0.09},
                {"label": R._LABELS_B["search"], "confidence": 0.01},
            ]}

    model = FakeModel()
    monkeypatch.setattr(R, "_schemas", (model, object(), object(), object()))
    scores = R._classify_sync("собери что-нибудь")
    # A: search .6/.4 · B: playlist .9/.1 → playlist 1.30 vs search 1.20 of 2.5
    assert scores["playlist"] > scores["search"]
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_classify_returns_empty_below_the_absolute_floor(monkeypatch):
    class FakeModel:
        def create_schema(self):
            return object()

        def extract(self, message, schema, include_confidence=False):
            return {"intent": [{"label": R._LABELS_A["search"], "confidence": 0.001}]}

    monkeypatch.setattr(R, "_schemas", (FakeModel(), object(), object(), object()))
    assert R._classify_sync("нечто бессмысленное") == {}


def test_normalise_handles_an_all_zero_dict():
    assert R._normalise({"a": 0.0, "b": 0.0}) == {}


# ── defensive parsing of GLiNER2 output shapes ───────────────────────────────


def test_scored_labels_accepts_bare_string():
    label = R._LABELS_A["facts"]
    assert R._scored_labels(label) == [(label, 1.0)]


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


# ── follow-up modifiers («а побыстрее?») ─────────────────────────────────────
# Detected in code from "short AND names nothing", because the classifier is
# CONFIDENTLY wrong on these — it reads them as lyric searches.


async def test_short_message_naming_nothing_sticks(fake_gliner):
    """«а побыстрее?» is not a search — it is "the last thing, but different"."""
    fake_gliner["scores"] = {"search": 0.6, "playlist": 0.3, "facts": 0.1}
    route = await R.route("а побыстрее?", AssistantSlots(last_intent="playlist"))
    assert route.intent == "playlist"
    assert route.source == "sticky"


async def test_stickiness_beats_a_confident_but_wrong_classification(fake_gliner):
    """The rule runs BEFORE the share gate on purpose: as a below-threshold
    fallback it never fired, because the model was sure it was a search."""
    fake_gliner["scores"] = {"search": 0.95, "playlist": 0.05}
    route = await R.route("ещё такого же", AssistantSlots(last_intent="facts"))
    assert route.intent == "facts"
    assert route.source == "sticky"


async def test_a_short_message_that_names_an_artist_is_a_real_request(fake_gliner):
    """«собери хиты Канье» is just as short, but it names someone — the previous
    intent must not swallow it."""
    fake_gliner["scores"] = {"playlist": 0.8, "search": 0.15, "facts": 0.05}
    fake_gliner["ents"] = _ents(artist="Канье")
    route = await R.route("собери хиты Канье", AssistantSlots(last_intent="facts"))
    assert route.intent == "playlist"
    assert route.source == "gliner"


async def test_a_short_message_naming_a_song_is_a_real_request(fake_gliner):
    fake_gliner["scores"] = {"facts": 0.8, "search": 0.2}
    fake_gliner["ents"] = _ents(song="Runaway")
    route = await R.route("про Runaway", AssistantSlots(last_intent="playlist"))
    assert route.intent == "facts"


async def test_a_short_message_with_a_count_is_a_real_request(fake_gliner):
    fake_gliner["scores"] = {"search": 0.9, "playlist": 0.1}
    fake_gliner["ents"] = _ents(count="10")
    route = await R.route("давай 10 штук", AssistantSlots(last_intent="facts"))
    assert route.intent == "playlist"
    assert route.source == "count_override"


async def test_no_previous_turn_means_nothing_to_refine(fake_gliner):
    """First message of a conversation — the classifier's call stands."""
    fake_gliner["scores"] = {"search": 0.8, "playlist": 0.2}
    route = await R.route("а побыстрее?", AssistantSlots())
    assert route.source == "gliner"


async def test_a_longer_message_is_never_treated_as_a_modifier(fake_gliner):
    """«сделай подборку под пробежку» names nothing either, but at 4 words it is
    a real request — length is what separates the two."""
    fake_gliner["scores"] = {"playlist": 0.75, "search": 0.2, "facts": 0.05}
    route = await R.route("сделай подборку под пробежку",
                          AssistantSlots(last_intent="search"))
    assert route.intent == "playlist"
    assert route.source == "gliner"
