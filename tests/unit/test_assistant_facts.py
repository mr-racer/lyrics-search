"""The citation gate of the facts executor.

``_verify`` is the single point that makes an ungrounded answer impossible: no
matter what a 12b model emits, an answer that doesn't cite real fact numbers is
thrown away and the deterministic fact rendering is served instead.
"""
from __future__ import annotations

import pytest

from app.services.assistant import facts_executor as F


# ── the gate ─────────────────────────────────────────────────────────────────


def test_valid_answer_passes():
    assert F._verify({"answer": "It samples an old soul record.", "used": [1, 3]}, 4) \
        == ("It samples an old soul record.", [1, 3])


@pytest.mark.parametrize("raw", [
    None,
    "just a string",
    {},
    {"answer": "", "used": [1]},              # empty answer
    {"answer": "   ", "used": [1]},           # whitespace only
    {"answer": "Something.", "used": []},     # cites nothing
    {"answer": "Something.", "used": "1"},    # not a list
    {"answer": "Something."},                 # no used at all
    {"answer": "Something.", "used": [9, 12]},  # all out of range
    {"answer": "Something.", "used": [0]},    # 1-indexed, 0 is invalid
])
def test_ungrounded_shapes_are_rejected(raw):
    assert F._verify(raw, 4) is None


def test_out_of_range_numbers_are_dropped_but_valid_ones_keep_it():
    assert F._verify({"answer": "A.", "used": [2, 99, 2]}, 3) == ("A.", [2])


def test_non_numeric_citations_are_ignored():
    assert F._verify({"answer": "A.", "used": ["2", "x", None]}, 3) == ("A.", [2])


# ── the deterministic fallback ───────────────────────────────────────────────


def test_fallback_lists_the_real_facts():
    subject = {"title": "Runaway"}
    items = [{"text": "Produced by Kanye West", "source": "credits"},
             {"text": "Released in 2010", "source": "facts"}]
    out = F._deterministic_answer(subject, items, "ru")
    assert "Runaway" in out
    assert "Produced by Kanye West" in out
    assert "Released in 2010" in out


def test_fallback_with_no_facts_admits_it():
    out = F._deterministic_answer({"title": "Nobody"}, [], "en")
    assert "don't have reliable information" in out


def test_fallback_language_follows_lang():
    ru = F._deterministic_answer({"title": "X"}, [{"text": "f", "source": "facts"}], "ru")
    en = F._deterministic_answer({"title": "X"}, [{"text": "f", "source": "facts"}], "en")
    assert "известно" in ru
    assert "known about" in en


# ── pack helpers ─────────────────────────────────────────────────────────────


def test_pack_is_numbered_from_one():
    items = [{"text": "alpha"}, {"text": "beta"}]
    assert F._render_pack(items, "en") == "[1] alpha\n[2] beta"


def test_clean_collapses_whitespace_and_truncates():
    assert F._clean("  a\n\n b  ") == "a b"
    long = "x" * (F.MAX_FACT_CHARS + 50)
    out = F._clean(long)
    assert len(out) <= F.MAX_FACT_CHARS + 1 and out.endswith("…")


def test_web_queries_are_built_from_the_subject_not_the_message():
    """The user's message may be conversational Russian; the web query must be
    a clean English lookup built from what we resolved."""
    artist = F._web_queries({"kind": "artist", "title": "Björk", "artist": "Björk"},
                            "а расскажи чё там у неё вообще")
    assert artist[0] == "Björk musician biography"
    song = F._web_queries({"kind": "song", "title": "Runaway",
                           "artist": "Kanye West"}, "?")
    assert all("Kanye West" in q and "Runaway" in q for q in song)


def test_web_queries_offer_a_second_angle():
    """The fallback query exists so a dry first search doesn't end the branch —
    the budget is spent by code, on a measured shortfall."""
    for kind in ("artist", "album", "song"):
        qs = F._web_queries({"kind": kind, "title": "X", "artist": "Y"}, "?")
        assert len(qs) == 2 and qs[0] != qs[1]
        assert len(qs) <= F.MAX_WEB_SEARCHES


def test_snippets_need_substance():
    """Short fragments are noise — a numbered pack item must be worth citing."""
    raw = "tiny\n\n" + ("a real paragraph with enough substance to matter " * 2)
    out = F._snippets_from_web(raw)
    assert len(out) == 1
    assert out[0]["source"] == "web"


def test_subject_from_hit_maps_each_kind():
    artist = F._subject_from_hit({"type": "artist", "artist": "Queen",
                                  "artist_slug": "queen", "image": "/a.jpg"})
    assert (artist["kind"], artist["title"], artist["image_path"]) == \
        ("artist", "Queen", "/a.jpg")

    song = F._subject_from_hit({"type": "song", "title": "Runaway",
                                "artist": "Kanye West", "track_id": "t1"})
    assert (song["kind"], song["track_id"], song["subtitle"]) == \
        ("song", "t1", "Kanye West")

    album = F._subject_from_hit({"type": "album", "album": "MBDTF",
                                 "artist": "Kanye West", "artist_slug": "kanye-west"})
    assert (album["kind"], album["title"], album["artist_slug"]) == \
        ("album", "MBDTF", "kanye-west")


def test_subject_query_prefers_extracted_spans():
    class Route:
        artist = "Kanye West"
        song = "Runaway"

    assert F._subject_query(Route(), "ignored") == "Runaway Kanye West"


def test_subject_query_falls_back_to_the_message():
    class Route:
        artist = None
        song = None

    assert F._subject_query(Route(), "что за трек такой") == "что за трек такой"


# ── subject resolution: an exactly named subject must not be put to a vote ────
# Every case below is a question from the production dry run that never reached
# the LLM because BM25F rivals cleared DISAMBIGUATE_RATIO.


class _Route:
    intent = "facts"
    artist = None
    song = None


def _hits(*names):
    """Catalog hits, descending score, shaped like search_catalog output."""
    return [{"type": "song", "title": n, "artist": "Someone", "track_id": f"t{i}",
             "score": 1.0 - i * 0.05} for i, n in enumerate(names)]


async def _resolve(hits, message):
    import app.services.assistant.facts_executor as F
    original = F._resolve_subject_sync
    F._resolve_subject_sync = lambda *a, **kw: hits
    try:
        return await F.resolve_subject(None, "acct_1", route=_Route(), message=message,
                                       slots=None)
    finally:
        F._resolve_subject_sync = original


async def test_exact_title_beats_a_high_scoring_rival():
    subject, options = await _resolve(_hits("Bohemian Rhapsody", "Bed Chem"),
                                      "о чём песня Bohemian Rhapsody")
    assert options == []
    assert subject["title"] == "Bohemian Rhapsody"


async def test_exact_album_name_beats_rivals():
    hits = [{"type": "album", "album": "OK Computer", "artist": "Radiohead", "score": 1.0},
            {"type": "song", "title": "Bed Chem", "artist": "Sabrina", "score": 0.95},
            {"type": "song", "title": "OK Pal", "artist": "Someone", "score": 0.9}]
    subject, options = await _resolve(hits, "чем интересен альбом OK Computer")
    assert options == []
    assert subject["title"] == "OK Computer"


async def test_several_exact_matches_still_ask():
    # Four different tracks really are called Runaway — that is real ambiguity.
    subject, options = await _resolve(_hits("Runaway", "Runaway", "Runaway", "Runaway"),
                                      "кто продюсировал Runaway")
    assert subject is None
    assert len(options) == 4


async def test_partial_name_is_not_an_exact_match():
    # "Hurting" must not claim a question about "Hurt"; nothing matches word for
    # word, so the usual score-ratio path decides.
    subject, options = await _resolve(_hits("Hurting", "Hurtin'"),
                                      "что за история у трека Hurt")
    assert subject is None and options      # ratio path → still a disambiguate


# ── the JSON the model actually returns ──────────────────────────────────────


def test_parses_a_bare_object():
    assert F._parse_json_object('{"answer": "A.", "used": [1]}') == {"answer": "A.", "used": [1]}


def test_parses_through_fences_and_prose():
    # Both shapes cost a whole answer on the prod dry run: ask_llm(parse_json)
    # raised and the deterministic fallback took over.
    assert F._parse_json_object('```json\n{"answer": "A.", "used": [2]}\n```') \
        == {"answer": "A.", "used": [2]}
    assert F._parse_json_object('Sure! Here it is:\n{"answer": "A.", "used": [2]} Hope it helps.') \
        == {"answer": "A.", "used": [2]}


@pytest.mark.parametrize("raw", ["", "no object here", None, 42, "{broken"])
def test_unparseable_stays_unparseable(raw):
    assert F._parse_json_object(raw) is None


def test_album_pack_leads_with_items_naming_the_record():
    items = [
        {"text": "Radiohead is an English rock band", "source": "bio"},
        {"text": "Selway got the nickname Mad Dog", "source": "facts"},
        {"text": "OK Computer was recorded in a mansion near Bath", "source": "facts"},
    ]
    out = F._prefer_items_naming(items, "OK Computer")
    assert out[0]["text"].startswith("OK Computer")
    assert len(out) == 3           # nothing is dropped, only reordered


def test_deterministic_answer_leads_with_facts_and_stays_short():
    items = [{"text": "a very long biography blob", "source": "bio"}] + [
        {"text": f"fact {i}", "source": "facts"} for i in range(1, 8)
    ]
    out = F._deterministic_answer({"title": "X"}, items, "ru")
    lines = out.splitlines()
    assert lines[1] == "- fact 1"          # the bio no longer opens it
    assert len(lines) == 6                 # heading + 5 bullets


# ── raw-fact selection: stories in, boilerplate out, duplicates collapsed ────


def test_clean_story_cuts_on_a_sentence_boundary():
    story = ("The band spent days in the studio. " * 30).strip()
    out = F._clean_story(story)
    assert len(out) <= F.RAW_FACT_CHARS
    assert out.endswith("studio.")          # a whole sentence, not "stu…"


def test_clean_story_falls_back_to_a_hard_cut_without_sentences():
    out = F._clean_story("x" * (F.RAW_FACT_CHARS + 200))
    assert out.endswith("…")


def test_annotation_boilerplate_is_rewritten():
    raw = "Lyrics string: I shoot the lights out. Fact: Light is a motif in Kanye's work."
    assert F._strip_annotation_boilerplate(raw) \
        == "Line «I shoot the lights out» — Light is a motif in Kanye's work."
    # Non-annotation texts pass through untouched.
    assert F._strip_annotation_boilerplate("A plain story.") == "A plain story."


def test_raw_selection_orders_stories_before_annotations_and_caps_both():
    topics = ["recording sessions overdubs", "video shoot budget", "label single fight",
              "wayne world revival", "opera harmonies phonebook", "gong ending live",
              "muppets viral cover", "draft lyrics mongolian", "guitar solo one take",
              "radio premiere everett"]
    rows = ([{"fact": f"Songfacts tells a long detailed story about the {t} of this song",
              "source": "songfacts.com", "category": ""} for t in topics]
            + [{"fact": f"Lyrics string: line about {t}. Fact: a good annotation on {t} here",
                "source": "genius.com", "category": "genius_annotation"} for t in topics])
    out = F._select_raw_facts(rows)
    assert len(out) == F.MAX_RAW_STORIES + F.MAX_RAW_ANNOTATIONS
    assert out[0]["text"].startswith("Songfacts tells")
    assert out[F.MAX_RAW_STORIES]["text"].startswith("Line «")
    assert all(it["source"] == "facts" for it in out)


def test_raw_selection_skips_stub_rows():
    # The prod Bowie pool opens with a bare date range — not a story.
    rows = [{"fact": "January 8, 1947 - January 10, 2016", "source": "songfacts.com",
             "category": ""},
            {"fact": "Bowie grew up fascinated by American culture and once wrote to "
                     "the US embassy, who sent him a football uniform",
             "source": "songfacts.com", "category": ""}]
    out = F._select_raw_facts(rows)
    assert len(out) == 1
    assert "football uniform" in out[0]["text"]


def test_overflowing_pool_keeps_the_lead_and_reaches_the_back():
    texts = [f"item {i}" for i in range(30)]
    out = F._lead_and_spread(texts, 7)
    assert len(out) == 7
    assert out[:3] == ["item 0", "item 1", "item 2"]     # the editorial lead
    assert "item 29" not in out[:4]
    assert any(int(t.split()[1]) > 20 for t in out[3:])  # the back is reached


def test_small_pool_passes_through_untouched():
    texts = [f"item {i}" for i in range(5)]
    assert F._lead_and_spread(texts, 7) == texts


def test_raw_selection_drops_near_duplicates_across_sources():
    # songfacts and the Genius description retell the same anecdote — one slot.
    rows = [
        {"fact": "Freddie Mercury refused to cut the six minute single despite label pressure",
         "source": "songfacts.com", "category": ""},
        {"fact": "Mercury refused to cut the six minute single despite the label pressure",
         "source": "genius.com", "category": "genius_description"},
        {"fact": "The video was shot in three hours at the band rehearsal space",
         "source": "genius.com", "category": "genius_description"},
    ]
    out = F._select_raw_facts(rows)
    assert len(out) == 2
    assert "video" in out[1]["text"]


def test_system_prompt_carries_the_few_shot_in_the_answer_language():
    ru = F._system_prompt("ru")
    en = F._system_prompt("en")
    assert "**Запись**" in ru and "Russian" in ru
    assert "**The recording**" in en and "English" in en
    # The format pass resolved every placeholder and kept the JSON contract.
    assert "{lang_name}" not in ru and "{example_answer}" not in ru
    assert '{"answer": "...", "used": [1, 3], "follow_ups": ["...", "..."]}' in ru


# ── follow-up chips: model wording, code caps ────────────────────────────────


def test_followups_survive_when_sane():
    raw = '{"answer":"A.","used":[1],"follow_ups":["Почему лейбл был против?","Как снимали видео?"]}'
    assert F._sane_followups(raw, "ru") == ["Почему лейбл был против?", "Как снимали видео?"]


def test_followups_drop_generic_dupes_and_junk():
    raw = ('{"follow_ups":["расскажи ещё","Почему лейбл был против?",'
           '"почему лейбл был против","x","' + "щ" * 120 + '",42]}')
    assert F._sane_followups(raw, "ru") == ["Почему лейбл был против?"]


def test_followups_cap_at_three():
    raw = '{"follow_ups":["Вопрос номер один?","Вопрос номер два?","Вопрос номер три?","Вопрос номер четыре?"]}'
    assert len(F._sane_followups(raw, "ru")) == 3


@pytest.mark.parametrize("raw", ["", None, "{}", '{"follow_ups":"short"}'])
def test_followups_empty_when_nothing_usable(raw):
    assert F._sane_followups(raw, "ru") in ([], ["short"])  # a lone string ≥8 chars would pass


# ── related tracks: only the sample counterparts the listener actually has ───


class _FakePoint:
    def __init__(self, pid, payload):
        self.id, self.payload = pid, payload


class _FakeQdrant:
    def __init__(self, full):
        self._full = full

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        return [_FakePoint(i, self._full[i]) for i in ids if i in self._full]


def _patch_related(monkeypatch, links, points):
    from app.resources.metadata_db import MetadataDB
    monkeypatch.setattr(MetadataDB, "get_sample_links",
                        classmethod(lambda cls, c, s: links))
    monkeypatch.setattr("app.resources.qdrant_utils.light_points",
                        lambda client, collection: points)


def test_related_tracks_are_sample_counterparts_only(monkeypatch):
    points = [("t1", {"title": "Runaway", "artist": "Kanye West"}),
              ("t2", {"title": "Expo 83", "artist": "Backyard Heavies"}),
              ("t3", {"title": "Bound 2", "artist": "Kanye West"})]
    full = {"t2": {"title": "Expo 83", "artist": "Backyard Heavies",
                   "file_path": "/x.mp3", "duration": 200.0}}
    _patch_related(monkeypatch, {"samples": [{"song": "Expo 83",
                                              "artist": "Backyard Heavies"}],
                                 "sampled_by": []}, points)
    subject = {"kind": "song", "title": "Runaway", "artist": "Kanye West",
               "track_id": "t1"}
    out = F._sample_related_sync(_FakeQdrant(full), "acct_1", subject)
    # The other side of the sample — and NOT the artist's other tracks.
    assert [t["track_id"] for t in out] == ["t2"]
    assert out[0]["file_path"] == "/x.mp3"


def test_artists_and_albums_suggest_no_tracks(monkeypatch):
    _patch_related(monkeypatch, {"samples": [], "sampled_by": []}, [])
    assert F._sample_related_sync(_FakeQdrant({}), "acct_1",
                                  {"kind": "artist", "title": "Queen",
                                   "artist": "Queen"}) == []


def test_song_without_sample_links_suggests_nothing(monkeypatch):
    _patch_related(monkeypatch, {"samples": [], "sampled_by": []},
                   [("t1", {"title": "Runaway", "artist": "Kanye West"})])
    subject = {"kind": "song", "title": "Runaway", "artist": "Kanye West",
               "track_id": "t1"}
    assert F._sample_related_sync(_FakeQdrant({}), "acct_1", subject) == []


# ── explain mode: one tapped fact, not the whole dossier ─────────────────────
# The bug this section pins: asked to explain "«A» сэмплирует «B»", the branch
# handed the model all eighteen pack items and got back the release year, the
# label and four unrelated trivia — everything except the link the listener
# tapped. Narrowing the pack is the fix, and silence is a legal outcome.


_FACT = "«Runaway» (Kanye West) сэмплирует «Expo 83» (Backyard Heavies)"
_SUBJECT = {"kind": "song", "title": "Runaway", "artist": "Kanye West"}
_SUBJECT_TOKENS = {"runaway", "kanye", "west"}


def test_the_fact_itself_is_always_evidence_one():
    out = F._fact_evidence([], _FACT, _SUBJECT_TOKENS)
    assert out[0]["text"] == _FACT
    assert len(out) == 1


def test_only_items_about_the_fact_join_the_evidence():
    items = [
        {"text": "Содержит сэмпл: Expo 83 by Backyard Heavies", "source": "credits"},
        {"text": "Runaway — Kanye West: album My Beautiful Dark Twisted Fantasy",
         "source": "catalog"},
        {"text": "Продюсер: Mike Dean", "source": "credits"},
    ]
    texts = [it["text"] for it in F._fact_evidence(items, _FACT, _SUBJECT_TOKENS)]
    assert "Содержит сэмпл: Expo 83 by Backyard Heavies" in texts
    assert not any("Mike Dean" in t for t in texts)
    assert not any("Twisted Fantasy" in t for t in texts)


def test_the_subjects_own_name_is_not_evidence_of_relevance():
    """Every item in a song's pack names the song. Counting those tokens would
    readmit the whole pack — which is the failure being fixed."""
    items = [{"text": "Runaway by Kanye West was released in 2010", "source": "catalog"}]
    assert len(F._fact_evidence(items, _FACT, _SUBJECT_TOKENS)) == 1


def test_the_fact_is_not_repeated_when_the_pack_already_holds_it():
    items = [{"text": _FACT, "source": "facts"}]
    assert len(F._fact_evidence(items, _FACT, _SUBJECT_TOKENS)) == 1


def test_evidence_is_capped():
    items = [{"text": f"Expo 83 Backyard Heavies detail {i}", "source": "facts"}
             for i in range(40)]
    out = F._fact_evidence(items, _FACT, _SUBJECT_TOKENS)
    assert len(out) == F.EXPLAIN_MAX_EVIDENCE


def test_a_fact_made_only_of_the_subject_name_still_matches_on_it():
    """Nothing distinctive left after removing the subject tokens — matching on
    the name beats admitting the entire pack."""
    items = [{"text": "Runaway by Kanye West runs nine minutes", "source": "facts"},
             {"text": "Продюсер: Mike Dean", "source": "credits"}]
    out = F._fact_evidence(items, "Runaway — Kanye West", _SUBJECT_TOKENS)
    assert len(out) == 2
    assert "nine minutes" in out[1]["text"]


# ── the queries the model writes for the web ─────────────────────────────────


def test_usable_queries_survive():
    raw = ('{"queries":["Kanye West Runaway Expo 83 sample",'
           '"Runaway Kanye West sample story",'
           '"Backyard Heavies Expo 83 sampled"]}')
    assert F._sane_queries(raw, _FACT, _SUBJECT) == [
        "Kanye West Runaway Expo 83 sample",
        "Runaway Kanye West sample story",
        "Backyard Heavies Expo 83 sampled",
    ]


def test_a_query_naming_nothing_from_the_fact_is_dropped():
    """The observed small-model failure: the listener's own sentence retyped
    into the search box. Searching it burns one of three budgeted searches."""
    raw = '{"queries":["а что это вообще значит","Kanye West Runaway sample"]}'
    assert F._sane_queries(raw, _FACT, _SUBJECT) == ["Kanye West Runaway sample"]


def test_quotes_and_operators_are_stripped():
    raw = '{"queries":["\\"Runaway\\" Kanye West sample? site:genius.com"]}'
    assert F._sane_queries(raw, _FACT, _SUBJECT) == ["Runaway Kanye West sample"]


def test_too_short_and_too_long_queries_are_dropped():
    raw = ('{"queries":["Runaway",'
           '"' + " ".join(["Runaway"] * 20) + '",'
           '"Kanye West Runaway sample"]}')
    assert F._sane_queries(raw, _FACT, _SUBJECT) == ["Kanye West Runaway sample"]


def test_rewordings_of_one_query_do_not_eat_the_budget():
    raw = ('{"queries":["Kanye West Runaway sample","kanye west runaway sample",'
           '"Kanye  West   Runaway  sample"]}')
    assert F._sane_queries(raw, _FACT, _SUBJECT) == ["Kanye West Runaway sample"]


def test_never_more_than_three_searches_are_offered():
    raw = '{"queries":[%s]}' % ",".join(
        f'"Kanye West Runaway sample {i}"' for i in range(9))
    assert len(F._sane_queries(raw, _FACT, _SUBJECT)) == F.EXPLAIN_MAX_WEB_SEARCHES


@pytest.mark.parametrize("raw", ["", "not json", '{"queries":[]}',
                                 '{"nope":["a b c"]}', None])
def test_unusable_generations_yield_nothing_to_search(raw):
    assert F._sane_queries(raw, _FACT, _SUBJECT) == []


def test_a_bare_string_is_accepted_as_one_query():
    assert F._sane_queries('{"queries":"Kanye West Runaway sample"}', _FACT, _SUBJECT) \
        == ["Kanye West Runaway sample"]


def test_code_built_queries_cover_the_same_three_angles():
    """An unreachable LLM still gets one honest attempt at the web."""
    out = F._fallback_fact_queries(_SUBJECT, _FACT)
    assert 1 <= len(out) <= F.EXPLAIN_MAX_WEB_SEARCHES
    assert all("kanye" in q.lower() or "expo" in q.lower() or "heavies" in q.lower()
               for q in out)


def test_code_built_queries_for_an_artist_ask_about_the_band():
    subject = {"kind": "artist", "title": "Radiohead", "artist": "Radiohead"}
    out = F._fallback_fact_queries(subject, "Radiohead formed in Abingdon")
    assert any("band history" in q for q in out)


# ── the deliberate refusal ───────────────────────────────────────────────────


def test_an_empty_answer_is_read_as_a_refusal_not_a_glitch():
    """`{"answer":"","used":[]}` is what the prompt asks for when nothing
    explains the fact. Retrying it is how a model gets talked into inventing."""
    assert F._declined('{"answer":"","used":[]}') is True
    assert F._declined('{"answer":"   ","used":[]}') is True


@pytest.mark.parametrize("raw", [
    "I think it means the song is about pride.",   # prose, no object → retry
    '{"used":[1]}',                                # no answer key at all
    "",
    '{"answer":"It samples an old funk record.","used":[2]}',
])
def test_everything_else_is_not_a_refusal(raw):
    assert F._declined(raw) is False


def test_the_honest_empty_answer_promises_nothing():
    ru = F._no_explanation("ru")
    en = F._no_explanation("en")
    assert "не нашёл" in ru and "придумывать" in ru
    assert "couldn't find" in en and "invent" in en


def test_a_lone_one_liner_goes_straight_to_the_web():
    """Nothing in the library matched and the fact is one line: the only thing
    the model could do with it is rephrase it, so don't ask."""
    assert F._needs_the_web_first([{"text": _FACT}], _FACT) is True


def test_a_long_stored_fact_is_read_before_searching():
    """A fact that carries its own story is worth a look before three searches."""
    long_fact = ("«Runaway» строится на сэмпле «Expo 83», который Канье нашёл на "
                 "виниле в чикагском магазине; партия фортепиано в интро — это "
                 "замедленный фрагмент той же записи, и именно из-за него трек "
                 "пришлось перезаписывать вживую для альбомной версии.")
    assert len(long_fact) >= F.EXPLAIN_SELF_CONTAINED_CHARS
    assert F._needs_the_web_first([{"text": long_fact}], long_fact) is False


def test_related_library_material_is_always_read_first():
    evidence = [{"text": _FACT}, {"text": "Содержит сэмпл: Expo 83"}]
    assert F._needs_the_web_first(evidence, _FACT) is False
