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
