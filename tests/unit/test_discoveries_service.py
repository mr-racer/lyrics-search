"""Unit tests for the assistant page's discovery rail.

MetadataDB and light_points are stubbed — what is under test is the filtering
(which links are worth showing) and the assembly, not SQLite.
"""

from __future__ import annotations

import pytest

from app.services import discoveries_service as ds
from app.services.song_facts_service import get_song_facts_key


def _point(track_id, title, artist, artist_slug="artist"):
    return (track_id, {"title": title, "artist": artist,
                       "cover_art_path": f"/covers/{track_id}.jpg",
                       "primary_artist_slug": artist_slug})


POINTS = [
    _point("t1", "Runaway", "Kanye West", "kanye-west"),
    _point("t2", "Expensive Shit", "Fela Kuti", "fela-kuti"),
    _point("t3", "Bound 2", "Kanye West", "kanye-west"),
    _point("t4", "Blue", "Mono", "mono"),
    _point("t5", "Grey", "Mono", "mono"),
    _point("t6", "White", "Mono", "mono"),
]


def _link(src, dst, relation):
    return {
        "src_slug": get_song_facts_key(src[1]["artist"], src[1]["title"]),
        "src_title": src[1]["title"], "src_artist_slug": "x",
        "dst_slug": get_song_facts_key(dst[1]["artist"], dst[1]["title"]),
        "dst_title": dst[1]["title"], "dst_artist_slug": "y",
        "relation": relation, "evidence": None,
    }


@pytest.fixture(autouse=True)
def _clear_cache():
    ds.invalidate()
    yield
    ds.invalidate()


@pytest.fixture
def stub_db(monkeypatch):
    state = {"links": [], "outgoing": [], "raw": {}, "relations": {}, "bios": []}

    monkeypatch.setattr(ds.MetadataDB, "get_in_library_sample_links",
                        classmethod(lambda cls, c: state["links"]))
    monkeypatch.setattr(ds.MetadataDB, "get_outgoing_sample_links",
                        classmethod(lambda cls, c: state["outgoing"]))
    monkeypatch.setattr(ds.MetadataDB, "get_song_relations_raw",
                        classmethod(lambda cls, slugs: state["raw"]))
    monkeypatch.setattr(ds.MetadataDB, "get_song_relations_bulk",
                        classmethod(lambda cls, slugs: state["relations"]))
    monkeypatch.setattr(ds.MetadataDB, "get_artist_slugs_with_bio",
                        classmethod(lambda cls, c, lang: state["bios"]))
    monkeypatch.setattr("app.resources.qdrant_utils.light_points",
                        lambda client, collection: POINTS)
    return state


def _cards(kind):
    return [c for c in ds.build_discoveries(None, "acct_1", lang="ru") if c["kind"] == kind]


def test_relation_card_needs_both_sides_in_library(stub_db):
    outside = ("tX", {"title": "Some 70s Funk", "artist": "Unknown", "cover_art_path": None,
                      "primary_artist_slug": "unknown"})
    stub_db["links"] = [_link(POINTS[0], POINTS[1], "sample"),
                        _link(POINTS[2], outside, "sample")]
    cards = _cards("relation")
    assert len(cards) == 1
    assert cards[0]["items"][0]["track_id"] == "t1"
    assert cards[0]["items"][1]["track_id"] == "t2"


@pytest.mark.parametrize("relation", ["lyrical_reference", "inspiration", "other", "remix"])
def test_weak_relation_kinds_are_dropped(stub_db, relation):
    stub_db["links"] = [_link(POINTS[0], POINTS[1], relation)]
    assert _cards("relation") == []


@pytest.mark.parametrize("relation", ["sample", "interpolation", "cover"])
def test_strong_relation_kinds_survive(stub_db, relation):
    stub_db["links"] = [_link(POINTS[0], POINTS[1], relation)]
    assert len(_cards("relation")) == 1


def test_self_link_and_duplicate_pair_are_dropped(stub_db):
    stub_db["links"] = [
        _link(POINTS[0], POINTS[0], "sample"),           # self
        _link(POINTS[0], POINTS[1], "sample"),
        _link(POINTS[1], POINTS[0], "interpolation"),    # same pair, other way
    ]
    assert len(_cards("relation")) == 1


def test_relation_card_sends_a_facts_turn(stub_db):
    stub_db["links"] = [_link(POINTS[0], POINTS[1], "sample")]
    card = _cards("relation")[0]
    assert card["intent"] == "general"
    assert "Runaway" in card["prompt"]


def test_relation_card_asks_about_its_own_finding(stub_db):
    """The card states a link, so tapping it must ask about THAT link. The old
    prompt («расскажи про «X»») asked about the track instead, and the answer
    came back as a list of everything known about it — with the sample the card
    was built on nowhere in it."""
    stub_db["links"] = [_link(POINTS[0], POINTS[1], "sample")]
    card = _cards("relation")[0]
    fact = card["fact"]
    assert "Runaway" in fact and "сэмплирует" in fact
    assert card["items"][1]["title"] in fact       # both sides are named
    assert card["prompt"] == f"объясни: {fact}"
    assert card["track_id"] == "t1"                # subject pinned by id


def test_relation_pairs_come_from_the_samples_json_cache_too(stub_db):
    # The production library predates the normalized table: its links live only
    # in songs.samples_json, entries as {song, artist}.
    stub_db["raw"] = {
        get_song_facts_key("Kanye West", "Runaway"): {
            "samples": [{"song": "Expensive Shit", "artist": "Fela Kuti"},
                        {"song": "Some 70s Funk", "artist": "Nobody Here"},
                        {"song": "A hawk chases a dove", "artist": None}],
            "sampled_by": [],
        },
    }
    cards = _cards("relation")
    assert len(cards) == 1                      # only the pair that resolves
    assert cards[0]["items"][1]["track_id"] == "t2"


def _out_link(src, dst_title, dst_artist, relation="sample", dst_slug=None):
    return {"src_slug": get_song_facts_key(src[1]["artist"], src[1]["title"]),
            "dst_title": dst_title, "dst_artist": dst_artist,
            "dst_slug": dst_slug, "relation": relation}


def test_sample_card_needs_two_links_and_counts_unresolved(stub_db):
    # Runaway is built from two records; one of them never resolved to a
    # library song — it still counts, «сколько сэмплов» is about the track.
    stub_db["outgoing"] = [
        _out_link(POINTS[0], "Expensive Shit", "Fela Kuti"),
        _out_link(POINTS[0], "Some 70s Funk", "Nobody Here"),
        _out_link(POINTS[2], "Only One Link", "Someone"),
    ]
    cards = _cards("samples")
    assert [c["headline"] for c in cards] == ["Runaway"]
    card = cards[0]
    assert card["count"] == 2
    assert card["badge"] == "2 сэмпла"
    assert card["intent"] == "general"
    assert card["track_id"] == "t1"
    assert card["prompt"] == "какие сэмплы использованы в «Runaway»?"


def test_sample_counts_merge_both_storages_without_double_counting(stub_db):
    # The same Fela Kuti link lives in BOTH the normalized table and the old
    # samples_json cache — it is one sample, not two. The cache adds a second,
    # different record, which clears the threshold.
    runaway = get_song_facts_key("Kanye West", "Runaway")
    stub_db["outgoing"] = [
        _out_link(POINTS[0], "Expensive Shit", "Fela Kuti",
                  dst_slug=get_song_facts_key("Fela Kuti", "Expensive Shit")),
    ]
    stub_db["raw"] = {runaway: {
        "samples": [{"song": "Expensive Shit", "artist": "Fela Kuti",
                     "slug": get_song_facts_key("Fela Kuti", "Expensive Shit")},
                    {"song": "Mind Playing Tricks", "artist": "Geto Boys"}],
        "sampled_by": [],
    }}
    cards = _cards("samples")
    assert len(cards) == 1
    assert cards[0]["count"] == 2


def test_sample_card_ignores_covers(stub_db):
    stub_db["outgoing"] = [
        _out_link(POINTS[0], "A", "B", relation="cover"),
        _out_link(POINTS[0], "C", "D", relation="cover"),
    ]
    assert _cards("samples") == []


def test_producer_needs_the_threshold(stub_db):
    slugs = [get_song_facts_key(p[1]["artist"], p[1]["title"]) for p in POINTS]
    stub_db["relations"] = {
        slugs[0]: {"producer": "Mike Dean"},
        slugs[1]: {"producer": "Mike Dean, Solo Guy"},
        slugs[2]: {"producer": "mike dean"},   # same person, other casing
        slugs[3]: {"producer": "Solo Guy"},
    }
    cards = _cards("producer")
    assert [c["headline"] for c in cards] == ["Mike Dean"]
    assert cards[0]["count"] == 3
    assert cards[0]["intent"] == "playlist"
    assert "Mike Dean" in cards[0]["prompt"]


def test_self_credited_performer_is_not_a_producer_finding(stub_db):
    slugs = [get_song_facts_key(p[1]["artist"], p[1]["title"]) for p in POINTS]
    # Every Mono track credits Mono — a real prod case («Lorde — 21 трек»).
    stub_db["relations"] = {s: {"producer": "Mono"} for s in slugs[3:]}
    assert _cards("producer") == []


def test_artist_card_needs_a_real_presence(stub_db):
    stub_db["bios"] = ["kanye-west", "mono", "someone-not-here"]
    cards = _cards("artist")
    # Kanye has 2 tracks here, Mono 3 — only Mono clears MIN_ARTIST_TRACKS.
    assert [c["headline"] for c in cards] == ["Mono"]
    assert cards[0]["count"] == 3
    assert cards[0]["artist_slug"] == "mono"


def test_empty_library_yields_no_cards(stub_db, monkeypatch):
    monkeypatch.setattr("app.resources.qdrant_utils.light_points",
                        lambda client, collection: [])
    assert ds.build_discoveries(None, "acct_1", lang="ru") == []


def test_kinds_are_interleaved_and_capped(stub_db):
    slugs = [get_song_facts_key(p[1]["artist"], p[1]["title"]) for p in POINTS]
    stub_db["links"] = [_link(POINTS[0], POINTS[1], "sample")]
    stub_db["relations"] = {s: {"producer": "Mike Dean"} for s in slugs}
    stub_db["bios"] = ["kanye-west", "mono"]
    cards = ds.build_discoveries(None, "acct_1", lang="ru", limit=3)
    assert len(cards) == 3
    assert {c["kind"] for c in cards} == {"relation", "producer", "artist"}


def test_english_strings(stub_db):
    stub_db["bios"] = ["mono"]
    card = [c for c in ds.build_discoveries(None, "acct_1", lang="en")
            if c["kind"] == "artist"][0]
    assert card["prompt"] == "tell me about Mono"
    assert "in your library" in card["subline"]
