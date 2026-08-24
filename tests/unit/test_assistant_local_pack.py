"""Unit tests for the library-only grounding pack (assistant/local_pack.py).

The pack this builds is what the assistant answers from BEFORE anything is
downloaded. Two properties matter enough to pin:

* the structural material — sample links, credits, gems — actually reaches it.
  It did not before: the samples card was built from the database and then
  answered from the web, with its own list nowhere in the prompt;
* both sample storages are read. The production library predates the normalized
  ``sample_links`` table and keeps every link in the ``songs.samples_json``
  cache, so a reader that consults only the table finds nothing there.
"""

import pytest

from app.services.assistant.contracts import Fact, Subject

pytestmark = pytest.mark.unit

COLLECTION = "acct_1"


class _DB:
    """A stand-in for MetadataDB with only the readers local_pack calls."""

    def __init__(self, *, song_facts=None, artist_facts=None, links=None,
                 raw=None, relations=None, gems=None):
        self._song_facts = song_facts or {}
        self._artist_facts = artist_facts or {}
        self._links = links or {}
        self._raw = raw or {}
        self._relations = relations or {}
        self._gems = gems or {}

    def get_song_facts_rich(self, slug, collection_name):
        return self._song_facts.get(slug, [])

    def get_artist_facts_rich(self, slug, collection_name):
        return self._artist_facts.get(slug, [])

    def get_sample_links(self, collection_name, slug):
        return self._links.get(slug, {"samples": [], "sampled_by": []})

    def get_song_relations_raw(self, slugs):
        return {s: self._raw[s] for s in slugs if s in self._raw}

    def get_song_relations_bulk(self, slugs):
        return {s: self._relations[s] for s in slugs if s in self._relations}

    def get_track_gems(self, track_id, collection_name):
        return self._gems.get(track_id, [])


def _pack(db, subject, *, query="what samples are in this?", lang="ru",
          ranker=None):
    from app.services.assistant import local_pack

    return local_pack.build(COLLECTION, subject, query, lang=lang, db=db,
                            ranker=ranker)


def _song_subject(**kw):
    base = dict(song_slug="kanye-west-runaway", artist_slug="kanye-west",
                artist_name="Kanye West", song_title="Runaway",
                track_id="t1", how="pinned")
    base.update(kw)
    return Subject(**base)


def _texts(pack):
    return [item.text for item in pack.items]


# ── structural material ──────────────────────────────────────────────────────


def test_sample_links_reach_the_pack_from_the_normalized_table():
    db = _DB(links={"kanye-west-runaway": {
        "samples": [{"song": "Expo 83", "artist": "Backyard Heavies",
                     "slug": "backyard-heavies-expo-83", "relation": "sample",
                     "evidence": "The drums come from Expo 83."}],
        "sampled_by": []}})

    pack = _pack(db, _song_subject())

    joined = " | ".join(_texts(pack))
    assert "Expo 83" in joined
    assert "Backyard Heavies" in joined
    # The evidence sentence is the story the card promised — it must travel too.
    assert "The drums come from Expo 83." in joined


def test_sample_links_reach_the_pack_from_samples_json_alone():
    """The production shape: sample_links empty, samples_json full."""
    db = _DB(raw={"kanye-west-runaway": {
        "samples": [{"song": "Expo 83", "artist": "Backyard Heavies"}],
        "sampled_by": []}})

    pack = _pack(db, _song_subject())

    assert "Expo 83" in " | ".join(_texts(pack))


def test_the_two_storages_do_not_double_up():
    """Same link in both places is one item, not two."""
    entry = {"song": "Expo 83", "artist": "Backyard Heavies",
             "slug": "backyard-heavies-expo-83"}
    db = _DB(links={"kanye-west-runaway": {"samples": [entry], "sampled_by": []}},
             raw={"kanye-west-runaway": {"samples": [dict(entry)],
                                         "sampled_by": []}})

    pack = _pack(db, _song_subject())

    assert sum("Expo 83" in t for t in _texts(pack)) == 1
    assert len(pack.links) == 1


def test_both_directions_are_kept_apart():
    db = _DB(links={"kanye-west-runaway": {
        "samples": [{"song": "Expo 83", "artist": "Backyard Heavies"}],
        "sampled_by": [{"song": "Some Later Track", "artist": "Someone"}]}})

    pack = _pack(db, _song_subject())

    joined = " | ".join(_texts(pack))
    assert "Expo 83" in joined and "Some Later Track" in joined
    directions = {link["direction"] for link in pack.links}
    assert directions == {"samples", "sampled_by"}


def test_credits_and_gems_reach_the_pack():
    db = _DB(relations={"kanye-west-runaway": {"producer": "Emile Haynie",
                                               "label": "Roc-A-Fella"}},
             gems={"t1": [{"kind": "namedrop", "canonical": "pusha t",
                           "display": "Pusha T", "quote": "…"}]})

    joined = " | ".join(_texts(_pack(db, _song_subject())))

    assert "Emile Haynie" in joined
    assert "Roc-A-Fella" in joined
    assert "Pusha T" in joined


def test_structural_items_carry_no_probability():
    """They are facts ABOUT the subject, not candidates FOR it — so they must
    not be judged by the cross-encoder veto that ranked facts are judged by."""
    db = _DB(relations={"kanye-west-runaway": {"producer": "Emile Haynie"}})

    pack = _pack(db, _song_subject())

    assert pack.items
    assert all(item.ce_prob is None for item in pack.items)


# ── neighbours ───────────────────────────────────────────────────────────────


def test_the_other_side_s_facts_are_pulled_in():
    """The explanation of a sample lives in the OTHER song's facts, and the
    other song is known by name — no similarity search needed to find it."""
    db = _DB(
        links={"kanye-west-runaway": {
            "samples": [{"song": "Expo 83", "artist": "Backyard Heavies",
                         "slug": "backyard-heavies-expo-83"}],
            "sampled_by": []}},
        song_facts={"backyard-heavies-expo-83": [
            {"fact": "Expo 83 was recorded in 1971 for a label that folded.",
             "source": "songfacts", "category": ""}]})

    seen = []

    def ranker(facts, query):
        seen.extend(f.text for f in facts)
        return facts

    pack = _pack(db, _song_subject(), ranker=ranker)

    assert any("recorded in 1971" in t for t in seen)
    assert any("recorded in 1971" in t for t in _texts(pack))


def test_subject_facts_are_ranked_not_dumped():
    db = _DB(song_facts={"kanye-west-runaway": [
        {"fact": "A", "source": "songfacts", "category": ""},
        {"fact": "B", "source": "songfacts", "category": ""}]})

    def ranker(facts, query):
        keep = [f for f in facts if f.text == "B"]
        for f in keep:
            f.ce_prob = 0.8
        return keep

    pack = _pack(db, _song_subject(), ranker=ranker)

    assert "B" in _texts(pack)
    assert "A" not in _texts(pack)


def test_ranked_facts_keep_their_probability():
    db = _DB(song_facts={"kanye-west-runaway": [
        {"fact": "A", "source": "songfacts", "category": ""}]})

    def ranker(facts, query):
        for f in facts:
            f.ce_prob = 0.42
        return facts

    pack = _pack(db, _song_subject(), ranker=ranker)

    assert [i.ce_prob for i in pack.items if i.text == "A"] == [0.42]


# ── shape and edges ──────────────────────────────────────────────────────────


def test_items_are_numbered_from_one_without_gaps():
    db = _DB(relations={"kanye-west-runaway": {"producer": "Emile Haynie",
                                               "label": "Roc-A-Fella"}})

    pack = _pack(db, _song_subject())

    assert [i.n for i in pack.items] == list(range(1, len(pack.items) + 1))


def test_structure_leads_the_pack():
    """A ranked fact must never push the verified link out of the model's view."""
    db = _DB(links={"kanye-west-runaway": {
                 "samples": [{"song": "Expo 83", "artist": "Backyard Heavies"}],
                 "sampled_by": []}},
             song_facts={"kanye-west-runaway": [
                 {"fact": "A", "source": "songfacts", "category": ""}]})

    def ranker(facts, query):
        for f in facts:
            f.ce_prob = 0.9
        return facts

    pack = _pack(db, _song_subject(), ranker=ranker)

    assert "Expo 83" in pack.items[0].text


def test_unresolved_subject_yields_an_empty_pack():
    pack = _pack(_DB(), Subject(how="none"))

    assert pack.items == []
    assert pack.links == []


def test_a_broken_reader_does_not_break_the_pack():
    """SQLite going wrong costs items, never the turn."""
    class _Angry(_DB):
        def get_sample_links(self, collection_name, slug):
            raise RuntimeError("no such table: sample_links")

    db = _Angry(relations={"kanye-west-runaway": {"producer": "Emile Haynie"}})

    joined = " | ".join(_texts(_pack(db, _song_subject())))

    assert "Emile Haynie" in joined


def test_english_and_russian_render_differently():
    db = _DB(relations={"kanye-west-runaway": {"producer": "Emile Haynie"}})

    ru = " | ".join(_texts(_pack(db, _song_subject(), lang="ru")))
    en = " | ".join(_texts(_pack(db, _song_subject(), lang="en")))

    assert ru != en
    assert "Emile Haynie" in ru and "Emile Haynie" in en


def test_artist_subject_takes_the_artist_path():
    """An artist has no sample links and no gems — just facts, and asking the
    song readers for them would be a slug-shaped lookup that matches nothing."""
    db = _DB(artist_facts={"kanye-west": [
        {"fact": "Born in Atlanta.", "source": "wiki", "category": ""}]})

    pack = _pack(db, Subject(artist_slug="kanye-west", artist_name="Kanye West",
                             how="pinned"))

    assert any("Born in Atlanta." in t for t in _texts(pack))
    assert pack.links == []


def test_facts_returned_as_plain_objects_are_accepted():
    """The ranker contract is a list of Fact in, a list of Fact out."""
    db = _DB(song_facts={"kanye-west-runaway": [
        {"fact": "A", "source": "songfacts", "category": ""}]})

    captured = {}

    def ranker(facts, query):
        captured["types"] = {type(f) for f in facts}
        return facts

    _pack(db, _song_subject(), ranker=ranker)

    assert captured["types"] == {Fact}
