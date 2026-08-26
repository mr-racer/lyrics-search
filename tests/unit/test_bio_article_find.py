"""The article ladder: rung zero must not throw away the artists it finds.

The title probe is the rung that finds the OBVIOUS article — `Metallica`,
`INXS`, `Jay-Z`. A crash there is invisible from the outside: the caller
catches every exception and quietly writes a web-researched bio instead, so
the artists with the best Wikipedia coverage are exactly the ones that never
get a Wikipedia bio. Production, 2026-08-22: 72 of 149 bios.
"""
import pytest

from app.resources import mediawiki
from app.resources.model_registry import ModelRegistry
from app.services.bio_v2 import article

pytestmark = pytest.mark.unit


def _row(title, lang="en", snippet="American heavy metal band."):
    return {"url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "title": title, "snippet": snippet}


@pytest.fixture
def stub_wiki(monkeypatch):
    """Wikipedia and the cross-encoder, off the network."""
    state = {"probe": [], "search": [], "probe_calls": [], "search_calls": []}

    monkeypatch.setattr(mediawiki, "probe_titles_batch",
                        lambda name, lang="en", **kw: (
                            state["probe_calls"].append((name, lang))
                            or list(state["probe"])))
    monkeypatch.setattr(mediawiki, "search",
                        lambda term, lang="en", **kw: (
                            state["search_calls"].append((term, lang))
                            or list(state["search"])))
    monkeypatch.setattr(ModelRegistry, "ce_probabilities",
                        staticmethod(lambda query, docs: [0.9] * len(docs)))
    return state


def test_title_probe_hit_is_returned(stub_wiki):
    """The rung-zero find is the answer, and the search rungs never run."""
    stub_wiki["probe"] = [_row("Metallica")]

    found, rejected = article.find("Metallica")

    assert found is not None, "the probed article was dropped"
    assert found["title"] == "Metallica"
    assert stub_wiki["search_calls"] == [], "the ladder advanced past a hit"


def test_probe_miss_falls_through_to_search(stub_wiki):
    """Nothing probed is not a failure — the search rungs still answer."""
    stub_wiki["search"] = [_row("Merk (musician)")]

    found, _ = article.find("Merk")

    assert found is not None and found["title"] == "Merk (musician)"


def test_the_whole_ladder_costs_two_requests(stub_wiki):
    """The budget IS the ladder: one probe, one search, one language.

    It used to be one request per disambiguator per language plus four searches
    — 9 to 10 requests per artist, measured on the production library, for a
    name most of them were about to discard anyway.
    """
    found, _ = article.find("Nobody At All")

    assert found is None
    assert stub_wiki["probe_calls"] == [("Nobody At All", "en")]
    assert stub_wiki["search_calls"] == [("Nobody At All", "en")]


def test_cyrillic_name_does_not_fall_back_to_en_wikipedia(stub_wiki):
    """The second language pass is gone; that artist goes to the web instead."""
    found, _ = article.find("Нобади")

    assert found is None
    assert [lang for _, lang in stub_wiki["probe_calls"]] == ["ru"]
    assert [lang for _, lang in stub_wiki["search_calls"]] == ["ru"]


def test_search_rung_asks_the_bare_name(stub_wiki):
    """Appending "band musician" is rung zero's job, and doing it in a query
    makes the answer worse: Phoenix returns people surnamed Phoenix."""
    stub_wiki["search"] = [_row("Phoenix (band)")]

    article.find("Phoenix")

    assert stub_wiki["search_calls"] == [("Phoenix", "en")]


def test_every_probed_title_is_a_candidate(stub_wiki, monkeypatch):
    """One request already paid for all the suffixes, so the gate — not the
    order of DISAMBIGUATORS — picks which of them is this artist."""
    stub_wiki["probe"] = [_row("Bullet (band)", snippet="Ghanaian afrobeats duo."),
                          _row("Bullet (musician)", snippet="Swedish hard rock.")]
    monkeypatch.setattr(
        ModelRegistry, "ce_probabilities",
        staticmethod(lambda query, docs: [0.9 if "Swedish" in d else 0.05
                                          for d in docs]))

    found, _ = article.find("Bullet")

    assert found is not None and found["title"] == "Bullet (musician)"
    assert stub_wiki["search_calls"] == [], "the ladder advanced past a hit"


def test_probed_wrong_entity_is_rejected_not_returned(stub_wiki, monkeypatch):
    """A same-name footballer clears every shape check; the gate is what stops
    it, and stopping it must leave the ladder able to keep climbing."""
    stub_wiki["probe"] = [_row("Bullet (footballer)", snippet="Ghanaian defender.")]
    stub_wiki["search"] = [_row("Bullet (Swedish band)", snippet="Swedish hard rock band.")]
    monkeypatch.setattr(
        ModelRegistry, "ce_probabilities",
        staticmethod(lambda query, docs: [0.9 if "band" in d else 0.05 for d in docs]))

    found, rejected = article.find("Bullet")

    assert found is not None and found["title"] == "Bullet (Swedish band)"
    assert any("footballer" in r[0] for r in rejected)
