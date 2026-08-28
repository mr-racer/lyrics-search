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
from app.resources.models import STATS, ModelUnavailable
from app.services.bio_v2 import article

pytestmark = pytest.mark.unit


def _cross_encoder_down(query, docs):
    """The leg is down: it raises, it does not answer ``None``."""
    raise ModelUnavailable("cross_encoder", "load", "no reranker here")


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


class TestTheGateWithoutAJudge:
    """The gate exists because a same-name footballer clears every OTHER check.

    It used to read a missing cross-encoder as ``[1.0] * len(pool)`` — a perfect
    score for every candidate, so the whole pool cleared a gate that had nothing
    to judge with. It cost a measurement: four artists "found" a Wikipedia
    article that does not exist, and the run reported success.
    """

    def test_no_cross_encoder_admits_nobody(self, monkeypatch):
        monkeypatch.setattr(ModelRegistry, "ce_probabilities",
                            staticmethod(_cross_encoder_down))
        pool = [{"rank": 0, "url": "https://en.wikipedia.org/wiki/Merk",
                 "title": "Merk", "snippet": "A Hungarian village."}]

        best, rejected = article.gate("Merk", pool)

        assert best is None, "an unjudged candidate is not an accepted candidate"
        assert len(rejected) == 1
        assert "cross-encoder unavailable" in rejected[0][2]

    def test_the_ladder_reports_no_article_rather_than_the_wrong_one(
            self, stub_wiki, monkeypatch):
        """``find`` must not fall through to "found something" either: both
        rungs advance on "nothing passed the gate", and nothing can pass it."""
        stub_wiki["probe"] = [_row("Merk", snippet="A village in Hungary.")]
        stub_wiki["search"] = [_row("Merk (coin)", snippet="A silver coin.")]
        monkeypatch.setattr(ModelRegistry, "ce_probabilities",
                            staticmethod(_cross_encoder_down))

        found, rejected = article.find("Merk")

        assert found is None
        assert rejected, "the refusal has to say why, or it reads as 'no results'"

    def test_the_degradation_is_counted(self, monkeypatch):
        monkeypatch.setattr(ModelRegistry, "ce_probabilities",
                            staticmethod(_cross_encoder_down))
        STATS.reset()
        article.gate("Merk", [{"rank": 0, "url": "https://en.wikipedia.org/wiki/M",
                               "title": "M", "snippet": "s"}])
        assert STATS.snapshot()["degradations"][
            "cross_encoder/bio.article_gate"] == 1
