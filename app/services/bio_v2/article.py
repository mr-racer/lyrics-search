"""Finding the one Wikipedia article an artist bio should be written from.

Three checks, in order of cost, and the order is the design. The shape checks
are free and prove the page is an article. Language preference decides WHICH
article. The cross-encoder is a GATE, not the ranking — it removes the wrong
entity and nothing else.

That last distinction was learned the expensive way: sorting candidates by
cross-encoder score outright picked the French M83 article (0.87) over the
English one (0.44), because a snippet in the reader's own language scores
higher — which says nothing about which article is the better source.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.resources import mediawiki
from app.resources.model_registry import ModelRegistry
from app.resources.models import STATS, ModelError

logger = logging.getLogger(__name__)

# Above this, the article is about a musical artist at all. Everything below is
# a coin, a bomb, a footballer or a pharmaceutical company — measured rejects,
# every one of them, at scores between 0.00 and 0.27.
CE_ARTICLE_GATE = 0.55

_TITLE_JUNK = re.compile(
    r"(?i)^(list of|special:|category:|template:|portal:|file:)"
    r"|\(disambiguation\)|\(значения\)")
# An article, just not the one we want.
_NOT_THE_ARTIST = re.compile(
    r"(?i)(discograph|_songs|/singles|album\)|song\)|_tour\b|awards_and)")


def preferred_lang(artist: str) -> str:
    """Which Wikipedia to try first. A Cyrillic name is usually thin on
    en.wikipedia and complete on ru — and that thinness is where the old bios
    started inventing."""
    return "ru" if re.search(r"[А-Яа-яЁё]", artist or "") else "en"


def _rank(url: str, want_lang: str) -> int:
    sub = (url.split("//", 1)[-1].split(".", 1)[0] or "").lower()
    if sub == want_lang:
        return 0
    if sub == "en":
        return 1
    return 3 if sub == "simple" else 2


def candidates(rows: list, want_lang: str) -> list:
    """Keep real artist articles, ranked by how much we want that wiki."""
    out = []
    for row in rows:
        url, title = row.get("url") or "", (row.get("title") or "").strip()
        if not url.split("//", 1)[-1].split("/", 1)[0].endswith("wikipedia.org"):
            continue
        if mediawiki.parse_article(url) is None or _TITLE_JUNK.search(title):
            continue
        if _NOT_THE_ARTIST.search(url):
            continue
        out.append({"rank": _rank(url, want_lang), "url": url, "title": title,
                    "snippet": row.get("snippet") or row.get("content") or ""})
    out.sort(key=lambda c: c["rank"])
    return out


def gate(artist: str, pool: list) -> tuple:
    """(best, rejected) — the article about THIS artist, or None.

    Host and path checks prove the page is a Wikipedia article; they prove
    nothing about WHOSE. A same-name footballer clears every one of them.
    """
    rejected: list = []
    if not pool:
        return None, rejected
    pool = pool[:6]
    docs = [f"{c['title']}\n{c['snippet']}".strip() for c in pool]
    try:
        probs = ModelRegistry.ce_probabilities(
            f"{artist} is a musical artist or band: their music, albums and "
            f"career.", docs)
    except ModelError as e:
        # This used to be ``or [1.0] * len(pool)``: with no cross-encoder every
        # candidate scored a perfect 1.0 and cleared the gate. A gate that
        # admits everyone when it breaks is not a gate — and this one exists
        # precisely because a same-name footballer clears every OTHER check.
        # It cost a measurement once: four artists "found" a Wikipedia article
        # that does not exist, and the run looked successful.
        STATS.degraded("cross_encoder", "bio.article_gate")
        logger.warning("[bio.article] no cross-encoder for '%s' — refusing to "
                       "pick an article rather than admitting %d unjudged "
                       "candidates: %s", artist, len(pool), e)
        rejected += [(c["title"], c["url"], "cross-encoder unavailable")
                     for c in pool]
        return None, rejected
    for cand, prob in zip(pool, probs):
        cand["ce"] = round(float(prob), 3)

    passed = [c for c in pool if (c.get("ce") or 0.0) >= CE_ARTICLE_GATE]
    for cand in pool:
        if cand not in passed:
            rejected.append((cand["title"], cand["url"],
                             f"ce={cand.get('ce')} below gate"))
    if not passed:
        return None, rejected
    passed.sort(key=lambda c: (c["rank"], -(c.get("ce") or 0.0)))
    for cand in passed[1:]:
        rejected.append((cand["title"], cand["url"],
                         f"ce={cand.get('ce')}, lower language rank"))
    return passed[0], rejected


def find(artist: str, *, proxies: Optional[dict] = None) -> tuple:
    """The artist's Wikipedia article: (article, rejected). TWO requests, max.

    Both rungs advance on "nothing PASSED THE GATE", not on "nothing was found".
    Advancing on the latter broke the obscure case: a bare search for `Merk`
    returns five articles — a coin, a Hungarian village, a football referee — so
    a "found something" test never reached the rung that finds
    `Merk (musician)`.

    One language, the preferred one. The second pass over en.wikipedia is gone
    along with the per-suffix requests: it doubled the cost of every artist to
    serve the case of a Cyrillic name that ru.wikipedia does not cover, and that
    case now falls through to the open web like any other uncovered artist.
    """
    want = preferred_lang(artist)
    rejected: list = []

    # Rung zero: the disambiguated titles, all of them in one request. Relevance
    # search answers "Merk" with a coin, a Hungarian village and a football
    # referee and never reaches "Merk (musician)"; the title probe goes straight
    # there. Candidates like any other rung — the gate still decides.
    pool = candidates(
        mediawiki.probe_titles_batch(artist, want, proxies=proxies), want)
    if pool:
        best, rejected = gate(artist, pool)
        if best is not None:
            return best, rejected

    # Rung one: the BARE name against Wikipedia's own relevance index. Appending
    # the disambiguating words here is what rung zero already did, and doing it
    # in a query makes the answer worse — "Phoenix band musician" returns Rain,
    # Summer and Liberty Phoenix, people surnamed Phoenix who are musicians,
    # while the band ranks nowhere.
    seen = {c["url"] for c in pool}
    pool += [c for c in candidates(mediawiki.search(artist, want,
                                                    proxies=proxies), want)
             if c["url"] not in seen]
    if not pool:
        return None, rejected

    best, rej = gate(artist, pool)
    return best, (rej or rejected)
