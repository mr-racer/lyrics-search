"""BM25 Okapi over a small in-memory corpus.

Written out rather than pulled from ``rank_bm25``: the corpus here is tens to a
few hundred short documents, the algorithm is thirty lines, and the app it will
be ported into avoids new dependencies for things this size (it already carries
two other hand-written BM25s for the same reason).

No transliteration variants and no pseudo-relevance feedback. Both existed in
``app/services/facts_retrieval.py`` to paper over one problem — a Russian query
scoring zero against English documents when BM25 was the only lexical signal.
It is no longer the only one: MILCO is a learned multilingual sparse model and
sits next to it at more than three times the weight. What is left for BM25 is
what it is genuinely good at — exact names, numbers and titles.
"""

from __future__ import annotations

import math
import re
from collections import Counter

K1 = 1.2
B = 0.75

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25:
    def __init__(self, docs: list[str]):
        self.docs = [tokenize(d) for d in docs]
        self.n = len(self.docs)
        self.lengths = [len(d) for d in self.docs]
        self.avglen = (sum(self.lengths) / self.n) if self.n else 0.0
        self.tf: list[Counter] = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        self.df = df

    def idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        return math.log(1 + (self.n - n_t + 0.5) / (n_t + 0.5))

    def scores(self, query: str) -> list[float]:
        out = [0.0] * self.n
        if not self.n:
            return out
        for term in tokenize(query):
            idf = self.idf(term)
            if idf <= 0:
                continue
            for i in range(self.n):
                freq = self.tf[i].get(term, 0)
                if not freq:
                    continue
                norm = 1 - B + B * (self.lengths[i] / (self.avglen or 1.0))
                out[i] += idf * freq * (K1 + 1) / (freq + K1 * norm)
        return out
