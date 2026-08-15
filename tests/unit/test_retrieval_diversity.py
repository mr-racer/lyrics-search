"""Near-duplicate collapse: what it must catch, and what it must never touch.

The second half is the important half. Dropping a passage that repeats another
saves a slot; dropping one that merely shares a subject loses a fact, silently,
in a way no log line downstream would show. So most of what is pinned here is
the refusal to collapse — the AND between two signals, the length rule, and the
"we could not tell, so keep it" default.

Everything below the seam test runs the POLICY on given numbers. The seam test
runs the pipeline's own `best_chunks` over a fake retriever, because a policy
that is right in isolation and never reached is what the source-voting bug
turned out to be.
"""

import pytest

from app.services.assistant.config import AgentConfig
from app.services.assistant.events import AgentSink as EventSink
from app.services.assistant.contracts import Chunk
from app.services.assistant.branches.general import GeneralBranch
from app.services.retrieval.diversity import (is_duplicate, pair_similarity,
                                              pick_diverse)
from app.services.retrieval.hybrid import Ranked
from app.services.assistant.timing import Timings

THRESHOLDS = {"dense": 0.95, "milco": 0.90}


def _sims(pairs: dict, n: int, *, signals=("dense", "milco"),
          background: float = 0.3) -> dict:
    """Similarity matrices where only the listed pairs are alike.

    ``pairs`` maps ``(i, j)`` to either one number for every signal, or a dict
    per signal when the point of the test is that the signals disagree.
    """
    out = {}
    for signal in signals:
        matrix = [[1.0 if i == j else background for j in range(n)]
                  for i in range(n)]
        for (i, j), value in pairs.items():
            score = value[signal] if isinstance(value, dict) else value
            matrix[i][j] = matrix[j][i] = score
        out[signal] = matrix
    return out


class TestTheRule:
    def test_both_signals_must_agree(self):
        assert is_duplicate({"dense": 0.97, "milco": 0.93}, THRESHOLDS)

    def test_one_signal_alone_is_not_enough(self):
        """The collapse the user is right to fear: two paragraphs about the
        same incident read as near-identical to a dense model even when one
        carries a date the other does not. The sparse view is what separates
        them, and it gets a veto."""
        assert not is_duplicate({"dense": 0.99, "milco": 0.62}, THRESHOLDS)

    def test_a_pair_nobody_scored_is_not_a_duplicate(self):
        assert not is_duplicate({}, THRESHOLDS)

    def test_a_signal_with_no_threshold_is_ignored_not_trusted(self):
        assert not is_duplicate({"bm25": 0.99}, THRESHOLDS)

    def test_a_ragged_matrix_does_not_raise(self):
        assert pair_similarity({"dense": [[1.0]]}, 0, 5) == {}
        assert pair_similarity({"dense": None}, 0, 1) == {}
        assert pair_similarity(None, 0, 1) == {}


class TestPicking:
    def test_a_copy_is_dropped_and_the_slot_refilled(self):
        """The whole point: the pack stays the same size and says more."""
        sims = _sims({(0, 1): 0.99}, 4)
        picked = pick_diverse([500] * 4, sims, thresholds=THRESHOLDS, limit=3)
        assert picked.kept == [0, 2, 3]
        assert [d.index for d in picked.duplicates] == [1]

    def test_nothing_is_dropped_when_nothing_repeats(self):
        picked = pick_diverse([500] * 4, _sims({}, 4),
                              thresholds=THRESHOLDS, limit=3)
        assert picked.kept == [0, 1, 2]
        assert picked.duplicates == []

    def test_a_topical_neighbour_survives(self):
        """0.96 dense and 0.71 sparse is the shape of "same subject, different
        content". It must take its own slot."""
        sims = _sims({(0, 1): {"dense": 0.96, "milco": 0.71}}, 3)
        picked = pick_diverse([500] * 3, sims, thresholds=THRESHOLDS, limit=3)
        assert picked.kept == [0, 1, 2]

    def test_the_longer_copy_takes_the_slot(self):
        """Same text, one host kept the paragraph the other truncated."""
        sims = _sims({(0, 2): 0.99}, 4)
        picked = pick_diverse([300, 500, 900, 500], sims,
                              thresholds=THRESHOLDS, limit=3)
        assert picked.kept == [2, 1, 3]
        assert picked.duplicates[0].index == 0
        assert picked.duplicates[0].replaced is True

    def test_a_marginally_longer_copy_does_not_displace(self):
        """Rank is information too — a rewording must not outrank a
        better-scoring passage over a handful of characters."""
        sims = _sims({(0, 1): 0.99}, 3)
        picked = pick_diverse([500, 520, 500], sims,
                              thresholds=THRESHOLDS, limit=2)
        assert picked.kept == [0, 2]

    def test_the_order_of_the_survivors_is_the_ranking(self):
        sims = _sims({(1, 2): 0.99}, 5)
        picked = pick_diverse([500] * 5, sims, thresholds=THRESHOLDS, limit=4)
        assert picked.kept == sorted(picked.kept)

    def test_a_full_pack_can_still_be_improved(self):
        """Scanning does not stop at the limit: candidate 3 is a fuller copy of
        candidate 0 and belongs in its slot."""
        sims = _sims({(0, 3): 0.99}, 4)
        picked = pick_diverse([300, 500, 500, 900], sims,
                              thresholds=THRESHOLDS, limit=3)
        assert picked.kept == [3, 1, 2]

    def test_three_copies_of_one_thing_leave_one(self):
        sims = _sims({(0, 1): 0.99, (0, 2): 0.99, (1, 2): 0.99}, 5)
        picked = pick_diverse([500] * 5, sims, thresholds=THRESHOLDS, limit=3)
        assert picked.kept == [0, 3, 4]

    def test_with_no_similarity_at_all_it_is_a_plain_top_k(self):
        picked = pick_diverse([500] * 5, {}, thresholds=THRESHOLDS, limit=3)
        assert picked.kept == [0, 1, 2]
        assert picked.duplicates == []

    def test_one_signal_alone_still_decides_when_it_is_all_there_is(self):
        """Degradation, documented rather than silent: with MILCO unavailable
        the guard is weaker, and the pipeline logs that it is."""
        sims = _sims({(0, 1): 0.99}, 3, signals=("dense",))
        picked = pick_diverse([500] * 3, sims, thresholds=THRESHOLDS, limit=2)
        assert picked.kept == [0, 2]

    def test_an_empty_pool_is_fine(self):
        assert pick_diverse([], {}, thresholds=THRESHOLDS, limit=5).kept == []


# The calibration report (``duplicate_report`` / ``margin``) did not travel to
# production: it existed to set these thresholds from a notebook, against a
# corpus you could read by eye. The thresholds are set; the rule they feed is
# what is tested above.

class _FakeRetriever:
    """Ranked results and document similarity, with no models in sight."""

    def __init__(self, probs, sims):
        self.probs = probs
        self.sims = sims

    def search(self, query, *, min_prob=None, limit=None, **kwargs):
        order = sorted(range(len(self.probs)), key=lambda i: -self.probs[i])
        out = [Ranked(index=i, rrf=0.0,
                      ce_prob=self.probs[i], final=self.probs[i])
               for i in order
               if min_prob is None or self.probs[i] >= min_prob]
        return out[:limit] if limit else out

    def similarity_matrix(self, indices):
        return {name: [[matrix[i][j] for j in indices] for i in indices]
                for name, matrix in self.sims.items()}


def _branch(**overrides):
    """A GeneralBranch wired to a fake retriever over five chunks.

    Chunks 0 and 2 are the same syndicated paragraph on two hosts, and both
    score near the top — which is exactly the run the user reported.
    """
    settings = {"max_chunks_in_context": 3, "ce_threshold_chunks": 0.5}
    cfg = AgentConfig(**{**settings, **overrides})
    branch = GeneralBranch.__new__(GeneralBranch)
    branch.cfg = cfg
    branch.sink = EventSink()
    branch.timings = Timings()
    branch.chunks = [
        Chunk(id=i, path=["Biography"], body=f"body {i}",
              url=f"https://host{i}.example/bio", title=f"page {i}")
        for i in range(5)
    ]
    branch.retriever = _FakeRetriever(
        probs=[0.95, 0.90, 0.93, 0.80, 0.70],
        sims=_sims({(0, 2): 0.99}, 5))
    return branch


class TestThroughTheSeam:
    """`best_chunks` is what the answer is actually built from.

    Testing `pick_diverse` alone would have proved nothing about the pack — the
    same way testing `merge_claims` alone proved nothing about source voting.
    """

    def test_the_pack_replaces_the_copy_with_new_material(self):
        pack = _branch().best_chunks("who is this artist")
        assert [c.id for c, _ in pack] == [0, 1, 3]

    def test_the_pack_is_still_full(self):
        assert len(_branch().best_chunks("q")) == 3

    def test_switching_it_off_restores_the_old_behaviour(self):
        pack = _branch(dedup_chunks=False).best_chunks("q")
        assert [c.id for c, _ in pack] == [0, 2, 1]

    def test_the_event_says_what_was_collapsed(self):
        branch = _branch()
        branch.best_chunks("q")
        [event] = branch.sink.of("dedup")
        assert event["duplicates"] == 1
        assert event["selected"] == 3
        assert event["signals"] == ["dense", "milco"]

    def test_no_event_when_nothing_repeated(self):
        branch = _branch()
        branch.retriever.sims = _sims({}, 5)
        assert len(branch.best_chunks("q")) == 3
        assert branch.sink.of("dedup") == []

    def test_the_notebook_entry_point_builds_the_same_pack(self):
        """`select_pack` is what a cell calls; `best_chunks` is what the agent
        calls. One of them being a reimplementation of the other is how a
        notebook stops describing the run it is supposed to measure."""
        from app.services.assistant.branches.base import select_pack

        branch = _branch()
        assert select_pack(branch.retriever, branch.chunks, "q",
                           config=branch.cfg) == branch.best_chunks("q")

    def test_it_survives_without_a_sink(self):
        branch = _branch()
        branch.sink = None
        assert len(branch.best_chunks("q")) == 3

    def test_the_threshold_still_rules(self):
        """Dedup changes WHICH passages fill the pack, never whether a passage
        was good enough to be in it."""
        branch = _branch(ce_threshold_chunks=0.92)
        assert [c.id for c, _ in branch.best_chunks("q")] == [0]
