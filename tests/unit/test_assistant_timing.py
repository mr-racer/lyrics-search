"""The run profile: aggregation, honesty about gaps, and the nesting guard.

The clock is injected rather than slept through — a timing test that sleeps is
slow and flaky at the same time, and none of the arithmetic here needs a real
one.
"""

import pytest

from app.services.assistant.timing import Timings, lane_for


class FakeClock:
    """A clock that only moves when the test says so."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def timings(clock):
    return Timings(clock=clock)


def record(timings, clock, name, seconds):
    with timings.span(name):
        clock.tick(seconds)


class TestLanes:
    @pytest.mark.parametrize("name,lane", [
        ("llm.answer", "llm"), ("llm.plan", "llm"),
        ("search.web", "net"), ("fetch", "net"), ("discography", "net"),
        ("rerank.ce", "gpu"), ("index.embed", "gpu"), ("select.pack", "gpu"),
        ("resolve.library", "cpu"),
    ])
    def test_the_prefix_decides_the_lane(self, name, lane):
        assert lane_for(name) == lane

    def test_an_unknown_prefix_has_no_lane(self):
        """Better blank than filed under the wrong lane — the lane totals are
        the number someone acts on."""
        assert lane_for("whatever.new") == ""


class TestAggregation:
    def test_repeat_calls_of_a_stage_are_summed_and_counted(self, timings, clock):
        with timings.measure():
            record(timings, clock, "search.web", 2.0)
            record(timings, clock, "search.web", 4.0)
        row = next(r for r in timings.rows() if r["stage"] == "search.web")
        assert row["calls"] == 2
        assert row["total"] == 6.0
        assert row["mean"] == 3.0

    def test_rows_come_back_slowest_first(self, timings, clock):
        with timings.measure():
            record(timings, clock, "fetch", 1.0)
            record(timings, clock, "llm.answer", 9.0)
            record(timings, clock, "rerank.ce", 3.0)
        assert [r["stage"] for r in timings.rows()] == [
            "llm.answer", "rerank.ce", "fetch"]

    def test_lane_totals_group_across_stages(self, timings, clock):
        with timings.measure():
            record(timings, clock, "llm.plan", 1.0)
            record(timings, clock, "llm.answer", 3.0)
            record(timings, clock, "search.web", 2.0)
        assert timings.by_lane() == {"llm": 4.0, "net": 2.0}


class TestHonesty:
    def test_untimed_work_shows_up_as_the_remainder(self, timings, clock):
        """The whole point of measuring wall time separately: a stage nobody
        instrumented has to be visible as a gap, not distributed silently over
        the stages that were."""
        with timings.measure():
            record(timings, clock, "llm.answer", 4.0)
            clock.tick(6.0)               # something nobody wrapped
        assert timings.wall == 10.0
        assert "unaccounted" in timings.report()
        measured = sum(r["total"] for r in timings.rows())
        assert timings.wall - measured == 6.0

    def test_shares_are_taken_against_the_wall_clock(self, timings, clock):
        with timings.measure():
            record(timings, clock, "llm.answer", 5.0)
            clock.tick(5.0)
        row = timings.rows()[0]
        assert row["share"] == 0.5

    def test_a_second_run_does_not_inherit_the_first(self, timings, clock):
        with timings.measure():
            record(timings, clock, "llm.answer", 4.0)
        with timings.measure():
            record(timings, clock, "llm.answer", 1.0)
        assert timings.rows()[0]["calls"] == 1
        assert timings.wall == 1.0

    def test_an_empty_run_says_so_instead_of_dividing_by_zero(self, timings):
        assert "nothing ran" in timings.report()

    def test_a_failing_stage_is_still_recorded(self, timings, clock):
        """A stage that raises took time too, and hiding it would make the
        remainder absorb exactly the case worth looking at."""
        with timings.measure():
            with pytest.raises(RuntimeError):
                with timings.span("fetch"):
                    clock.tick(3.0)
                    raise RuntimeError("boom")
        assert timings.rows()[0]["stage"] == "fetch"
        assert timings.rows()[0]["total"] == 3.0


class TestNestingGuard:
    def test_a_nested_span_is_reported_loudly(self, timings, clock, caplog):
        with timings.measure():
            with timings.span("fetch"):
                with timings.span("llm.answer"):
                    clock.tick(1.0)
        assert "nested" in caplog.text
        assert "llm.answer" in caplog.text and "fetch" in caplog.text

    def test_overlapping_spans_are_called_out_in_the_report(self, timings, clock):
        """Double-counted time makes the remainder negative. Saying why beats
        printing a minus sign and letting it read as rounding."""
        with timings.measure():
            with timings.span("fetch"):
                with timings.span("llm.answer"):
                    clock.tick(4.0)
        assert timings.wall == 4.0
        assert sum(r["total"] for r in timings.rows()) == 8.0
        assert "spans overlap" in timings.report()

    def test_sequential_spans_are_not_nesting(self, timings, clock, caplog):
        with timings.measure():
            record(timings, clock, "fetch", 1.0)
            record(timings, clock, "llm.answer", 1.0)
        assert "nested" not in caplog.text
