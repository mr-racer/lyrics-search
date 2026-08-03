"""Unit tests for app.services.stream.calibration (design §5).

The whole point of the quantile table is that a threshold means the same thing
in every library. These cases pin the mapping's monotonicity and its endpoints,
plus the two ways it is allowed to degrade: a collection too small to estimate a
distribution, and Qdrant being unreachable.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from app.resources.metadata_db import MetadataDB
from app.services.stream import calibration as calib

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 3, 12, 0, 0)


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


class _Point:
    def __init__(self, tid, vector):
        self.id = tid
        self.vector = {"clap": list(vector)}
        self.payload = {}


class _FakeQdrant:
    """Scrolls a fixed set of CLAP vectors; ``count`` mirrors their number."""

    def __init__(self, vectors, *, fail=False):
        self.points = [_Point(f"t{i}", v) for i, v in enumerate(vectors)]
        self.fail = fail

    def scroll(self, collection_name, limit, offset=None, **kw):
        if self.fail:
            raise RuntimeError("qdrant down")
        return self.points, None

    def count(self, collection_name, exact=True):
        return SimpleNamespace(count=len(self.points))


def _spread_vectors(n=64, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestQuantileTable:
    def test_is_monotone_and_spans_the_percentile_range(self):
        table = calib.quantiles_from_vectors(_spread_vectors())
        assert len(table) == calib.N_QUANTILES
        assert table == sorted(table)
        assert table[0] < table[-1]

    def test_identical_vectors_give_a_flat_table(self):
        """A library where everything sounds the same must not explode — the
        table collapses to a constant and sim_pct degrades gracefully."""
        v = np.tile(np.array([1.0, 0.0], dtype=np.float32), (40, 1))
        table = calib.quantiles_from_vectors(v)
        assert table[0] == pytest.approx(table[-1])
        c = calib.Calibration(table, source="table")
        assert 0.0 <= c.sim_pct(1.0) <= 1.0

    def test_single_vector_has_no_pairs(self):
        table = calib.quantiles_from_vectors(np.array([[1.0, 0.0]], dtype=np.float32))
        assert table == [0.0] * calib.N_QUANTILES


class TestSimPct:
    def _calibration(self):
        # cosines spread evenly over [0, 1] → percentile ≈ the cosine itself
        return calib.Calibration([i / 100.0 for i in range(101)], source="table")

    def test_endpoints_clamp(self):
        c = self._calibration()
        assert c.sim_pct(-1.0) == 0.0
        assert c.sim_pct(0.0) == 0.0
        assert c.sim_pct(1.0) == 1.0
        assert c.sim_pct(5.0) == 1.0

    def test_monotone_in_between(self):
        c = self._calibration()
        vals = [c.sim_pct(x / 20.0) for x in range(21)]
        assert vals == sorted(vals)

    def test_narrow_band_is_stretched(self):
        """The actual complaint: raw cosines 0.82 vs 0.86 are indistinguishable;
        in percentile space they must be far apart."""
        c = calib.Calibration([0.75 + i * 0.002 for i in range(101)], source="table")
        assert c.sim_pct(0.86) - c.sim_pct(0.82) > 0.15

    def test_matrix_form_matches_scalar(self):
        c = self._calibration()
        arr = c.sim_pct_matrix([0.1, 0.5, 0.9])
        assert list(arr) == pytest.approx([c.sim_pct(0.1), c.sim_pct(0.5), c.sim_pct(0.9)], abs=1e-5)

    def test_raw_fallback_maps_cosine_onto_unit_range(self):
        c = calib.Calibration(None, source="raw")
        assert c.sim_pct(-1.0) == pytest.approx(0.0)
        assert c.sim_pct(0.0) == pytest.approx(0.5)
        assert c.sim_pct(1.0) == pytest.approx(1.0)


class TestBuildAndLoad:
    def test_build_persists_and_load_reuses(self):
        q = _FakeQdrant(_spread_vectors())
        built = calib.build(q, "col", now=NOW)
        assert built is not None
        assert MetadataDB.get_clap_calibration("col")["quantiles"] == built["quantiles"]

        # a second load must not rebuild — break the client to prove it
        loaded = calib.load(_FakeQdrant([], fail=True), "col", n_tracks=64, now=NOW)
        assert loaded.source == "table"
        assert loaded.quantiles == built["quantiles"]

    def test_too_small_a_collection_stays_on_raw(self):
        q = _FakeQdrant(_spread_vectors(n=calib.CALIB_MIN_TRACKS - 1))
        assert calib.build(q, "col", now=NOW) is None
        assert calib.load(q, "col", now=NOW).source == "raw"

    def test_unreachable_qdrant_degrades_instead_of_raising(self):
        assert calib.load(_FakeQdrant([], fail=True), "col", now=NOW).source == "raw"

    def test_track_count_drift_forces_a_rebuild(self):
        q = _FakeQdrant(_spread_vectors())
        calib.build(q, "col", now=NOW)
        stored = MetadataDB.get_clap_calibration("col")
        assert calib._is_fresh(stored, 64, NOW) is True
        # +50% tracks — the distribution moved, the table is stale
        assert calib._is_fresh(stored, 96, NOW) is False

    def test_ttl_expiry_forces_a_rebuild(self):
        q = _FakeQdrant(_spread_vectors())
        calib.build(q, "col", now=NOW)
        stored = MetadataDB.get_clap_calibration("col")
        later = NOW + timedelta(days=calib.CALIB_TTL_DAYS + 1)
        assert calib._is_fresh(stored, 64, later) is False

    def test_version_bump_invalidates(self):
        stored = {"version": calib.CALIB_VERSION - 1, "quantiles": [0.0] * 101,
                  "n_tracks": 64, "built_at": NOW.isoformat()}
        assert calib._is_fresh(stored, 64, NOW) is False

    def test_build_if_missing_false_never_touches_qdrant(self):
        assert calib.load(_FakeQdrant([], fail=True), "col",
                          build_if_missing=False, now=NOW).source == "raw"
