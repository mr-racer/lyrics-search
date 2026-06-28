"""Consolidated backfill / axis-norm-reference script tests.

Merged from:
  - test_backfill_artist_slugs.py    -> class TestBackfillArtistSlugs
  - test_backfill_sonic_payload.py   -> class TestBackfillSonicPayload
  - test_build_axis_norm_reference.py -> class TestBuildAxisNormReference
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.resources.clap_features import AXIS_NAMES, AXIS_PROMPTS
from scripts.backfill_artist_slugs import backfill_collection
from scripts.build_axis_norm_reference import build_reference


# --------------------------------------------------------------------------- #
# Helpers / fixtures (hoisted from the source modules)
# --------------------------------------------------------------------------- #
def _pt(pid, artist):
    pt = MagicMock()
    pt.id = pid
    pt.payload = {"artist": artist, "title": "T"}
    return pt


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    from app.resources.metadata_db import MetadataDB
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield MetadataDB
    MetadataDB._reset_for_tests()


DIM = 16


class FakeClapModel:
    def get_text_embedding(self, texts, use_tensor=False):
        rng = np.random.default_rng(42)
        return rng.normal(size=(len(texts), DIM)).astype(np.float32)


class _Point:
    def __init__(self, vec):
        self.vector = {"clap": vec}


class FakeQdrant:
    """Two-page scroll to exercise the offset loop."""

    def __init__(self, vectors):
        self._pages = [vectors[: len(vectors) // 2], vectors[len(vectors) // 2:]]

    def scroll(self, collection_name, limit, with_payload, with_vectors, offset=None):
        if offset is None:
            return [_Point(v) for v in self._pages[0]], "page2"
        return [_Point(v) for v in self._pages[1]], None


# --------------------------------------------------------------------------- #
# test_backfill_artist_slugs.py
# --------------------------------------------------------------------------- #
class TestBackfillArtistSlugs:
    """Backfill writes artists/artist_slugs/primary_artist_slug onto existing points."""

    def test_backfill_sets_payload_for_each_point(self):
        pts = [_pt("1", "Kanye West, Sia"), _pt("2", "Radiohead")]
        qdrant = MagicMock()
        qdrant.scroll.return_value = (pts, None)

        n = backfill_collection(qdrant, "c", dry_run=False)

        assert n == 2
        # Each point got a set_payload with the three fields.
        calls = {c.kwargs["points"][0]: c.kwargs["payload"]
                 for c in qdrant.set_payload.call_args_list}
        assert calls["1"]["artist_slugs"] == ["kanye-west", "sia"]
        assert calls["1"]["primary_artist_slug"] == "kanye-west"
        assert calls["2"]["artists"] == ["Radiohead"]
        # Keyword index ensured.
        qdrant.create_payload_index.assert_called_once()

    def test_dry_run_writes_nothing(self):
        qdrant = MagicMock()
        qdrant.scroll.return_value = ([_pt("1", "Drake")], None)
        n = backfill_collection(qdrant, "c", dry_run=True)
        assert n == 1
        qdrant.set_payload.assert_not_called()
        qdrant.create_payload_index.assert_not_called()

    def test_skips_already_backfilled_points(self):
        pt = MagicMock()
        pt.id = "1"
        pt.payload = {"artist": "Drake", "artist_slugs": ["drake"]}
        qdrant = MagicMock()
        qdrant.scroll.return_value = ([pt], None)
        n = backfill_collection(qdrant, "c", dry_run=False)
        assert n == 0
        qdrant.set_payload.assert_not_called()


# --------------------------------------------------------------------------- #
# test_backfill_sonic_payload.py
#   Smoke test for scripts/backfill_sonic_payload.py — exercises the core
#   collection_iter -> set_payload logic against a mocked Qdrant client.
# --------------------------------------------------------------------------- #
class TestBackfillSonicPayload:
    def test_backfill_writes_sonic_tags_to_qdrant_payload(self, db):
        from app.resources.metadata_db import MetadataDB
        # slug = _slugify("A") + "-" + _slugify("X") = "a-x"
        song_slug = "a-x"
        MetadataDB.upsert_artist("a", "A", "test_col")
        MetadataDB.upsert_song(song_slug, "X", "a", "test_col")
        MetadataDB.upsert_sonic_descriptor(
            song_slug=song_slug,
            tags=[{"tag": "melancholic", "score": 0.8}, {"tag": "lo-fi", "score": 0.7}],
        )

        qdrant = MagicMock()
        qdrant.scroll.return_value = (
            [MagicMock(id="point-1", payload={"artist": "A", "title": "X"})],
            None,  # next page offset
        )

        from scripts.backfill_sonic_payload import backfill_collection
        n = backfill_collection(qdrant, collection="test_col", dry_run=False)

        assert n == 1
        qdrant.set_payload.assert_called_once()
        _, kwargs = qdrant.set_payload.call_args
        assert kwargs["collection_name"] == "test_col"
        assert kwargs["points"] == ["point-1"]
        assert sorted(kwargs["payload"]["sonic_tags"]) == ["lo-fi", "melancholic"]

    def test_backfill_skips_when_no_sqlite_row(self, db):
        qdrant = MagicMock()
        qdrant.scroll.return_value = (
            [MagicMock(id="point-1", payload={"artist": "A", "title": "X"})],
            None,
        )

        from scripts.backfill_sonic_payload import backfill_collection
        n = backfill_collection(qdrant, collection="test_col", dry_run=False)

        assert n == 0
        qdrant.set_payload.assert_not_called()

    def test_backfill_dry_run_makes_no_writes(self, db):
        from app.resources.metadata_db import MetadataDB
        song_slug = "a-x"
        MetadataDB.upsert_artist("a", "A", "test_col")
        MetadataDB.upsert_song(song_slug, "X", "a", "test_col")
        MetadataDB.upsert_sonic_descriptor(
            song_slug=song_slug,
            tags=[{"tag": "melancholic", "score": 0.8}],
        )

        qdrant = MagicMock()
        qdrant.scroll.return_value = (
            [MagicMock(id="point-1", payload={"artist": "A", "title": "X"})],
            None,
        )

        from scripts.backfill_sonic_payload import backfill_collection
        n = backfill_collection(qdrant, collection="test_col", dry_run=True)

        assert n == 1  # counts what _would_ be written
        qdrant.set_payload.assert_not_called()


# --------------------------------------------------------------------------- #
# test_build_axis_norm_reference.py
#   build_axis_norm_reference: Qdrant scroll -> axis stats JSON (fake Qdrant + CLAP).
# --------------------------------------------------------------------------- #
class TestBuildAxisNormReference:
    def test_build_reference_stats_shape(self):
        rng = np.random.default_rng(0)
        vectors = [rng.normal(size=DIM).astype(np.float32).tolist() for _ in range(10)]

        stats = build_reference(FakeQdrant(vectors), "big_lib", clap_model=FakeClapModel())

        assert stats["n"] == 10
        assert stats["source_collection"] == "big_lib"
        assert len(stats["version"]) == 12
        assert set(stats["mean"]) == set(AXIS_NAMES)
        assert set(stats["std"]) == set(AXIS_NAMES)
        assert all(v > 0 for v in stats["std"].values())  # random vectors → nonzero spread

    def test_build_reference_too_few_vectors_exits(self):
        rng = np.random.default_rng(0)
        vectors = [rng.normal(size=DIM).astype(np.float32).tolist()]
        with pytest.raises(SystemExit, match="CLAP vector"):
            build_reference(FakeQdrant(vectors), "tiny", clap_model=FakeClapModel())

    def test_points_without_clap_vector_skipped(self):
        rng = np.random.default_rng(0)
        vectors = [rng.normal(size=DIM).astype(np.float32).tolist() for _ in range(4)]
        fake = FakeQdrant(vectors)
        # первый поинт второй страницы — без clap-вектора
        orig_pages = fake._pages
        fake._pages = [orig_pages[0], orig_pages[1]]

        class _NoClap:
            vector = {}

        real_scroll = fake.scroll

        def scroll_with_gap(collection_name, limit, with_payload, with_vectors, offset=None):
            pts, nxt = real_scroll(collection_name, limit, with_payload, with_vectors, offset)
            if offset is None:
                return pts + [_NoClap()], nxt
            return pts, nxt

        fake.scroll = scroll_with_gap
        stats = build_reference(fake, "lib", clap_model=FakeClapModel())
        assert stats["n"] == 4  # gap point ignored
