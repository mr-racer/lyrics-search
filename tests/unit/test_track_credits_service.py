"""Unit tests for track_credits_service (producer credits + record labels)."""
import sqlite3

import pytest

import app.resources.metadata_db as mod
from app.resources.metadata_db import MetadataDB
from app.services.track_credits_service import (
    aggregate_labels,
    label_key,
    resolve_producers,
    split_credit_names,
    split_labels,
    tracks_produced_by,
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Repoint MetadataDB at a fresh temp SQLite file for the test."""
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "meta.db")
    MetadataDB._instance = None
    orig_connect = MetadataDB._connect

    def _connect(cls):
        conn = sqlite3.connect(str(mod.DB_PATH), check_same_thread=False)
        conn.row_factory = None
        return conn

    monkeypatch.setattr(MetadataDB, "_connect", classmethod(_connect))
    MetadataDB.init()
    yield
    MetadataDB._instance = None
    MetadataDB._connect = orig_connect


def _upsert(track_id, **payload):
    MetadataDB.upsert_track_metadata("c", track_id, payload)


# --------------------------------------------------------------------------- #
# Splitters
# --------------------------------------------------------------------------- #
class TestSplitCreditNames:
    def test_splits_on_commas_semicolons_slashes_and_amp(self):
        assert split_credit_names("Dr. Dre, Mel-Man") == ["Dr. Dre", "Mel-Man"]
        assert split_credit_names("A; B") == ["A", "B"]
        assert split_credit_names("A & B") == ["A", "B"]
        assert split_credit_names("A / B") == ["A", "B"]

    def test_splits_on_feat(self):
        assert split_credit_names("Kanye West feat. Emile") == ["Kanye West", "Emile"]

    def test_caps_at_three_names(self):
        assert split_credit_names("A, B, C, D") == ["A", "B", "C"]

    def test_empty_and_none(self):
        assert split_credit_names("") == []
        assert split_credit_names(None) == []
        assert split_credit_names("   ") == []


class TestLabels:
    def test_split_labels_on_comma_and_semicolon_only(self):
        assert split_labels("GOOD Music, Def Jam") == ["GOOD Music", "Def Jam"]
        assert split_labels("Roc-A-Fella; Def Jam") == ["Roc-A-Fella", "Def Jam"]
        # "&" and "/" stay inside label names.
        assert split_labels("A&M Records") == ["A&M Records"]

    def test_label_key_normalizes_case_and_whitespace(self):
        assert label_key("  Def   Jam ") == "def jam"
        assert label_key("DEF JAM") == "def jam"

    def test_aggregate_labels_dedupes_and_orders_by_frequency(self):
        got = aggregate_labels([
            "Def Jam",
            "GOOD Music, def jam",
            "GOOD Music",
            "GOOD Music",
        ])
        # GOOD Music: 3 mentions; Def Jam: 2 — frequency order, first-seen casing.
        assert got == ["GOOD Music", "Def Jam"]

    def test_aggregate_labels_skips_empties(self):
        assert aggregate_labels([None, "", "  ", "Stones Throw"]) == ["Stones Throw"]


# --------------------------------------------------------------------------- #
# resolve_producers
# --------------------------------------------------------------------------- #
class TestResolveProducers:
    def test_resolves_artist_slug_and_counts_other_tracks(self, temp_db):
        # Kanye is both a library artist and produces two other tracks.
        _upsert("t1", title="Runaway", artist="Kanye West",
                artist_slugs=["kanye-west"], producer="Kanye West, Emile")
        _upsert("t2", title="Power", artist="Kanye West",
                artist_slugs=["kanye-west"], producer="Kanye West")
        _upsert("t3", title="Other", artist="Someone",
                artist_slugs=["someone"], producer="kanye west & Q-Tip")

        got = {p.name: p for p in resolve_producers("c", "t1")}

        assert set(got) == {"Kanye West", "Emile"}
        assert got["Kanye West"].artist_slug == "kanye-west"
        # t2 + t3 credit Kanye; t1 itself is excluded.
        assert got["Kanye West"].produced_count == 2
        assert got["Emile"].artist_slug is None
        assert got["Emile"].produced_count == 0

    def test_unknown_track_and_no_producer_yield_empty(self, temp_db):
        _upsert("t1", title="X", artist="A", artist_slugs=["a"])
        assert resolve_producers("c", "missing") == []
        assert resolve_producers("c", "t1") == []

    def test_no_cross_collection_leak(self, temp_db):
        _upsert("t1", title="X", artist="A", artist_slugs=["a"], producer="P")
        MetadataDB.upsert_track_metadata(
            "other_collection", "t9", {"title": "Y", "artist": "B", "producer": "P"},
        )
        got = resolve_producers("c", "t1")
        assert len(got) == 1
        assert got[0].produced_count == 0


# --------------------------------------------------------------------------- #
# tracks_produced_by
# --------------------------------------------------------------------------- #
class TestTracksProducedBy:
    def test_matches_case_insensitively_and_sorts(self, temp_db):
        _upsert("t1", title="Beta", artist="Zeta", producer="Emile")
        _upsert("t2", title="Alpha", artist="Alpha Artist", producer="emile, Other")
        _upsert("t3", title="Nope", artist="X", producer="Someone Else")

        got = tracks_produced_by("c", "EMILE")

        assert [t.track_id for t in got] == ["t2", "t1"]  # artist, then title
        assert got[0].artist_refs  # collab tag split into refs

    def test_empty_name_and_no_match(self, temp_db):
        _upsert("t1", title="Beta", artist="Zeta", producer="Emile")
        assert tracks_produced_by("c", "") == []
        assert tracks_produced_by("c", "Nobody") == []
