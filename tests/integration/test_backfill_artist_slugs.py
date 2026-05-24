"""Backfill writes artists/artist_slugs/primary_artist_slug onto existing points."""

from unittest.mock import MagicMock

from scripts.backfill_artist_slugs import backfill_collection


def _pt(pid, artist):
    pt = MagicMock()
    pt.id = pid
    pt.payload = {"artist": artist, "title": "T"}
    return pt


def test_backfill_sets_payload_for_each_point():
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


def test_dry_run_writes_nothing():
    qdrant = MagicMock()
    qdrant.scroll.return_value = ([_pt("1", "Drake")], None)
    n = backfill_collection(qdrant, "c", dry_run=True)
    assert n == 1
    qdrant.set_payload.assert_not_called()
    qdrant.create_payload_index.assert_not_called()


def test_skips_already_backfilled_points():
    pt = MagicMock()
    pt.id = "1"
    pt.payload = {"artist": "Drake", "artist_slugs": ["drake"]}
    qdrant = MagicMock()
    qdrant.scroll.return_value = ([pt], None)
    n = backfill_collection(qdrant, "c", dry_run=False)
    assert n == 0
    qdrant.set_payload.assert_not_called()
