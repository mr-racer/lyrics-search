"""Album tracks are ordered by real position (disc, track_number).

If every track in an album has a track_number, order by (disc_number or 1,
track_number). Otherwise fall back to title order for the whole album — the
agreed behaviour for incomplete numbering (no separate trailing bucket for
un-numbered tracks).
"""

from app.api.routes.artists import _album_track_sort_key
from app.domain.models import TrackMetadata


def _t(title, track_number=None, disc_number=None):
    return TrackMetadata(
        track_id=title, title=title, artist="A",
        duration_sec=1.0, file_path="/x",
        track_number=track_number, disc_number=disc_number,
    )


def test_all_numbered_single_disc_ordered_by_number():
    out = _album_track_sort_key([_t("C", 3), _t("A", 1), _t("B", 2)])
    assert [t.title for t in out] == ["A", "B", "C"]


def test_missing_disc_treated_as_disc_one():
    out = _album_track_sort_key([_t("t2", 2), _t("t1", 1)])
    assert [t.title for t in out] == ["t1", "t2"]


def test_mixed_numbering_falls_back_to_title():
    out = _album_track_sort_key([_t("Beta", 2), _t("Alpha", None), _t("Gamma", 1)])
    assert [t.title for t in out] == ["Alpha", "Beta", "Gamma"]


def test_multi_disc_ordered_by_disc_then_track():
    out = _album_track_sort_key([
        _t("d2t1", 1, 2), _t("d1t2", 2, 1), _t("d1t1", 1, 1),
    ])
    assert [t.title for t in out] == ["d1t1", "d1t2", "d2t1"]
