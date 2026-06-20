"""<30s duration filter helper for the recommendation surfaces."""
import pytest

from app.services import stream_service as ss


@pytest.mark.parametrize("dur,ok", [
    (None, True),      # unknown → keep
    (0, True),         # zero/unknown → keep
    (5, False),        # short → drop
    (29.9, False),
    (30.0, True),      # exactly 30 → keep
    (180, True),
    ("45", True),      # coerced
    ("12", False),
])
def test_duration_ok(dur, ok):
    assert ss._duration_ok({"duration": dur}) is ok


def test_duration_ok_missing_key():
    assert ss._duration_ok({}) is True
    assert ss._duration_ok(None) is True
