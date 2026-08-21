"""Why a batch produced no indexed tracks — and why that has to be said out loud.

``prepare``/``prepare_metadata`` drop tracks on two hard, silent gates. The path
diff that offered those files knows nothing about either, so it offers the same
files on the next rescan, and the one after that. A user pressing "add music"
four times on a library of long DJ sets is not doing anything wrong — nothing
ever told them the files were rejected.
"""

from app.resources.qdrant_payload import MAX_DURATION, prepare_metadata
from app.services.indexing_service import MAX_LYRICS_WORDS, explain_rejections
from app.services.library_service import _rejection_message


def _track(**kw):
    base = {"artist": "A", "title": "T", "duration": 200, "lyrics": "la la"}
    base.update(kw)
    return base


class TestExplainRejections:
    def test_a_long_track_is_named_as_too_long(self):
        batch = {"A — T": _track(duration=MAX_DURATION + 1)}
        assert explain_rejections(batch) == {"too_long": 1}

    def test_the_cap_itself_is_allowed(self):
        batch = {"A — T": _track(duration=MAX_DURATION)}
        assert explain_rejections(batch) == {}

    def test_an_overlong_text_is_named_separately(self):
        batch = {"A — T": _track(lyrics="word " * MAX_LYRICS_WORDS)}
        assert explain_rejections(batch) == {"lyrics_too_long": 1}

    def test_a_missing_duration_is_its_own_reason(self):
        batch = {"A — T": _track(duration=None)}
        assert explain_rejections(batch) == {"no_duration": 1}

    def test_reasons_are_counted_per_kind(self):
        batch = {
            "A — 1": _track(duration=MAX_DURATION + 1),
            "A — 2": _track(duration=MAX_DURATION + 2),
            "A — 3": _track(lyrics="word " * MAX_LYRICS_WORDS),
            "A — 4": _track(),
        }
        assert explain_rejections(batch) == {"too_long": 2, "lyrics_too_long": 1}

    def test_it_agrees_with_the_gate_it_explains(self):
        """The helper must not drift from ``prepare_metadata``'s own filter."""
        batch = {
            "A — keep": _track(duration=MAX_DURATION),
            "A — drop": _track(duration=MAX_DURATION + 1),
        }
        kept = prepare_metadata(batch)
        assert len(kept) == 1
        assert sum(explain_rejections(batch).values()) == len(batch) - len(kept)


class TestRejectionMessage:
    def test_it_names_the_dominant_reason(self):
        msg = _rejection_message(94, {"too_long": 93, "lyrics_too_long": 1})
        assert "94" in msg and "93" in msg
        assert "7 минут" in msg

    def test_no_reasons_means_everything_was_already_indexed(self):
        assert "уже в библиотеке" in _rejection_message(12, {})
