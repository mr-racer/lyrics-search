"""Unit tests for deprecated_collection_warning — verifies the right log
level / message shape so operators can grep for migration progress."""

import logging

from app.api.helpers import deprecated_collection_warning


class TestDeprecatedCollectionWarning:
    def test_no_log_when_supplied_is_none(self, caplog):
        with caplog.at_level(logging.INFO, logger="app.api.helpers"):
            deprecated_collection_warning(None, "acct_X", "/foo")
        assert not caplog.records

    def test_info_when_supplied_matches_derived(self, caplog):
        with caplog.at_level(logging.INFO, logger="app.api.helpers"):
            deprecated_collection_warning("acct_X", "acct_X", "/foo")
        assert any(r.levelno == logging.INFO for r in caplog.records)
        assert "/foo" in caplog.text
        assert "acct_X" in caplog.text

    def test_warning_when_supplied_mismatches(self, caplog):
        with caplog.at_level(logging.WARNING, logger="app.api.helpers"):
            deprecated_collection_warning("acct_OTHER", "acct_ME", "/library/stats")
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert "acct_OTHER" in caplog.text
        assert "acct_ME" in caplog.text
        assert "/library/stats" in caplog.text
        assert "IGNORED" in caplog.text
