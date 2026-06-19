"""Unit tests for the MEMBER_INDEX_ROOT opt-in helpers (app/api/helpers.py).

These guard the security boundary that lets server-mode MEMBERS index a single
operator-approved mounted folder by reference, without being able to escape it
to arbitrary host paths.
"""
import pytest

from app.api.helpers import member_index_root, path_within_root

pytestmark = pytest.mark.unit


def test_member_index_root_unset_is_empty(monkeypatch):
    monkeypatch.delenv("MEMBER_INDEX_ROOT", raising=False)
    assert member_index_root() == ""


def test_member_index_root_trims_whitespace(monkeypatch):
    monkeypatch.setenv("MEMBER_INDEX_ROOT", "  /music  ")
    assert member_index_root() == "/music"


def test_within_allows_root_and_children(tmp_path):
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    assert path_within_root(str(tmp_path), str(tmp_path))                    # root itself
    assert path_within_root(str(tmp_path / "sub"), str(tmp_path))            # direct child
    assert path_within_root(str(tmp_path / "sub" / "deep"), str(tmp_path))   # nested


def test_within_rejects_sibling_and_dotdot_escape(tmp_path):
    root = tmp_path / "music"
    root.mkdir()
    (tmp_path / "etc").mkdir()
    assert not path_within_root(str(tmp_path / "etc"), str(root))            # sibling dir
    assert not path_within_root(str(root / ".." / "etc"), str(root))         # ../ escape
    assert not path_within_root(str(tmp_path), str(root))                    # parent dir


def test_within_empty_root_is_disabled(tmp_path):
    # No opt-in configured → nothing is allowed, even an otherwise-valid path.
    assert not path_within_root(str(tmp_path), "")
