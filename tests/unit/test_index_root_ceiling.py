"""Unit tests for the folder-index confinement helpers (app/api/helpers.py).

MEMBER_INDEX_ROOT used to GRANT: setting it let every server-mode member index
that folder. It now only CONFINES what the owner may grant per account, and is
read through ``index_root_ceiling``. ``path_within_root`` is unchanged and is
still what keeps a granted account from escaping to arbitrary host paths.
"""
import pytest

from app.api.helpers import index_root_ceiling, path_within_root

pytestmark = pytest.mark.unit


def test_ceiling_unset_is_empty(monkeypatch):
    monkeypatch.delenv("MEMBER_INDEX_ROOT", raising=False)
    assert index_root_ceiling() == ""


def test_ceiling_trims_whitespace(monkeypatch):
    monkeypatch.setenv("MEMBER_INDEX_ROOT", "  /music  ")
    assert index_root_ceiling() == "/music"


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
