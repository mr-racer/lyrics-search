"""Who may point the host indexer at a path.

Before this, ``MEMBER_INDEX_ROOT`` was an instance-wide env var: setting it let
EVERY member index that root, and the path itself was published through the
unauthenticated ``GET /config``. The grant is now per-account, and the env var
survives only as a ceiling on what the owner is allowed to hand out.
"""
from app.api.helpers import grant_within_ceiling, index_grant_allows


def test_owner_may_index_any_path_without_a_grant(tmp_path):
    assert index_grant_allows(
        role="owner", index_root=None, candidate=str(tmp_path / "anything")
    )


def test_member_without_a_grant_may_index_nothing(tmp_path):
    assert not index_grant_allows(
        role="member", index_root=None, candidate=str(tmp_path)
    )


def test_member_with_an_empty_grant_may_index_nothing(tmp_path):
    assert not index_grant_allows(
        role="member", index_root="", candidate=str(tmp_path)
    )


def test_member_may_index_inside_their_grant(tmp_path):
    album = tmp_path / "Music" / "Album"
    album.mkdir(parents=True)

    assert index_grant_allows(
        role="member", index_root=str(tmp_path / "Music"), candidate=str(album)
    )


def test_member_may_index_the_grant_root_itself(tmp_path):
    root = tmp_path / "Music"
    root.mkdir()

    assert index_grant_allows(
        role="member", index_root=str(root), candidate=str(root)
    )


def test_member_may_not_index_a_sibling_of_their_grant(tmp_path):
    (tmp_path / "Music").mkdir()
    other = tmp_path / "Secrets"
    other.mkdir()

    assert not index_grant_allows(
        role="member", index_root=str(tmp_path / "Music"), candidate=str(other)
    )


def test_member_may_not_escape_their_grant_with_dotdot(tmp_path):
    root = tmp_path / "Music"
    root.mkdir()

    assert not index_grant_allows(
        role="member",
        index_root=str(root),
        candidate=str(root / ".." / ".." / "etc"),
    )


def test_grant_inside_the_ceiling_is_accepted(tmp_path):
    ceiling = tmp_path / "music"
    ceiling.mkdir()

    assert grant_within_ceiling(str(ceiling / "Music"), ceiling=str(ceiling))


def test_grant_outside_the_ceiling_is_refused(tmp_path):
    ceiling = tmp_path / "music"
    ceiling.mkdir()

    assert not grant_within_ceiling(str(tmp_path / "etc"), ceiling=str(ceiling))


def test_without_a_ceiling_any_grant_is_accepted(tmp_path):
    assert grant_within_ceiling(str(tmp_path / "anywhere"), ceiling="")


def test_an_empty_grant_is_not_a_grant(tmp_path):
    """Revoking is its own operation; it must not slip through the ceiling check."""
    assert not grant_within_ceiling("", ceiling=str(tmp_path))
    assert not grant_within_ceiling("", ceiling="")
