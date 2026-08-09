"""Only Apple's own playlists count as a source of songs.

A stranger's public playlist has the same URL shape, the same parser output and
the same air of authority. The difference is the curator, and it has to be
checked, because "2000s club hits" curated by Apple Music Dance is an editorial
answer while the same title by a listener is one person's evening.
"""

from lab.agent.extraction import apple_page_kind, is_usable_apple_page


PLAYLIST_URL = ("https://music.apple.com/us/playlist/muse-essentials/"
                "pl.5d8ac6f0282445bb8a53a17c7995d52f")
TOP_SONGS_URL = ("https://music.apple.com/us/artist/kanye-west/2715720/top-songs")


class _Playlist:
    def __init__(self, author=None, playlist_id=None, title=""):
        self.author = author
        self.playlist_id = playlist_id
        self.title = title


class TestEditorialCheck:
    def test_the_muse_essentials_playlist_passes(self):
        """The exact page the owner asked about:
        music.apple.com/us/playlist/muse-essentials/pl.5d8ac6f0282445bb8a53a17c7995d52f
        — curated by "Apple Music Alternative"."""
        assert is_usable_apple_page(_Playlist(
            author="Apple Music Alternative",
            playlist_id="pl.5d8ac6f0282445bb8a53a17c7995d52f",
            title="Muse Essentials"), PLAYLIST_URL)

    def test_a_genre_desk_passes(self):
        for desk in ("Apple Music", "Apple Music Hip-Hop", "Apple Music Dance",
                     "Apple Music Pop"):
            assert is_usable_apple_page(
                _Playlist(author=desk, playlist_id="pl.abc123"),
                PLAYLIST_URL), desk


class TestCataloguePages:
    """Apple's own catalogue is not a playlist and has no curator to check.

    Conflating the two is what made an artist's top-songs page come back empty:
    it has no author and no `pl.` id because it is not a playlist, and the
    editorial gate rejected it for failing to be one.
    """

    def test_an_artist_top_songs_page_is_accepted(self):
        """The page the owner hit: /artist/kanye-west/2715720/top-songs."""
        assert is_usable_apple_page(_Playlist(author=None, playlist_id=None),
                                    TOP_SONGS_URL)

    def test_an_album_page_is_accepted(self):
        assert is_usable_apple_page(
            _Playlist(), "https://music.apple.com/us/album/yeezus/1440851894")

    def test_the_kinds_are_told_apart_by_url(self):
        assert apple_page_kind(PLAYLIST_URL) == "playlist"
        assert apple_page_kind(TOP_SONGS_URL) == "catalogue"
        assert apple_page_kind("https://music.apple.com/us/browse") == "other"

    def test_an_unknown_apple_path_is_refused(self):
        """Neither a playlist nor catalogue — nothing to trust it on."""
        assert not is_usable_apple_page(
            _Playlist(author="Apple Music"),
            "https://music.apple.com/us/browse")

    def test_a_listener_playlist_is_refused(self):
        assert not is_usable_apple_page(_Playlist(
            author="Ivan", playlist_id="pl.u-8aAbBcC"), PLAYLIST_URL)

    def test_a_user_id_is_refused_even_if_the_author_claims_otherwise(self):
        """The id needs no parsing to have succeeded, so it still holds when
        the page shape changes and the author field goes wrong."""
        assert not is_usable_apple_page(_Playlist(
            author="Apple Music Alternative", playlist_id="pl.u-deadbeef"), PLAYLIST_URL)

    def test_a_missing_author_is_refused(self):
        """Not "probably fine": a missing author means the parser did not find
        one, and guessing turns a parser regression into silent bad data."""
        assert not is_usable_apple_page(_Playlist(
            author=None, playlist_id="pl.abc123"), PLAYLIST_URL)

    def test_another_brand_is_refused(self):
        assert not is_usable_apple_page(_Playlist(
            author="Rap Life", playlist_id="pl.abc123"), PLAYLIST_URL)

    def test_case_and_spacing_do_not_matter(self):
        assert is_usable_apple_page(_Playlist(
            author="  apple music alternative ", playlist_id="PL.ABC123"), PLAYLIST_URL)
