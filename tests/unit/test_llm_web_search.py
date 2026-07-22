"""Unit tests for the playlist web-search result re-ranker.

The playlist agent asks "list" questions and SearXNG's `genius` engine buries the
real chart/tracklist pages under bogus same-name entities. ``rank_playlist_results``
drops that junk, floats authoritative list domains up, and keeps genius ALBUM
tracklists. See the module note in ``llm_web_search.py``.
"""

import pytest

from app.services.llm_web_search import rank_playlist_results

pytestmark = pytest.mark.unit


def _urls(results):
    return [r["url"] for r in results]


def test_drops_genius_junk_keeps_album_tracklists():
    raw = [
        {"title": "Kanye West", "url": "https://genius.com/artists/Kanye-west"},
        {"title": "x annotated", "url": "https://genius.com/Some-thing-annotated"},
        {"title": "a lyrics", "url": "https://genius.com/Artist-song-lyrics"},
        {"title": "Watch Dogs OST", "url": "https://genius.com/albums/Various/Watch-dogs-soundtrack"},
        {"title": "wiki", "url": "https://en.wikipedia.org/wiki/Kanye_West"},
    ]
    out = _urls(rank_playlist_results(raw))
    # genius artist/annotation/lyric pages gone; album tracklist survives
    assert "https://genius.com/artists/Kanye-west" not in out
    assert "https://genius.com/Some-thing-annotated" not in out
    assert "https://genius.com/Artist-song-lyrics" not in out
    assert "https://genius.com/albums/Various/Watch-dogs-soundtrack" in out


def test_authority_domains_float_above_random_blog():
    raw = [
        {"title": "blog", "url": "https://randomblog.example.com/kanye"},
        {"title": "billboard", "url": "https://www.billboard.com/lists/kanye-hits"},
        {"title": "mb", "url": "https://musicbrainz.org/release/abc"},
    ]
    out = _urls(rank_playlist_results(raw))
    # authoritative sources outrank the position-1 blog
    assert out[0].startswith("https://www.billboard.com") or out[0].startswith("https://musicbrainz.org")
    assert out[-1] == "https://randomblog.example.com/kanye"


def test_list_path_bonus_prefers_dated_discography():
    raw = [
        {"title": "artist", "url": "https://en.wikipedia.org/wiki/Kanye_West"},
        {"title": "disco", "url": "https://en.wikipedia.org/wiki/Kanye_West_singles_discography"},
    ]
    out = _urls(rank_playlist_results(raw))
    assert out[0].endswith("singles_discography")


def test_non_list_noise_dropped():
    raw = [
        {"title": "insta", "url": "https://www.instagram.com/reel/x"},
        {"title": "yt", "url": "https://www.youtube.com/channel/UC123"},
        {"title": "tix", "url": "https://www.ticketmaster.com/ye-tickets/artist/1"},
        {"title": "wiki", "url": "https://en.wikipedia.org/wiki/Kanye_West"},
    ]
    out = _urls(rank_playlist_results(raw))
    assert out == ["https://en.wikipedia.org/wiki/Kanye_West"]


def test_all_junk_returns_original_head_never_blank():
    raw = [{"title": "a", "url": "https://genius.com/artists/x"}]
    # everything filtered → fall back to original so the agent never gets nothing
    assert rank_playlist_results(raw) == raw


def test_empty_in_empty_out():
    assert rank_playlist_results([]) == []
