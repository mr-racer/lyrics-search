"""Unit tests for the playlist web-search result re-ranker.

The playlist agent asks "list" questions and SearXNG's `genius` engine buries the
real chart/tracklist pages under bogus same-name entities. ``rank_playlist_results``
drops that junk, floats authoritative list domains up, and keeps genius ALBUM
tracklists. See the module note in ``llm_web_search.py``.
"""

import pytest

from app.services.llm_web_search import (
    _extract_tracklines,
    rank_playlist_results,
)

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


# ── _extract_tracklines ──────────────────────────────────────────────────────
# Fandom wikitables come out of readability as one line PER CELL: the artist
# line ends with a dash, the title is the next line, the year "(2012)" the one
# after. The extractor must re-join those before filtering, so a 26-station
# soundtrack page survives as compact "Artist — Title (year)" lines instead of
# being cut off by the flat 4000-char prefix cap.

_FANDOM_STYLE = "\n".join(
    line
    for i in range(12)
    for line in (f"Artist Number{i} feat. Guest —", f"Song Title {i}", f"(201{i % 10})")
)

_FLAT_STYLE = "\n".join(f"Artist {i} - Song {i}" for i in range(15))

_PROSE = (
    "Grand Theft Auto V is an action-adventure game played from either a "
    "third-person or first-person perspective. Players complete missions to "
    "progress the story.\n"
    "The game is set within the fictional state of San Andreas, based on "
    "Southern California. Criminal activities may gain a wanted level for "
    "the player character.\n"
) * 6


def test_tracklines_merges_fandom_table_cells():
    out = _extract_tracklines(_FANDOM_STYLE)
    assert "Artist Number3 feat. Guest — Song Title 3 (2013)" in out
    # every one of the 12 entries survives on its own line
    assert sum(1 for ln in out.splitlines() if "—" in ln) == 12


def test_tracklines_keeps_flat_single_dash_lists():
    out = _extract_tracklines(_FLAT_STYLE)
    assert "Artist 7 - Song 7" in out
    assert len(out.splitlines()) == 15


def test_tracklines_rejects_prose_pages():
    # a page with no track-like lines must return "" so the caller falls back
    # to the old prefix behaviour instead of feeding the model filtered noise
    assert _extract_tracklines(_PROSE) == ""


def test_tracklines_below_threshold_returns_empty():
    # 3 track-like lines in a sea of prose is not a tracklist page
    text = _PROSE + "\nA - B\nC - D\nE - F\n"
    assert _extract_tracklines(text) == ""


def test_tracklines_caps_output_at_line_boundary():
    huge = "\n".join(f"Artist {i} — Song number {i} (2010)" for i in range(1000))
    out = _extract_tracklines(huge, max_chars=2000)
    assert len(out) <= 2100
    # cut on a line boundary — no half-line garbage
    body = out.rstrip("…").rstrip()
    assert body.splitlines()[-1].startswith("Artist ")


def test_tracklines_over_budget_samples_across_whole_page():
    """Soundtrack pages group songs by radio station; a prefix cut silently
    drops every station after the first. Over-budget extraction must sample
    evenly across the WHOLE page — head, middle and tail all represented."""
    huge = "\n".join(f"Artist {i} — Song number {i} (2010)" for i in range(1000))
    out = _extract_tracklines(huge, max_chars=2000)
    nums = [int(ln.split(" ")[1]) for ln in out.splitlines() if ln.startswith("Artist ")]
    assert min(nums) < 100          # head present
    assert any(400 < n < 600 for n in nums)   # middle present
    assert max(nums) > 900          # tail present
    # and the model is told it sees a sample, not the full list
    assert "of 1000" in out


def test_tracklines_mixed_page_keeps_only_tracks():
    text = _PROSE + "\n" + _FLAT_STYLE + "\n" + _PROSE
    out = _extract_tracklines(text)
    assert "Artist 3 - Song 3" in out
    assert "wanted level" not in out


# ── engines fan-out for the playlist path ────────────────────────────────────


def test_search_searxng_passes_explicit_engines(monkeypatch):
    from app.services import llm_web_search as m

    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"title": "t", "url": "https://x.example/", "content": "s"}]}

    def fake_get(url, params=None, **kw):
        captured.update(params or {})
        return _Resp()

    monkeypatch.setattr(m, "_WEB_SEARCH_AVAILABLE", True)
    monkeypatch.setattr(m.httpx, "get", fake_get)
    m.search_searxng("q", max_results=5, engines="brave,presearch")
    assert captured.get("engines") == "brave,presearch"


def test_playlist_rank_uses_redundant_engine_set(monkeypatch):
    from app.services import llm_web_search as m

    seen = {}

    def fake_search(query, max_results=5, engines=None):
        seen["engines"] = engines
        return []

    monkeypatch.setattr(m, "search_searxng", fake_search)
    m.smart_web_search("gta 5 songs", fetch_content=False, max_results=3, rank="playlist")
    assert seen["engines"] == m.SEARXNG_PLAYLIST_ENGINES
    assert "presearch" in seen["engines"] and "brave" in seen["engines"]


def test_playlist_fetch_prefers_tracklist_page_over_readable_prose(monkeypatch):
    """Two readable prose pages must NOT exhaust the full-read budget: the
    tracklist page further down the ranking is the one worth inlining."""
    from app.services import llm_web_search as m

    tracklist_body = "\n".join(f"Artist {i} — Song {i} (2005)" for i in range(20))
    prose_body = _PROSE

    results = [
        {"title": "wiki EN", "url": "https://en.wikipedia.org/wiki/GTA", "content": "sn1"},
        {"title": "wiki DE", "url": "https://de.wikipedia.org/wiki/GTA", "content": "sn2"},
        {"title": "fandom", "url": "https://gta-songs.fandom.com/wiki/Radio", "content": "sn3"},
    ]

    def fake_search(query, max_results=5, engines=None):
        return results

    def fake_fetch(url, max_chars=4000):
        if "fandom" in url:
            return tracklist_body
        return prose_body

    monkeypatch.setattr(m, "search_searxng", fake_search)
    monkeypatch.setattr(m, "fetch_full_content", fake_fetch)
    monkeypatch.setattr(m, "rank_playlist_results", lambda pool, q="": pool)

    out = m.smart_web_search("gta 5 songs", fetch_content=True, max_results=3, rank="playlist")
    # the fandom tracklist is inlined in extracted form
    assert "Artist 7 — Song 7 (2005)" in out
    # prose pages appear, but the tracklist page was not crowded out
    assert "fandom" in out


# ── GLiNER2 salvage pairing (pure logic; the model itself is not loaded) ─────

from app.services.llm_web_search import _gliner_tracklines, _pair_entities


def test_pair_entities_handles_both_orientations():
    text = ("The station features Radio Ga Ga by Queen and also plays\n"
            "M83 - Midnight City every night.")
    out = _pair_entities(text, ["Radio Ga Ga", "Midnight City"], ["Queen", "M83"])
    assert "Queen — Radio Ga Ga" in out
    assert "M83 — Midnight City" in out


def test_pair_entities_respects_distance_gap():
    text = "Radio Ga Ga is a great song." + (" filler" * 60) + " Queen formed in 1970."
    assert _pair_entities(text, ["Radio Ga Ga"], ["Queen"]) == []


def test_gliner_tracklines_chunks_and_dedupes(monkeypatch):
    from app.services import llm_web_search as m

    def fake_extract(chunk):
        return {"songs": ["Radio Ga Ga"], "artists": ["Queen"]}

    monkeypatch.setattr(m, "_gliner_extract_entities", fake_extract)
    text = "Queen plays Radio Ga Ga tonight.\n" * 200   # many chunks, same pair
    out = m._gliner_tracklines(text)
    assert out == ["Queen — Radio Ga Ga"]


def test_playlist_fetch_collects_tracklines_for_automatch(monkeypatch):
    """The FULL extracted tracklist (not the sampled excerpt) must reach the
    caller via tracklines_out — code-side library intersection needs it all."""
    from app.services import llm_web_search as m

    tracklist_body = "\n".join(f"Artist {i} — Song {i} (2005)" for i in range(300))
    results = [{"title": "fandom", "url": "https://gta-songs.fandom.com/wiki/R",
                "content": "sn"}]

    monkeypatch.setattr(m, "search_searxng", lambda q, max_results=5, engines=None: results)
    monkeypatch.setattr(m, "fetch_full_content", lambda url, max_chars=4000: tracklist_body)
    monkeypatch.setattr(m, "rank_playlist_results", lambda pool, q="": pool)

    lines: list = []
    out = m.smart_web_search("gta 5 songs", fetch_content=True, max_results=3,
                             rank="playlist", tracklines_out=lines)
    assert len(lines) == 300                      # full list collected
    assert "Artist 299 — Song 299 (2005)" in lines[-1]
    assert len(out) < 6000                        # model-visible text stays compact


def test_tracklines_dedupes_repeated_junk_below_threshold():
    """Wikipedia nav/reference junk repeats one pseudo-track line many times;
    after dedupe the page falls below the tracklist threshold and goes the
    prose-fallback route instead of wasting a full-read slot."""
    junk = "\n".join(["The Cinematographic Score — GTA"] * 8
                     + ["Franz Kafka — The Castle"] * 3)
    assert _extract_tracklines(junk) == ""


def test_playlist_fetch_walks_past_max_results_to_find_tracklist(monkeypatch):
    """The tracklist page may rank anywhere in the pool — position 5 with
    max_results=3 must still be fetched and inlined; the model-visible output
    stays compact (snippets capped at max_results entries)."""
    from app.services import llm_web_search as m

    tracklist_body = "\n".join(f"Artist {i} — Song {i} (2005)" for i in range(20))
    results = [
        {"title": f"prose{i}", "url": f"https://site{i}.example/a", "content": f"sn{i}"}
        for i in range(4)
    ] + [{"title": "fandom", "url": "https://gta-songs.fandom.com/wiki/R", "content": "sn"}]

    monkeypatch.setattr(m, "search_searxng", lambda q, max_results=5, engines=None: results)
    monkeypatch.setattr(m, "fetch_full_content",
                        lambda url, max_chars=4000:
                        tracklist_body if "fandom" in url else _PROSE)
    monkeypatch.setattr(m, "rank_playlist_results", lambda pool, q="": pool)

    lines: list = []
    out = m.smart_web_search("gta 5 songs", fetch_content=True, max_results=3,
                             rank="playlist", tracklines_out=lines)
    assert "Artist 7 — Song 7 (2005)" in out    # position-5 tracklist inlined
    assert len(lines) == 20                      # and collected for auto-match
