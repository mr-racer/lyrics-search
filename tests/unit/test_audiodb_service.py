"""Tests for audiodb_service HTTP fetch and per-artist enrichment."""
import asyncio
import pytest
import requests
from unittest.mock import MagicMock, patch

from app.services.audiodb_service import _http_get_json


def _run(coro):
    # Python 3.12+ no longer auto-creates an event loop in MainThread.
    # Use a fresh loop per call so we don't reuse a closed one across tests.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_http_get_json_happy_path():
    fake_response = MagicMock()
    fake_response.json.return_value = {"artists": [{"strArtist": "Kanye West"}]}
    fake_response.raise_for_status.return_value = None
    with patch("app.services.audiodb_service.requests.get", return_value=fake_response):
        result = _run(_http_get_json("http://example.com"))
    assert result == {"artists": [{"strArtist": "Kanye West"}]}


def test_http_get_json_retries_on_connection_error():
    """First call raises ConnectionError, second succeeds."""
    fake_response = MagicMock()
    fake_response.json.return_value = {"artists": [{"strArtist": "x"}]}
    fake_response.raise_for_status.return_value = None
    side_effects = [requests.ConnectionError("boom"), fake_response]
    with patch("app.services.audiodb_service.requests.get", side_effect=side_effects):
        with patch("app.services.audiodb_service.asyncio.sleep", return_value=None):
            result = _run(_http_get_json("http://example.com"))
    assert result == {"artists": [{"strArtist": "x"}]}


def test_http_get_json_returns_none_after_two_failures():
    """Both attempts time out; helper returns None (caller treats as 'not found')."""
    with patch(
        "app.services.audiodb_service.requests.get",
        side_effect=requests.Timeout("slow"),
    ):
        with patch("app.services.audiodb_service.asyncio.sleep", return_value=None):
            result = _run(_http_get_json("http://example.com"))
    assert result is None


def test_http_get_json_returns_none_on_http_404_no_retry():
    """4xx errors are not retried — they indicate the response was received."""
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = requests.HTTPError("404")
    with patch("app.services.audiodb_service.requests.get", return_value=fake_response):
        result = _run(_http_get_json("http://example.com"))
    assert result is None


from app.services.audiodb_service import _download_image


def test_download_image_writes_content_addressed_file(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    png_bytes = b"\x89PNG\r\n\x1a\nfake_image_data_padding_to_make_unique"
    fake_response = MagicMock()
    fake_response.content = png_bytes
    fake_response.raise_for_status.return_value = None
    with patch("app.services.audiodb_service.requests.get", return_value=fake_response):
        url = _run(_download_image("http://audiodb.com/x.png"))
    assert url is not None
    assert url.startswith("/covers/artists/")
    assert url.endswith(".png")
    # File written
    fname = url.rsplit("/", 1)[-1]
    assert (tmp_path / fname).exists()
    assert (tmp_path / fname).read_bytes() == png_bytes


def test_download_image_returns_none_on_empty_url():
    result = _run(_download_image(None))
    assert result is None
    result = _run(_download_image(""))
    assert result is None


def test_download_image_returns_none_on_http_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)
    with patch(
        "app.services.audiodb_service.requests.get",
        side_effect=requests.ConnectionError("boom"),
    ):
        result = _run(_download_image("http://audiodb.com/x.png"))
    assert result is None


def test_download_image_dedups_by_content(tmp_path, monkeypatch):
    """Two calls with same image bytes produce the same URL (content-addressed)."""
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    png_bytes = b"\x89PNG\r\n\x1a\nsame_image_bytes"
    fake_response = MagicMock()
    fake_response.content = png_bytes
    fake_response.raise_for_status.return_value = None
    with patch("app.services.audiodb_service.requests.get", return_value=fake_response):
        url1 = _run(_download_image("http://example.com/a.png"))
        url2 = _run(_download_image("http://example.com/b.png"))   # different URL, same bytes
    assert url1 == url2


from app.resources.metadata_db import MetadataDB
from app.services.audiodb_service import fetch_audiodb_for_artist


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _audiodb_response(**fields):
    base = {
        "strBiographyEN": "Bio in English.",
        "strBiography": "Default bio.",
        "strMood": "energetic",
        "strCountryCode": "US",
        "strCountry": "Chicago, USA",
        "strLabel": "Roc-A-Fella",
        "strArtistCutout": "http://audiodb.com/cutout.png",
        "strArtistThumb": "http://audiodb.com/thumb.png",
        "strMusicBrainzID": "mbid-123",
    }
    base.update(fields)
    return {"artists": [base]}


def test_fetch_audiodb_for_artist_happy_path(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    json_response = MagicMock()
    json_response.json.return_value = _audiodb_response()
    json_response.raise_for_status.return_value = None

    img_response = MagicMock()
    img_response.content = b"\x89PNG\r\n\x1a\nfake"
    img_response.raise_for_status.return_value = None

    with patch(
        "app.services.audiodb_service.requests.get",
        side_effect=[json_response, img_response, img_response],
    ):
        _run(fetch_audiodb_for_artist("Kanye West", "test_col"))

    row = MetadataDB.get_artist_audiodb("kanye-west", "test_col")
    assert row is not None
    assert row["audiodb_bio"] == "Bio in English."
    assert row["mood"] == "energetic"
    assert row["country_code"] == "US"
    assert row["country"] == "Chicago, USA"
    assert row["label"] == "Roc-A-Fella"
    assert row["audiodb_mbid"] == "mbid-123"
    assert row["cutout_path"] and row["cutout_path"].startswith("/covers/artists/")
    assert row["thumb_path"] and row["thumb_path"].startswith("/covers/artists/")
    assert row["audiodb_fetched_at"]


def test_fetch_audiodb_for_artist_strips_feat_in_query_and_persists_under_canonical(
    isolated_db, tmp_path, monkeypatch,
):
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    json_response = MagicMock()
    json_response.json.return_value = _audiodb_response(strArtist="Dua Lipa")
    json_response.raise_for_status.return_value = None
    img_response = MagicMock()
    img_response.content = b"\x89PNG\r\n\x1a\nfake"
    img_response.raise_for_status.return_value = None

    captured_urls = []

    def fake_get(url, timeout=None):
        captured_urls.append(url)
        return img_response if "audiodb.com/cutout" in url or "audiodb.com/thumb" in url else json_response

    with patch("app.services.audiodb_service.requests.get", side_effect=fake_get):
        _run(fetch_audiodb_for_artist("Dua Lipa feat. Angele", "test_col"))

    # URL must be 'dua+lipa', not 'dua+lipa+feat+angele'
    assert any("s=dua+lipa" in u for u in captured_urls)
    # Persisted under canonical slug
    assert MetadataDB.get_artist_audiodb("dua-lipa", "test_col") is not None
    # NOT under feat-suffixed slug
    feat_row = MetadataDB.get_artist_audiodb("dua-lipa-feat-angele", "test_col")
    # get_artist_audiodb returns None when the row doesn't exist at all
    assert feat_row is None


def test_fetch_audiodb_for_artist_marks_fetched_when_artist_not_found(
    isolated_db, tmp_path, monkeypatch,
):
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)
    json_response = MagicMock()
    json_response.json.return_value = {"artists": None}
    json_response.raise_for_status.return_value = None
    with patch("app.services.audiodb_service.requests.get", return_value=json_response):
        _run(fetch_audiodb_for_artist("Nonexistent Band", "test_col"))

    row = MetadataDB.get_artist_audiodb("nonexistent-band", "test_col")
    assert row is not None
    assert row["audiodb_bio"] is None
    assert row["audiodb_fetched_at"]  # IS set — prevents re-fetch loop


def test_fetch_audiodb_for_artist_skips_if_already_fetched(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)
    # Pre-populate
    MetadataDB.upsert_artist_audiodb(
        slug="cached-artist", collection_name="test_col",
        audiodb_bio="old", mood=None, country_code=None, country=None,
        label=None, cutout_path=None, thumb_path=None, audiodb_mbid=None,
    )
    mock_get = MagicMock()
    with patch("app.services.audiodb_service.requests.get", mock_get):
        _run(fetch_audiodb_for_artist("Cached Artist", "test_col"))
    mock_get.assert_not_called()


def test_fetch_audiodb_for_artist_handles_total_network_failure(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)
    with patch(
        "app.services.audiodb_service.requests.get",
        side_effect=requests.ConnectionError("boom"),
    ):
        with patch("app.services.audiodb_service.asyncio.sleep", return_value=None):
            # Should not raise
            _run(fetch_audiodb_for_artist("Unreachable Artist", "test_col"))

    # No row written; next index will retry.
    assert MetadataDB.get_artist_audiodb("unreachable-artist", "test_col") is None

# ---------------------------------------------------------------------------
# Deezer fallback — artist image when AudioDB yields none
# ---------------------------------------------------------------------------


def _deezer_json_response(name="Kanye West", picture_xl="http://cdn.deezer.example/pic_xl.jpg"):
    resp = MagicMock()
    resp.json.return_value = {"data": [{"name": name, "picture_xl": picture_xl}], "total": 1}
    resp.raise_for_status.return_value = None
    return resp


def _jpeg_response():
    resp = MagicMock()
    resp.content = b"\xff\xd8\xfffake_jpeg_bytes"
    resp.raise_for_status.return_value = None
    return resp


def test_deezer_fallback_when_audiodb_misses(isolated_db, tmp_path, monkeypatch):
    """AudioDB doesn't know the artist -> Deezer search by the same +slug fills thumb_path."""
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    audiodb_json = MagicMock()
    audiodb_json.json.return_value = {"artists": None}
    audiodb_json.raise_for_status.return_value = None
    deezer_json = _deezer_json_response()
    img_response = _jpeg_response()

    captured = []

    def fake_get(url, timeout=None):
        captured.append(url)
        if "search.php" in url:
            return audiodb_json
        if "api.deezer.com" in url:
            return deezer_json
        return img_response

    with patch("app.services.audiodb_service.requests.get", side_effect=fake_get):
        _run(fetch_audiodb_for_artist("Kanye West", "test_col"))

    # Deezer queried with the lowercase+plus slug, not the display name
    assert any("api.deezer.com/search/artist?q=kanye+west" in u for u in captured)
    row = MetadataDB.get_artist_audiodb("kanye-west", "test_col")
    assert row is not None
    assert row["thumb_path"] and row["thumb_path"].startswith("/covers/artists/")
    assert row["thumb_path"].endswith(".jpg")
    assert row["audiodb_fetched_at"]  # still marked fetched — no re-fetch loop


def test_deezer_fallback_when_audiodb_image_downloads_fail(isolated_db, tmp_path, monkeypatch):
    """AudioDB found the artist but both image downloads fail -> Deezer fills thumb_path,
    while AudioDB text fields are still persisted."""
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    audiodb_json = MagicMock()
    audiodb_json.json.return_value = _audiodb_response()
    audiodb_json.raise_for_status.return_value = None
    deezer_json = _deezer_json_response()
    img_response = _jpeg_response()

    def fake_get(url, timeout=None):
        if "search.php" in url:
            return audiodb_json
        if "audiodb.com" in url:  # cutout + thumb downloads
            raise requests.ConnectionError("image host down")
        if "api.deezer.com" in url:
            return deezer_json
        return img_response  # deezer picture_xl

    with patch("app.services.audiodb_service.requests.get", side_effect=fake_get):
        _run(fetch_audiodb_for_artist("Kanye West", "test_col"))

    row = MetadataDB.get_artist_audiodb("kanye-west", "test_col")
    assert row["audiodb_bio"] == "Bio in English."  # AudioDB text intact
    assert row["cutout_path"] is None
    assert row["thumb_path"] and row["thumb_path"].startswith("/covers/artists/")


def test_deezer_skipped_when_audiodb_image_downloaded(isolated_db, tmp_path, monkeypatch):
    """AudioDB gave an image and the download succeeded -> never knock on Deezer."""
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    audiodb_json = MagicMock()
    audiodb_json.json.return_value = _audiodb_response()
    audiodb_json.raise_for_status.return_value = None
    img_response = MagicMock()
    img_response.content = b"\x89PNG\r\n\x1a\nfake"
    img_response.raise_for_status.return_value = None

    captured = []

    def fake_get(url, timeout=None):
        captured.append(url)
        return audiodb_json if "search.php" in url else img_response

    with patch("app.services.audiodb_service.requests.get", side_effect=fake_get):
        _run(fetch_audiodb_for_artist("Kanye West", "test_col"))

    assert not any("deezer" in u for u in captured)


def test_deezer_skipped_when_only_cutout_downloaded(isolated_db, tmp_path, monkeypatch):
    """One successfully downloaded AudioDB image (cutout) is enough to skip Deezer."""
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    audiodb_json = MagicMock()
    audiodb_json.json.return_value = _audiodb_response(strArtistThumb=None)
    audiodb_json.raise_for_status.return_value = None
    img_response = MagicMock()
    img_response.content = b"\x89PNG\r\n\x1a\nfake"
    img_response.raise_for_status.return_value = None

    captured = []

    def fake_get(url, timeout=None):
        captured.append(url)
        return audiodb_json if "search.php" in url else img_response

    with patch("app.services.audiodb_service.requests.get", side_effect=fake_get):
        _run(fetch_audiodb_for_artist("Kanye West", "test_col"))

    assert not any("deezer" in u for u in captured)
    row = MetadataDB.get_artist_audiodb("kanye-west", "test_col")
    assert row["cutout_path"] is not None
    assert row["thumb_path"] is None


def test_deezer_fallback_rejects_name_mismatch(isolated_db, tmp_path, monkeypatch):
    """First Deezer result name != pre-slugify query -> picture is NOT downloaded."""
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    audiodb_json = MagicMock()
    audiodb_json.json.return_value = {"artists": None}
    audiodb_json.raise_for_status.return_value = None
    deezer_json = _deezer_json_response(name="Kanye East")

    captured = []

    def fake_get(url, timeout=None):
        captured.append(url)
        if "search.php" in url:
            return audiodb_json
        if "api.deezer.com" in url:
            return deezer_json
        raise AssertionError(f"unexpected download: {url}")

    with patch("app.services.audiodb_service.requests.get", side_effect=fake_get):
        _run(fetch_audiodb_for_artist("Kanye West", "test_col"))

    assert not any("pic_xl" in u for u in captured)
    row = MetadataDB.get_artist_audiodb("kanye-west", "test_col")
    assert row is not None
    assert row["thumb_path"] is None
    assert row["audiodb_fetched_at"]


def test_deezer_fallback_name_match_is_case_insensitive(isolated_db, tmp_path, monkeypatch):
    """'KANYE WEST' from Deezer matches the query 'Kanye West'."""
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    audiodb_json = MagicMock()
    audiodb_json.json.return_value = {"artists": None}
    audiodb_json.raise_for_status.return_value = None
    deezer_json = _deezer_json_response(name="KANYE WEST")
    img_response = _jpeg_response()

    def fake_get(url, timeout=None):
        if "search.php" in url:
            return audiodb_json
        if "api.deezer.com" in url:
            return deezer_json
        return img_response

    with patch("app.services.audiodb_service.requests.get", side_effect=fake_get):
        _run(fetch_audiodb_for_artist("Kanye West", "test_col"))

    row = MetadataDB.get_artist_audiodb("kanye-west", "test_col")
    assert row["thumb_path"] is not None


def test_deezer_fallback_survives_deezer_failure(isolated_db, tmp_path, monkeypatch):
    """Deezer down / empty results -> row is still written (graceful, thumb stays None)."""
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    audiodb_json = MagicMock()
    audiodb_json.json.return_value = {"artists": None}
    audiodb_json.raise_for_status.return_value = None

    def fake_get(url, timeout=None):
        if "search.php" in url:
            return audiodb_json
        raise requests.ConnectionError("deezer down")

    with patch("app.services.audiodb_service.requests.get", side_effect=fake_get):
        with patch("app.services.audiodb_service.asyncio.sleep", return_value=None):
            _run(fetch_audiodb_for_artist("Kanye West", "test_col"))

    row = MetadataDB.get_artist_audiodb("kanye-west", "test_col")
    assert row is not None
    assert row["thumb_path"] is None
    assert row["audiodb_fetched_at"]

    # Empty result list is equally graceful
    deezer_empty = MagicMock()
    deezer_empty.json.return_value = {"data": [], "total": 0}
    deezer_empty.raise_for_status.return_value = None

    def fake_get2(url, timeout=None):
        return audiodb_json if "search.php" in url else deezer_empty

    with patch("app.services.audiodb_service.requests.get", side_effect=fake_get2):
        _run(fetch_audiodb_for_artist("Other Band", "test_col"))

    row2 = MetadataDB.get_artist_audiodb("other-band", "test_col")
    assert row2 is not None
    assert row2["thumb_path"] is None


from app.services.audiodb_service import fetch_audiodb_for_artists


def test_fetch_audiodb_for_artists_dedups_by_canonical_name(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    json_response = MagicMock()
    json_response.json.return_value = _audiodb_response()
    json_response.raise_for_status.return_value = None
    img_response = MagicMock()
    img_response.content = b"\x89PNG\r\n\x1a\nfake"
    img_response.raise_for_status.return_value = None

    captured = []
    def fake_get(url, timeout=None):
        captured.append(url)
        return img_response if ("cutout" in url or "thumb" in url) else json_response

    with patch("app.services.audiodb_service.requests.get", side_effect=fake_get):
        with patch("app.services.audiodb_service.asyncio.sleep", return_value=None):
            await_result = _run(fetch_audiodb_for_artists(
                ["Dua Lipa", "Dua Lipa feat. Angele", "Dua Lipa ft. Other"],
                "test_col",
            ))

    # Exactly one JSON-fetch URL (the others are image downloads)
    json_urls = [u for u in captured if "search.php" in u]
    assert len(json_urls) == 1, json_urls
    # Result keyed by canonical name
    assert list(await_result.keys()) == ["Dua Lipa"]


def test_fetch_audiodb_for_artists_calls_progress_callback(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    json_response = MagicMock()
    json_response.json.return_value = _audiodb_response()
    json_response.raise_for_status.return_value = None
    img_response = MagicMock()
    img_response.content = b"\x89PNG\r\n\x1a\nfake"
    img_response.raise_for_status.return_value = None
    with patch(
        "app.services.audiodb_service.requests.get",
        side_effect=lambda u, timeout=None: img_response if ("cutout" in u or "thumb" in u) else json_response,
    ):
        with patch("app.services.audiodb_service.asyncio.sleep", return_value=None):
            calls = []
            _run(fetch_audiodb_for_artists(
                ["Kanye West", "Dua Lipa"],
                "test_col",
                progress_callback=lambda idx, total, label, found: calls.append((idx, total, label, found)),
            ))
    assert len(calls) == 2
    assert calls[0][:3] == (1, 2, "Kanye West")
    assert calls[1][:3] == (2, 2, "Dua Lipa")
    assert all(c[3] is True for c in calls)


def test_fetch_audiodb_for_artists_per_artist_failure_does_not_break_batch(
    isolated_db, tmp_path, monkeypatch,
):
    monkeypatch.setattr("app.services.audiodb_service.ARTIST_COVERS_DIR", tmp_path)

    img_response = MagicMock()
    img_response.content = b"\x89PNG\r\n\x1a\nfake"
    img_response.raise_for_status.return_value = None
    good_json = MagicMock()
    good_json.json.return_value = _audiodb_response()
    good_json.raise_for_status.return_value = None

    def fake_get(url, timeout=None):
        if "good+artist" in url:
            return good_json
        if "cutout" in url or "thumb" in url:
            return img_response
        raise requests.ConnectionError("artist 2 unreachable")

    with patch("app.services.audiodb_service.requests.get", side_effect=fake_get):
        with patch("app.services.audiodb_service.asyncio.sleep", return_value=None):
            _run(fetch_audiodb_for_artists(
                ["Good Artist", "Bad Artist"],
                "test_col",
            ))

    assert MetadataDB.get_artist_audiodb("good-artist", "test_col") is not None
    # Bad artist: total network failure → no row written
    assert MetadataDB.get_artist_audiodb("bad-artist", "test_col") is None
