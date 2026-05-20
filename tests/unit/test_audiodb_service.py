"""Tests for audiodb_service HTTP fetch and per-artist enrichment."""
import asyncio
import pytest
import requests
from unittest.mock import MagicMock, patch

from app.services.audiodb_service import _http_get_json


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


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
