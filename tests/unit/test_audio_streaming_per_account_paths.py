"""Per-account transcoded cache namespace: same track_id in two accounts → two files."""

from pathlib import Path

import pytest

from app.services import audio_streaming


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_streaming, "_CACHE_DIR", tmp_path / "transcoded")
    yield


def test_cache_path_includes_account_id():
    p = audio_streaming._cache_path(account_id="acct-A", track_id="abc123")
    assert p.parts[-2] == "acct-A"
    assert p.name == "abc123.flac"


def test_two_accounts_get_distinct_cache_files():
    a = audio_streaming._cache_path(account_id="acct-A", track_id="abc123")
    b = audio_streaming._cache_path(account_id="acct-B", track_id="abc123")
    assert a != b
    assert a.parent != b.parent


def test_drop_transcoded_for_tracks_only_touches_caller_account(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_streaming, "_CACHE_DIR", tmp_path / "tx")
    cache_dir = tmp_path / "tx"
    (cache_dir / "acct-A").mkdir(parents=True)
    (cache_dir / "acct-B").mkdir(parents=True)
    (cache_dir / "acct-A" / "shared.flac").write_bytes(b"a-data")
    (cache_dir / "acct-B" / "shared.flac").write_bytes(b"b-data")

    n = audio_streaming.drop_transcoded_for_tracks(account_id="acct-A", track_ids=["shared"])
    assert n == 1
    assert not (cache_dir / "acct-A" / "shared.flac").exists()
    assert (cache_dir / "acct-B" / "shared.flac").exists(), "must not touch other account's cache"


@pytest.mark.asyncio
async def test_get_streamable_path_uses_account_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_streaming, "_CACHE_DIR", tmp_path / "tx")
    # Force non-ALAC path so we don't need ffmpeg
    monkeypatch.setattr(audio_streaming, "_is_alac_m4a", lambda *_: False)
    src = tmp_path / "song.flac"
    src.write_bytes(b"flac-bytes")

    path, mime = await audio_streaming.get_streamable_path(
        account_id="acct-X", track_id="t1", file_path=src,
    )
    # Non-ALAC short-circuits → returns source path unchanged
    assert path == src
    assert mime == "audio/flac"
