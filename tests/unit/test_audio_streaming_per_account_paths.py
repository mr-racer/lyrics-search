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
    # .flac short-circuits codec detection (suffix check) — no ffprobe/ffmpeg needed.
    src = tmp_path / "song.flac"
    src.write_bytes(b"flac-bytes")

    path, mime = await audio_streaming.get_streamable_path(
        account_id="acct-X", track_id="t1", file_path=src,
    )
    # Non-ALAC short-circuits → returns source path unchanged
    assert path == src
    assert mime == "audio/flac"


@pytest.mark.asyncio
async def test_codec_hint_skips_ffprobe(tmp_path, monkeypatch):
    """A payload codec hint must decide ALAC-ness without ever running ffprobe."""
    def _boom(*_):
        raise AssertionError("ffprobe must not run when a codec hint is present")
    monkeypatch.setattr(audio_streaming, "_detect_codec_ffprobe", _boom)

    src = tmp_path / "song.m4a"
    src.write_bytes(b"m4a-bytes")

    # AAC hint → served as-is, no ffprobe, no transcode.
    path, mime = await audio_streaming.get_streamable_path(
        account_id="acct-X", track_id="t1", file_path=src, codec="mp4a.40.2",
    )
    assert path == src
    assert mime == "audio/mp4"

    assert await audio_streaming._needs_alac_transcode(src, codec_hint="alac") is True


@pytest.mark.asyncio
async def test_codec_detection_cached_per_file_identity(tmp_path, monkeypatch):
    """Without a hint, ffprobe runs once per (path, mtime, size), not per request."""
    calls = []
    monkeypatch.setattr(
        audio_streaming, "_detect_codec_ffprobe",
        lambda p: calls.append(p) or "aac",
    )
    src = tmp_path / "song.m4a"
    src.write_bytes(b"m4a-bytes")

    assert await audio_streaming._needs_alac_transcode(src) is False
    assert await audio_streaming._needs_alac_transcode(src) is False
    assert len(calls) == 1


def test_source_cache_roundtrip_and_scoped_drop():
    audio_streaming.put_cached_source("acct-A", "t1", "/x/a.flac", "alac")
    audio_streaming.put_cached_source("acct-A", "t2", "/x/b.flac", None)
    audio_streaming.put_cached_source("acct-B", "t1", "/y/a.flac", None)

    assert audio_streaming.get_cached_source("acct-A", "t1") == ("/x/a.flac", "alac")

    audio_streaming.drop_source_for_tracks("acct-A", ["t1"])
    assert audio_streaming.get_cached_source("acct-A", "t1") is None
    assert audio_streaming.get_cached_source("acct-A", "t2") is not None
    assert audio_streaming.get_cached_source("acct-B", "t1") is not None

    audio_streaming.drop_account_sources("acct-A")
    assert audio_streaming.get_cached_source("acct-A", "t2") is None
    assert audio_streaming.get_cached_source("acct-B", "t1") == ("/y/a.flac", None)


def test_source_cache_ttl_expiry(monkeypatch):
    audio_streaming.put_cached_source("acct-T", "t1", "/x/a.flac", None)
    real_monotonic = audio_streaming.time.monotonic
    monkeypatch.setattr(
        audio_streaming.time, "monotonic",
        lambda: real_monotonic() + audio_streaming._SOURCE_TTL_SECONDS + 1,
    )
    assert audio_streaming.get_cached_source("acct-T", "t1") is None
