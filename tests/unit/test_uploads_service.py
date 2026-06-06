"""Unit tests for the upload pipeline helpers."""

import pytest

from app.services._magic_sniff import sniff_audio_mime

# libmagic may be absent on some hosts; the MIME assertions only hold when it's
# present. Skip the whole module rather than fail the baseline if so.
pytest.importorskip("magic")


class TestSniffAudioMime:
    def test_flac_header(self):
        # "fLaC" magic at offset 0 is the FLAC stream signature.
        data = b"fLaC\x00\x00\x00\x22" + b"\x00" * 64
        mime = sniff_audio_mime(data)
        assert mime in ("audio/flac", "audio/x-flac")

    def test_mp3_id3v2_header(self):
        # "ID3" magic at offset 0 marks an MP3 with ID3v2 tags, followed by an
        # MPEG-1 Layer III frame sync (\xff\xfb). libmagic needs the actual frame
        # to classify as audio — a bare ID3 header alone sniffs as octet-stream.
        data = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x44" + b"\x00" * 1024
        mime = sniff_audio_mime(data)
        assert mime.startswith("audio/")  # libmagic varies — audio/mpeg or audio/mp3

    def test_rejects_text(self):
        data = b"#!/bin/sh\necho hello\n"
        mime = sniff_audio_mime(data)
        assert not mime.startswith("audio/")

    def test_rejects_jpeg(self):
        data = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 64
        mime = sniff_audio_mime(data)
        assert not mime.startswith("audio/")


class TestFixtures:
    def test_flac_fixture_loadable(self, audio_bytes):
        data = audio_bytes("tiny.flac")
        assert data[:4] == b"fLaC"

    def test_mp3_fixture_loadable(self, audio_bytes):
        data = audio_bytes("tiny.mp3")
        # MP3 files start with either an ID3 tag block or an MPEG frame sync.
        assert data[:3] == b"ID3" or data[0] == 0xFF


import io

from app.services.uploads_service import (
    MAX_UPLOAD_BYTES, EXT_BY_MIME,
    UploadRejected, UploadOversize, UploadWrongType,
    write_to_quarantine, atomic_promote_to_managed, choose_extension,
)


class TestChooseExtension:
    def test_flac(self):
        assert choose_extension("audio/flac", original="song.flac") == ".flac"

    def test_mp3(self):
        assert choose_extension("audio/mpeg", original="song.mp3") == ".mp3"

    def test_m4a_falls_back_to_filename(self):
        # libmagic reports audio/mp4 for m4a; we need to disambiguate via extension.
        assert choose_extension("audio/mp4", original="song.m4a") == ".m4a"

    def test_unknown_mime_uses_original_extension(self):
        assert choose_extension("application/octet-stream", original="song.flac") == ".flac"

    def test_no_extension_returns_bin(self):
        assert choose_extension("application/octet-stream", original="weirdfile") == ".bin"


class TestWriteToQuarantine:
    def test_writes_bytes_and_returns_sha(self, tmp_path, audio_bytes):
        data = audio_bytes("tiny.flac")
        quarantine_path = tmp_path / "u1.tmp"
        stream = io.BytesIO(data)
        sha, size, head = write_to_quarantine(
            stream, quarantine_path, max_bytes=MAX_UPLOAD_BYTES,
        )
        import hashlib
        assert sha == hashlib.sha256(data).hexdigest()
        assert size == len(data)
        assert head[:4] == b"fLaC"
        assert quarantine_path.exists()
        assert quarantine_path.read_bytes() == data

    def test_oversize_raises_and_cleans_up(self, tmp_path):
        big = io.BytesIO(b"x" * 1024)
        out = tmp_path / "u1.tmp"
        with pytest.raises(UploadOversize):
            write_to_quarantine(big, out, max_bytes=100)
        assert not out.exists()


class TestPromote:
    def test_atomic_move_into_managed_layout(self, tmp_path):
        src = tmp_path / "q.tmp"
        src.write_bytes(b"hello")
        media_root = tmp_path / "media"
        dst = atomic_promote_to_managed(
            quarantine_path=src,
            media_root=media_root,
            account_id="acct_abc",
            sha256="deadbeef",
            extension=".flac",
        )
        assert dst == media_root / "acct_abc" / "audio" / "deadbeef.flac"
        assert dst.exists()
        assert dst.read_bytes() == b"hello"
        assert not src.exists()

    def test_promote_idempotent_when_dst_already_exists(self, tmp_path):
        media_root = tmp_path / "media"
        existing = media_root / "acct_abc" / "audio" / "deadbeef.flac"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"existing")
        src = tmp_path / "q.tmp"
        src.write_bytes(b"new bytes")
        dst = atomic_promote_to_managed(
            quarantine_path=src,
            media_root=media_root,
            account_id="acct_abc",
            sha256="deadbeef",
            extension=".flac",
        )
        # Existing file untouched (content-addressed → identity), quarantine cleaned.
        assert dst.read_bytes() == b"existing"
        assert not src.exists()
