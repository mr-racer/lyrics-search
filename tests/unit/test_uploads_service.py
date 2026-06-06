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
