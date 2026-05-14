"""Unit tests for cluster_curator CLI tool — exercise the pure-logic parts."""

import sys, types
sys.modules.setdefault("laion_clap", types.ModuleType("laion_clap"))

from unittest.mock import MagicMock, patch

import pytest

from scripts.cluster_curator import format_cluster_summary, parse_command


def test_format_cluster_summary_basic():
    cluster = {
        "cluster_id": 0,
        "size": 12,
        "representative_tracks": [
            {"track_id": "t1", "title": "Karma Police", "artist": "Radiohead"},
            {"track_id": "t2", "title": "Atmosphere", "artist": "Joy Division"},
        ],
        "current_label": None,
    }
    out = format_cluster_summary(cluster)
    assert "Cluster 0" in out
    assert "12 tracks" in out
    assert "Karma Police" in out
    assert "Radiohead" in out
    assert "(unlabeled)" in out


def test_format_cluster_summary_with_label():
    cluster = {
        "cluster_id": 5,
        "size": 3,
        "representative_tracks": [],
        "current_label": "Indie melancholy",
    }
    out = format_cluster_summary(cluster)
    assert "Indie melancholy" in out


def test_parse_command_label():
    cmd, args = parse_command("label 3 Lo-fi indie")
    assert cmd == "label"
    assert args == ("3", "Lo-fi indie")


def test_parse_command_skip():
    cmd, args = parse_command("skip")
    assert cmd == "skip"
    assert args == ()


def test_parse_command_quit():
    cmd, args = parse_command("q")
    assert cmd == "quit"


def test_parse_command_train():
    cmd, args = parse_command("train")
    assert cmd == "train"


def test_parse_command_unknown_returns_unknown():
    cmd, args = parse_command("xyzzy")
    assert cmd == "unknown"
