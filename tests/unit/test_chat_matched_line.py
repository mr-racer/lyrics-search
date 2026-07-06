"""Unit tests for the matched-line picker used by the chat search loop."""

import pytest

from app.api.routes.chat import _pick_matched_line

pytestmark = pytest.mark.unit

LYRICS = "\n".join([
    "Ночь холодна и пуста",
    "Я иду по мокрым улицам города",
    "Неон отражается в лужах",
    "",
    "Ты далеко, но я помню голос",
])


def test_picks_line_with_max_overlap():
    # exact-token overlap: «неон» + «лужах» → third line wins
    assert _pick_matched_line(LYRICS, "неон в лужах") == "Неон отражается в лужах"


def test_no_overlap_returns_none():
    assert _pick_matched_line(LYRICS, "solar flare quantum") is None


def test_empty_lyrics_returns_none():
    assert _pick_matched_line("", "мокрые улицы") is None


def test_short_stopword_terms_ignored():
    # all query tokens ≤ 2 chars → no terms → None
    assert _pick_matched_line(LYRICS, "я и ты") is None
