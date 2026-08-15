"""The new assistant imports and wires up under the conftest stubs.

Not a formality. Every module in the pipeline is required to be importable on a
machine with no torch — the heavy imports live inside functions — and this is
what catches the one that quietly moved back to the top of a file.
"""

from __future__ import annotations

import pytest


def test_every_module_imports_without_torch():
    import app.services.assistant.agent as agent
    import app.services.assistant.branches.audio  # noqa: F401
    import app.services.assistant.branches.general  # noqa: F401
    import app.services.assistant.branches.lyrics  # noqa: F401
    import app.services.assistant.branches.playlist  # noqa: F401
    import app.services.assistant.service  # noqa: F401
    import app.services.assistant.tracklists  # noqa: F401
    import app.services.library_catalog  # noqa: F401
    import app.services.retrieval  # noqa: F401

    assert hasattr(agent, "Assistant")


def test_config_defaults_match_the_module_constants():
    """The dataclass exists so one run can override a knob; the constants at the
    top of the module are where the knob actually lives. A field that drifts from
    its constant is a trap for whoever tunes the wrong one."""
    from app.services.assistant import config as c

    cfg = c.AgentConfig()
    assert cfg.ce_threshold_docs == c.CE_THRESHOLD_DOCS == 0.35
    assert cfg.ce_threshold_chunks == c.CE_THRESHOLD_CHUNKS == 0.65
    assert cfg.ce_threshold_facts == c.CE_THRESHOLD_FACTS == 0.25
    assert cfg.fetch_refill_attempts == c.FETCH_REFILL_ATTEMPTS == 4
    assert cfg.dedup_chunks is c.DEDUP_CHUNKS is True
    assert cfg.dedup_pool_factor == c.DEDUP_POOL_FACTOR == 3
    assert cfg.dedup_thresholds == {"dense": 0.95, "milco": 0.90}
    assert cfg.dedup_prefer_longer == c.DEDUP_PREFER_LONGER == 1.2
    assert cfg.clap_queries == c.CLAP_QUERIES == 4


def test_dedup_signal_names_agree_with_the_retriever():
    """The duplicate rule ignores a signal it has no threshold for, so a rename on
    either side silently disables the two-signal guard instead of failing."""
    from app.services.assistant.config import DEDUP_THRESHOLDS
    from app.services.retrieval.hybrid import DEFAULT_WEIGHTS

    assert set(DEDUP_THRESHOLDS) <= set(DEFAULT_WEIGHTS)


@pytest.mark.parametrize("intent", ["lyrics_search", "audio_search",
                                    "playlist", "general"])
def test_every_intent_has_a_caption_and_a_clarify_label(intent):
    from app.services.assistant.humanize import clarify_labels, human

    assert human("route", "ru", intent=intent)
    assert human("route", "en", intent=intent)
    assert clarify_labels("ru")[intent]
    assert clarify_labels("en")[intent]
