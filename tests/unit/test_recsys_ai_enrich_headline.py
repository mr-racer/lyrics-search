import pytest
from unittest.mock import AsyncMock, patch

from app.services import recsys_ai_service

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_enrich_profile_surfaces_headline():
    fake_profile = {
        "islands": [
            {"track_id": "t1", "weight": 2.0,
             "tracks": [{"track_id": "t1", "title": "A", "artist": "X", "cover_art_path": None}]},
        ],
        "axes": {"energy": {"z": 0.5, "level": "high"}},
        "n_signals": 10,
    }
    llm_json = {
        "portrait": "A concrete portrait.",
        "island_names": {"t1": "Ночной синтвейв"},
        "headline": "Поздний неон",
    }
    with patch.object(recsys_ai_service.stream_service, "long_term_profile", return_value=fake_profile), \
         patch.object(recsys_ai_service, "ask_llm", new=AsyncMock(return_value=llm_json)), \
         patch.object(recsys_ai_service.MetadataDB, "set_recsys_llm_text") as set_text:
        result = await recsys_ai_service.enrich_profile(
            qdrant_client=object(), collection_name="acct_test", lang="ru",
        )

    assert result["headline"] == "Поздний неон"
    assert result["portrait"] == "A concrete portrait."
    # headline must be persisted in the cached content blob
    stored_content = set_text.call_args.args[4]
    assert stored_content["headline"] == "Поздний неон"
