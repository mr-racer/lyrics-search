"""A music assistant that reads the web and the library, and makes no decisions
it can be talked out of.

    from lab.agent import AgentConfig, Assistant

    cfg = AgentConfig(
        searxng_url="http://192.168.0.168:8088",
        llm_base_url="http://192.168.0.168:8082/v1",
        llm_model="gemma-4-12b-it-qat-q4_0",
        db_path=r"C:\\dumps\\metadata.db",
    )
    agent = Assistant(cfg, verbose=True)

    result = await agent.run("Почему Эминем взял себе такой псевдоним?")
    print(result.answer)

    playlist = await agent.run("Песни из Test Drive Unlimited 2")
    for t in playlist.tracks:
        print(t.artist, "—", t.title, t.match, t.weight, t.reason)

    print(agent.sink.summary())      # every stage, in order

The shape of it:

* ``planner`` reads the request once and hands over a plan every field of which
  code has checked.
* ``sources`` + ``fetch`` gather pages: hosts pinned for Wikipedia / Apple /
  Fandom, and the open web reranked by cross-encoder before anything is
  downloaded.
* ``chunking`` + ``retrieval`` turn pages into the ten passages that matter.
* ``tables`` and ``extraction`` pull track titles — structure first, the model
  only where there is no structure.
* ``catalog`` decides what exists: a title the model invented matches nothing
  and disappears.
* ``pipeline`` runs the two branches and owns every stop/continue decision.

``lab.agent.retrieval`` is self-contained and imports nothing from the rest of
the package — it is the part meant to move into ``app/`` first.
"""

from lab.agent.catalog import LibraryCatalog
from lab.agent.config import AgentConfig
from lab.agent.events import EventSink
from lab.agent.models import (Chunk, ClarifyRequest, Evidence, Filters,
                              GeneralResult, Page, Plan, PlaylistResult,
                              ResolvedTrack, SearchHit, TrackRef)
from lab.agent.pipeline import Assistant
from lab.agent.retrieval import HybridRetriever, ModelHub
from lab.agent.retrieval.facts import FactsRetriever, SqliteFactSource

__all__ = [
    "AgentConfig", "Assistant", "EventSink", "LibraryCatalog",
    "HybridRetriever", "ModelHub", "FactsRetriever", "SqliteFactSource",
    "Chunk", "ClarifyRequest", "Evidence", "Filters", "GeneralResult", "Page",
    "Plan", "PlaylistResult", "ResolvedTrack", "SearchHit", "TrackRef",
]
