"""Fetch Genius.com song facts (description + line annotations) and
producer/label credits for the live indexing pipeline.

Mirrors ``song_facts_service.fetch_facts_for_songs``'s shape so it drops into
the same FACTS-stage ``asyncio.gather`` and progress-callback machinery.
Reuses ``genius_service`` for the actual fetch/parse; facts are stored under
the same ``get_song_facts_key`` slug that ``song_facts_service`` and
``track_chat_service.resolve_song_facts_list`` already use, so Genius facts
show up in the "chat about song" / "explain this line" prompts unchanged.

Producer/label can't be written to ``track_metadata`` here directly: at
FACTS-stage time (concurrent with the encode pipeline) no Qdrant point id
exists yet for these tracks — ids are minted (``uuid.uuid4()``) deep inside
``indexing_service``'s upsert step, which runs after this stage starts.
Instead ``fetch_genius_facts_for_songs`` returns the producer/label per song
keyed by ``"{artist} — {title}"`` (the same key ``IndexPipeline._resolve_track_ids``
uses), and the caller in ``library_service`` applies it once track ids are
resolved post-upsert — mirroring how ``_apply_upload_track_ids`` stamps
``pending_uploads.track_id`` from the same resolved mapping.

The label additionally goes straight into ``songs.label`` (slug-keyed, so no
track id is needed) via ``MetadataDB.set_song_label`` — the same row the
GLiNER2+LLM pipeline fills with producers/samples, which
``song_facts_service.apply_song_relations`` overlays onto every read path.

No skip-check for already-processed songs: this only runs against freshly
indexed collections, which never have prior Genius facts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, List, Optional, Tuple

from app.resources.metadata_db import MetadataDB
from app.services import genius_service
from app.services.song_facts_service import get_song_facts_key

logger = logging.getLogger(__name__)


async def fetch_genius_facts_for_song(
    artist: str,
    title: str,
    collection_name: str,
    delay: float = 0.3,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Fetch + store Genius facts for one song.

    Returns ``(found, producer, label)`` — ``found`` is True if any facts were
    stored; ``producer``/``label`` are the credits parsed from the page (the
    caller writes these to ``track_metadata`` once a track id is known).
    """
    song_slug = get_song_facts_key(artist, title)
    url = genius_service.build_genius_url(artist, title)
    try:
        parsed = await asyncio.to_thread(genius_service.fetch_song_data, url, delay)
    except Exception as e:
        logger.warning("[genius] failed for '%s — %s' (%s): %s", artist, title, url, e)
        return False, None, None

    facts = genius_service.build_song_facts(parsed)
    description_facts = [f for f, cat in facts if cat == "genius_description"]
    annotation_facts = [f for f, cat in facts if cat == "genius_annotation"]
    if description_facts:
        MetadataDB.add_song_facts_batch(
            song_slug, collection_name, description_facts,
            source="genius.com", category="genius_description",
        )
    if annotation_facts:
        MetadataDB.add_song_facts_batch(
            song_slug, collection_name, annotation_facts,
            source="genius.com", category="genius_annotation",
        )

    producer, label = genius_service.build_producer_label(parsed)
    if label:
        # Unlike producer (applied to track_metadata post-upsert, once ids
        # exist), the label lands in songs.label right here: the slug is known
        # already, and song_facts_service.apply_song_relations overlays it
        # onto every read path from that row.
        MetadataDB.set_song_label(song_slug, label, collection_name)
    logger.info(
        "[genius] ok: %s — %s (%d facts, producer=%r, label=%r)",
        artist, title, len(facts), producer, label,
    )
    return bool(facts), producer, label


async def fetch_genius_facts_for_songs(
    songs: List[Tuple[str, str]],
    collection_name: str,
    delay: float = 0.3,
    progress_callback: Optional[Callable[[int, int, str, bool], None]] = None,
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Fetch Genius facts for multiple songs sequentially with delay between requests.

    Args:
        songs: list of (artist, title) tuples.
        collection_name: collection scope.
        delay: seconds between requests.
        progress_callback: optional callback(current, total, label, found) after each song.
    Returns:
        dict of ``"{artist} — {title}"`` -> ``(producer, label)`` for songs
        where either credit was found. The caller resolves track ids from
        this same key and calls ``MetadataDB.update_track_producer_label``.
    """
    producer_label: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    total = len(songs)
    for idx, (artist, title) in enumerate(songs, 1):
        key = f"{artist} — {title}"
        found, producer, label = await fetch_genius_facts_for_song(
            artist, title, collection_name, delay=delay,
        )
        if producer or label:
            producer_label[key] = (producer, label)
        if progress_callback:
            progress_callback(idx, total, key, found)
        await asyncio.sleep(delay)
    return producer_label
