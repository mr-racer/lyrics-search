"""Artist Bio task — a biography written from the artist's Wikipedia article.

The previous version handed the whole job to an agent that chose its own
searches, and the result read like a Wikipedia lead paragraph — when it was not
worse. Of 836 production bios the one for Андрей Губин invented his death
("трагическая гибель в 2011 году"; he is alive) and several rendered performer
names in Cyrillic against an explicit instruction.

Now the article is found in Wikipedia's own index, fetched once, and asked five
questions whose answers are fused into one biography; four more questions
produce the facts shown beside it. The agent survives as the fallback for an
artist Wikipedia does not cover.

Cache key is (artist_slug, collection, lang). This module walks the collection;
``services/bio_v2`` does the work.
"""

from __future__ import annotations

import logging

from app.resources.metadata_db import MetadataDB
from app.services.llm_client import ask_llm
from app.services.proxy_config import get_proxy
from app.services import ai_indexing_service
from app.services.artist_split import (
    artist_slugs, display_name_for_slug,
)
from app.services.bio_v2 import pipeline as bio2
from app.services.llm_web_search import web_research_bio

logger = logging.getLogger(__name__)

_LANG_NAME = {"ru": "Russian", "en": "English"}


def _asker(job):
    """`ask(prompt, temperature) -> str` bound to this job's model."""
    async def ask(prompt: str, temperature: float = 0.3) -> str:
        return await ask_llm(prompt, temperature=temperature,
                             base_url=job.llm_base_url, model=job.llm_model)
    return ask


def _web_rows(query: str) -> list:
    """Open-web rows for the last-resort widen, in the shape bio_v2 expects.

    Kept behind a function so the pipeline never imports the search stack —
    and so a probe or a test can pass its own.
    """
    from app.services.assistant.config import AgentConfig
    from app.services.assistant.web_sources import SearchSources

    try:
        hits = SearchSources(AgentConfig()).web(query)
    except Exception:                               # noqa: BLE001
        return []
    return [{"url": h.url, "title": h.title, "snippet": h.snippet}
            for h in hits]


async def run(job, db_client, llm) -> None:
    """Iterate distinct artists in the collection; bio each one via web search."""
    n_done = 0
    n_failed = 0
    n_skipped = 0

    async def _process(artist_slug: str, artist_name: str) -> None:
        nonlocal n_done, n_failed, n_skipped

        # Skip if bio already cached for this artist+collection+lang.
        existing = MetadataDB.get_artist_bio(
            artist_slug, job.collection_name, job.lang,
        )
        if existing is not None:
            n_done += 1
            MetadataDB.update_ai_job(job_id=job.job_id, n_done=n_done)
            return

        audiodb_data = MetadataDB.get_artist_audiodb(artist_slug, job.collection_name)
        seed_bio = (audiodb_data or {}).get("audiodb_bio")

        logger.info(
            "[artist_bio] searching web for: %s (slug=%s, seed_bio=%s)",
            artist_name, artist_slug,
            "yes" if seed_bio else "no",
        )
        logger.info("[artist_bio] %s (slug=%s, seed_bio=%s)",
                    artist_name, artist_slug, "yes" if seed_bio else "no")
        bio, facets = "", {}
        try:
            result = await bio2.build(
                _asker(job), artist_name,
                lang_name=_LANG_NAME.get(job.lang, "Russian"),
                lang_code=job.lang, proxies=get_proxy(),
                web_search=_web_rows,
            )
            bio, facets = result.get("bio") or "", result.get("facets") or {}
            if result.get("error"):
                logger.info("[artist_bio] %s: %s", artist_name, result["error"])
        except Exception as e:                       # noqa: BLE001
            logger.warning("[artist_bio] wiki pipeline failed for %s: %s",
                           artist_name, e, exc_info=True)

        if not bio:
            # No article, or nothing cleared the chunk gate. The agent that used
            # to do the whole job is a reasonable last resort for exactly this
            # case — an artist Wikipedia does not cover.
            try:
                bio = await web_research_bio(
                    artist_name=artist_name, lang=job.lang,
                    base_url=job.llm_base_url, model_name=job.llm_model,
                    seed_bio=seed_bio,
                )
                if bio:
                    facets = {"source_kind": "web"}
            except Exception as e:                   # noqa: BLE001
                logger.warning("[artist_bio] web fallback failed for %s: %s",
                               artist_name, e)
                n_failed += 1
                MetadataDB.update_ai_job(job_id=job.job_id, n_failed=n_failed)
                return

        if not bio:
            logger.warning("[artist_bio] empty result for %s", artist_name)
            n_skipped += 1
            MetadataDB.update_ai_job(job_id=job.job_id, n_skipped=n_skipped)
            return

        MetadataDB.set_artist_bio(artist_slug, job.collection_name, job.lang,
                                  bio, facets=facets)
        n_done += 1
        MetadataDB.update_ai_job(job_id=job.job_id, n_done=n_done)

    # Incremental auto-run (append/upload): only the artists of the tracks
    # this run just indexed. A brand-new artist gets a web-researched bio; an
    # artist already in the library hits the bio-cache skip in _process and
    # costs nothing. A payload retrieve is enough — no collection-wide walk.
    if job.new_track_ids is not None:
        qdrant = db_client.qdrant
        seen_new_slugs: set[str] = set()
        points = qdrant.retrieve(
            collection_name=job.collection_name,
            ids=list(job.new_track_ids),
            # ``artists`` rides along because it is the ONLY payload field that
            # names a guest credited in the title — the raw ``artist`` tag does
            # not mention them at all.
            with_payload=["artist", "artists", "artist_slugs"],
            with_vectors=False,
        )
        for p in points:
            p = (p.payload or {}) if hasattr(p, "payload") else p
            raw = (p.get("artist") or "").strip()
            slugs = p.get("artist_slugs") or (artist_slugs(raw) if raw else [])
            for artist_slug in slugs:
                if not artist_slug or artist_slug in seen_new_slugs:
                    continue
                seen_new_slugs.add(artist_slug)
                artist_name = display_name_for_slug(
                    artist_slug, participants=p.get("artists"), raw=raw,
                )
                await _process(artist_slug, artist_name)
        logger.info(
            "[artist_bio] done (incremental, %d new artists): %d written, "
            "%d skipped, %d failed — collection=%s",
            len(seen_new_slugs), n_done, n_skipped, n_failed, job.collection_name,
        )
        return

    # SQLite mirror first: one indexed query for {slug, name} instead of
    # scrolling every Qdrant payload. Empty result (pre-backfill) falls back
    # to the payload scan below.
    mirror_rows = MetadataDB.get_distinct_artist_slugs_from_sqlite(job.collection_name)
    if mirror_rows:
        for row in mirror_rows:
            slug = row.get("slug")
            if not slug:
                continue
            # The mirror already resolves the name per slug; the guard keeps a
            # stale/legacy row from re-introducing a foreign name here.
            name = display_name_for_slug(
                slug, participants=[row.get("name")], raw=row.get("name"),
            )
            await _process(slug, name)
    else:
        qdrant = db_client.qdrant
        seen_artist_slugs: set[str] = set()
        offset = None
        while True:
            points, offset = qdrant.scroll(
                collection_name=job.collection_name,
                limit=64,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            for pt in points:
                p = pt.payload or {}

                # Prefer artist_slugs from payload (split at index time); fallback
                # to splitting the raw "artist" tag ourselves — uses artist_slugs()
                # which includes alias resolution, matching the indexing path.
                slugs = p.get("artist_slugs") or []
                if not slugs:
                    raw = (p.get("artist") or "").strip()
                    if raw:
                        slugs = artist_slugs(raw)

                for artist_slug in slugs:
                    if not artist_slug or artist_slug in seen_artist_slugs:
                        continue
                    seen_artist_slugs.add(artist_slug)

                    # Display name for the web search — taken from the aligned
                    # participant list, so a title-credited guest is researched
                    # as themselves and not as the tag's headline artist.
                    artist_name = display_name_for_slug(
                        artist_slug,
                        participants=p.get("artists"),
                        raw=p.get("artist"),
                    )

                    await _process(artist_slug, artist_name)

            if offset is None:
                break

    logger.info(
        "[artist_bio] done: %d written, %d skipped, %d failed — collection=%s",
        n_done, n_skipped, n_failed, job.collection_name,
    )


# Register at import time
ai_indexing_service.register_task("artist_bio", run)
