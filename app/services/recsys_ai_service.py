"""LLM layer for the Recommend tab (AI mode).

Two features:
- profile enrichment — one LLM call writes a listener portrait AND names the
  taste islands; cached in ``recsys_llm_texts`` keyed by a hash of the island
  members, so the cache self-invalidates when taste drifts;
- prompt-to-playlist — a deterministic plan→execute→select pipeline (more
  reliable on small local models than free-form tool calling): the LLM first
  emits a JSON plan of 1–3 search actions, the backend executes them against
  real services, then a second LLM call curates the final ordered playlist
  with per-track reasons.
"""

from __future__ import annotations

import hashlib
import json
import logging

from app.resources.metadata_db import MetadataDB
from app.services import stream_service
from app.services.llm_client import ask_llm

logger = logging.getLogger(__name__)

PROFILE_ENRICH_KIND = "profile_enrich"

_LANG_NAMES = {"en": "English", "ru": "Russian"}

# Prompt-to-playlist guardrails.
ALLOWED_TOOLS = ("clap_search", "library_search", "similar_tracks")
MAX_PLAN_ACTIONS = 3
MAX_ACTION_LIMIT = 25
MAX_SELECT_CANDIDATES = 60


def _lang_name(lang: str) -> str:
    return _LANG_NAMES.get(lang, _LANG_NAMES["en"])


# ── Profile enrichment: portrait + island names ─────────────────────────────

_ENRICH_SYSTEM = """You write a short music-taste portrait for a music player and name the listener's taste clusters.

Language for ALL output strings: {lang_name}.

Input: the listener's taste islands (clusters of tracks they demonstrably love, with weights) and their sound-axis profile (z-scores: energy, vocal_lead, spacious, experimental, brightness, acousticness).

Write:
1. "portrait" — 2-4 sentences about what this person actually loves in music. Be concrete: name sounds, moods, scenes, contrasts between their islands. NEVER use filler like "eclectic taste", "diverse listener", "music lover", "wide range". If the data shows a contradiction (e.g. loves both harsh noise and soft folk) — say it, contradictions are the interesting part.
2. "island_names" — a punchy 1-4 word name for EACH island, in the listener's language. Name the SOUND, not the artist (e.g. "Ночной синтвейв", not "Группа X"). Use the member tracks as evidence.

Output ONLY minified JSON, no prose, no fences:
{{"portrait": "...", "island_names": {{"<island_track_id>": "...", ...}}}}"""


def profile_source_hash(islands: list[dict]) -> str:
    """Fingerprint of the profile inputs the LLM texts were generated from.

    Order-insensitive over islands and members: a reshuffle of the same tracks
    must NOT invalidate the cache — only actual membership changes do.
    """
    membership = sorted(
        sorted(m["track_id"] for m in isl.get("tracks", [])) for isl in islands
    )
    blob = json.dumps(membership)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def get_cached_enrichment(collection_name: str, lang: str, islands: list[dict]) -> dict | None:
    """Cached {portrait, island_names} if it matches the CURRENT profile."""
    return MetadataDB.get_recsys_llm_text(
        collection_name, PROFILE_ENRICH_KIND, lang,
        source_hash=profile_source_hash(islands),
    )


def _enrich_user_prompt(profile: dict) -> str:
    lines = ["TASTE ISLANDS (strongest first):"]
    for i, isl in enumerate(profile["islands"], 1):
        members = "; ".join(f"{m['artist']} — {m['title']}" for m in isl["tracks"])
        lines.append(f"{i}. id={isl['track_id']} weight={isl['weight']}: {members}")
    axes = profile.get("axes")
    if axes:
        lines.append("\nSOUND AXES (z-scores, + = first pole):")
        for name, ax in axes.items():
            lines.append(f"- {name}: {ax['z']} ({ax['level']})")
    lines.append(f"\nSignal volume: {profile.get('n_signals', 0)} events/reactions.")
    return "\n".join(lines)


async def enrich_profile(
    *,
    qdrant_client,
    collection_name: str,
    lang: str = "en",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict:
    """Generate + cache the portrait and island names for the current profile.

    Returns ``{"portrait", "island_names", "islands"}`` (islands included so the
    frontend can re-render immediately without refetching the profile).
    """
    profile = stream_service.long_term_profile(
        qdrant_client=qdrant_client, collection_name=collection_name,
    )
    if not profile["islands"]:
        return {"portrait": None, "island_names": {}, "islands": []}

    raw = await ask_llm(
        _enrich_user_prompt(profile),
        system_prompt=_ENRICH_SYSTEM.format(lang_name=_lang_name(lang)),
        base_url=llm_base_url,
        model=llm_model,
        temperature=0.6,
        parse_json=True,
    )
    portrait = (raw.get("portrait") or "").strip() or None
    valid_ids = {i["track_id"] for i in profile["islands"]}
    island_names = {
        k: str(v).strip()
        for k, v in (raw.get("island_names") or {}).items()
        if k in valid_ids and str(v).strip()
    }

    content = {"portrait": portrait, "island_names": island_names}
    MetadataDB.set_recsys_llm_text(
        collection_name, PROFILE_ENRICH_KIND, lang,
        profile_source_hash(profile["islands"]), content,
    )
    return {**content, "islands": profile["islands"]}


# ── Prompt-to-playlist: plan → execute → select ─────────────────────────────

_PLAN_SYSTEM = """You translate a music listener's wish into search actions over their indexed personal library.

Available tools:
- "clap_search" — search by SOUND. The query must be an English description of sound/mood/instrumentation, written like an audio caption (e.g. "a calm late-night jazz track with soft female vocals and brushed drums"). Use for any wish about vibe, genre, energy, instruments.
- "library_search" — search by LYRICS THEME and metadata. English query about what the songs are ABOUT (e.g. "songs about heartbreak and moving on"). Use for lyrical/thematic wishes.
- "similar_tracks" — tracks that SOUND like one specific track the user explicitly named. Query = "Artist Title" exactly as the user named it. Use ONLY when a concrete song is named.

Rules:
- 1 to {max_actions} actions. One focused action beats three vague ones.
- A wish can mix kinds: "энергичный рок про любовь" → clap_search (energetic rock sound) + library_search (love lyrics).
- "title": a short playlist name in {lang_name} that captures the wish.

Output ONLY minified JSON, no prose, no fences:
{{"title": "...", "actions": [{{"tool": "clap_search", "query": "...", "limit": 20}}]}}"""

_SELECT_SYSTEM = """You curate the final playlist from candidate tracks found in the listener's library.

The listener's wish and a numbered candidate list follow. Pick up to {limit} tracks in a good listening order (flow matters: don't whiplash between extremes unless the wish asks for it).

For each pick give a SHORT reason in {lang_name} (max 12 words) tied to the wish — concrete sound or lyrics, never filler like "great track" or "fits the mood".

Output ONLY minified JSON, no prose, no fences:
{{"picks": [{{"n": 3, "reason": "..."}}, ...]}}"""


def _validate_plan(raw: dict) -> tuple[str, list[dict]]:
    title = str(raw.get("title") or "").strip() or "Playlist"
    actions = []
    # Filter FIRST, cap after — an invalid action must not consume a slot
    # that a valid one later in the list deserves.
    for a in (raw.get("actions") or []):
        if len(actions) >= MAX_PLAN_ACTIONS:
            break
        tool = str(a.get("tool") or "").strip()
        query = str(a.get("query") or "").strip()
        if tool not in ALLOWED_TOOLS or not query:
            continue
        try:
            limit = max(1, min(MAX_ACTION_LIMIT, int(a.get("limit") or 20)))
        except (TypeError, ValueError):
            limit = 20
        actions.append({"tool": tool, "query": query, "limit": limit})
    return title, actions


async def _execute_action(
    action: dict, *, search_service, qdrant_client, collection_name: str,
) -> list[dict]:
    """Run one plan action → list of candidate dicts {track(TrackMetadata-ish), tool}."""
    tool, query, limit = action["tool"], action["query"], action["limit"]

    if tool == "similar_tracks":
        # Resolve the named track first (text mode: bm25 over metadata+lyrics).
        seed_hits = await search_service.search(
            query, mode="text", limit=1, collection_name=collection_name,
        )
        if not seed_hits:
            return []
        seed_id = seed_hits[0].track.track_id
        result = stream_service.similar_tracks(
            qdrant_client=qdrant_client, collection_name=collection_name,
            seed_track_id=seed_id, limit=limit,
        )
        out = []
        for c in result["tracks"]:
            p = c.payload or {}
            out.append({
                "track_id": c.track_id,
                "title": p.get("title") or "—",
                "artist": p.get("artist") or "—",
                "album": p.get("album"),
                "year": p.get("year"),
                "genre": p.get("genre"),
                "duration": p.get("duration"),
                "file_path": p.get("file_path") or "",
                "cover_art_path": p.get("cover_art_path"),
                "tool": tool,
            })
        return out

    mode = "audio" if tool == "clap_search" else "hybrid"
    hits = await search_service.search(
        query, mode=mode, limit=limit, collection_name=collection_name,
    )
    out = []
    for h in hits:
        t = h.track
        out.append({
            "track_id": t.track_id,
            "title": t.title,
            "artist": t.artist,
            "album": t.album,
            "year": t.year,
            "genre": t.genre,
            "duration": t.duration_sec,
            "file_path": t.file_path,
            "cover_art_path": t.cover_art_path,
            "tool": tool,
        })
    return out


async def ai_playlist(
    *,
    search_service,
    qdrant_client,
    collection_name: str,
    prompt: str,
    lang: str = "en",
    limit: int = 15,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict:
    """One wish → curated playlist. Returns {title, steps, tracks}.

    ``steps`` narrate what the pipeline actually did (tool, query, found) —
    the frontend animates them. Selection failures degrade gracefully to the
    un-curated candidate order instead of erroring the whole request.
    """
    llm_kw = {"base_url": llm_base_url, "model": llm_model}

    # 1. Plan.
    raw_plan = await ask_llm(
        prompt,
        system_prompt=_PLAN_SYSTEM.format(
            lang_name=_lang_name(lang), max_actions=MAX_PLAN_ACTIONS,
        ),
        temperature=0.3, parse_json=True, **llm_kw,
    )
    title, actions = _validate_plan(raw_plan)
    if not actions:
        # Degenerate plan — fall back to treating the whole wish as a sound query.
        actions = [{"tool": "clap_search", "query": prompt, "limit": 20}]

    # 2. Execute (dedup across actions, first tool wins the attribution).
    steps = []
    candidates: dict[str, dict] = {}
    for action in actions:
        try:
            found = await _execute_action(
                action, search_service=search_service,
                qdrant_client=qdrant_client, collection_name=collection_name,
            )
        except Exception:
            logger.exception("[ai-playlist] action failed: %s", action)
            found = []
        for c in found:
            candidates.setdefault(c["track_id"], c)
        steps.append({"tool": action["tool"], "query": action["query"], "found": len(found)})

    ordered = list(candidates.values())[:MAX_SELECT_CANDIDATES]
    if not ordered:
        return {"title": title, "steps": steps, "tracks": []}

    # 3. Select + justify (index-based: small ints survive small models).
    numbered = "\n".join(
        f"{i}. {c['artist']} — {c['title']}"
        + (f" [{c['genre']}]" if c.get("genre") else "")
        + f" (via {c['tool']})"
        for i, c in enumerate(ordered, 1)
    )
    picked: list[dict] = []
    try:
        raw_sel = await ask_llm(
            f"WISH: {prompt}\n\nCANDIDATES:\n{numbered}",
            system_prompt=_SELECT_SYSTEM.format(
                limit=limit, lang_name=_lang_name(lang),
            ),
            temperature=0.4, parse_json=True, **llm_kw,
        )
        seen: set[int] = set()
        for p in (raw_sel.get("picks") or [])[:limit]:
            try:
                n = int(p.get("n"))
            except (TypeError, ValueError):
                continue
            if not (1 <= n <= len(ordered)) or n in seen:
                continue
            seen.add(n)
            picked.append({**ordered[n - 1], "reason": str(p.get("reason") or "").strip() or None})
    except Exception:
        logger.exception("[ai-playlist] selection failed — falling back to raw candidate order")

    if not picked:
        picked = [{**c, "reason": None} for c in ordered[:limit]]

    return {"title": title, "steps": steps, "tracks": picked}
