"""The ``facts`` intent — "расскажи про этот трек / про этого артиста".

The one genuinely new branch of the assistant. Design principle: on a 12b model
the LLM decides NOTHING except the wording. Five steps, four of them pure code:

1. **Resolve the subject** — ``catalog_search_service`` (entity mode) over the
   spans GLiNER already extracted. A thin margin between the top two candidates
   produces a ``disambiguate`` frame instead of a guess.
2. **Build a numbered grounding pack** — refined facts → raw facts, credits,
   gems, bio, AudioDB, lyrics. Zero LLM involvement. The ``[n]`` numbering is
   the whole anti-hallucination mechanism.
3. **One ``ask_llm(parse_json=True)`` call** returning
   ``{"answer": ..., "used": [n, …]}``. Not pydantic-ai, not tool calling — the
   repo has been burned twice by small models mangling tool-call syntax.
4. **Code verifies the citations.** Empty ``used``, out-of-range indices or an
   empty answer ⇒ the answer is thrown away and the pack is rendered verbatim.
   An ungrounded paragraph physically cannot reach the user.
5. **The web is a code decision, not a model decision.** A thin pack triggers
   ``smart_web_search`` before step 3 and its snippets join the pack numbered.
   "Should I search?" is exactly the judgement small models fail at.

``track_chat_service`` is deliberately left alone: inside the player it already
has the track in hand and works.
"""

from __future__ import annotations

import asyncio
import logging
import re

from app.services.assistant.humanize import human

logger = logging.getLogger(__name__)

# A pack thinner than this is treated as "we don't really know anything" and
# earns a web lookup before the LLM ever sees it.
MIN_PACK_ITEMS = 3
MAX_WEB_SEARCHES = 2
# Hard ceiling on what goes into the prompt — a 12b context filled with 60 facts
# produces worse answers than one filled with the best 18.
MAX_PACK_ITEMS = 18
MAX_FACT_CHARS = 400
MAX_LYRICS_CHARS = 700
MAX_RELATED_TRACKS = 8
# Below this score ratio against the top hit, a runner-up is not a real rival
# and we resolve silently instead of asking.
DISAMBIGUATE_RATIO = 0.75

_LANG_NAMES = {"ru": "Russian", "en": "English"}


def _lang_name(lang: str | None) -> str:
    return _LANG_NAMES.get((lang or "en").strip().lower(), "English")


_SYSTEM = """You answer a listener's question about a specific artist or song, using ONLY the numbered facts provided.

Write the answer in {lang_name}.

HARD RULES:
- Every statement you make must come from a numbered fact below. If the facts do not cover something, do not say it — silence is correct, invention is not.
- Never add dates, numbers, chart positions, collaborators or backstory that are not in the facts.
- Cite by listing in "used" every fact number you actually relied on. A fact you did not use must not appear there.
- If the facts are too thin to answer the question, say so plainly in one sentence and put the numbers of whatever facts you did mention in "used".
- Any artist, producer or song name must be copied EXACTLY as it appears in the facts — never translated, transliterated, localized, or grammatically declined, regardless of the answer language.

STYLE:
- Lead with the answer. No preamble, no "This artist is a fascinating figure…".
- 2-5 sentences for a normal question. Match length to the question; most answers are short.
- Sound like a well-read friend talking, not an encyclopedia entry. No bullet lists unless the facts genuinely split into separate threads.
- Say each thing once. No closing summary.

Output ONLY minified JSON, no prose, no fences:
{{"answer": "...", "used": [1, 3]}}"""


# ── Step 1: subject resolution ───────────────────────────────────────────────


def _resolve_subject_sync(qdrant, collection_name: str, query: str, limit: int = 6) -> list:
    from app.services import catalog_search_service

    try:
        return catalog_search_service.search_catalog(qdrant, collection_name, query, limit)
    except Exception:
        logger.warning("[assistant/facts] catalog search failed", exc_info=True)
        return []


def _subject_from_hit(hit: dict) -> dict:
    """Normalise a catalog hit into the subject shape the rest of this module uses."""
    kind = hit.get("type") or "song"
    if kind == "artist":
        return {
            "kind": "artist",
            "title": hit.get("artist") or "",
            "subtitle": None,
            "artist_slug": hit.get("artist_slug"),
            "artist": hit.get("artist") or "",
            "track_id": None,
            "image_path": hit.get("image") or hit.get("cover_art_path"),
        }
    if kind == "album":
        return {
            "kind": "album",
            "title": hit.get("album") or "",
            "subtitle": hit.get("artist"),
            "artist_slug": hit.get("artist_slug"),
            "artist": hit.get("artist") or "",
            "track_id": None,
            "image_path": hit.get("cover_art_path"),
        }
    return {
        "kind": "song",
        "title": hit.get("title") or "",
        "subtitle": hit.get("artist"),
        "artist_slug": None,
        "artist": hit.get("artist") or "",
        "track_id": hit.get("track_id"),
        "image_path": hit.get("cover_art_path"),
    }


def _subject_query(route, message: str) -> str:
    """What to look the subject up by: the extracted spans if GLiNER found any,
    otherwise the raw message (the catalog index tolerates noise better than a
    wrong-but-confident span would)."""
    parts = [p for p in (getattr(route, "song", None), getattr(route, "artist", None)) if p]
    return " ".join(parts) if parts else (message or "")


async def resolve_subject(qdrant, collection_name: str, *, route, message: str, slots,
                          subject_track_id=None, subject_artist_slug=None,
                          now_playing_track_id=None):
    """Return ``(subject, options)``.

    ``subject`` set and ``options`` empty → resolved.
    ``subject`` None and ``options`` non-empty → ask the user (disambiguate).
    Both empty → nothing in the library matches.
    """
    # The user already picked a card from a previous disambiguate frame.
    if subject_track_id or subject_artist_slug:
        subject = await _subject_from_ids(qdrant, collection_name,
                                          subject_track_id, subject_artist_slug)
        if subject:
            return subject, []

    query = _subject_query(route, message)
    hits = await asyncio.to_thread(_resolve_subject_sync, qdrant, collection_name, query)

    if not hits:
        # No entity in the message at all — fall back to what the player is on,
        # then to the last subject discussed. "расскажи про этот трек" needs no
        # entity extraction to work.
        fallback_id = now_playing_track_id or getattr(slots, "last_track_id", None)
        if fallback_id:
            subject = await _subject_from_ids(qdrant, collection_name, fallback_id, None)
            if subject:
                return subject, []
        return None, []

    top = hits[0]
    rivals = [
        h for h in hits[1:4]
        if float(h.get("score") or 0) >= float(top.get("score") or 0) * DISAMBIGUATE_RATIO
    ]
    if rivals:
        return None, [_subject_from_hit(h) for h in [top, *rivals]]
    return _subject_from_hit(top), []


async def _subject_from_ids(qdrant, collection_name: str, track_id, artist_slug):
    """Build a subject straight from an id the client supplied."""
    if artist_slug:
        from app.resources.metadata_db import MetadataDB

        def _load():
            row = MetadataDB.get_artist_audiodb(artist_slug, collection_name) or {}
            conn = MetadataDB.get()
            name_row = conn.execute(
                "SELECT name FROM artists WHERE slug = ?", (artist_slug,),
            ).fetchone()
            return row, (name_row[0] if name_row else None)

        try:
            row, name = await asyncio.to_thread(_load)
        except Exception:
            logger.warning("[assistant/facts] artist lookup failed", exc_info=True)
            row, name = {}, None
        title = name or artist_slug.replace("-", " ").title()
        return {
            "kind": "artist", "title": title, "subtitle": None,
            "artist_slug": artist_slug, "artist": title, "track_id": None,
            "image_path": row.get("thumb_path") or row.get("cutout_path"),
        }

    if not track_id:
        return None
    payload = await asyncio.to_thread(_track_payload, qdrant, collection_name, track_id)
    if not payload:
        return None
    return {
        "kind": "song",
        "title": payload.get("title") or "",
        "subtitle": payload.get("artist"),
        "artist_slug": payload.get("primary_artist_slug"),
        "artist": payload.get("artist") or "",
        "track_id": track_id,
        "image_path": payload.get("cover_art_path"),
    }


def _track_payload(qdrant, collection_name: str, track_id: str) -> dict:
    """Full payload (lyrics included) for one track. Blocking — use to_thread."""
    try:
        pts = qdrant.retrieve(collection_name=collection_name, ids=[track_id],
                              with_payload=True, with_vectors=False)
    except Exception:
        logger.warning("[assistant/facts] qdrant retrieve failed for %s", track_id,
                       exc_info=True)
        return {}
    return (pts[0].payload or {}) if pts else {}


# ── Step 2: the numbered grounding pack ──────────────────────────────────────


def _clean(text: str, limit: int = MAX_FACT_CHARS) -> str:
    text = " ".join((text or "").split())
    return text[:limit].rstrip() + "…" if len(text) > limit else text


def _build_song_pack(subject: dict, collection_name: str, lang: str, payload: dict) -> list[dict]:
    """Facts, credits and gems for one song. Pure SQLite + the Qdrant payload."""
    from app.resources.metadata_db import MetadataDB
    from app.services.song_facts_service import get_song_facts_key

    items: list[dict] = []
    artist = subject.get("artist") or payload.get("artist") or ""
    title = subject.get("title") or payload.get("title") or ""
    slug = get_song_facts_key(artist, title) if (artist and title) else ""

    # Refined facts first — the LLM-cleaned, categorised versions. An explicit
    # empty list means "AI-indexed, judged nothing interesting", so it must NOT
    # fall through to the raw facts; only a missing row (None) does.
    facts: list[str] = []
    if slug:
        refined = MetadataDB.get_refined_facts(
            scope="song", scope_key=slug, collection_name=collection_name, lang=lang,
        )
        facts = refined if refined is not None else MetadataDB.get_song_facts(slug, collection_name)
    for f in facts:
        if f and f.strip():
            items.append({"text": _clean(f), "source": "facts"})

    if slug:
        rel = (MetadataDB.get_song_relations_bulk([slug]) or {}).get(slug) or {}
        if rel.get("producer"):
            items.append({"text": f"Продюсер: {rel['producer']}" if lang == "ru"
                                  else f"Produced by {rel['producer']}", "source": "credits"})
        if rel.get("label"):
            items.append({"text": f"Лейбл: {rel['label']}" if lang == "ru"
                                  else f"Label: {rel['label']}", "source": "credits"})
        for s in (rel.get("samples") or [])[:4]:
            items.append({"text": (f"Содержит сэмпл: {s}" if lang == "ru"
                                   else f"Contains a sample of {s}"), "source": "credits"})
        for s in (rel.get("sampled_by") or [])[:4]:
            items.append({"text": (f"Был засэмплирован в: {s}" if lang == "ru"
                                   else f"Sampled by {s}"), "source": "credits"})

    if subject.get("track_id"):
        for gem in MetadataDB.get_track_gems(subject["track_id"], collection_name)[:5]:
            display = gem.get("display") or gem.get("canonical")
            quote = gem.get("quote")
            if not display:
                continue
            text = (f"В тексте упоминается {display}" if lang == "ru"
                    else f"The lyrics reference {display}")
            if quote:
                text += f" — «{quote}»"
            items.append({"text": _clean(text), "source": "gems"})

    # Release metadata the facts rarely repeat but the user often asks for.
    bits = []
    if payload.get("album"):
        bits.append(f"album {payload['album']}")
    if payload.get("year"):
        bits.append(f"released {payload['year']}")
    if payload.get("genre"):
        bits.append(f"genre {payload['genre']}")
    if bits:
        items.append({"text": f"{title} — {artist}: " + ", ".join(bits), "source": "catalog"})

    lyrics = (payload.get("lyrics") or "").strip()
    if lyrics:
        items.append({"text": _clean(lyrics, MAX_LYRICS_CHARS), "source": "lyrics"})

    return items


def _build_artist_pack(subject: dict, collection_name: str, lang: str) -> list[dict]:
    """Bio, facts and AudioDB metadata for one artist."""
    from app.resources.metadata_db import MetadataDB

    items: list[dict] = []
    slug = subject.get("artist_slug")
    if not slug:
        return items

    bio = MetadataDB.get_artist_bio(slug, collection_name, lang)
    if bio and bio.strip():
        items.append({"text": _clean(bio, 900), "source": "bio"})

    refined = MetadataDB.get_refined_facts(
        scope="artist", scope_key=slug, collection_name=collection_name, lang=lang,
    )
    facts = refined if refined is not None else MetadataDB.get_artist_facts(slug, collection_name)
    for f in facts:
        if f and f.strip():
            items.append({"text": _clean(f), "source": "facts"})

    row = MetadataDB.get_artist_audiodb(slug, collection_name) or {}
    bits = []
    if row.get("country"):
        bits.append(f"from {row['country']}")
    if row.get("label"):
        bits.append(f"label {row['label']}")
    if row.get("mood"):
        bits.append(f"mood {row['mood']}")
    if bits:
        items.append({"text": f"{subject.get('title')}: " + ", ".join(bits),
                      "source": "catalog"})
    if row.get("audiodb_bio") and not bio:
        items.append({"text": _clean(row["audiodb_bio"], 900), "source": "bio"})

    return items


def _related_tracks_sync(qdrant, collection_name: str, subject: dict) -> list:
    """Library tracks to show under the answer — the artist's own, or the song itself."""
    from app.services import catalog_search_service

    query = subject.get("artist") or subject.get("title") or ""
    if not query:
        return []
    try:
        return catalog_search_service.search_catalog_tracks(
            qdrant, collection_name, query, MAX_RELATED_TRACKS,
        )
    except Exception:
        logger.warning("[assistant/facts] related tracks failed", exc_info=True)
        return []


# ── Step 5: web fill-in (a CODE decision) ────────────────────────────────────


def _web_query(subject: dict, message: str) -> str:
    """A web query built from the resolved subject, not from the raw message —
    the message may be in Russian and phrased conversationally."""
    title = subject.get("title") or ""
    artist = subject.get("artist") or ""
    if subject.get("kind") == "artist":
        return f"{title} musician biography"
    if subject.get("kind") == "album":
        return f"{title} {artist} album"
    return f"{artist} {title} song meaning background"


def _web_search_sync(query: str) -> str:
    from app.services.llm_web_search import smart_web_search

    try:
        return smart_web_search(query, False, 4) or ""
    except Exception:
        logger.warning("[assistant/facts] web search failed", exc_info=True)
        return ""


def _snippets_from_web(raw: str, limit: int = 5) -> list[dict]:
    """Split the web-search blob into individually numberable pack items."""
    out: list[dict] = []
    for chunk in re.split(r"\n{2,}", raw or ""):
        text = _clean(chunk)
        if len(text) >= 60:
            out.append({"text": text, "source": "web"})
        if len(out) >= limit:
            break
    return out


# ── Step 3+4: the single LLM call and its verification ───────────────────────


def _render_pack(items: list[dict], lang: str) -> str:
    return "\n".join(f"[{i}] {it['text']}" for i, it in enumerate(items, 1))


def _deterministic_answer(subject: dict, items: list[dict], lang: str) -> str:
    """The fallback served whenever the LLM answer fails verification.

    Not an error message — a real, useful reply built only from stored facts.
    """
    ru = (lang or "en").startswith("ru")
    name = subject.get("title") or ""
    if not items:
        return (f"Про «{name}» у меня пока нет достоверных сведений."
                if ru else f"I don't have reliable information about “{name}” yet.")
    head = (f"Вот что известно про «{name}»:" if ru
            else f"Here's what is known about “{name}”:")
    bullets = "\n".join(f"- {it['text']}" for it in items[:8])
    return f"{head}\n{bullets}"


def _verify(raw: object, n_items: int) -> tuple[str, list[int]] | None:
    """Accept the LLM answer only if it is non-empty and cites real fact numbers.

    Returns ``(answer, used)`` or None — None means "throw it away, render the
    pack instead". This is the gate that makes an ungrounded paragraph
    impossible, no matter what the model produced.
    """
    if not isinstance(raw, dict):
        return None
    answer = str(raw.get("answer") or "").strip()
    if not answer:
        return None
    used_raw = raw.get("used")
    if not isinstance(used_raw, list) or not used_raw:
        return None
    used: list[int] = []
    for value in used_raw:
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= n_items and n not in used:
            used.append(n)
    if not used:
        return None
    return answer, used


# ── Orchestration ────────────────────────────────────────────────────────────


async def run(*, qdrant, collection_name: str, search_service, message: str, route, slots,
              lang: str = "en", subject_track_id=None, subject_artist_slug=None,
              now_playing_track_id=None, llm_base_url=None, llm_model=None, emit=None):
    """Run the facts branch. Returns ``(payload_dict | None, options list)``.

    ``payload_dict`` matches :class:`~app.domain.models.AssistantFactsPayload`.
    A non-empty ``options`` list means the caller should emit ``disambiguate``.
    """
    from app.services.llm_client import ask_llm

    async def _say(stage: str, **kw) -> None:
        if emit is not None:
            await emit({"type": "status", "stage": stage,
                        "human": human(stage, lang, **kw), **kw})

    await _say("resolving")
    subject, options = await resolve_subject(
        qdrant, collection_name, route=route, message=message, slots=slots,
        subject_track_id=subject_track_id, subject_artist_slug=subject_artist_slug,
        now_playing_track_id=now_playing_track_id,
    )
    if options:
        return None, options
    if subject is None:
        return None, []

    await _say("resolved", subject=subject.get("title") or "")

    # ── pack ──
    payload: dict = {}
    if subject["kind"] == "song" and subject.get("track_id"):
        payload = await asyncio.to_thread(
            _track_payload, qdrant, collection_name, subject["track_id"],
        )

    def _pack_sync() -> list[dict]:
        if subject["kind"] == "artist":
            return _build_artist_pack(subject, collection_name, lang)
        items = _build_song_pack(subject, collection_name, lang, payload)
        # An album subject also gets its artist's facts — that's what the user
        # is usually after when naming a record.
        if subject["kind"] == "album" and subject.get("artist_slug"):
            items += _build_artist_pack(subject, collection_name, lang)
        return items

    try:
        items = await asyncio.to_thread(_pack_sync)
    except Exception:
        logger.exception("[assistant/facts] pack build failed")
        items = []
    await _say("collecting", found=len(items))

    # ── web fill-in: code decides, not the model ──
    web_used = False
    if len(items) < MIN_PACK_ITEMS:
        query = _web_query(subject, message)
        for _ in range(min(1, MAX_WEB_SEARCHES)):
            await _say("web_search", query=query)
            raw_web = await asyncio.to_thread(_web_search_sync, query)
            snippets = _snippets_from_web(raw_web)
            if snippets:
                items += snippets
                web_used = True
            break

    items = items[:MAX_PACK_ITEMS]

    related = await asyncio.to_thread(_related_tracks_sync, qdrant, collection_name, subject)

    if not items:
        return {
            "subject_kind": subject["kind"],
            "subject_title": subject.get("title") or "",
            "subject_subtitle": subject.get("subtitle"),
            "artist_slug": subject.get("artist_slug"),
            "track_id": subject.get("track_id"),
            "image_path": subject.get("image_path"),
            "answer": _deterministic_answer(subject, [], lang),
            "grounded": False,
            "web_search_used": web_used,
            "items": [],
            "related_tracks": _as_track_models(related),
        }, []

    # ── one LLM call, then code verifies its citations ──
    await _say("thinking", step="answer")
    subject_line = f"{subject['kind'].upper()}: {subject.get('title')}"
    if subject.get("subtitle"):
        subject_line += f" — {subject['subtitle']}"
    user_prompt = (
        f"{subject_line}\n\nQUESTION: {message}\n\nFACTS:\n{_render_pack(items, lang)}"
    )
    verified = None
    try:
        raw = await ask_llm(
            user_prompt,
            system_prompt=_SYSTEM.format(lang_name=_lang_name(lang)),
            parse_json=True, temperature=0.3,
            base_url=llm_base_url, model=llm_model,
            extra_body={"enable_thinking": False},
        )
        verified = _verify(raw, len(items))
        if verified is None:
            logger.info("[assistant/facts] answer rejected by citation check — "
                        "serving the deterministic fact rendering instead")
    except Exception:
        logger.exception("[assistant/facts] LLM call failed")

    if verified is not None:
        answer, used = verified
        grounded = True
    else:
        answer, used, grounded = _deterministic_answer(subject, items, lang), [], False

    used_set = set(used)
    return {
        "subject_kind": subject["kind"],
        "subject_title": subject.get("title") or "",
        "subject_subtitle": subject.get("subtitle"),
        "artist_slug": subject.get("artist_slug"),
        "track_id": subject.get("track_id"),
        "image_path": subject.get("image_path"),
        "answer": answer,
        "grounded": grounded,
        "web_search_used": web_used,
        "items": [
            {"n": i, "text": it["text"], "source": it["source"], "used": i in used_set}
            for i, it in enumerate(items, 1)
        ],
        "related_tracks": _as_track_models(related),
    }, []


def _as_track_models(tracks: list) -> list:
    """Catalog track dicts → ``TrackMetadata`` models for the response.

    Mirrors ``_build_ai_playlist_response``: the catalog dicts carry Qdrant
    payload fields only, while producer/label/samples live in SQLite — hence the
    same ``apply_song_relations`` overlay, without which the player shows no
    producers chevron and no samples pill for these tracks.
    """
    from app.domain.models import TrackMetadata
    from app.services._payload_coerce import coerce_float, coerce_year
    from app.services.artist_split import artist_refs_for_track, display_title_for_track
    from app.services.song_facts_service import apply_song_relations

    out = []
    for t in tracks or []:
        try:
            out.append(TrackMetadata(
                track_id=t["track_id"],
                title=t.get("title") or "—",
                title_display=display_title_for_track(t),
                artist=t.get("artist") or "—",
                album=t.get("album"),
                year=coerce_year(t.get("year")),
                genre=t.get("genre"),
                duration_sec=coerce_float(t.get("duration")) or 0.0,
                file_path=t.get("file_path") or "",
                cover_art_path=t.get("cover_art_path"),
                artist_refs=artist_refs_for_track(t),
            ))
        except Exception:
            continue
    apply_song_relations(out)
    return out
