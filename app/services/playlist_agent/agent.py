"""Playlist-builder pydantic-ai agent.

Handles two prompt shapes with one instruction set + three tools:
  * "artist hits" — collect an artist's best-known songs;
  * "film/game soundtrack" — collect a title's soundtrack.

Tools:
  * ``fetch_filters(artist_query)`` — map how the user named an artist to the
    real names in the library (transliteration-aware difflib over the
    collection's artist values), so "Канье" resolves to "Kanye West" before
    searching.
  * ``web_search(query)`` — SearXNG/DDG web search (capped at
    ``_MAX_WEB_SEARCHES`` per run via a closure counter).
  * ``get_songs(items)`` — resolve proposed titles against the library
    (exact → fuzzy), returning per-item match info so the model can react.

The agent output is a :class:`PlaylistDraft`; ``get_songs`` also records every
resolved track into the shared ``state`` dict so the caller (the recsys
``web_hits`` delegation) can build the track list (and drop any track_id the
model might hallucinate) without trusting the LLM to echo IDs perfectly.

Progress: every tool emits a structured event through ``state["on_status"]``
(``{"stage": ..., "query"/"count"/"found"/...}``) so the streaming route can
show the user a human-readable chain of what the agent is doing — which tool,
with which query, and what it found.

Uses ``instructions=`` (not ``system_prompt=``) per the repo convention: a
``system_prompt`` is dropped once ``message_history`` is passed.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import (
    IncompleteToolCall,
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.usage import UsageLimits

from app.domain.models import PlaylistDraft
from app.services.agents import _create_pydantic_model
from app.services.llm_web_search import smart_web_search
from app.services.name_match import score_names
from app.services.playlist_agent.resolver import resolve_songs, resolve_tracklines

logger = logging.getLogger(__name__)

_MAX_WEB_SEARCHES = 6
_MAX_LLM_REQUESTS = 15

_INSTRUCTIONS = """You build a music playlist from the user's request by finding real songs and matching them against THIS user's music library. Common request shapes:

1) ARTIST SONGS ("собери хиты X", "best of X", "Kanye hits", "свежие песни X"):
   - FIRST call fetch_filters with the artist name as the user wrote it, to learn how the artist is actually spelled in the library (e.g. "Канье" -> "Kanye West"; it transliterates across alphabets). Use the returned real name from then on.
   - If fetch_filters returned nothing, the artist may still be in the library under the canonical name — do NOT give up and do NOT stop: continue with the canonical name you know (or confirm it via web_search).
   - Then ALWAYS call web_search 1-2 times, phrased for what was asked: hits → "<artist> greatest hits / most popular songs"; recent releases → "<artist> new songs <current year>" / "<artist> latest album tracklist".
   - Then call get_songs with those {title, artist} pairs.

2) FILM / GAME SOUNDTRACK ("саундтрек к <фильм/игра>", "песни из <игра>"):
   - FIRST call web_search once (snippets) to confirm the EXACT official title of the film/game in the right language (titles are often localized or abbreviated).
   - Then call web_search with fetch_content=true for the tracklist. Phrase it "<exact title> soundtrack tracklist" or "<exact title> licensed music songs" — do NOT use the bare abbreviation "OST", it surfaces game-rip playlists and score cues instead of the licensed songs.
   - The user wants songs FROM the title — the tracklist itself, NOT a popularity ranking. Never phrase these searches as "famous / popular / best known songs from X": that surfaces listicles instead of the soundtrack. Take the tracklist as-is; get_songs decides what the library has.
   - Games with radio stations (GTA, Watch Dogs…) keep their song lists on fan wikis, one page per station or one combined page — "<game> radio stations song list" reaches them.
   - Then call get_songs with the song titles + artists extracted from the page text.

3) THEME / ERA / CHARTS ("популярные клубные песни 90-х", "летние хиты 2010-х", "top eurodance"):
   - A genre/era wish with NO country or language named means the INTERNATIONALLY popular songs of that era — e.g. "хиты поп музыки 00-х" → global 2000s pop (Britney Spears, Beyoncé, Rihanna, Lady Gaga, Justin Timberlake…), NOT your own country's domestic pop. Never invent a nationality the user did not give, and never spend searches re-asking the same wish in another language.
   - Search in ENGLISH for the ranked list ("best pop songs of the 2000s", "top eurodance songs of the 90s") — such lists live on full pages, so use fetch_content=true. One or two good lists usually suffice.
   - Collect song titles WITH their artists from those lists, then call get_songs.

Any other shape: same principle — web_search for real song titles that answer the request, then get_songs.

Rules:
- fetch_content=true downloads the full text of the top pages; use it whenever you need a LIST from a page (tracklist, chart, "best of" ranking) — snippets almost never contain the actual songs. Snippets (fetch_content=false) are enough for confirming names and facts.
- When a web_search response contains a "LIBRARY MATCHES" section, those songs were ALREADY verified to be in the user's library. Put their keys (e.g. "M3") directly into the final track_ids — do NOT re-check them with get_songs, and prefer them over anything you would propose from memory. If LIBRARY MATCHES already cover ~15 songs, skip further searching and finish.
- Album/OST/compilation names ("Original Soundtrack", "OST Vol. 2", "Now That's What I Call Music") are NOT songs — never pass them to get_songs.
- Song titles MUST come from web_search results — never from your own memory, and never by reinterpreting the user's request text as a song title or a vibe/mood to search for. get_songs refuses to run until web_search has been called.
- Only put a track into the playlist if get_songs returned it with match "exact" or "fuzzy" (never "none"). Use the track_id from get_songs verbatim.
- If get_songs matched FEWER than 8 library tracks and you still have web searches left, dig deeper: run MORE web_search calls with different angles (deep cuts, "full album tracklist", the other language, other soundtrack volumes/parts), then call get_songs again with the NEW titles. Stop digging once ~15 library tracks are matched or the search budget runs out.
- If the request constrains an era or years ("хиты 80-х", "2000s hits"), keep only songs whose ORIGINAL release year (per the web results) fits; a covers/remaster year in the library does not disqualify a song, but a song originally from another era must be dropped.
- If fewer than 3 tracks were found in the library, say so honestly in the comment; still return whatever was found.
- "missing": ALWAYS output an empty list []. Songs that were not in the library are tracked automatically — never echo them back.
- web_search is limited; don't waste calls. Do not call get_songs before you have real song titles, and never pass more than 30 items per get_songs call.
- Never run two searches with near-identical wording — a rephrase of the same question returns the same pages. If two consecutive searches produced no NEW song titles, STOP searching: call get_songs with what you have, or finish the draft honestly.
- Write the playlist "title" and "comment" in the user's language (given as [lang=..] at the start of the prompt).
- Output strictly the PlaylistDraft structure."""


def _score_artists(query: str, values) -> list[dict]:
    """Cross-script artist candidates via the shared ``name_match`` scoring
    ("канье" finds "Kanye West"). Up to 5 with score >= 0.5, best first."""
    scored = score_names(query, values)
    return [{"artist": v, "score": round(s, 2)} for s, v in scored[:5] if s >= 0.5]


class SongQuery(BaseModel):
    title: str
    artist: Optional[str] = None


def create_playlist_agent(model, deps, catalog, state):
    """Construct the agent with its three tools bound to ``deps``/``catalog``.

    ``state`` accumulates cross-tool data: ``web`` (search counter),
    ``resolved`` (track_id -> {title, artist, match}), ``missing`` (list of
    "Title — Artist" not in the library), and an optional ``on_status`` sync
    callback for SSE progress. The callback receives event dicts —
    ``{"stage": str, ...}`` with human-relevant details (the query a tool ran
    with, how many songs matched) so the UI can render a progress chain.
    """
    # max_tokens: потолок одной генерации. Здоровые ответы агента (tool-call
    # или финальный PlaylistDraft на ~15 треков) много меньше; зациклившаяся
    # локальная модель (gemma повторяет одно и то же) упирается в потолок за
    # секунды вместо минут — обрезанный вывод даёт структурную ошибку, и
    # run_playlist_agent отдаёт частичный плейлист штатным фоллбэком.
    agent = Agent(model, output_type=PlaylistDraft, instructions=_INSTRUCTIONS,
                  model_settings={"max_tokens": 3000})

    def _emit(stage: str, **info) -> None:
        cb = state.get("on_status")
        if cb is None:
            return
        try:
            cb({"stage": stage, **info})
        except Exception:  # pragma: no cover — a broken callback must not break the agent
            logger.debug("[playlist_agent] on_status failed", exc_info=True)

    @agent.tool_plain
    async def fetch_filters(artist_query: str) -> list:
        """Resolve how the user named an artist to the real artist names present
        in the library (transliteration-aware, e.g. "Канье" -> "Kanye West").
        Returns up to 5 candidates with a similarity score."""
        _emit("filters", query=artist_query)
        try:
            values = await deps.resolve_filter_values("artist", artist_query)
        except Exception as exc:
            logger.warning("[playlist_agent] fetch_filters failed: %s", exc)
            _emit("filters_done", query=artist_query, best=None)
            return []
        scored = _score_artists(artist_query, values)
        logger.info("[playlist_agent] fetch_filters query=%r scanned=%d artists -> candidates=%s",
                    artist_query, len(values), scored)
        _emit("filters_done", query=artist_query,
              best=scored[0]["artist"] if scored else None)
        return scored

    async def _do_web_search(query: str, fetch_content: bool = False) -> str:
        if state["web"] >= _MAX_WEB_SEARCHES:
            return (
                f"(web search limit reached — {_MAX_WEB_SEARCHES} searches already used. "
                "Build the playlist from the titles you already have.)"
            )
        # Looping models re-ask the same query with cosmetic variations; a
        # repeat returns the same pages, so refuse it without burning budget.
        norm = " ".join(query.lower().split())
        seen = state.setdefault("seen_queries", set())
        if norm in seen:
            logger.info("[playlist_agent] web_search repeat refused: %r", query)
            return (
                "(already searched exactly this — the results are above in the "
                "conversation. Use them, or search a genuinely DIFFERENT angle.)"
            )
        seen.add(norm)
        state["web"] += 1
        _emit("web_search", query=query)
        logger.info("[playlist_agent] web_search #%d query=%r fetch_content=%s",
                    state["web"], query, fetch_content)
        try:
            # Полные страницы длинные — при fetch_content берём 3 результата
            # вместо 5, чтобы не раздувать контекст модели (~4k знаков каждая).
            # rank="playlist": глубокая выборка + переранжирование в коде
            # (убирает genius-мусор списочных запросов) + авто-чтение топ-1
            # авторитетного источника, даже если модель просила только сниппеты.
            # lines_out собирает ПОЛНЫЙ извлечённый треклист (не сэмпл) —
            # библиотечное пересечение делает код ниже, не модель.
            n = 3 if fetch_content else 5
            lines_out: list = []
            res = await asyncio.to_thread(
                smart_web_search, query, fetch_content, n, "playlist", lines_out
            ) or "(no web results)"
        except Exception as exc:
            logger.warning("[playlist_agent] web_search failed: %s", exc)
            return "(web search unavailable)"

        # Auto-matching: пересечь весь извлечённый треклист с библиотекой.
        # 500-песенный саундтрек не «перепечатать» через маленькую модель —
        # она сэмплирует пару десятков строк и пересечение выходит пустым.
        if lines_out:
            try:
                matches = await asyncio.to_thread(
                    resolve_tracklines, lines_out, catalog)
            except Exception:
                logger.warning("[playlist_agent] auto-match failed", exc_info=True)
                matches = []
            if matches:
                automap = state.setdefault("automatch", {})   # "M1" -> track_id
                by_tid = {tid: key for key, tid in automap.items()}
                section = []
                for m in matches:
                    tid = m["track_id"]
                    key = by_tid.get(tid)
                    if key is None:
                        key = f"M{len(automap) + 1}"
                        automap[key] = tid
                        by_tid[tid] = key
                    state["resolved"][tid] = {
                        "title": m["title"], "artist": m["artist"],
                        "match": m["match"], "year": m.get("year"),
                    }
                    yr = f" ({m['year']})" if m.get("year") else ""
                    section.append(f"{key}. {m['artist']} — {m['title']}{yr}")
                _emit("auto_matched", query=query, found=len(matches))
                logger.info("[playlist_agent] auto-matched %d/%d tracklist lines: %s",
                            len(matches), len(lines_out),
                            "; ".join(section[:40]))
                res += (
                    "\n\nLIBRARY MATCHES — every line of the found tracklists was "
                    "auto-checked against the user's library; THESE songs are in it. "
                    "Put the keys (e.g. \"M3\") straight into track_ids — no need to "
                    "call get_songs for them:\n" + "\n".join(section[:40])
                )
        logger.info("[playlist_agent] web_search #%d -> %d chars: %.300s",
                    state["web"], len(res), res.replace("\n", " · "))
        return res

    # Доступ для seed-поиска (run_playlist_agent): тот же путь, что и у
    # инструмента — бюджет, дедуп, авто-матчинг, события.
    state["_do_web_search"] = _do_web_search

    @agent.tool_plain
    async def web_search(query: str, fetch_content: bool = False) -> str:
        """Search the web for real song titles (artist hits, soundtrack tracklists).

        fetch_content=true downloads the full text of the top pages (use for
        soundtrack tracklists — snippets don't contain them); false returns
        snippets only (fast, enough for artist hits).
        """
        return await _do_web_search(query, fetch_content)

    @agent.tool_plain
    async def get_songs(items: list[SongQuery]) -> list | str:
        """Match proposed {title, artist} songs against the user's library.

        Returns one entry per item with ``match`` ("exact"|"fuzzy"|"none") and,
        when found, the library ``track_id``/``title``/``artist``."""
        if state["web"] == 0:
            # Web-first is the whole point: without it the model pattern-matches
            # the request text against the library ("semantic" garbage).
            logger.info("[playlist_agent] get_songs refused (no web_search yet), items=%s",
                        [f"{i.title} — {i.artist}" for i in items])
            return (
                "(refused: call web_search first to get REAL song titles from "
                "the internet, then call get_songs with those titles.)"
            )
        if len(items) > 30:
            # Копипаст целого треклиста в один tool-call — то, на чём слабые
            # модели деградируют к хвосту генерации (наблюдалось: 90 позиций →
            # питон-грамматика сервинга падает 500-кой). Треклисты со страниц
            # сверяются автоматически — модели незачем их пересылать.
            logger.info("[playlist_agent] get_songs refused (%d items > 30)", len(items))
            return (
                "(refused: too many items — at most 30 per call. Songs from "
                "tracklist pages are checked automatically (see LIBRARY "
                "MATCHES); do not re-send them.)"
            )
        _emit("matching", count=len(items))
        logger.info("[playlist_agent] get_songs items=%s",
                    [f"{i.title} — {i.artist}" for i in items])
        payload = [i.model_dump() for i in items]
        try:
            resolved = await asyncio.to_thread(resolve_songs, payload, catalog)
        except Exception as exc:
            logger.warning("[playlist_agent] get_songs failed: %s", exc)
            return [{"query_title": i.get("title", ""), "match": "none"} for i in payload]
        logger.info("[playlist_agent] get_songs results=%s",
                    [(r["query_title"], r["match"],
                      f'{r.get("title")} — {r.get("artist")}' if r["match"] != "none" else None)
                     for r in resolved])
        for r in resolved:
            if r["match"] != "none" and r.get("track_id"):
                state["resolved"][r["track_id"]] = {
                    "title": r.get("title"), "artist": r.get("artist"), "match": r["match"],
                }
            elif r["match"] == "none" and r.get("query_title"):
                state["missing"].append(r["query_title"])
        found = sum(1 for r in resolved if r["match"] != "none")
        _emit("matching_done", found=found, total=len(resolved))
        return resolved

    return agent


async def run_playlist_agent(prompt, lang, deps, catalog, state,
                             llm_base_url=None, llm_model=None, on_status=None):
    """Run the playlist agent to completion and return its :class:`PlaylistDraft`.

    ``state`` (caller-provided dict) is populated with ``resolved``/``missing``
    so the route can build a track preview and filter hallucinated ids.
    """
    state.setdefault("web", 0)
    state.setdefault("resolved", {})
    state.setdefault("missing", [])
    state["on_status"] = on_status

    logger.info("[playlist_agent] start prompt=%r lang=%s model=%s base_url=%s",
                prompt, lang, llm_model or "(default)", llm_base_url or "(default)")
    model = _create_pydantic_model(llm_base_url, llm_model)
    agent = create_playlist_agent(model, deps, catalog, state)

    # Seed-поиск: первый веб-поиск запускает КОД с дословным текстом желания,
    # до старта модели. Формулировка первого запроса — то, на чём слабые
    # модели проваливают весь прогон («popular songs…» вместо треклиста);
    # поисковики же прекрасно понимают сырое желание на любом языке. Модель
    # стартует, уже имея LIBRARY MATCHES, и ищет только недостающее.
    seed_note = ""
    seed_fn = state.get("_do_web_search")
    if seed_fn is not None:
        try:
            await seed_fn((prompt or "").strip(), True)
        except Exception:
            logger.warning("[playlist_agent] seed search failed", exc_info=True)
        automap = state.get("automatch") or {}
        if automap:
            def _fmt(tid):
                r = state["resolved"].get(tid, {})
                yr = f" ({r['year']})" if r.get("year") else ""
                return f"{r.get('artist')} — {r.get('title')}{yr}"
            found = "; ".join(
                f"{key}. {_fmt(tid)}" for key, tid in list(automap.items())[:40]
            )
            seed_note = (
                "\n\n(A web search for this wish already ran. LIBRARY MATCHES so "
                f"far — put these keys into track_ids: {found}. Search further "
                "only for what is still missing.)"
            )
        else:
            seed_note = (
                "\n\n(A web search for the raw wish text already ran and matched "
                "nothing in the library yet — phrase your own searches "
                "differently, don't repeat the wish verbatim.)"
            )

    try:
        result = await agent.run(
            f"[lang={lang}] {prompt}{seed_note}",
            usage_limits=UsageLimits(request_limit=_MAX_LLM_REQUESTS),
        )
        draft = result.output
        # Короткие M-ключи авто-матчинга ("M3") → реальные track_id. Модель
        # оперирует ключами, потому что копирование длинных UUID — то, на чём
        # слабые модели ошибаются.
        automap = state.get("automatch") or {}
        if automap:
            draft.track_ids = [automap.get(t, t) for t in draft.track_ids]
        # missing собирает КОД (get_songs пишет промахи в state["missing"]),
        # модель его не перепечатывает: гемма, следуя старой инструкции,
        # вываливала в missing весь треклист (80+ строк) — длинный структурный
        # вывод и был триггером её грамматических 500-ок и ухода в thinking.
        draft.missing = list(dict.fromkeys(state["missing"]))[:12]
    except (UsageLimitExceeded, ModelAPIError, ModelHTTPError,
            UnexpectedModelBehavior, IncompleteToolCall) as exc:
        # Агент не довёл структурный вывод до конца. Либо зациклился и сжёг бюджет
        # запросов (UsageLimitExceeded), либо модель сорвалась на финальном
        # PlaylistDraft — частая беда слабых/сильно-квантованных моделей: сервинг
        # возвращает 500 «output does not match grammar» (ModelHTTPError) или
        # невалидный/обрезанный tool-call. ModelAPIError — сам LLM-бэкенд умер
        # посреди прогона (gemma в цикле генерации кладёт сервинг → connection
        # refused на следующем запросе). Всё, что get_songs уже сматчил, лежит в
        # state["resolved"] — отдаём частичный плейлист вместо падения всего стрима.
        logger.warning(
            "[playlist_agent] agent aborted (%s: %s) — returning partial draft "
            "with %d resolved tracks", type(exc).__name__, exc, len(state["resolved"]),
        )
        ru = str(lang).lower().startswith("ru")
        draft = PlaylistDraft(
            title=(prompt or "").strip()[:60] or ("Плейлист" if ru else "Playlist"),
            track_ids=list(state["resolved"].keys()),
            comment=("Не удалось довести подбор до конца — вот что уже нашлось." if ru
                     else "Couldn't finish building the playlist — here is what was found so far."),
            missing=list(dict.fromkeys(state["missing"]))[:12],
        )
    logger.info(
        "[playlist_agent] done title=%r draft_track_ids=%s (resolved_in_library=%d, "
        "missing=%d, web_searches=%d)",
        draft.title, draft.track_ids, len(state["resolved"]),
        len(state["missing"]), state["web"],
    )
    return draft
