"""Plain-language labels for every assistant progress stage.

The backend ships a ready-to-render ``human`` string with each frame so the UI
only animates it — no phrasing lives in the frontend. Grew out of
``_human()`` in ``app/api/routes/chat.py`` (search stages) and now also covers
the playlist stages emitted by ``recsys_ai_service`` / ``playlist_agent`` and
the facts stages of ``facts_executor``.

Adding a stage: add a branch here, never a string in the SPA. An unknown stage
returns "" — the frame still streams, it just renders without a caption.
"""

from __future__ import annotations

_MODE_LABEL_RU = {"text": "по тексту", "audio": "по звучанию", "hybrid": "по тексту и звуку"}
_MODE_LABEL_EN = {"text": "by lyrics", "audio": "by sound", "hybrid": "by lyrics and sound"}

_INTENT_LABEL_RU = {
    "lyrics_search": "Понял — ищу трек по тексту",
    "audio_search": "Понял — ищу по звучанию",
    "playlist": "Понял — собираю подборку",
    "general": "Понял — рассказываю",
}
_INTENT_LABEL_EN = {
    "lyrics_search": "Got it — finding the track by its words",
    "audio_search": "Got it — finding it by sound",
    "playlist": "Got it — building a playlist",
    "general": "Got it — looking it up",
}

# Labels for the buttons of a `clarify` frame (the planner could not settle).
CLARIFY_LABELS_RU = {
    "lyrics_search": "Найти по строчке",
    "audio_search": "Найти по звучанию",
    "playlist": "Собрать подборку",
    "general": "Рассказать подробнее",
}
CLARIFY_LABELS_EN = {
    "lyrics_search": "Find it by a line",
    "audio_search": "Find it by sound",
    "playlist": "Build a playlist",
    "general": "Tell me more",
}


def is_ru(lang: str | None) -> bool:
    return (lang or "en").lower().startswith("ru")


def plural_ru(n: int, one: str, few: str, many: str) -> str:
    """Russian plural form for ``n`` («1 совпадение / 2 совпадения / 5 совпадений»)."""
    r = abs(n) % 100
    if 11 <= r <= 14:
        return many
    d = r % 10
    if d == 1:
        return one
    if 2 <= d <= 4:
        return few
    return many


def _matches_ru(n: int) -> str:
    return f"{n} " + plural_ru(n, "совпадение", "совпадения", "совпадений")


def _tracks_ru(n: int) -> str:
    return f"{n} " + plural_ru(n, "трек", "трека", "треков")


def _facts_ru(n: int) -> str:
    return f"{n} " + plural_ru(n, "факт", "факта", "фактов")


def clarify_labels(lang: str | None) -> dict:
    return CLARIFY_LABELS_RU if is_ru(lang) else CLARIFY_LABELS_EN


def human(stage: str, lang: str | None = None, **kw) -> str:
    """Label for one progress stage. Unknown stages return ""."""
    ru = is_ru(lang)

    # ── routing ───────────────────────────────────────────────────────────
    if stage == "route":
        intent = kw.get("intent") or "search"
        return (_INTENT_LABEL_RU if ru else _INTENT_LABEL_EN).get(intent, "")
    if stage == "clarify":
        return ("Не уверен, что именно нужно — уточни"
                if ru else "Not sure what you need — pick one")
    if stage == "disambiguate":
        return "Уточни, о ком речь" if ru else "Which one did you mean?"
    if stage == "thinking":
        step = kw.get("step")
        if step == "plan":
            return "Продумываю план" if ru else "Working out a plan"
        if step == "curate":
            return "Отбираю треки" if ru else "Picking the tracks"
        if step == "answer":
            return "Формулирую ответ" if ru else "Writing the answer"
        return "Думаю" if ru else "Thinking"

    # ── the web agent ─────────────────────────────────────────────────────
    # One caption per stage the pipeline actually emits. A stage with no branch
    # here still streams — it just renders without a caption — so this list is
    # allowed to lag behind the pipeline without breaking anything.
    if stage == "start":
        return "Разбираю запрос" if ru else "Reading the request"
    if stage == "plan_failed":
        return ("Не удалось составить план — модель не ответила"
                if ru else "Couldn't plan the run — the model didn't answer")
    if stage == "iteration":
        n = int(kw.get("n") or 1)
        if n <= 1:
            return "Иду в интернет" if ru else "Going to the web"
        return (f"Захожу на второй круг (попытка {n})"
                if ru else f"Another round (attempt {n})")
    if stage == "rerank":
        kept = int(kw.get("kept") or 0)
        total = int(kw.get("candidates") or 0)
        return (f"Из {total} ссылок стоит читать {kept}"
                if ru else f"{kept} of {total} links are worth reading")
    if stage == "fetch":
        n = int(kw.get("count") or 0)
        if kw.get("refill"):
            return ("Часть страниц не открылась — беру следующие"
                    if ru else "Some pages refused — taking the next ones")
        return (f"Читаю {n} " + plural_ru(n, "страницу", "страницы", "страниц")
                if ru else f"Reading {n} page{'' if n == 1 else 's'}")
    if stage == "fetch_done":
        got = int(kw.get("fetched") or 0)
        return (f"Прочитал {got} " + plural_ru(got, "страницу", "страницы", "страниц")
                if ru else f"Read {got} page{'' if got == 1 else 's'}")
    if stage == "index":
        return "Раскладываю прочитанное" if ru else "Indexing what I read"
    if stage == "chunks":
        n = int(kw.get("selected") or 0)
        if not n:
            return ("Ничего по делу на этих страницах" if ru
                    else "Nothing on those pages was about it")
        return (f"Отобрал {n} " + plural_ru(n, "фрагмент", "фрагмента", "фрагментов")
                if ru else f"Picked {n} passage{'' if n == 1 else 's'}")
    if stage == "dedup":
        n = int(kw.get("duplicates") or 0)
        return (f"Схлопнул {n} " + plural_ru(n, "повтор", "повтора", "повторов")
                if ru else f"Collapsed {n} near-duplicate{'' if n == 1 else 's'}")
    if stage == "seeded":
        n = int(kw.get("chunks") or 0)
        return (f"Беру {n} " + plural_ru(n, "фрагмент", "фрагмента", "фрагментов")
                + " из прошлого ответа"
                if ru else f"Reusing {n} passage{'' if n == 1 else 's'} "
                           f"from the last answer")
    if stage == "verdict":
        if kw.get("local"):
            # The whole point of the local iteration is that the listener sees
            # it stop here — an answer with no "going to the web" line is the
            # visible difference between the two paths.
            return (("Ответил по библиотеке" if ru else "Answered from your library")
                    if kw.get("stop") else
                    ("В библиотеке этого нет — иду в интернет"
                     if ru else "Your library doesn't cover it — going to the web"))
        if kw.get("stop"):
            return "Материала достаточно" if ru else "That's enough material"
        return "Материала мало — ищу ещё" if ru else "Not enough yet — searching again"
    if stage == "subject":
        return ("Понял, о ком речь" if kw.get("artist") or kw.get("song")
                else "Не понял, о ком речь — иду в интернет") if ru else (
            "Worked out who this is about" if kw.get("artist") or kw.get("song")
            else "Couldn't tell who this is about — going to the web")
    if stage == "facts_done":
        n = int(kw.get("kept") or 0)
        if n:
            return (f"В библиотеке нашлось {_facts_ru(n)}"
                    if ru else f"Found {n} fact{'' if n == 1 else 's'} in the library")
        return ("В библиотеке фактов нет — иду в интернет"
                if ru else "No facts stored — going to the web")
    if stage == "reddit_rescue":
        return ("Больше нигде нет — смотрю обсуждения на Reddit"
                if ru else "Nothing anywhere else — checking Reddit threads")
    if stage == "structured":
        n = int(kw.get("tracks") or 0)
        return (f"Разобрал таблицу: {_tracks_ru(n)}"
                if ru else f"Parsed a table: {n} track{'' if n == 1 else 's'}")
    if stage == "extract":
        n = int(kw.get("claims") or 0)
        return (f"Выписал {_tracks_ru(n)} из текста"
                if ru else f"Pulled {n} track{'' if n == 1 else 's'} out of the text")
    if stage == "matched":
        n = int(kw.get("resolved") or 0)
        return (f"В твоей библиотеке из них есть {_tracks_ru(n)}"
                if ru else f"Your library has {n} of them")
    if stage == "discography":
        return (f"Мало треков — читаю дискографию {kw.get('artist') or ''}"
                if ru else f"Thin so far — reading {kw.get('artist') or ''}'s discography")
    if stage == "triage":
        n = int(kw.get("candidates") or 0)
        return (f"Отсеиваю лишнее из {_tracks_ru(n)}"
                if ru else f"Weeding out {n} candidate{'' if n == 1 else 's'}")
    if stage == "curated":
        n = int(kw.get("tracks") or 0)
        return (f"Собрал подборку из {_tracks_ru(n)}"
                if ru else f"Built a playlist of {n} track{'' if n == 1 else 's'}")
    if stage == "clap_rephrase":
        n = len(kw.get("queries") or [])
        return (f"Перевожу звучание на язык модели: {n} формулировки"
                if ru else f"Rewriting the sound into {n} prompts")
    if stage == "result":
        n = int(kw.get("tracks") or 0)
        return (f"Готово: {_tracks_ru(n)}"
                if ru else f"Done: {n} track{'' if n == 1 else 's'}")
    if stage == "engines_down":
        return ("Часть поисковиков не ответила"
                if ru else "Some search engines didn't answer")

    # ── search branch (was chat.py::_human) ───────────────────────────────
    if stage == "classify":
        mode = kw.get("mode", "hybrid")
        return (f"Ищу {_MODE_LABEL_RU.get(mode, 'по тексту и звуку')}"
                if ru else f"Searching {_MODE_LABEL_EN.get(mode, 'by lyrics and sound')}")
    if stage == "plan":
        return "Составил план поиска" if ru else "Planned the search"
    if stage == "search":
        found = int(kw.get("found") or 0)
        if ru:
            return f"Ищу в библиотеке… нашёл {_matches_ru(found)}"
        return f"Searching the library… found {found} match{'' if found == 1 else 'es'}"
    if stage == "validate":
        if kw.get("valid", True):
            return "Проверил — трек подходит" if ru else "Checked — the track fits"
        return "Проверяю точность совпадения" if ru else "Double-checking the match"
    if stage == "retry":
        return "Уточняю запрос и пробую снова" if ru else "Refining the query and retrying"
    if stage == "answer":
        return "Готовлю ответ" if ru else "Preparing the answer"

    # ── playlist branch (recsys_ai_service + playlist_agent) ──────────────
    if stage == "plan_done":
        n = int(kw.get("actions") or 0)
        return (f"План готов: {n} " + plural_ru(n, "шаг", "шага", "шагов")
                if ru else f"Plan ready: {n} step{'' if n == 1 else 's'}")
    if stage == "action":
        tool, query = kw.get("tool"), kw.get("query") or ""
        if tool == "clap_search":
            return (f"Ищу по звучанию: {query}" if ru else f"Searching by sound: {query}")
        if tool == "library_search":
            return (f"Ищу в библиотеке: {query}" if ru else f"Searching the library: {query}")
        if tool == "similar_tracks":
            return (f"Ищу похожее на «{query}»" if ru else f"Finding tracks like “{query}”")
        return (f"Выполняю поиск: {query}" if ru else f"Running a search: {query}")
    if stage == "action_done":
        found = int(kw.get("found") or 0)
        return (f"Нашёл {_tracks_ru(found)}" if ru else f"Found {found} track{'' if found == 1 else 's'}")
    if stage == "select":
        n = int(kw.get("candidates") or 0)
        return (f"Отбираю лучшее из {_tracks_ru(n)}"
                if ru else f"Curating from {n} candidate{'' if n == 1 else 's'}")
    if stage == "select_done":
        n = int(kw.get("picked") or 0)
        return (f"Собрал подборку из {_tracks_ru(n)}"
                if ru else f"Built a playlist of {n} track{'' if n == 1 else 's'}")
    if stage == "filters":
        return (f"Ищу «{kw.get('query') or ''}» в библиотеке"
                if ru else f"Looking up “{kw.get('query') or ''}” in the library")
    if stage == "filters_done":
        best = kw.get("best")
        if best:
            return f"Нашёл артиста: {best}" if ru else f"Matched the artist: {best}"
        return ("В библиотеке такого артиста не нашёл"
                if ru else "No such artist in the library")
    if stage == "web_search":
        return (f"Ищу в интернете: {kw.get('query') or ''}"
                if ru else f"Searching the web: {kw.get('query') or ''}")
    if stage == "auto_matched":
        found = int(kw.get("found") or 0)
        return (f"Сверил с библиотекой — {_matches_ru(found)}"
                if ru else f"Cross-checked the library — {found} match{'' if found == 1 else 'es'}")
    if stage == "matching":
        n = int(kw.get("count") or 0)
        return (f"Сверяю {_tracks_ru(n)} с библиотекой"
                if ru else f"Checking {n} track{'' if n == 1 else 's'} against the library")
    if stage == "matching_done":
        found = int(kw.get("found") or 0)
        return (f"В библиотеке есть {_tracks_ru(found)}"
                if ru else f"The library has {found} of them")

    # ── facts branch ──────────────────────────────────────────────────────
    if stage == "resolving":
        return "Определяю, о чём речь" if ru else "Working out what you mean"
    if stage == "resolved":
        subject = kw.get("subject") or ""
        return (f"Собираю, что известно про {subject}"
                if ru else f"Gathering what's known about {subject}")
    if stage == "collecting":
        n = int(kw.get("found") or 0)
        if n:
            return (f"Нашёл {_facts_ru(n)} в библиотеке"
                    if ru else f"Found {n} fact{'' if n == 1 else 's'} in the library")
        return ("В библиотеке фактов нет — иду в интернет"
                if ru else "No facts stored — going to the web")
    # ── explain mode: one tapped fact, not the whole subject ──
    if stage == "explaining":
        n = int(kw.get("found") or 0)
        if n:
            return (f"Разбираю факт — рядом {_facts_ru(n)}"
                    if ru else f"Working out the fact — {n} related note{'' if n == 1 else 's'}")
        return ("Разбираюсь, что значит этот факт"
                if ru else "Working out what this fact means")
    if stage == "no_explanation":
        return ("Объяснения не нашлось — придумывать не буду"
                if ru else "No explanation found — I won't invent one")

    if stage == "error":
        return "Ошибка" if ru else "Error"
    return ""
