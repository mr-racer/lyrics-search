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
    "search": "Понял — ищу трек",
    "playlist": "Понял — собираю подборку",
    "facts": "Понял — рассказываю",
}
_INTENT_LABEL_EN = {
    "search": "Got it — finding a track",
    "playlist": "Got it — building a playlist",
    "facts": "Got it — looking it up",
}

# Labels for the buttons of a `clarify` frame (routing was inconclusive).
CLARIFY_LABELS_RU = {
    "search": "Найти трек",
    "playlist": "Собрать подборку",
    "facts": "Рассказать подробнее",
}
CLARIFY_LABELS_EN = {
    "search": "Find a track",
    "playlist": "Build a playlist",
    "facts": "Tell me more",
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

    if stage == "error":
        return "Ошибка" if ru else "Error"
    return ""
