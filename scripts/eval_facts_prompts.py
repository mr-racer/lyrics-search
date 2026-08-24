"""Прогнать классификатор фактов по эталонному набору и посчитать три числа.

Зачем именно три. «Точность классификации» одним числом прячет ровно тот обмен,
который здесь и решается: промпт, отбрасывающий всё подряд, получает идеальную
отбраковку и выкашивает библиотеку. Поэтому отдельно считаются

  ОТБРАКОВКА — доля банальных фактов, которые НЕ показаны (чем выше, тем чище);
  СОХРАННОСТЬ — доля хороших фактов, которые показаны (чем выше, тем полнее);
  КЛАСС       — доля показанных, попавших в верную рубрику.

Эталон — `tests/data/facts_gold.json`: 55 СЫРЫХ фактов из живой коллекции
`acct_c2b5b12d…`, размеченных вручную (39 «показать», 2 off-scope, 14
«отбросить»). В каждом лежит и то, что выдал прод на qwen 9b со старым
промптом (`was`), так что колонка «до» бесплатна.

Оговорка про честность замера: у правила b («глоссарий прозвищ») в промпте есть
свой пример, а в эталоне три позиции того же типа — на них отбраковка даётся
легче, чем в среднем. Пример оставлен намеренно: в проде это самый частый брак
(у Jay-Z из 12 фактов «откуда название» девять были глоссами прозвищ).

Каждый факт классифицируется в одиночку. В проде батч — шесть фактов ОДНОЙ
песни, а эталонные факты все из разных: подложить их под общий заголовок
значило бы нарушить правило «речь о другой песне → other» и испортить замер.
Одиночный вызов — это ровно тот путь, которым в проде идёт и фолбэк.

Запуск (внутри контейнера, чтобы взялись настройки инстанса):

    docker exec -i musix python scripts/eval_facts_prompts.py --out /tmp/eval.json
    docker exec -i musix python scripts/eval_facts_prompts.py --refine --repeat 2

Сравнение двух прогонов:

    docker exec -i musix python scripts/eval_facts_prompts.py --diff a.json b.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.facts_v2 import pipeline as fv2      # noqa: E402
from app.services.llm_client import ask_llm, resolve_model   # noqa: E402

GOLD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "tests", "data", "facts_gold.json")


def use_legacy_prompts() -> str:
    """Подменить два классификатора их версией из HEAD.

    Нужно, чтобы отделить «стало лучше от промпта» от «стало лучше от модели»:
    прод-цифры сняты на qwen 9b, и без этого режима сравнивать пришлось бы
    промпт и модель разом. Маршрутизация (`route`) остаётся новой — меняется
    ровно текст промпта, ровно то, что и замеряется.
    """
    import subprocess
    import types

    from app.services.facts_v2 import prompts as P
    src = subprocess.check_output(
        ["git", "show", "HEAD:app/services/facts_v2/prompts.py"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ).decode("utf-8")
    ns = types.ModuleType("legacy_prompts")
    exec(compile(src, "<HEAD prompts.py>", "exec"), ns.__dict__)   # noqa: S102
    P.SONG_CLASSIFY = ns.SONG_CLASSIFY
    P.ARTIST_CLASSIFY = ns.ARTIST_CLASSIFY
    return "HEAD"


def _asker(model, base_url):
    async def ask(prompt: str, temperature: float = 0.3) -> str:
        return await ask_llm(prompt, temperature=temperature,
                             base_url=base_url, model=model)
    return ask


def verdict(labels: list, gold) -> tuple:
    """(что случилось, верно ли) для одного факта.

    Решает не метка, а исход: показан факт или нет. Off-scope метка ничего не
    пишет, поэтому «about_artist» — это скрытие, а не показ.
    """
    shown = fv2.route(labels)["primary"] is not None
    if gold == "other":
        return ("скрыт" if not shown else "ПОКАЗАН"), not shown
    if isinstance(gold, list) and set(gold) & fv2.OFF_SCOPE:
        ok = not shown and bool(set(labels) & set(gold))
        return ("off-scope" if ok else "не off-scope"), ok
    if not shown:
        return "ПОТЕРЯН", False
    hit = bool(set(labels) & set(gold))
    return ("верный класс" if hit else f"класс {','.join(labels) or '—'}"), hit


async def run_once(items, model, base_url, refine: bool) -> list:
    ask = _asker(model, base_url)
    out = []
    for it in items:
        entity = ({"artist": it["artist"], "title": it["title"] or ""}
                  if it["scope"] == "song" else {"name": it["artist"]})
        fact = {"id": 0, "fact": it["fact"], "category": it["category"]}
        recs = await fv2.classify_entity(ask, entity, it["scope"], [fact])
        rec = recs[0] if recs else {"labels": []}
        row = {"id": it["id"], "labels": rec.get("labels", [])}
        if refine and fv2.route(row["labels"])["primary"]:
            try:
                await fv2.refine_one(ask, rec, entity, it["scope"])
                row["refined"] = rec.get("refined") or ""
            except Exception as exc:                        # noqa: BLE001
                row["refined"] = f"<ошибка: {type(exc).__name__}: {exc}>"
        out.append(row)
        print(".", end="", flush=True)
    print()
    return out


def score(items, runs: list) -> dict:
    """Свести N прогонов в отчёт. Факт считается верным, если верен в большинстве."""
    per_id = {it["id"]: it for it in items}
    merged = {}
    for run in runs:
        for row in run:
            merged.setdefault(row["id"], []).append(row)

    buckets = {"отбраковка": [0, 0], "сохранность": [0, 0],
               "класс": [0, 0], "off-scope": [0, 0]}
    misses, details = [], []
    for fid, rows in merged.items():
        it = per_id[fid]
        oks = [verdict(r["labels"], it["gold"])[1] for r in rows]
        ok = sum(oks) * 2 >= len(oks)                    # большинство прогонов
        note, _ = verdict(rows[0]["labels"], it["gold"])
        if it["gold"] == "other":
            key = "отбраковка"
        elif isinstance(it["gold"], list) and set(it["gold"]) & fv2.OFF_SCOPE:
            key = "off-scope"
        else:
            key = "сохранность"
        buckets[key][1] += 1
        buckets[key][0] += ok
        if key == "сохранность" and ok:
            buckets["класс"][1] += 1
            buckets["класс"][0] += 1
        elif key == "сохранность":
            buckets["класс"][1] += 1
        details.append({"id": fid, "gold": it["gold"], "got": rows[0]["labels"],
                        "ok": ok, "note": note, "why": it["why"],
                        "refined": rows[0].get("refined", "")})
        if not ok:
            misses.append(details[-1] | {"fact": it["fact"][:220],
                                         "was": it["was"]})
    return {"buckets": buckets, "misses": misses, "details": details}


def show(report: dict, items, title: str) -> None:
    b = report["buckets"]
    print(f"\n=== {title} ===")
    for name in ("отбраковка", "сохранность", "класс", "off-scope"):
        ok, n = b[name]
        if n:
            print(f"  {name:12} {ok:3}/{n:<3} {100 * ok / n:5.1f}%")
    was = Counter()
    for it in items:
        labels = [x for x in it["was"]["labels"] if not x.startswith("gate:")]
        shown = fv2.route(labels)["primary"] is not None
        if it["gold"] == "other":
            was["банальных показано в проде"] += shown
        elif not (isinstance(it["gold"], list) and set(it["gold"]) & fv2.OFF_SCOPE):
            was["хороших показано в проде"] += shown
    print(f"  для сравнения — прод (qwen 9b, старый промпт): "
          f"{dict(was)}")
    if report["misses"]:
        print(f"\n  --- расхождения ({len(report['misses'])}) ---")
        for m in report["misses"]:
            print(f"  [{m['id']}] эталон={m['gold']} получено={m['got'] or '—'} "
                  f"({m['note']})")
            print(f"        почему эталон такой: {m['why']}")
            print(f"        было в проде: {m['was']['labels']}"
                  f"{' (ПЕРЕНЕСЁН)' if m['was']['moved'] else ''}")
            print(f"        факт: {m['fact']}")


def show_refined(report: dict) -> None:
    rows = [d for d in report["details"] if d.get("refined")]
    if not rows:
        return
    print(f"\n=== переписанные тексты ({len(rows)}) ===")
    for d in rows:
        print(f"  [{d['id']}] {','.join(d['got'])}: {d['refined']}")


def diff(path_a: str, path_b: str) -> None:
    a = json.load(open(path_a, encoding="utf-8"))
    b = json.load(open(path_b, encoding="utf-8"))
    print(f"{'метрика':14}{'A':>12}{'B':>12}{'Δ':>9}")
    for name in ("отбраковка", "сохранность", "класс", "off-scope"):
        ka, kb = a["buckets"].get(name), b["buckets"].get(name)
        if not ka or not ka[1]:
            continue
        pa, pb = 100 * ka[0] / ka[1], 100 * kb[0] / kb[1]
        print(f"{name:14}{pa:11.1f}%{pb:11.1f}%{pb - pa:+8.1f}")
    da = {d["id"]: d for d in a["details"]}
    db = {d["id"]: d for d in b["details"]}
    changed = [i for i in da if i in db and da[i]["ok"] != db[i]["ok"]]
    for i in sorted(changed):
        arrow = "почин" if db[i]["ok"] else "СЛОМАН"
        print(f"  [{i}] {arrow}: {da[i]['got']} → {db[i]['got']}  ({da[i]['why']})")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="переопределить модель")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--repeat", type=int, default=1,
                    help="сколько прогонов усреднить (temperature 0.2 шумит)")
    ap.add_argument("--only", choices=["song", "artist"], default=None)
    ap.add_argument("--refine", action="store_true",
                    help="дописать вторую стадию и показать русский текст")
    ap.add_argument("--out", default=None, help="куда сложить отчёт JSON")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"),
                    help="сравнить два готовых отчёта и выйти")
    ap.add_argument("--legacy", action="store_true",
                    help="взять классификаторы из HEAD — базовая линия промпта")
    args = ap.parse_args()

    if args.diff:
        diff(*args.diff)
        return

    items = json.load(open(GOLD_PATH, encoding="utf-8"))
    if args.only:
        items = [i for i in items if i["scope"] == args.only]
    model = resolve_model(args.model)
    tag = "промпт HEAD" if args.legacy else "промпт рабочей копии"
    if args.legacy:
        use_legacy_prompts()
    print(f"модель: {model} | {tag} | примеров: {len(items)} | "
          f"прогонов: {args.repeat}")

    runs = [await run_once(items, args.model, args.base_url, args.refine)
            for _ in range(args.repeat)]
    report = score(items, runs)
    show(report, items, f"{model}, {tag}, {args.repeat} прогон(ов)")
    if args.refine:
        show_refined(report)
    if args.out:
        report["model"] = model
        report["prompts"] = "HEAD" if args.legacy else "worktree"
        json.dump(report, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\nотчёт: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
