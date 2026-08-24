"""Прогнать промпт биографии по фиксированным отрывкам.

Замеряется ПРОМПТ, а не поиск. Отрывки лежат в `tests/data/bio_passages.json`
(головы статей Википедии, нарезанные детерминированно) — ни кросс-энкодера, ни
сети в прогоне нет, поэтому 27b и 9b получают ровно один и тот же вход и
разница в ответах — это разница моделей, а не разных кусков статьи.

Что считается автоматически — только то, что проверяется точно:

  ПУСТО      — ответ равен сентинелу NO_DATA или пуст. Ровное сравнение строк.
  НЕ ТОТ ЯЗЫК — ответ целиком не на целевом языке (по алфавиту, не по словам).
               В проде такой ответ сначала переспрашивают, потом сбрасывают.
  ДЛИНА      — медиана, ловит схлопывание в одну фразу.

«Интересно/пресно» отсюда не считается и считаться не может: это смысл, а не
форма. Скрипт печатает тексты целиком — сравнивать две ветки глазами. Так была
поймана регрессия, которую счётчик не увидел бы: инструкция «одна конкретная
вещь… число» вытащила во второй абзац тиражи продаж вместо историй.

Запуск:

    python scripts/eval_bio_prompt.py --model … --base-url … --out bio.json
    python scripts/eval_bio_prompt.py --legacy --out bio_legacy.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import text_quality as tq                    # noqa: E402
from app.services.bio_v2 import prompts as P                   # noqa: E402
from app.services.llm_client import ask_llm, resolve_model     # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tests", "data", "bio_passages.json")


def use_legacy_prompt() -> None:
    """Взять BIO_PROMPT из HEAD — базовая линия для сравнения."""
    import subprocess
    import types

    src = subprocess.check_output(
        ["git", "show", "HEAD:app/services/bio_v2/prompts.py"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ).decode("utf-8")
    ns = types.ModuleType("legacy_bio_prompts")
    exec(compile(src, "<HEAD bio prompts.py>", "exec"), ns.__dict__)   # noqa: S102
    P.BIO_PROMPT = ns.BIO_PROMPT


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--lang", default="Russian")
    ap.add_argument("--lang-code", default="ru")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.legacy:
        use_legacy_prompt()
    rows = json.load(open(DATA, encoding="utf-8"))
    model = resolve_model(args.model)
    tag = "промпт HEAD" if args.legacy else "промпт рабочей копии"
    print(f"модель: {model} | {tag} | артистов: {len(rows)}")

    out = []
    for row in rows:
        text = (await ask_llm(
            P.BIO_PROMPT.format(artist=row["artist"], lang=args.lang,
                                passages=row["passages"]),
            temperature=0.35, base_url=args.base_url, model=args.model) or "").strip()
        out.append({
            "artist": row["artist"], "text": text,
            "empty": tq.is_refusal(text),
            "wrong_lang": tq.no_target_script(text, args.lang_code),
            "chars": len(text),
        })
        print(".", end="", flush=True)
    print()

    print(f"\n=== {model}, {tag} ===")
    print(f"  пусто        {sum(r['empty'] for r in out)}/{len(out)}")
    print(f"  не тот язык  {sum(r['wrong_lang'] for r in out)}/{len(out)} "
          f"(в проде уходит на переспрос)")
    print(f"  длина        медиана "
          f"{statistics.median(r['chars'] for r in out):.0f} символов")
    print("\nдальше — тексты целиком, читать глазами:")
    for r in out:
        flags = " ".join(f for f, on in (("ПУСТО", r["empty"]),
                                         ("НЕ ТОТ ЯЗЫК", r["wrong_lang"])) if on)
        print(f"\n--- {r['artist']} {flags}")
        print(r["text"])
    if args.out:
        json.dump({"model": model, "prompts": "HEAD" if args.legacy else "worktree",
                   "rows": out}, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\nотчёт: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
