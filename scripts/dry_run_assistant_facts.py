"""Run the assistant's facts branch over a list of questions and print the
grounding pack next to the answer.

This is the measurement half of tuning that branch. Reading an answer alone
cannot tell you whether it is thin because the model is weak or because the
pack it was handed is nothing but release dates — printing both side by side
can, and the pack is exactly what the prompt work has to react to.

    docker exec musix python scripts/dry_run_assistant_facts.py --collection acct_2
    docker exec musix python scripts/dry_run_assistant_facts.py --questions my.txt --lang ru
    docker exec musix python scripts/dry_run_assistant_facts.py --json before.json

Compare two runs (before/after a prompt change) with the same --questions file
and diff the JSON dumps.

``--facts FILE`` measures the OTHER branch — explain mode, where each line is a
statement the listener tapped rather than a question they typed. That branch is
allowed to answer nothing, so read its summary differently: "explained 4/9" with
five honest blanks is a better run than 9/9 of invented context. What to check
in a blank is the WEB QUERIES line — a fact that went unexplained because the
model typed a useless query is a prompt problem, not a missing-source problem.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.domain.models import AssistantSlots  # noqa: E402
from app.resources.metadata_db import MetadataDB  # noqa: E402
from app.services.assistant import facts_executor  # noqa: E402

# Deliberately a mix: named artists, named songs, and questions with no entity
# at all (those exercise the resolver, which is where thin answers often start).
DEFAULT_QUESTIONS_RU = [
    "расскажи про Radiohead",
    "что известно про Kanye West",
    "расскажи про этот трек",
    "кто продюсировал Runaway",
    "о чём песня Bohemian Rhapsody",
    "чем интересен альбом OK Computer",
    "расскажи про Massive Attack",
    "что за история у трека Hurt",
]
DEFAULT_QUESTIONS_EN = [
    "tell me about Radiohead",
    "what is known about Kanye West",
    "who produced Runaway",
    "what is Bohemian Rhapsody about",
]


class _Route:
    """The router's output, stubbed: this script measures the executor, and
    calling GLiNER2 here would only add latency and a second failure mode."""
    intent = "facts"
    artist = None
    song = None
    count = None
    era = None
    confidence = 1.0


async def one(qdrant, collection: str, question: str, lang: str,
              focus_fact: str | None = None) -> dict:
    # The web queries are the thing to inspect when a fact goes unexplained, and
    # they are otherwise only visible in the server log.
    queries: list = []
    emit = _query_recorder(queries)
    payload, options = await facts_executor.run(
        qdrant=qdrant, collection_name=collection, message=question,
        route=_Route(), slots=AssistantSlots(), lang=lang,
        focus_fact=focus_fact, emit=emit,
    )
    return {"question": question, "focus_fact": focus_fact, "payload": payload,
            "web_queries": queries,
            "options": [o.get("title") for o in (options or [])]}


def _query_recorder(sink: list):
    async def emit(event: dict) -> None:
        if event.get("stage") == "web_search" and event.get("query"):
            sink.append(event["query"])
    return emit


def render(result: dict, verbose: bool) -> None:
    print("\n" + "═" * 78)
    if result.get("focus_fact"):
        print(f"FACT: {result['focus_fact']}")
    else:
        print(f"Q: {result['question']}")
    payload = result["payload"]
    if payload is None:
        if result["options"]:
            print(f"  → disambiguate: {', '.join(result['options'])}")
        else:
            print("  → subject not resolved")
        return

    items = payload.get("items") or []
    used = {i["n"] for i in items if i.get("used")}
    print(f"SUBJECT: {payload.get('subject_kind')} · {payload.get('subject_title')}"
          f"{' — ' + payload['subject_subtitle'] if payload.get('subject_subtitle') else ''}")
    print(f"PACK: {len(items)} items · used {sorted(used) or '—'}"
          f" · grounded={payload.get('grounded')} · web={payload.get('web_search_used')}"
          + (f" · explained={payload.get('explained')}"
             if payload.get("focus_fact") else ""))
    if result.get("web_queries"):
        print("WEB QUERIES: " + " | ".join(result["web_queries"]))
    for item in items:
        mark = "*" if item.get("n") in used else " "
        text = item.get("text") or ""
        print(f" {mark}[{item.get('n')}] ({item.get('source')}) "
              f"{text if verbose else text[:150]}")
    print("-" * 78)
    print(payload.get("answer") or "")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=os.environ.get("MUSIX_COLLECTION", "acct_1"))
    ap.add_argument("--lang", default="ru", choices=["ru", "en"])
    ap.add_argument("--questions", help="file with one question per line")
    ap.add_argument("--facts", help="file with one STATEMENT per line — runs "
                                    "explain mode instead of the question mode")
    ap.add_argument("--json", help="also dump raw results to this file")
    ap.add_argument("--verbose", action="store_true", help="print pack items in full")
    args = ap.parse_args()

    facts: list[str] = []
    if args.facts:
        with open(args.facts, encoding="utf-8") as fh:
            facts = [ln.strip() for ln in fh if ln.strip()]
        # The message a tapped card sends, built the same way the SPA builds it.
        questions = [(f"объясни: {f}" if args.lang == "ru" else f"explain this: {f}")
                     for f in facts]
    elif args.questions:
        with open(args.questions, encoding="utf-8") as fh:
            questions = [ln.strip() for ln in fh if ln.strip()]
    else:
        questions = DEFAULT_QUESTIONS_RU if args.lang == "ru" else DEFAULT_QUESTIONS_EN

    MetadataDB.init()
    from app.resources.db_client import DbClient

    results = []
    with DbClient(collection_name=args.collection) as db:
        if db.qdrant is None:
            print("Qdrant unavailable — the facts branch needs it to resolve subjects.")
            return
        for i, question in enumerate(questions):
            focus = facts[i] if facts else None
            try:
                result = await one(db.qdrant, args.collection, question, args.lang, focus)
            except Exception as exc:  # one bad question must not end the run
                result = {"question": question, "focus_fact": focus, "payload": None,
                          "options": [], "web_queries": [],
                          "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            render(result, args.verbose)

    print("\n" + "═" * 78)
    if facts:
        explained = [r for r in results if r["payload"] and r["payload"].get("explained")]
        from_web = [r for r in explained if r["payload"].get("web_search_used")]
        print(f"{len(explained)}/{len(results)} explained · {len(from_web)} of those "
              f"needed the web · {len(results) - len(explained)} honestly blank")
    else:
        answered = [r for r in results
                    if r["payload"] and (r["payload"].get("answer") or "").strip()]
        grounded = [r for r in answered if r["payload"].get("grounded")]
        print(f"{len(answered)}/{len(results)} answered · {len(grounded)} written by the LLM · "
              f"{len(answered) - len(grounded)} fell back to the deterministic render")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            # related_tracks are pydantic models — default=str keeps the dump
            # working without pulling model_dump through three nesting levels.
            json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
        print(f"raw results → {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
