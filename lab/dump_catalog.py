"""Фоллбэк: выгрузка каталога в файл-кэш, если Qdrant сервера не виден с ноутбука.

Обычный путь — `Catalog()` ходит в Qdrant сама (QDRANT_URL). Этот скрипт нужен,
только если порт 6333 наружу не отдан:

    docker compose exec musix python /app/lab/dump_catalog.py          # авто-выбор acct_*
    docker compose exec musix python /app/lab/dump_catalog.py acct_3
    docker compose cp musix:/app/catalog_cache.json ./catalog_cache.json

Формат совпадает с CATALOG_CACHE, поэтому на ноутбуке достаточно `Catalog()` —
она подхватит кэш и в Qdrant не пойдёт.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab.websearch_lab import load_songs_from_qdrant  # noqa: E402

OUT = os.environ.get("CATALOG_CACHE", "catalog_cache.json")

collection, points = load_songs_from_qdrant(
    collection=sys.argv[1] if len(sys.argv) > 1 else None)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"collection": collection, "points": points}, f, ensure_ascii=False)
print(f"{len(points)} треков ({collection}) → {OUT}")
