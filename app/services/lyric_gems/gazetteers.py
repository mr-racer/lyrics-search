"""Curated gazetteers for the gems pipeline.

- CAPSULE: bygone items matched by plain regex over the cleaned lyrics (no
  GLiNER involved). Each entry carries its Russian display name, whether the
  match must appear capitalized in the original text (brand-like words that
  double as common nouns), and an optional nearby-context regex for the
  genuinely ambiguous ones (measured false positive: "two-way highway lines").
- STOPWORDS: GLiNER span junk seen in the 2026-07-19 experiment (pronouns and
  generic address words scored up to 0.97 as "celebrity").
- GENERIC_ARTIST_WORDS: library-artist names that are common English words —
  a plain-text match on these is meaningless ("The Game" matched 204 tracks
  as the phrase "the game"). They may only enter via LLM verification.
- Pop-culture entries live in app/resources/data/popculture_gazetteer.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "resources" / "data"

# item -> {ru, cap (require capitalized occurrence), near (context regex or None)}
CAPSULE: Dict[str, dict] = {
    "payphone":          {"ru": "таксофон", "en": "payphone", "cap": False, "near": None},
    "pay phone":         {"ru": "таксофон", "en": "payphone", "cap": False, "near": None},
    "phone booth":       {"ru": "телефонная будка", "en": "phone booth", "cap": False, "near": None},
    "pager":             {"ru": "пейджер", "en": "pager", "cap": False, "near": None},
    "beeper":            {"ru": "пейджер", "en": "pager", "cap": False, "near": None},
    "two-way":           {"ru": "двусторонний пейджер", "en": "two-way pager", "cap": False,
                          "near": r"pager|page|beep"},
    "cassette":          {"ru": "кассета", "en": "cassette tape", "cap": False, "near": None},
    "tape deck":         {"ru": "кассетная дека", "en": "tape deck", "cap": False, "near": None},
    "walkman":           {"ru": "плеер Walkman", "en": "Walkman", "cap": False, "near": None},
    "discman":           {"ru": "плеер Discman", "en": "Discman", "cap": False, "near": None},
    "boombox":           {"ru": "бумбокс", "en": "boombox", "cap": False, "near": None},
    "8-track":           {"ru": "восьмидорожечная кассета", "en": "8-track tape", "cap": False, "near": None},
    "vcr":               {"ru": "видеомагнитофон", "en": "VCR", "cap": False, "near": None},
    "vhs":               {"ru": "кассета VHS", "en": "VHS tape", "cap": False, "near": None},
    "betamax":           {"ru": "Betamax", "en": "Betamax", "cap": False, "near": None},
    "camcorder":         {"ru": "видеокамера", "en": "camcorder", "cap": False, "near": None},
    "floppy disk":       {"ru": "дискета", "en": "floppy disk", "cap": False, "near": None},
    "dial-up":           {"ru": "dial-up интернет", "en": "dial-up internet", "cap": False, "near": None},
    "myspace":           {"ru": "MySpace", "en": "MySpace", "cap": True, "near": None},
    "msn":               {"ru": "MSN", "en": "MSN", "cap": True, "near": None},
    "limewire":          {"ru": "LimeWire", "en": "LimeWire", "cap": True, "near": None},
    "napster":           {"ru": "Napster", "en": "Napster", "cap": True, "near": None},
    "ipod":              {"ru": "iPod", "en": "iPod", "cap": False, "near": None},
    "blackberry":        {"ru": "BlackBerry", "en": "BlackBerry", "cap": True, "near": None},
    "motorola":          {"ru": "Motorola", "en": "Motorola", "cap": True, "near": None},
    "razr":              {"ru": "Motorola Razr", "en": "Motorola Razr", "cap": True, "near": None},
    "nokia":             {"ru": "Nokia", "en": "Nokia", "cap": True, "near": None},
    "nextel":            {"ru": "Nextel", "en": "Nextel", "cap": True, "near": None},
    "sidekick":          {"ru": "Sidekick", "en": "Sidekick", "cap": True, "near": None},
    "ichat":             {"ru": "iChat", "en": "iChat", "cap": False, "near": None},
    "blockbuster":       {"ru": "видеопрокат Blockbuster", "en": "Blockbuster", "cap": True, "near": None},
    "video store":       {"ru": "видеопрокат", "en": "video rental store", "cap": False, "near": None},
    "polaroid":          {"ru": "полароид", "en": "Polaroid", "cap": False, "near": None},
    "answering machine": {"ru": "автоответчик", "en": "answering machine", "cap": False, "near": None},
    "rotary phone":      {"ru": "дисковый телефон", "en": "rotary phone", "cap": False, "near": None},
    "flip phone":        {"ru": "раскладушка", "en": "flip phone", "cap": False, "near": None},
    "jukebox":           {"ru": "музыкальный автомат", "en": "jukebox", "cap": False, "near": None},
    "phonograph":        {"ru": "фонограф", "en": "phonograph", "cap": False, "near": None},
    "gramophone":        {"ru": "граммофон", "en": "gramophone", "cap": False, "near": None},
}

# Lowercased span junk: never a person/brand no matter the confidence score.
STOPWORDS = frozenset(
    """
    i you me he she it we they him her them us my your his their our
    baby babe girl boy man men woman women lady shawty honey darling sweetheart
    mama momma mom dad papa daddy brother sister son daughter
    nigga niggas bitch bitches homie homies dawg bro sis
    love heart soul mind body eyes night day time life world
    mr mrs dj mc
    """.split()
)

# Library-artist names that are ordinary words/phrases — plain matching is
# meaningless for them (dry-run counter-examples in comments).
GENERIC_ARTIST_WORDS = frozenset({
    "the game",      # 204 hits as the phrase "the game"
    "game", "offset", "brick", "the family", "family", "the doors", "doors",
    "justice", "topic", "in flames", "bizarre",  # Tame Impala's "feeling is bizarre"
    "cars", "air", "war", "free", "love", "police", "the police", "genesis",
    "queen", "madness", "poison", "heart", "kiss", "berlin", "america",
    "boston", "chicago", "europe", "asia", "hole", "blur", "pulp", "cream",
    "traffic", "sublime", "garbage", "live", "fuel", "train", "eagles",
    "journey", "rush", "muse", "seal", "pink", "logic", "future", "common",
    "tank", "monica", "brandy", "eve", "wow", "yes", "no", "who", "the who",
})

_popculture_cache: Optional[Dict[str, dict]] = None


def load_popculture() -> Dict[str, dict]:
    """{alias_lower: {canonical, type, work?}} from the curated JSON."""
    global _popculture_cache
    if _popculture_cache is None:
        path = _DATA_DIR / "popculture_gazetteer.json"
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        _popculture_cache = {k.lower(): v for k, v in raw.items()}
    return _popculture_cache


def capsule_hits(cleaned_lyrics: str, original_lyrics: str, title: str) -> list[dict]:
    """All capsule matches in a track's lyrics, guards applied."""
    from app.services.lyric_gems.preprocess import appears_capitalized, find_quote

    low = cleaned_lyrics.lower()
    title_low = (title or "").lower()
    hits = []
    for item, meta in CAPSULE.items():
        if item in title_low:
            continue  # "Payphone" by Maroon 5 — the mention is not a surprise
        m = re.search(r"(?<![\w'])" + re.escape(item) + r"(?![\w'])", low)
        if not m:
            continue
        if meta["near"]:
            lo, hi = max(0, m.start() - 60), m.end() + 60
            if not re.search(meta["near"], low[lo:hi]):
                continue
        if meta["cap"] and not appears_capitalized(item, original_lyrics):
            continue
        quote = find_quote(m.group(0), original_lyrics)
        if not quote:
            continue
        hits.append({
            "kind": "capsule",
            "canonical": meta["en"],
            "display": meta["en"],
            "quote": quote,
            "detail": {"item": item, "ru": meta["ru"]},
            "score": 1.0,
        })
    return hits
