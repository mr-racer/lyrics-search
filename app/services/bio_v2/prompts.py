"""Prompts for the Wikipedia-first bio and its facets.

Conventions from ``assistant/prompts.py``: worked examples beat rules, the
prompt asks for material rather than decisions, and the output contract is
repeated at the end.
"""

from __future__ import annotations

BIO_PROMPT = """You write the biography paragraph shown on an artist's page in a
music player. Everything you may use is in the PASSAGES below; they come from
that artist's Wikipedia article.

CRITICAL RULE — NAMES. The artist is written exactly "{artist}". Copy that
spelling character for character wherever it appears. Names of other people,
bands, albums, songs, films, labels and brands also keep the script the passages
use: a name in Latin letters stays in Latin letters, undeclined. Countries and
cities are ordinary words — translate those.

WHAT TO WRITE

Two short blocks in {lang}, separated by a blank line.

1. Who they are: where they came from, when they started, what they play, and
   what they are best known for. Three or four sentences.
2. What makes them worth a listen: the one detail from the passages a listener
   would repeat to a friend — a habit, an incident, a way of working, a
   contradiction. One or two sentences. If the passages hold nothing like that,
   leave this block out entirely rather than padding it.

RULES

- Every fact, name, date and number must come from the PASSAGES. If they do not
  say where the artist was born, you do not say it. Inventing a death, a
  break-up or an award is the worst failure possible here.
- If the passages say the artist has died or the band has split, write in the
  past tense. If they are active, write in the present.
- No opening filler, no "по данным Википедии", no source line — the interface
  shows the source itself.
- Do not list a discography. A year matters only when something happened in it.

PASSAGES

{passages}

Write the biography now. Plain text, no headings, no markdown other than a blank
line between the two blocks. Names spelled in Latin letters stay in Latin."""


LANG_RETRY = """ЯЗЫК ОТВЕТА — РУССКИЙ. Ты уже один раз написал этот текст
по-английски. Весь текст, каждое предложение, пишется по-русски. По-английски
остаются ТОЛЬКО имена людей, групп, песен, альбомов и фильмов.

"""


FACET_PROMPTS = {
    "grammy": """From the PASSAGES below, how many Grammy Awards has {artist} WON,
and how many nominations have they received? Count awards to the artist and to
their songs and albums alike.

Answer only from the passages. If they do not state a number, that number is
null — an artist with no Grammy is the normal case, and guessing one is worse
than saying nothing. "Nominated for three" is nominations=3, wins=null, not
wins=3.

PASSAGES

{passages}

Answer with STRICT JSON and nothing else.
{{"wins": <number or null>, "nominations": <number or null>, "evidence": "<the sentence you took it from, or null>"}}""",

    "formed": """From the PASSAGES below, in what YEAR and in what PLACE was
{artist} formed — or, for a solo artist, when and where did their music career
begin?

Answer only from the passages. A year you cannot find is null. A first album is
not a formation date unless the passages say so.

PASSAGES

{passages}

Answer with STRICT JSON and nothing else.
{{"year": <4-digit year or null>, "place": "<city, country or null>", "evidence": "<the sentence you took it from, or null>"}}""",

    "name_origin": """From the PASSAGES below, where does the name "{artist}" come
from, and who chose it?

An origin is often stated obliquely — as an inspiration, a decision, or a thing
the name was taken from. All of those are origins. Only answer null when the
passages say nothing at all about where the name came from.

  passage: "Once the members came together, they settled on the name Franz
  Ferdinand. The name was originally inspired by a racehorse called Archduke
  Ferdinand. After seeing the horse win the Northumberland Plate in 2001, the
  band began to discuss Archduke Franz Ferdinand and thought it would be a good
  band name because of the alliteration."
  → {{"origin": "Название пришло от скаковой лошади Archduke Ferdinand: увидев её победу на Northumberland Plate в 2001-м, музыканты заговорили об эрцгерцоге и оценили аллитерацию.", "evidence": "The name was originally inspired by a racehorse called Archduke Ferdinand."}}

  passage: "M83 is a French electronic group formed in Antibes in 1999.
  Initially a duo of multi-instrumentalists Nicolas Fromageau and Anthony
  Gonzalez…"
  → {{"origin": null, "evidence": null}}

Do not reason about what the words of the name might mean on their own. If the
members disagree about it, that disagreement is the answer.

Write `origin` in {lang}, one sentence, at most 200 characters. Names of people,
places, works and brands stay in the script the passages use.

PASSAGES

{passages}

Answer with STRICT JSON and nothing else.
{{"origin": "<one sentence in {lang}, or null>", "evidence": "<the sentence you took it from, or null>"}}""",

    "years_active": """From the PASSAGES below, when was {artist} active, and what
is their status now?

status is one of:
  "active"    — still working
  "disbanded" — the band has split up
  "hiatus"    — paused, not ended
  "deceased"  — the artist has died

Answer only from the passages. Anything they do not state is null. `to` stays
null while the artist is still active.

PASSAGES

{passages}

Answer with STRICT JSON and nothing else.
{{"from": <4-digit year or null>, "to": <4-digit year or null>, "status": "<one of the four, or null>", "evidence": "<the sentence you took it from, or null>"}}""",
}


