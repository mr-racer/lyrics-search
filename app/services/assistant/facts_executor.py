"""The ``facts`` intent — "расскажи про этот трек / про этого артиста".

The one genuinely new branch of the assistant. Design principle: on a 12b model
the LLM decides NOTHING except the wording. Five steps, four of them pure code:

1. **Resolve the subject** — ``catalog_search_service`` (entity mode) over the
   spans GLiNER already extracted. A thin margin between the top two candidates
   produces a ``disambiguate`` frame instead of a guess.
2. **Build a numbered grounding pack** — RAW source facts (songfacts.com
   stories, Genius descriptions and line annotations — selected, deduped and
   sentence-cropped in code), credits, gems, bio, AudioDB, lyrics. Zero LLM
   involvement. The ``[n]`` numbering is the whole anti-hallucination
   mechanism. The refined one-liners are a display store, not grounding: packs
   built from them produced answers that recited the cleaned list back.
3. **One ``ask_llm(parse_json=True)`` call** returning
   ``{"answer": ..., "used": [n, …]}``. Not pydantic-ai, not tool calling — the
   repo has been burned twice by small models mangling tool-call syntax.
4. **Code verifies the citations.** Empty ``used``, out-of-range indices or an
   empty answer ⇒ the answer is thrown away and the pack is rendered verbatim.
   An ungrounded paragraph physically cannot reach the user.
5. **The web is a code decision, not a model decision.** A thin pack triggers
   ``smart_web_search`` before step 3 and its snippets join the pack numbered.
   "Should I search?" is exactly the judgement small models fail at.

**Explain mode** (``focus_fact`` set — the listener tapped one statement and
asked what it means) is the same five steps with three differences, because the
question is different: *this line*, not *this subject*.

* The pack is **narrowed to the fact** before the model sees it. Handing over
  all eighteen items is what produced the failure this mode exists to fix: asked
  to explain "«A» сэмплирует «B»", the model dutifully recited the release year,
  the label and four unrelated trivia, none of which explain anything.
* **Silence is a valid answer.** The prompt tells the model to return an empty
  answer when the evidence only restates the fact, and the citation gate turns
  that into an honest "no explanation found" rather than a fallback fact dump.
* **The web search is targeted and query-generated.** Up to three searches,
  their queries written by the model under a spec with worked examples — a small
  model left to invent a query types the listener's Russian sentence into a
  search box and gets nothing back.

``track_chat_service`` is deliberately left alone: inside the player it already
has the track in hand and works.
"""

from __future__ import annotations

import asyncio
import logging
import re

from app.services.assistant.humanize import human
from app.services.assistant.llm_json import parse_json_object as _parse_json_object
from app.services.text_normalize import fold, tokenize

logger = logging.getLogger(__name__)

# A pack thinner than this is treated as "we don't really know anything" and
# earns a web lookup before the LLM ever sees it.
MIN_PACK_ITEMS = 3
MAX_WEB_SEARCHES = 2

# ── explain mode ─────────────────────────────────────────────────────────────
# The listener asked what ONE statement means, so the budget is spent on that
# statement instead of on breadth.
# Three searches, because the angles genuinely differ (the relationship, the
# story behind it, the subject itself) and a fact that none of the three
# explains is a fact the web does not explain.
EXPLAIN_MAX_WEB_SEARCHES = 3
# Enough snippets to stop early: past this another query buys context the 12b
# will not read anyway.
EXPLAIN_ENOUGH_SNIPPETS = 5
# Ceiling on the narrowed pack. Deliberately far below MAX_PACK_ITEMS: the
# failure mode being fixed here is precisely "too much unrelated material".
EXPLAIN_MAX_EVIDENCE = 10
# A library item joins the evidence when it shares this many content tokens with
# the fact. One is noise ("the", "песня"); two means it is about the same thing.
EXPLAIN_MIN_OVERLAP = 2
# Below this length a fact standing alone cannot contain its own explanation —
# "«A» сэмплирует «B»" states a thing, it does not account for it. Asking the
# model to explain a lone one-liner buys a paraphrase and a wasted round-trip,
# so the web is asked first instead. A longer stored fact often DOES carry the
# story ("…, потому что продюсер записывал партию в подвале"), and that one is
# worth reading before spending three searches on it.
EXPLAIN_SELF_CONTAINED_CHARS = 180
# Hard ceiling on what goes into the prompt — a 12b context filled with 60 facts
# produces worse answers than one filled with the best 18.
MAX_PACK_ITEMS = 18
MAX_FACT_CHARS = 400
# Raw source facts (songfacts.com stories, Genius descriptions/annotations) are
# the material the answer is written FROM, so they get a bigger budget than the
# 400-char cap used for credit lines — cropping a story mid-anecdote is exactly
# the "уже кропнутые факты" failure this branch was rebuilt to avoid. Cut on a
# sentence boundary, never mid-word.
RAW_FACT_CHARS = 700
# A fact-rich song carries 60-80 raw rows (measured on prod: Bohemian Rhapsody
# 71, Monster 78). The pack takes the best few of each shape and leaves room
# for credits, samples, gems, catalog bits and lyrics under MAX_PACK_ITEMS.
MAX_RAW_STORIES = 7        # songfacts.com paragraphs + Genius descriptions
MAX_RAW_ANNOTATIONS = 5    # Genius line annotations
# Two sources retell the same anecdote often enough to matter; a near-duplicate
# burns a pack slot AND makes the model cite the same story twice.
DEDUPE_OVERLAP = 0.6
# Below this a "story" is a stub, not a story — the prod Bowie pool opens with
# the row "January 8, 1947 - January 10, 2016", which would burn a slot.
MIN_RAW_CHARS = 60
# When a pool overflows its cap, keep the first rows (the source's own lead —
# songfacts puts the headline anecdotes first) and spread the rest evenly, so
# an artist's chronological pool isn't all childhood and a song's annotations
# aren't all the first verse. Measured on prod: Bowie's first seven rows were
# birth, name, school; Bohemian Rhapsody's first five annotations never left
# the intro.
RAW_LEAD_KEEP = 3
MAX_LYRICS_CHARS = 700
MAX_RELATED_TRACKS = 8
# Below this score ratio against the top hit, a runner-up is not a real rival
# and we resolve silently instead of asking.
DISAMBIGUATE_RATIO = 0.75

_LANG_NAMES = {"ru": "Russian", "en": "English"}


def _lang_name(lang: str | None) -> str:
    return _LANG_NAMES.get((lang or "en").strip().lower(), "English")


def _is_ru(lang: str | None) -> bool:
    """Tolerates "RU" / "ru-RU" — a bare ``lang == "ru"`` silently emitted
    English captions inside an otherwise Russian answer."""
    return (lang or "").strip().lower().startswith("ru")


_SYSTEM = """You answer a listener's question about a specific artist or song, using ONLY the numbered facts provided.

Write the answer in {lang_name}.

HARD RULES:
- Every statement you make must come from a numbered fact below. If the facts do not cover something, do not say it — silence is correct, invention is not.
- Never add dates, numbers, chart positions, collaborators or backstory that are not in the facts.
- Cite by listing in "used" every fact number you actually relied on. A fact you did not use must not appear there.
- If the facts are too thin to answer the question, say so plainly in one sentence and put the numbers of whatever facts you did mention in "used".
- Song, album, band and label names must be copied EXACTLY as they appear in the facts — never translated, transliterated or declined, regardless of the answer language. A PERSON's name may follow the natural grammar of the answer language (Меркьюри отказался; для Брайана Мэя), but never change WHICH name it is and never invent one.

HOW TO BUILD THE ANSWER — three steps, in this order:
1. CONNECT. Several facts usually tell parts of ONE story: the same recording session, the same feud, the same sample, the same film. Find those threads first and merge each into a single narrative — never retell the facts one by one in list order.
2. SELECT. Keep only the 2-4 most interesting threads. Interesting = a story with people acting (someone refused, found, hid, fought), a surprise, a vivid concrete detail. Boring = generic praise, awards lists, chart numbers without a story, encyclopedic summaries. Drop boring threads entirely and do not cite them.
3. SHAPE.
   - Broad question («расскажи про», "tell me about", «чем интересен») → 2-4 short thematic blocks. Each block: a bold mini-heading of 1-3 words in {lang_name} (**Запись**, **Слова**, **Клип** — name it after the thread, these are examples, not a fixed set), then 1-3 sentences weaving that thread's facts together. Blank line between blocks.
   - Narrow question (who produced it, what year, the story of the video, what a line means) → answer ONLY that thread: 1-4 sentences, direct, no headings. Facts about other sides of the song are not an answer to a narrow question — do not add them as bonus blocks or closing context, and do not cite them.

STYLE:
- Lead with the substance. No preamble, and never open with a general description ("an English rock band formed in 1985" is the least interesting sentence you can write).
- Sound like a well-read friend telling stories, not an encyclopedia entry.
- Say each thing once. No closing summary.

FOLLOW-UPS:
- In "follow_ups", write 2 short questions in {lang_name} (each under 60 characters) that the listener would naturally ask NEXT — digging deeper into a thread you mentioned, or opening a strong fact you had no room for.
- Each must be specific to THIS subject and answerable from the facts above. Generic ones ("расскажи ещё", "что дальше?") are useless — never write them.

Output ONLY minified JSON, no prose, no fences. "answer" is a JSON string — use \\n for line breaks and **bold** for block headings:
{{"answer": "...", "used": [1, 3], "follow_ups": ["...", "..."]}}

## Example (broad question)

SONG: Bohemian Rhapsody — Queen
QUESTION: расскажи про эту песню
FACTS:
[1] Freddie Mercury wrote the lyrics, and there has been a lot of speculation as to their meaning. Many of the words appear in the Qu'ran: "Bismillah" literally means "In the name of Allah". "Scaramouch" is a boastful coward, "Beelzebub" one of the names of the Devil. Mercury was always vague about the meaning, admitting only that it was "about relationships".
[2] The backing track came together quickly, but Queen spent days overdubbing vocals on a 24-track machine: about 180 tracks were layered and bounced into sub-mixes. Brian May recalled being able to see through the tape, worn thin by overdubs.
[3] Producer Roy Thomas Baker recalls Mercury coming into the studio proclaiming: "oh, I've got a few more 'Galileos' dear!"
[4] Queen's manager played it to Elton John, who declared: "are you mad? You'll never get that on the radio!" The label pleaded to cut the six-minute single; Mercury refused.
[5] In the UK it went to #1 on November 29, 1975 and stayed for nine weeks, a record at the time.
[6] It got a whole new audience when it was used in Wayne's World (1992): re-released, it charted at #2 in the US.
[7] The video was shot in three hours for £3,500 and started the UK trend of making videos instead of live TV appearances.
[8] Queen fans, and also Brian May, colloquially refer to the song as "Bo Rhap".

GOOD ANSWER (structure to imitate; write yours in {lang_name}):
{example_answer}

Why it is good: facts 2 and 3 merged into one recording thread; 4, 5 and 6 became one arc about the single's fate; 7 and 8 were left out as weaker — and are NOT in "used"."""

# The few-shot answer in the listener's language: a 12b model imitates the
# example's language as eagerly as its structure, so showing it a Russian
# answer under an English instruction is how English replies to «расскажи про»
# stop happening.
# NB: single braces here on purpose — these strings are the VALUE substituted
# into _SYSTEM.format(), not part of the format template itself.
_EXAMPLE_ANSWERS = {
    "ru": ('{"answer":"**Запись**\\nПесню собирали как оперу: около 180 вокальных дорожек '
           'наложили друг на друга, и плёнка местами протёрлась насквозь [2]. Меркьюри всё '
           'приходил в студию со словами: «у меня тут ещё пара „Галилео“» [3].\\n\\n'
           '**Слова**\\nТекст полон загадок — Bismillah из Корана, Scaramouch, Beelzebub; '
           'сам Меркьюри так и не объяснил смысл, отделываясь фразой «это про отношения» [1].'
           '\\n\\n**Судьба сингла**\\nЛейбл умолял урезать шесть минут, Elton John пророчил, '
           'что радио это не возьмёт — Меркьюри отказался [4]. Итог: девять недель на первом '
           'месте в Британии [5], а в 1992-м «Wayne\'s World» вернул песню в чарты США [6].",'
           '"used":[1,2,3,4,5,6],'
           '"follow_ups":["Почему лейбл был против шести минут?",'
           '"Как снимали то самое видео?"]}'),
    "en": ('{"answer":"**The recording**\\nThe song was built like an opera: about 180 vocal '
           'tracks were layered until the tape wore thin enough to see through [2], and Mercury '
           'kept walking in announcing \\"a few more \'Galileos\'\\" [3].\\n\\n**The words**\\n'
           'The lyrics are a riddle — Bismillah from the Qu\'ran, Scaramouch, Beelzebub — and '
           'Mercury never explained them beyond \\"it\'s about relationships\\" [1].\\n\\n'
           '**The single\'s fate**\\nThe label begged to cut the six minutes and Elton John said '
           'radio would never play it — Mercury refused [4]. It sat at UK #1 for nine weeks [5], '
           'and in 1992 Wayne\'s World sent it back up the US charts [6].",'
           '"used":[1,2,3,4,5,6],'
           '"follow_ups":["Why did the label fight the six minutes?",'
           '"How was the famous video shot?"]}'),
}


def _system_prompt(lang: str) -> str:
    """The main-branch system prompt with the few-shot answer in the right language."""
    key = "ru" if _is_ru(lang) else "en"
    return _SYSTEM.format(lang_name=_lang_name(lang), example_answer=_EXAMPLE_ANSWERS[key])

# Appended to the prompt for the single retry after an uncitable first answer.
_RETRY_SUFFIX = (
    'Your previous reply was not usable. Output the JSON object and NOTHING else: '
    'no sentence before it, no sentence after it, no code fence. '
    'The object must have exactly two keys, "answer" and "used", and "used" must '
    'list the numbers of the facts your answer relies on.'
)


# ── explain mode prompts ─────────────────────────────────────────────────────

# The one rule that matters here is the empty answer. Without it a small model
# always produces *something*: asked what "«A» сэмплирует «B»" means, it lists
# the release year, the label and three unrelated trivia, because every one of
# those is a true sentence about the same track. An explanation it does not have
# is exactly what the listener asked for, so "nothing" has to be a legal reply.
_EXPLAIN_SYSTEM = """A listener tapped ONE statement in the app and asked what it means. Explain THAT statement — nothing else.

Write the answer in {lang_name}.

The statement is given under FACT. The numbered lines under EVIDENCE are the only material you may use.

HARD RULES:
- Explain the FACT itself: what it means, how it came about, what it changes for someone listening. Nothing else is being asked.
- Every claim must come from a numbered evidence line. Never add a date, a number, a name, a chart position or a backstory that is not there.
- Evidence about the same artist or song but NOT about this fact is not material. Ignore it completely. Do not list other facts, do not "add context" with them, do not close on a general remark about the artist.
- If the evidence does not actually explain the FACT — if it only repeats it in other words, or only talks about other things — then answer with exactly {{"answer":"","used":[]}}. This is a correct outcome, not a failure. An honest nothing beats a plausible invention.
- Copy any artist, song, album, producer or label name EXACTLY as it appears in the evidence — never translated, transliterated or declined.

STYLE (when you do have an explanation):
- 1-4 sentences. Open with the substance, not with a restatement of the fact.
- No preamble, no "this is interesting because", no closing summary.
- Sound like a well-read friend telling you the story, not an encyclopedia entry.

Output ONLY minified JSON, no prose, no fences:
{{"answer": "...", "used": [2, 4]}}"""

# Small models are bad at this specific thing: asked for a search query they
# retype the listener's conversational Russian ("а что это вообще значит") into
# the box and get nothing. So the spec is concrete — a length, a shape, three
# named angles — and every rule comes with a worked example.
_WEB_QUERY_SYSTEM = """You turn ONE statement about music into web-search queries that would find a page explaining it. You output queries only. You never answer the question yourself, you never comment.

Output ONLY minified JSON, no prose, no fences:
{"queries": ["...", "...", "..."]}

## What a usable query looks like
- 3 to 8 words. No question words, no punctuation, no quotation marks, no operators (AND, OR, site:).
- Worded the way a music journalist would TITLE the page you want — not the way a person asks a friend.
- Contains at least one proper name taken from the statement: the artist, the song, the album, the producer, the label.
- In English, unless a name is natively written in another script — then keep that name in its own script and put the rest of the query in English.
- The three queries must attack from three different angles. Three rewordings of the same query waste all three searches.

## The three angles, in this order
1. THE RELATIONSHIP — the two named things plus the word that connects them: sample, interpolation, cover, remix, produced, wrote, featured.
2. THE STORY — the thing that needs explaining plus a word that promises an account of it: meaning, story, behind, origin, interview, history, making of.
3. THE SUBJECT — the song or artist alone plus the topic of the statement.

## Never
- Never conversational: not "why did Kanye sample this", not "what does it mean that".
- Never the listener's own sentence, and never a query that names nothing.
- Never invent a name, a year, an album or a label that is not in the statement.
- Never add "lyrics", "mp3", "download", "listen", "official video".

## Examples

FACT: "Runaway" by Kanye West contains a sample of "Expo 83" by Backyard Heavies
{"queries": ["Kanye West Runaway Expo 83 sample", "Runaway Kanye West sample story", "Backyard Heavies Expo 83 sampled"]}

FACT: «Smells Like Teen Spirit» спродюсировал Butch Vig
{"queries": ["Butch Vig Smells Like Teen Spirit production", "Smells Like Teen Spirit recording sessions story", "Nevermind Butch Vig production interview"]}

FACT: The lyrics of "Hurt" reference a needle
{"queries": ["Nine Inch Nails Hurt lyrics meaning", "Trent Reznor Hurt song origin interview", "Hurt Nine Inch Nails behind the song"]}

FACT: «Кино» — «Группа крови» вышла в 1988 году на лейбле Мелодия
{"queries": ["Кино Группа крови album 1988 Melodiya", "Группа крови album recording history", "Viktor Tsoi Gruppa krovi album story"]}"""


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


def _hit_name(hit: dict) -> str:
    """The one string a listener would have typed to mean this hit."""
    kind = hit.get("type") or "song"
    if kind == "artist":
        return hit.get("artist") or ""
    if kind == "album":
        return hit.get("album") or ""
    return hit.get("title") or ""


def _named_in_query(hit: dict, query: str) -> bool:
    """True when the hit's own name appears in the question word for word.

    Token-contiguous rather than substring: "Hurt" must not claim a question
    about "Hurting", and a one-word title must not match a random word of a
    long sentence unless it stands there as itself.
    """
    name_tokens = tokenize(_hit_name(hit))
    query_tokens = tokenize(query)
    if not name_tokens or len(name_tokens) > len(query_tokens):
        return False
    span = len(name_tokens)
    return any(query_tokens[i:i + span] == name_tokens
               for i in range(len(query_tokens) - span + 1))


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

    # A name the user typed in full outranks BM25F arithmetic. Measured on the
    # production library: «о чём песня Bohemian Rhapsody» asked the listener to
    # choose between "Bohemian Rhapsody" and "Bed Chem", and «чем интересен
    # альбом OK Computer» offered four rivals — four questions out of eight
    # never reached the LLM at all. Two or more exact matches (a library with
    # four different tracks called "Runaway") is real ambiguity and still asks.
    exact = [h for h in hits if _named_in_query(h, query)]
    if len(exact) == 1:
        return _subject_from_hit(exact[0]), []
    if len(exact) > 1:
        return None, [_subject_from_hit(h) for h in exact[:4]]

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


def _clean_story(text: str, limit: int = RAW_FACT_CHARS) -> str:
    """Whitespace-collapse and cut on a sentence boundary.

    A songfacts paragraph clipped mid-anecdote reads worse than a shorter but
    complete one — and tempts the model to invent the ending. Falls back to the
    hard cut only when no sentence end lands in the back 60% of the budget.
    """
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "), head.rfind(".»"))
    if cut >= limit * 0.4:
        return head[:cut + 1]
    return head.rstrip() + "…"


# Genius annotations arrive as "Lyrics string: <line>. Fact: <story>" — useful
# structure, noisy wording. Rewritten to a compact "Line «…» — story" so the
# pack spends its characters on the story, not on boilerplate.
_ANNOTATION_RE = re.compile(r"^\s*Lyrics string:\s*(?P<line>.*?)\.?\s*Fact:\s*",
                            re.DOTALL)


def _strip_annotation_boilerplate(text: str) -> str:
    m = _ANNOTATION_RE.match(text or "")
    if not m:
        return text or ""
    line = " ".join((m.group("line") or "").split())
    rest = (text or "")[m.end():]
    return f"Line «{line}» — {rest}" if line else rest


def _select_raw_facts(rows: list[dict]) -> list[dict]:
    """Pick the raw source facts worth a pack slot. Pure code, deliberately:

    * stories first (songfacts.com paragraphs, then Genius descriptions) — they
      carry the anecdotes the answer is supposed to be built from;
    * then Genius line annotations, capped harder (a fact-rich song has dozens);
    * near-duplicates across the two sources collapse (Jaccard on content
      tokens): the same story cited twice reads as padding, not grounding.
    """
    stories: list[str] = []
    annotations: list[str] = []
    for row in rows:
        text = (row.get("fact") or "").strip()
        if len(text) < MIN_RAW_CHARS:
            continue
        if (row.get("category") or "") == "genius_annotation":
            annotations.append(_clean_story(_strip_annotation_boilerplate(text)))
        else:
            stories.append(_clean_story(text))
    picked: list[dict] = []
    seen_tokens: list[set[str]] = []

    def _take(text: str, cap: int, taken: int) -> int:
        if taken >= cap:
            return taken
        tokens = set(_content_tokens(text))
        for prev in seen_tokens:
            union = tokens | prev
            if union and len(tokens & prev) / len(union) >= DEDUPE_OVERLAP:
                return taken
        seen_tokens.append(tokens)
        picked.append({"text": text, "source": "facts"})
        return taken + 1

    # A couple of spread candidates beyond the cap, so a dedupe hit doesn't
    # leave a slot empty.
    taken = 0
    for text in _lead_and_spread(stories, MAX_RAW_STORIES + 2):
        taken = _take(text, MAX_RAW_STORIES, taken)
    taken = 0
    for text in _lead_and_spread(annotations, MAX_RAW_ANNOTATIONS + 2):
        taken = _take(text, MAX_RAW_ANNOTATIONS, taken)
    return picked


def _lead_and_spread(texts: list[str], cap: int) -> list[str]:
    """First RAW_LEAD_KEEP rows as-is, the rest sampled evenly to fill ``cap``.

    Keeps the source's own lead (songfacts orders by editorial weight) while
    still reaching the back of a long chronological pool. Deterministic — the
    same subject always builds the same pack.
    """
    if len(texts) <= cap:
        return texts
    head = texts[:RAW_LEAD_KEEP]
    rest = texts[RAW_LEAD_KEEP:]
    want = cap - len(head)
    step = len(rest) / want
    return head + [rest[min(len(rest) - 1, int(i * step))] for i in range(want)]


def _build_song_pack(subject: dict, collection_name: str, lang: str, payload: dict) -> list[dict]:
    """Facts, credits and gems for one song. Pure SQLite + the Qdrant payload."""
    from app.resources.metadata_db import MetadataDB
    from app.services.song_facts_service import get_song_facts_key

    items: list[dict] = []
    ru = _is_ru(lang)
    artist = subject.get("artist") or payload.get("artist") or ""
    title = subject.get("title") or payload.get("title") or ""
    slug = get_song_facts_key(artist, title) if (artist and title) else ""

    # RAW source facts, not the refined one-liners. The refined store keeps the
    # home strip readable, but as LLM grounding it produced answers that just
    # recited the cleaned list back — the stories the answer should be built
    # from (studio anecdotes, quotes, the why) only live in the originals.
    if slug:
        items.extend(_select_raw_facts(
            MetadataDB.get_song_facts_rich(slug, collection_name)))

    if slug:
        rel = (MetadataDB.get_song_relations_bulk([slug]) or {}).get(slug) or {}
        if rel.get("producer"):
            items.append({"text": f"Продюсер: {rel['producer']}" if ru
                                  else f"Produced by {rel['producer']}", "source": "credits"})
        if rel.get("label"):
            items.append({"text": f"Лейбл: {rel['label']}" if ru
                                  else f"Label: {rel['label']}", "source": "credits"})
        for s in (rel.get("samples") or [])[:4]:
            items.append({"text": (f"Содержит сэмпл: {s}" if ru
                                   else f"Contains a sample of {s}"), "source": "credits"})
        for s in (rel.get("sampled_by") or [])[:4]:
            items.append({"text": (f"Был засэмплирован в: {s}" if ru
                                   else f"Sampled by {s}"), "source": "credits"})

    if subject.get("track_id"):
        for gem in MetadataDB.get_track_gems(subject["track_id"], collection_name)[:5]:
            display = gem.get("display") or gem.get("canonical")
            quote = gem.get("quote")
            if not display:
                continue
            text = (f"В тексте упоминается {display}" if ru
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

    # RAW facts here too (see _build_song_pack) — the artist pool is
    # songfacts.com biography episodes, full stories the refined pass shrank.
    items.extend(_select_raw_facts(
        MetadataDB.get_artist_facts_rich(slug, collection_name)))

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

    # An album question gets the ARTIST's pack — there is no per-album fact
    # store — so the handful of items that actually name the record must lead.
    # Prod run: «чем интересен альбом OK Computer» handed the model seven
    # Radiohead trivia items with the one about OK Computer buried at [4].
    if subject.get("kind") == "album":
        items = _prefer_items_naming(items, subject.get("title") or "")

    return items


def _prefer_items_naming(items: list[dict], name: str) -> list[dict]:
    """Stable-sort ``items`` so the ones naming ``name`` come first."""
    tokens = tokenize(name)
    if not tokens:
        return items
    needle = " ".join(tokens)

    def _names_it(item: dict) -> bool:
        return needle in " ".join(tokenize(item.get("text") or ""))

    return [it for it in items if _names_it(it)] + [it for it in items if not _names_it(it)]


def _sample_related_sync(qdrant, collection_name: str, subject: dict) -> list:
    """In-library counterparts of the subject's sample links — nothing else.

    The old behaviour (the artist's whole catalogue under every answer) read as
    filler: tracks that had nothing to do with what was just said. A track
    earns its place under a facts answer only when it is the OTHER SIDE of a
    sample / interpolation / cover of this song and the listener actually has
    it. Artists and albums get no track suggestions at all.
    """
    from app.resources.metadata_db import MetadataDB
    from app.resources.qdrant_utils import light_points
    from app.services.song_facts_service import get_song_facts_key

    if subject.get("kind") != "song":
        return []
    artist = subject.get("artist") or ""
    title = subject.get("title") or ""
    if not (artist and title):
        return []
    subject_slug = get_song_facts_key(artist, title)
    # Both storages, same as everywhere else sample links are read: the
    # normalized table AND the older ``songs.samples_json`` cache — the
    # production library predates the table and keeps ALL its links in the
    # cache (measured: sample_links is empty there, samples_json is not).
    entries: list[dict] = []
    try:
        rel = MetadataDB.get_sample_links(collection_name, subject_slug)
        entries += (rel.get("samples") or []) + (rel.get("sampled_by") or [])
    except Exception:
        logger.warning("[assistant/facts] sample links unavailable", exc_info=True)
    try:
        raw = (MetadataDB.get_song_relations_raw([subject_slug]) or {}).get(subject_slug) or {}
        entries += (raw.get("samples") or []) + (raw.get("sampled_by") or [])
    except Exception:
        logger.warning("[assistant/facts] samples_json unavailable", exc_info=True)
    want: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug") or (
            get_song_facts_key(entry["artist"], entry["song"])
            if entry.get("artist") and entry.get("song") else None)
        if slug and slug != subject_slug and slug not in want:
            want.append(slug)
    if not want:
        return []

    try:
        points = light_points(qdrant, collection_name)
    except Exception:
        return []
    by_slug: dict = {}
    for track_id, payload in points:
        payload = payload or {}
        t = (payload.get("title") or "").strip()
        a = (payload.get("artist") or "").strip()
        if t and a:
            by_slug.setdefault(get_song_facts_key(a, t), track_id)
    ids = [by_slug[s] for s in want
           if s in by_slug and by_slug[s] != subject.get("track_id")]
    if not ids:
        return []
    # Full payloads for the handful of matches: the light mirror strips
    # file_path/duration, and a track row without them is not playable.
    try:
        pts = qdrant.retrieve(collection_name=collection_name,
                              ids=ids[:MAX_RELATED_TRACKS],
                              with_payload=True, with_vectors=False)
    except Exception:
        logger.warning("[assistant/facts] related retrieve failed", exc_info=True)
        return []
    return [{"track_id": str(p.id), **(p.payload or {})} for p in pts or []]


def _sane_followups(raw: object, lang: str) -> list[str]:
    """Model-written next questions, filtered to the ones worth a chip.

    Code owns the caps (the LLM decides the wording, nothing else): ≤3 chips,
    each a real question of sane length, no duplicates, no generic filler.
    """
    obj = _parse_json_object(raw) if not isinstance(raw, dict) else raw
    values = obj.get("follow_ups") if isinstance(obj, dict) else None
    if isinstance(values, str):
        values = [values]
    generic = {fold(g).rstrip("?") for g in
               ("расскажи ещё", "что дальше", "tell me more", "what else")}
    out: list[str] = []
    seen: set[str] = set()
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, str):
            continue
        q = " ".join(value.split()).strip()
        if not (8 <= len(q) <= 90):
            continue
        key = fold(q).rstrip("?")
        if not key or key in seen or key in generic:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= 3:
            break
    return out


# ── Step 5: web fill-in (a CODE decision) ────────────────────────────────────


def _web_queries(subject: dict, message: str) -> list[str]:
    """Web queries built from the RESOLVED subject, not the raw message.

    The message is often conversational Russian ("а расскажи чё там у неё") —
    useless as a search query. Ordered best-angle-first; the caller stops at the
    first one that returns anything.
    """
    title = subject.get("title") or ""
    artist = subject.get("artist") or ""
    kind = subject.get("kind")
    if kind == "artist":
        return [f"{title} musician biography", f"{title} band history"]
    if kind == "album":
        return [f"{title} {artist} album", f"{title} {artist} album review"]
    return [f"{artist} {title} song meaning background",
            f"{artist} {title} song facts"]


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


# ── explain mode: narrowing the pack and building its queries ────────────────

# Words that carry no topic. Kept deliberately short: this list only has to stop
# a shared "песня"/"the" from counting as evidence that two texts are about the
# same thing, and every word added here is a word the overlap can no longer see.
_STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "у", "о", "об", "от", "до", "за", "по", "из",
    "что", "это", "эта", "этот", "как", "для", "же", "а", "но", "не", "то", "тот",
    "был", "была", "было", "были", "есть", "его", "её", "их", "там", "тут",
    "песня", "песни", "песне", "трек", "треке", "трека", "альбом", "альбома",
    "группа", "группы", "артист", "артиста", "исполнитель",
    "the", "a", "an", "of", "in", "on", "at", "by", "for", "to", "and", "or",
    "is", "was", "were", "are", "with", "from", "this", "that", "it", "its", "as",
    "song", "songs", "track", "tracks", "album", "band", "artist",
}

# Stripped out of a generated query before it is sent: engine operators and the
# punctuation a chatty model likes to decorate its queries with.
_QUERY_NOISE = re.compile(r'(?i)\b(?:site|inurl|filetype|intitle):\S*|["“”«»?!,;:()\[\]]')


def _content_tokens(text: str) -> list[str]:
    """Topic-bearing tokens: folded, de-noised, short words dropped."""
    return [t for t in tokenize(text) if len(t) > 2 and t not in _STOPWORDS]


def _fact_evidence(items: list[dict], focus_fact: str,
                   subject_tokens: set[str]) -> list[dict]:
    """The fact first, then only the pack items that are about THAT fact.

    ``subject_tokens`` (the artist and title) are removed from the needle on
    purpose: every item in a song's pack names the song, so counting those
    tokens would readmit the whole pack and rebuild the exact failure this
    narrowing exists to prevent.
    """
    fact_text = _clean(focus_fact)
    needle = set(_content_tokens(focus_fact)) - subject_tokens
    if not needle:
        # A fact made only of the subject's own name ("Runaway — Kanye West")
        # has no distinctive words; fall back to matching on the name itself
        # rather than admitting everything.
        needle = set(_content_tokens(focus_fact))

    fact_key = fold(fact_text)
    scored: list[tuple[int, dict]] = []
    for item in items:
        text = item.get("text") or ""
        if fold(text) == fact_key:
            continue                      # the fact itself is already item [1]
        overlap = len(needle & set(_content_tokens(text)))
        if overlap >= EXPLAIN_MIN_OVERLAP:
            scored.append((overlap, item))
    scored.sort(key=lambda pair: -pair[0])
    return ([{"text": fact_text, "source": "facts"}]
            + [item for _, item in scored[:EXPLAIN_MAX_EVIDENCE - 1]])


def _sane_queries(raw: object, focus_fact: str, subject: dict) -> list[str]:
    """Model-written queries, filtered down to the ones worth a search.

    The rejected shapes are the ones observed from small models: the listener's
    own sentence retyped verbatim, a single word, a paragraph, and three
    rewordings of the same query. A query naming nothing from the fact is
    dropped outright — it would spend one of three searches on a guess.
    """
    obj = _parse_json_object(raw)
    values = obj.get("queries") if isinstance(obj, dict) else None
    if isinstance(values, str):
        values = [values]
    allowed = (set(_content_tokens(focus_fact))
               | set(_content_tokens(subject.get("title") or ""))
               | set(_content_tokens(subject.get("artist") or "")))

    out: list[str] = []
    seen: set[str] = set()
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, str):
            continue
        query = " ".join(_QUERY_NOISE.sub(" ", value).split())
        words = query.split()
        if not (2 <= len(words) <= 12):
            continue
        if not (set(_content_tokens(query)) & allowed):
            continue
        key = fold(query)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(query)
        if len(out) >= EXPLAIN_MAX_WEB_SEARCHES:
            break
    return out


def _fallback_fact_queries(subject: dict, focus_fact: str) -> list[str]:
    """Deterministic queries for when the model wrote none we could use.

    Same three angles as the prompt asks for, assembled in code, so an
    unreachable or unhelpful LLM still gets one honest attempt at the web.
    """
    artist = subject.get("artist") or ""
    title = subject.get("title") or ""
    head = " ".join(part for part in (artist, title) if part).strip()
    head_tokens = set(_content_tokens(head))
    distinctive = [t for t in _content_tokens(focus_fact) if t not in head_tokens][:4]
    tail = " ".join(distinctive)

    queries = []
    if head and tail:
        queries.append(f"{head} {tail}")
    if head:
        queries.append(f"{head} band history" if subject.get("kind") == "artist"
                       else f"{head} song story meaning")
    if tail:
        queries.append(f"{tail} music history")
    return [q for q in queries if q][:EXPLAIN_MAX_WEB_SEARCHES]


# ── Step 3+4: the single LLM call and its verification ───────────────────────


def _render_pack(items: list[dict], lang: str) -> str:
    return "\n".join(f"[{i}] {it['text']}" for i, it in enumerate(items, 1))


def _deterministic_answer(subject: dict, items: list[dict], lang: str) -> str:
    """The fallback served whenever the LLM answer fails verification.

    Not an error message — a real, useful reply built only from stored facts.
    """
    ru = _is_ru(lang)
    name = subject.get("title") or ""
    if not items:
        return (f"Про «{name}» у меня пока нет достоверных сведений."
                if ru else f"I don't have reliable information about “{name}” yet.")
    head = (f"Вот что известно про «{name}»:" if ru
            else f"Here's what is known about “{name}”:")
    # Facts first and only five of them: the fallback used to open with the
    # 900-character biography blob and then list everything, which read as a
    # database dump rather than an answer.
    ranked = ([it for it in items if it.get("source") == "facts"]
              + [it for it in items if it.get("source") != "facts"])
    # Raw facts run to 700 chars now — as bullets they must stay skimmable.
    bullets = "\n".join(f"- {_clean(it['text'], 300)}" for it in ranked[:5])
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


def _declined(raw: object) -> bool:
    """Did the model deliberately answer "nothing here explains this"?

    ``{"answer": "", "used": []}`` is the reply the explain prompt asks for when
    the evidence does not explain the fact. It fails :func:`_verify` exactly like
    a malformed answer does, and the difference matters: a malformed answer is
    worth one stricter retry, a deliberate refusal is the correct outcome and
    retrying it is how a small model gets talked into inventing one.
    """
    obj = _parse_json_object(raw)
    return isinstance(obj, dict) and "answer" in obj and not str(obj["answer"] or "").strip()


def _no_explanation(lang: str | None) -> str:
    """The honest empty answer. Not an error — the requested outcome."""
    if _is_ru(lang):
        return ("Объяснения этому факту я не нашёл — ни в твоей библиотеке, ни в "
                "интернете. Сам факт остаётся, а придумывать причину не буду.")
    return ("I couldn't find an explanation for this — not in your library, not on "
            "the web. The fact stands; I'm not going to invent a reason for it.")


# ── Orchestration: explain one fact ──────────────────────────────────────────


async def _ask_explain(evidence: list[dict], *, subject: dict, focus_fact: str,
                       message: str, lang: str, llm_base_url, llm_model):
    """One explain call. Returns ``(answer, used)``, or None for "no explanation".

    None covers both "the model declined" (the correct, expected outcome when
    nothing explains the fact) and "the model produced something uncitable" —
    the second gets one stricter retry, the first does not.
    """
    from app.services.llm_client import ask_llm

    subject_line = f"{subject['kind'].upper()}: {subject.get('title')}"
    if subject.get("subtitle"):
        subject_line += f" — {subject['subtitle']}"
    prompt = (
        f"{subject_line}\n\nFACT: {focus_fact}\n\n"
        f"LISTENER ASKED: {(message or '').strip()[:200]}\n\n"
        f"EVIDENCE:\n{_render_pack(evidence, lang)}"
    )
    system = _EXPLAIN_SYSTEM.format(lang_name=_lang_name(lang))

    try:
        raw = await ask_llm(prompt, system_prompt=system, parse_json=False,
                            temperature=0.2, base_url=llm_base_url, model=llm_model,
                            extra_body={"enable_thinking": False})
        verified = _verify(_parse_json_object(raw), len(evidence))
        if verified is not None:
            return verified
        if _declined(raw):
            logger.info("[assistant/facts] model declined to explain — evidence is thin")
            return None
        # Uncitable but not a refusal: the same non-deterministic wrapping the
        # main branch retries once, at temperature 0.
        raw = await ask_llm(prompt + "\n\n" + _RETRY_SUFFIX, system_prompt=system,
                            parse_json=False, temperature=0.0,
                            base_url=llm_base_url, model=llm_model,
                            extra_body={"enable_thinking": False})
        return _verify(_parse_json_object(raw), len(evidence))
    except Exception:
        logger.exception("[assistant/facts] explain call failed")
        return None


async def _fact_web_queries(subject: dict, focus_fact: str, *,
                            llm_base_url, llm_model) -> list[str]:
    """Up to three search queries for this fact — model-written, code-filtered."""
    from app.services.llm_client import ask_llm

    try:
        raw = await ask_llm(
            f"FACT: {focus_fact}\n"
            f"SUBJECT: {subject.get('title') or ''}"
            + (f" — {subject.get('artist')}" if subject.get("artist") else ""),
            system_prompt=_WEB_QUERY_SYSTEM, parse_json=False, temperature=0.0,
            base_url=llm_base_url, model=llm_model,
            extra_body={"enable_thinking": False},
        )
        queries = _sane_queries(raw, focus_fact, subject)
    except Exception:
        logger.warning("[assistant/facts] query generation failed", exc_info=True)
        queries = []
    if queries:
        return queries
    logger.info("[assistant/facts] no usable model queries — falling back to code-built ones")
    return _fallback_fact_queries(subject, focus_fact)


def _needs_the_web_first(evidence: list[dict], focus_fact: str) -> bool:
    """True when the library provably has nothing to explain this fact with.

    A code decision, like every other "should we search?" in this module: the
    narrowing found no related item, and the fact is a one-liner, so the only
    thing the model could do with it is rephrase it.
    """
    return len(evidence) <= 1 and len(focus_fact) < EXPLAIN_SELF_CONTAINED_CHARS


async def _explain_fact(*, subject: dict, items: list[dict], focus_fact: str,
                        message: str, qdrant, collection_name: str, lang: str,
                        llm_base_url, llm_model, say) -> dict:
    """The ``focus_fact`` branch: explain one statement, or say nothing.

    The library is asked first because an explanation already stored beats three
    web searches; the web is the fill-in, not the default.
    """
    subject_tokens = (set(_content_tokens(subject.get("title") or ""))
                      | set(_content_tokens(subject.get("artist") or "")))
    evidence = _fact_evidence(items, focus_fact, subject_tokens)

    verified = None
    if not _needs_the_web_first(evidence, focus_fact):
        await say("explaining", found=max(0, len(evidence) - 1))
        verified = await _ask_explain(evidence, subject=subject, focus_fact=focus_fact,
                                      message=message, lang=lang,
                                      llm_base_url=llm_base_url, llm_model=llm_model)

    web_used = False
    if verified is None:
        # Nothing stored explains it. Three angles at the web, stopping as soon
        # as there is enough to read — the budget is a ceiling, not a quota.
        queries = await _fact_web_queries(subject, focus_fact,
                                          llm_base_url=llm_base_url, llm_model=llm_model)
        snippets: list[dict] = []
        for query in queries[:EXPLAIN_MAX_WEB_SEARCHES]:
            await say("web_search", query=query)
            raw_web = await asyncio.to_thread(_web_search_sync, query)
            snippets += _snippets_from_web(raw_web, 4)
            if len(snippets) >= EXPLAIN_ENOUGH_SNIPPETS:
                break
        if snippets:
            web_used = True
            evidence = (evidence + snippets)[:MAX_PACK_ITEMS]
            await say("explaining", found=max(0, len(evidence) - 1))
            verified = await _ask_explain(evidence, subject=subject,
                                          focus_fact=focus_fact, message=message,
                                          lang=lang, llm_base_url=llm_base_url,
                                          llm_model=llm_model)

    related = await asyncio.to_thread(_sample_related_sync, qdrant, collection_name, subject)
    base = {
        "subject_kind": subject["kind"],
        "subject_title": subject.get("title") or "",
        "subject_subtitle": subject.get("subtitle"),
        "artist_slug": subject.get("artist_slug"),
        "track_id": subject.get("track_id"),
        "image_path": subject.get("image_path"),
        "focus_fact": focus_fact,
        "web_search_used": web_used,
        "related_tracks": _as_track_models(related),
    }

    if verified is None:
        await say("no_explanation")
        # No items: the whole point of this branch is that an unexplained fact
        # must NOT turn into a list of the other things we happen to know.
        return {**base, "answer": _no_explanation(lang), "grounded": False,
                "explained": False, "items": []}

    answer, used = verified
    used_set = set(used)
    return {
        **base, "answer": answer, "grounded": True, "explained": True,
        "items": [{"n": i, "text": it["text"], "source": it["source"],
                   "used": i in used_set}
                  for i, it in enumerate(evidence, 1)],
    }


# ── Orchestration ────────────────────────────────────────────────────────────


async def run(*, qdrant, collection_name: str, message: str, route, slots,
              lang: str = "en", subject_track_id=None, subject_artist_slug=None,
              now_playing_track_id=None, focus_fact=None,
              llm_base_url=None, llm_model=None, emit=None):
    """Run the facts branch. Returns ``(payload_dict | None, options list)``.

    ``payload_dict`` matches :class:`~app.domain.models.AssistantFactsPayload`.
    A non-empty ``options`` list means the caller should emit ``disambiguate``.

    ``focus_fact`` switches the branch to explain mode: the listener tapped one
    statement and wants THAT explained, so the pack is narrowed to it and an
    unexplained fact stays unexplained (see :func:`_explain_fact`).
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
        if subject["kind"] == "album":
            # There is no per-album fact store, and running the SONG lookup on an
            # album title would either find nothing or — worse — collide with a
            # same-named track. The artist's own facts are what the question is
            # actually about.
            return _build_artist_pack(subject, collection_name, lang)
        return _build_song_pack(subject, collection_name, lang, payload)

    try:
        items = await asyncio.to_thread(_pack_sync)
    except Exception:
        logger.exception("[assistant/facts] pack build failed")
        items = []
    await _say("collecting", found=len(items))

    # ── explain mode: a different question, so a different pack and prompt ──
    focus = (focus_fact or "").strip()
    if focus:
        payload = await _explain_fact(
            subject=subject, items=items, focus_fact=focus, message=message,
            qdrant=qdrant, collection_name=collection_name, lang=lang,
            llm_base_url=llm_base_url, llm_model=llm_model, say=_say,
        )
        return payload, []

    # ── web fill-in: code decides, not the model ──
    # A second angle only when the first one came back empty — the budget is
    # spent by code on a measured shortfall, never on the model's hunch.
    web_used = False
    if len(items) < MIN_PACK_ITEMS:
        for query in _web_queries(subject, message)[:MAX_WEB_SEARCHES]:
            await _say("web_search", query=query)
            raw_web = await asyncio.to_thread(_web_search_sync, query)
            snippets = _snippets_from_web(raw_web)
            if snippets:
                items += snippets
                web_used = True
                break

    items = items[:MAX_PACK_ITEMS]

    related = await asyncio.to_thread(_sample_related_sync, qdrant, collection_name, subject)

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
    followups: list[str] = []
    try:
        # parse_json=False on purpose: ``ask_llm``'s own parse raises on the
        # slightest prose around the object, and a 12b model wrapping its JSON
        # in one polite sentence cost a whole answer on the prod dry run.
        # ``_parse_json_object`` digs the object out instead.
        raw_text = await ask_llm(
            user_prompt,
            system_prompt=_system_prompt(lang),
            parse_json=False, temperature=0.3,
            base_url=llm_base_url, model=llm_model,
            extra_body={"enable_thinking": False},
        )
        parsed = _parse_json_object(raw_text)
        verified = _verify(parsed, len(items))
        if verified is None:
            # The model answers well but wraps it inconsistently: the same
            # question returns prose+JSON one run and bare prose the next, and
            # bare prose has no citations to check, so it must be thrown away.
            # One stricter retry recovers it; a second would just cost seconds.
            logger.info("[assistant/facts] no citable answer — retrying once, strictly")
            raw_text = await ask_llm(
                user_prompt + "\n\n" + _RETRY_SUFFIX,
                system_prompt=_system_prompt(lang),
                parse_json=False, temperature=0.0,
                base_url=llm_base_url, model=llm_model,
                extra_body={"enable_thinking": False},
            )
            parsed = _parse_json_object(raw_text)
            verified = _verify(parsed, len(items))
        if verified is None:
            logger.info("[assistant/facts] answer rejected by citation check — "
                        "serving the deterministic fact rendering instead")
        else:
            # Follow-up chips ride the same call — no second round-trip. They
            # only make sense next to a real answer, never under the fallback.
            followups = _sane_followups(parsed, lang)
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
        "follow_ups": followups,
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
