"""Script and shape control for LLM-written text. Shared by facts and bios.

Both pipelines write short Russian prose out of English sources, and both fail
the same three ways, at rates measured on production data:

  * a performer's name comes back in Cyrillic         — 33% before any repair
  * the whole answer comes back in English            — 4.5% of facts, 2 of 4 bios
  * a name that was never in the source appears        — 1.0%

So the rules live in one place and the two callers behave identically. Nothing
here imports the app; it is pure text.

The layering is deliberate. The prompt asks, a repair call is told exactly what
it got wrong, and only then does code substitute — because each layer is
cheaper than the one before it and none of them is sufficient alone.
"""

from __future__ import annotations

import difflib
import re

# ── transliteration, for recognising a name the model rewrote ────────────────

_DIGRAPHS = [("sch", "щ"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"),
             ("ph", "ф"), ("th", "т"), ("ck", "к"), ("ts", "ц"), ("ee", "и"),
             ("oo", "у"), ("ou", "у"), ("ea", "и"), ("ya", "я"), ("yu", "ю"),
             ("ju", "ю"), ("ai", "эй"), ("ey", "ей"), ("ie", "и")]
_MONO = {"a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
         "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
         "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
         "v": "в", "w": "в", "x": "кс", "y": "и", "z": "з"}

# Two thresholds, not one. A two-word match cannot happen by accident, so it can
# be loose; a single word can, so it must be tight. The tight value was chosen
# by the case it has to reject: the bio run matched the ordinary Russian word
# «Теплое» (from the translated film title «Тёплые тела») against the Latin
# `Triple` at 0.667 and rewrote it, producing "фильму «Triple тело»". Raising the
# single-word bar to 0.72 rejects it. It also loses «Гонсалеса»→`Gonzalez` at
# 0.706, and that trade is the right way round: a missed fix leaves a
# transliterated name, a wrong fix invents text.
RATIO_MULTIWORD = 0.62
RATIO_SINGLE = 0.72

_LATIN_TOKEN_RE = re.compile(r"[A-Z][A-Za-z0-9'&.\-]{2,}")
_NAME_RE = re.compile(r"\b[A-Z][A-Za-z0-9'&.\-]+(?:\s+[A-Z][A-Za-z0-9'&.\-]+)*")
_CYR_WORD_RE = re.compile(r"[А-ЯЁ][а-яё]{2,}")
_LETTER_RUN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")
_CYR_RE = re.compile(r"[А-Яа-яЁё]")
_LAT_RE = re.compile(r"[A-Za-z]")
_TOKEN_STRIP = ".,;:!?»«\"'"

# Sentence-initial English words the detectors kept reporting as invented names.
_EN_COMMON = frozenset("""The This That These Those There Their They Them Then Thus
He She His Her Him Hers It Its Are And But For Not You Your Yours When Where
While What Which Who Whom Whose Why How Here Because Since Also Even Just Only
Still Both Each Every Some Was Were Been Being Have Has Had Will Would Could
Should Might May Can One Two Three Now Ever Never Many Much Most More Less Least
Such Another Instead However Although Though Basically Essentially Note Notice
Perhaps Maybe Probably Possibly Likely Rather Actually Literally Additionally
Listeners Throughout During After Before Above Below Between Within Without
Lyrics Fact Line Chorus Verse Bridge Outro Intro Hook Refrain Pre Suggestion
Credit Song Songs Album Track Video Band Music Records Record""".split())

# Russian translates these; they are not performer names and must not be forced
# back into Latin.
_GEO = frozenset("""America American Ireland Irish England English Britain British
Scotland Scottish Wales Welsh France French Germany German Italy Italian Spain
Spanish Japan Japanese China Chinese Russia Russian Canada Canadian Australia
Australian Mexico Mexican Brazil Sweden Swedish Norway Denmark Netherlands Dutch
Poland Polish Africa African Europe European Asia Asian London Paris Berlin
Moscow Tokyo Rome Madrid Vienna Chicago Detroit Seattle Boston Miami Atlanta
Hollywood Manhattan Brooklyn Queens Bronx Vegas Angeles Francisco Orleans York
Christmas Easter Catholic Christian Jewish Muslim Bible God Jesus Heaven Devil
Earth Moon Sun Grammy Grammys Oscar Oscars Guinness Billboard Navy Army
January February March April June July August September October November
December Monday Friday Saturday Sunday Internet""".split())


# Wikipedia section headings ride along inside every chunk (the chunker prepends
# the heading path), so they look exactly like capitalised source words.
_WIKI_HEADINGS = frozenset("""Style Group Career History Members Legacy Life Early
Background Formation Influences Artistry Music Art Personal Reception Impact
Tours Singles Albums Discography Awards Filmography References Overview
Biography Origins Sound Production Songwriting Themes Composition Release
Critical Commercial Charts Certifications Personnel Track Listing Credits
Bibliography Notes Sources External Links Contents Sections Studio Live
Compilation Video Videos Concert Concerts Band Solo Media Press Public Image
Activism Philanthropy Controversies Death Illness Family Education Marriage
Влияние Стиль Группа Биография Творчество Награды Дискография""".split())


def transliterate(word: str) -> str:
    w = word.lower().replace("'", "").replace(" ", "")
    for a, b in _DIGRAPHS:
        w = w.replace(a, b)
    return "".join(_MONO.get(c, c if not c.isascii() else "") for c in w)


# ── detectors ────────────────────────────────────────────────────────────────

def latin_tokens(text: str) -> set:
    return {t.strip(_TOKEN_STRIP) for t in _LATIN_TOKEN_RE.findall(text or "")
            if t.strip(_TOKEN_STRIP)}


def invented_names(out: str, source: str) -> list:
    """Latin names in the output that are absent from the source."""
    src = (source or "").lower()
    return sorted({t for t in latin_tokens(out)
                   if t not in _EN_COMMON and t.lower() not in src})


def garbled_script(text: str) -> bool:
    """A single word mixing alphabets — «Джagger». A hyphen splits runs, so
    "Grammy-номинация" is fine."""
    for run in _LETTER_RUN_RE.findall(text or ""):
        if _CYR_RE.search(run) and _LAT_RE.search(run):
            return True
    return False


_SENT_END = re.compile(r"[.!?:;»\"')\]]\s*$")


def _is_real_proper_noun(word: str, source: str) -> bool:
    """Does this single capitalised word behave like a NAME in the source?

    A real proper noun turns up capitalised in the MIDDLE of a sentence. A
    section heading and a sentence opener do not, and both were mistaken for
    names: the bio run pulled "Style" and "Group" out of Wikipedia's heading
    path — they ride along with every chunk — and then rewrote the ordinary
    Russian words «Стиль» and «Группа» into them, producing "Style группы
    эволюционировал". Enumerating heading words would be endless; the position
    test is the property itself.
    """
    for line in (source or "").split("\n"):
        words = line.split()
        if len(words) <= 3:
            continue                      # a heading, not a sentence
        for i, w in enumerate(words):
            if w.strip(_TOKEN_STRIP) != word:
                continue
            if i == 0:
                continue                  # sentence-initial proves nothing
            if _SENT_END.search(words[i - 1]):
                continue                  # first word of a new sentence
            return True
    return False


def source_names(text: str) -> list:
    """Full names from the source, longest first, possessives normalised.

    Full names, not tokens: offering "Bennett" alone gets «Тони Bennett» back,
    a mixed name that is worse than the transliterated one.
    """
    found, seen = [], []
    for m in _NAME_RE.finditer(text or ""):
        # A name never crosses a sentence boundary. The period is inside the
        # token class so initialisms survive ("D.M.C.", "Dr."), and that let the
        # pattern swallow "Mudcrutch. The Group" and "Florida. Petty" as single
        # names — which is how the ordinary word "Group" reached the candidate
        # list as a "surname", bypassing the heading stop-list entirely.
        for raw in _split_sentences_in_match(m.group(0)):
            _collect_name(raw, found, seen)
    # A lone capitalised word only counts as a name if it behaves like one in
    # the source — capitalised mid-sentence, not merely at a heading or a
    # sentence opening.
    found = [n for n in found
             if len(n.split()) > 1 or _is_real_proper_noun(n, text)]
    return sorted(found, key=lambda n: (-len(n.split()), -len(n)))


def _split_sentences_in_match(raw: str) -> list:
    out, cur = [], []
    for tok in raw.split():
        cur.append(tok)
        if tok.endswith(".") and len(tok.rstrip(".")) > 2:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def _collect_name(raw: str, found: list, seen: list) -> None:
    if True:
        raw = re.sub(r"'s\b", "", raw).strip(" .,;:!?'")
        toks = [t for t in raw.split() if t]
        while toks and toks[-1] in _EN_COMMON:
            toks.pop()
        while toks and toks[0] in _EN_COMMON:
            toks.pop(0)
        if not toks:
            return
        if len(toks) == 1 and not _usable_single(toks[0]):
            return
        name = " ".join(toks)
        if name.lower() in seen:
            return
        seen.append(name.lower())
        found.append(name)


def _usable_single(word: str) -> bool:
    return not (word in _EN_COMMON or word in _GEO or word in _WIKI_HEADINGS
                or len(word) < 4)


def transliterated_names(out: str, source: str) -> list:
    """Source names the model rewrote in Cyrillic, as "Latin→Cyrillic" pairs."""
    names = source_names(source)
    if not names or not _CYR_WORD_RE.search(out or ""):
        return []
    hits = []
    for name in names:
        if name.lower() in (out or "").lower():
            continue
        floor = RATIO_MULTIWORD if len(name.split()) > 1 else RATIO_SINGLE
        target = transliterate(name)
        for cw in set(_CYR_WORD_RE.findall(out)):
            if difflib.SequenceMatcher(None, target, cw.lower()).ratio() >= floor:
                hits.append(f"{name}→{cw}")
                break
    return sorted(set(hits))


def no_target_script(text: str, lang: str) -> bool:
    """For a Russian target, an answer with no Cyrillic at all is a failure."""
    if (lang or "").lower() not in ("ru", "russian"):
        return False
    return not _CYR_RE.search(text or "")


# ── the deterministic layer ──────────────────────────────────────────────────

def restore_latin(text: str, source: str) -> tuple:
    """Put transliterated source names back in Latin. Returns (text, swaps).

    Matches n-grams of Cyrillic capitalised words, longest name first, so
    "Tony Bennett" is tried as a pair before "Bennett" can consume half of it.
    A declension tail is absorbed into the match; a substitution that would
    leave a mixed-script token is refused.
    """
    names = source_names(source)
    if not names or not _CYR_WORD_RE.search(text or ""):
        return text, []

    # Russian prose names a person by surname alone after the first mention, so
    # a two-word source name routinely comes back as one Cyrillic word:
    # "Roger Glover" → «Глэвер». A two-word n-gram cannot match one word, which
    # left those unfixed. Register the surname as its own candidate.
    candidates = list(names)
    for name in names:
        parts = name.split()
        # The surname goes through the same filters as any lone word: without
        # that, "Group" arrived as the surname of a two-word match and rewrote
        # the ordinary Russian «Группа».
        if len(parts) > 1 and _usable_single(parts[-1]):
            candidates.append(parts[-1])
    # No "already present" filter. It looks like a saving and is a bug: a bio
    # can name the band "Tom Petty and the Heartbreakers" in Latin further down
    # while the sentence above says «Том Петти», and excluding the candidate
    # left that span to match "Earl Petty" out of "Thomas Earl Petty".
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return text, []

    # Iterate over the SPANS and take the best candidate for each — not over the
    # candidates taking the first span that clears the floor. First-match cost
    # two invented facts in one run: the Tom Petty article names his mother
    # Kitty Petty, which is longer and so was tried first, and the bio came back
    # as "Kitty Petty — американский певец"; the same way «Майи Харт» became
    # `Mariah Carey`. A margin over the runner-up is what keeps two similar
    # names from swapping places.
    words = list(re.finditer(r"[А-ЯЁ][а-яё]+(?:['’][а-яё]+)?", text))
    spans = []
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            grp = words[i:i + n]
            gaps = [text[grp[k].end():grp[k + 1].start()]
                    for k in range(len(grp) - 1)]
            if any(g.strip() for g in gaps):
                continue                       # not consecutive words
            spans.append((grp[0].start(), grp[-1].end(), n,
                          "".join(w.group(0) for w in grp).lower()))

    chosen, taken = [], []
    for start, end, n, phrase in spans:
        if any(start < e and s < end for s, e, _, _ in taken):
            continue                           # overlaps a longer span already used
        floor = RATIO_MULTIWORD if n > 1 else RATIO_SINGLE
        # A candidate may not be SHORTER than the span it replaces, or the
        # extra Russian word is deleted: «Сегодня Энтони Гонсалес» matched the
        # two-word "Anthony Gonzalez" and the sentence lost «Сегодня».
        scored = sorted(
            ((difflib.SequenceMatcher(None, transliterate(c), phrase).ratio(), c)
             for c in candidates if n <= len(c.split()) <= n + 1),
            reverse=True)
        if not scored or scored[0][0] < floor:
            continue
        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.05:
            continue                           # two names fit equally — pick neither
        chosen.append((start, end, scored[0][1], scored[0][0]))
        taken.append((start, end, n, phrase))

    swaps = []
    for start, end, name, ratio in sorted(chosen, reverse=True):
        tail = re.match(r"[а-яё]+", text[end:])
        real_end = end + (tail.end() if tail else 0)
        candidate = text[:start] + name + text[real_end:]
        if garbled_script(candidate):
            continue                           # refuse to make it worse
        swaps.append((text[start:real_end], name, round(ratio, 2)))
        text = candidate
    return text, list(reversed(swaps))


# ── the one entry point both pipelines use ───────────────────────────────────

def check(text: str, *, source: str, lang: str = "ru",
          max_chars: int | None = None, forbid: str = "") -> dict:
    """Every issue with one piece of generated text. Empty dict means clean.

    ``forbid`` is text that must not appear — the song's own title for a
    song-scope fact, since the listener is already looking at it.
    """
    out = {}
    if not (text or "").strip():
        return {"empty": True}
    inv = invented_names(text, source)
    if inv:
        out["invented"] = inv
    tr = transliterated_names(text, source)
    if tr:
        out["translit"] = tr
    if garbled_script(text):
        out["garbled"] = True
    if no_target_script(text, lang):
        out["wrong_language"] = True
    f = (forbid or "").strip().lower()
    if len(f) >= 4 and f in text.lower():
        out["forbidden"] = forbid
    if max_chars and len(text) > max_chars:
        out["too_long"] = len(text)
    return out


def complaints(issues: dict, lang_name: str = "Russian") -> str:
    """The repair prompt's body: what went wrong, named concretely.

    Naming the exact word is the whole point. The abstract rule is already in
    the model's context and it has already been ignored once.
    """
    lines = []
    if issues.get("wrong_language"):
        lines.append(f"- The whole answer is in the wrong language. Write it in "
                     f"{lang_name}. Names of people, bands and works still stay "
                     f"in Latin letters — only the sentences around them change.")
    if issues.get("translit"):
        pairs = "; ".join(f'you wrote «{v.split("→")[1]}», the source spells it '
                          f'"{v.split("→")[0]}"' for v in issues["translit"][:6])
        lines.append(f"- A name was rewritten in Cyrillic: {pairs}. Put every one "
                     "of those back in Latin letters, exactly as the source "
                     "spells them, undeclined.")
    if issues.get("garbled"):
        lines.append("- One word mixes Latin and Cyrillic letters inside itself "
                     "(for example «Джagger»). Write that word in one alphabet.")
    if issues.get("invented"):
        lines.append("- These names do not appear in the source and must be "
                     f"removed: {', '.join(issues['invented'][:6])}.")
    if issues.get("forbidden"):
        lines.append(f"- You wrote «{issues['forbidden']}». Remove it; the "
                     "listener already sees it. Point at it with an ordinary "
                     "noun phrase, or drop the pointer entirely.")
    if issues.get("too_long"):
        lines.append(f"- Too long ({issues['too_long']} characters). Cut it by "
                     "dropping the least important detail — not by summarising "
                     "everything into vagueness.")
    return "\n".join(lines)


def repair_is_safe(before: str, after: str) -> bool:
    """Did the repair fix the named fault without wrecking the sentence?

    Counting issues is not enough. The run produced "Видео снято в Лос-объекте
    притяжения почти фатальной Puth" — a repair that removed the transliteration
    and destroyed the grammar around it. Every check passed afterwards, because
    no check looks at whether the text still reads. These two do the cheap part
    of that: a repair is a correction, so it keeps roughly the same length and
    keeps the facts it was carrying.
    """
    if not after.strip():
        return False
    lo, hi = 0.65 * len(before), 1.35 * len(before)
    if not (lo <= len(after) <= hi):
        return False
    nums_before = set(re.findall(r"\d+", before))
    if nums_before and len(nums_before - set(re.findall(r"\d+", after))) > len(nums_before) / 2:
        return False
    lat_before = latin_tokens(before)
    if lat_before and len(lat_before - latin_tokens(after)) > len(lat_before) / 2:
        return False
    return True


# ── two kinds of repair, because they are two different jobs ─────────────────
#
# A name in the wrong script does NOT need the sentence rewritten, and asking for
# a rewrite is what destroyed two outputs in the v2 run: a bio collapsed from two
# paragraphs to one sentence, and a fact came back as "Видео снято в Лос-объекте
# притяжения почти фатальной Puth". Neither was caught by counting issues — the
# names were fixed, so every check passed.
#
# So for a script fault the model returns a REPLACEMENT LIST and code applies it.
# The sentence cannot change, because the model never touches it. A full rewrite
# is kept only where the sentence genuinely has to change.

SCRIPT_FAULTS = ("translit", "garbled")


def needs_rewrite(issues: dict) -> bool:
    return any(k in issues for k in
               ("wrong_language", "too_long", "forbidden", "invented"))


REPLACE_PROMPT = """Some names in the text below are written in the wrong script.
You will not rewrite the text. You will only say which words to swap.

THE TEXT

{text}

WHAT IS WRONG

{complaints}

For each faulty name give the exact substring as it appears in the text, and what
it must become. Copy the replacement spelling character for character from the
source spelling given above; a name never gets declined once it is in Latin.
Give nothing else — no rephrasing, no additions, no punctuation changes.

EXAMPLE

text: «Микку Джaggerу и Ричарду Эшкрофту вернули гонорары.»
{{"replace":[{{"from":"Микку Джaggerу","to":"Mick Jagger"}},
             {{"from":"Ричарду Эшкрофту","to":"Richard Ashcroft"}}]}}

Answer with STRICT JSON and nothing else — no markdown fence, no text around it.
{{"replace":[{{"from":"...","to":"..."}}]}}"""


def apply_replacements(text: str, pairs: list, source: str = "") -> tuple:
    """Apply a model-supplied replacement list. Code owns the edit.

    The `to` side must be a name that EXISTS IN THE SOURCE, verbatim. Without
    that check the model supplies plausible inventions and partial spans, both
    measured: it answered `Фромажо → Fromage` for "Nicolas Fromageau" (a
    shortening that means "cheese"), and `Гонсалеса → Gonzalez` for the full
    «Антони Гонсалеса», which leaves the hybrid «Антони Gonzalez». The model is
    good at spotting WHICH words are wrong and bad at spelling them; the source
    has the spelling.
    """
    known = {n.lower(): n for n in source_names(source)} if source else {}
    done, skipped = [], []
    for p in pairs or []:
        if not isinstance(p, dict):
            continue
        frm, to = (p.get("from") or "").strip(), (p.get("to") or "").strip()
        if not frm or not to or frm not in text:
            continue
        if known:
            exact = known.get(to.lower())
            if exact is None:
                # accept only if the offered form is the head of a known name
                exact = next((n for n in known.values()
                              if n.lower().startswith(to.lower())
                              and len(n) - len(to) <= 3), None)
            if exact is None:
                skipped.append((frm, to, "not_in_source"))
                continue
            to = exact
        candidate = text.replace(frm, to)
        if garbled_script(candidate):
            skipped.append((frm, to, "would_garble"))
            continue
        text = candidate
        done.append((frm, to))
    return text, done, skipped


REPAIR_PROMPT = """Your previous answer broke rules the text must follow. Fix it.

YOUR ANSWER

{text}

WHAT IS WRONG

{complaints}

Repair exactly those points and change nothing else — same facts, same meaning.
Do not add any detail that is absent from your answer above.

Answer with STRICT JSON and nothing else — no markdown fence, no text around it.
{{"text":"..."}}"""
