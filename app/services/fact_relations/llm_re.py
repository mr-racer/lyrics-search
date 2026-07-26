"""LLM leg of the producers/samples relation-extraction pipeline.

Takes the GLiNER candidates that Task 2's triage bucketed into ``LLM`` (needs
confirmation), marks them inline in the fact text, builds the chat messages
for an OpenAI-compatible LLM call, parses the strict-JSON reply, and merges
it with the ``AS_IS`` (already-confident) results from triage.

The system prompt below is taken verbatim from
``.superpowers/sdd/reference_llm_re_test.py``, where it was validated on 50
production facts (100% valid JSON on gemma-12b), plus four extra trap rules
(8-11) added for this task. No torch/gliner2/network imports here -- pure
``re`` + ``json`` + stdlib.
"""
import json
import re

from .triage import norm

SYSTEM = """You read ONE fact about a KNOWN subject song and label how it links that song to other music. Entity candidates are pre-marked inline as [Person: name], [Song: title], [Artist: name]. Markers come from an automatic NER system: they can be wrong or incomplete. Trust the sentence meaning, not the markers.

Return ONLY a JSON object, no explanations, exactly this shape:
{"producers": ["name", ...], "links": [{"song": "title or null", "artist": "name or null", "direction": "source|usage", "relation": "sample|interpolation|cover|remix|lyrical_reference|inspiration|other"}, ...]}

"producers": people or teams the fact EXPLICITLY credits as producer or co-producer of the subject song (or of the album containing it).

"links": one entry per OTHER musical work the fact connects to the subject song.

"direction" — which side took from which:
- "source": the other work is the older one and the SUBJECT took from it.
- "usage": the other work is the newer one and IT took from the subject.

"relation" — pick exactly one, and do not stretch the first two:
- "sample": part of the other recording's AUDIO is used.
- "interpolation": the other work's melody or part is re-played or re-sung.
- "cover": one is a cover or a version of the other.
- "remix": one is a remix, edit or live version of the other.
- "lyrical_reference": the lyrics only mention, quote, allude to or diss it. No sound was taken.
- "inspiration": "inspired by", "reminiscent of", "matches the melody of", "tribute to", "in the style of".
- "other": anything else, including lawsuits and accusations of copying.

Rules you MUST follow:
1. "sample" and "interpolation" require the fact to SAY that sound was taken. If it describes a mention in the lyrics, a diss, a resemblance, a homage or a legal claim, use the label that fits — never "sample".
2. Facts beginning with "Lyrics string:" are annotations explaining a LINE. Default to "lyrical_reference" unless the same fact explicitly states the audio was sampled or interpolated.
3. If the fact links two OTHER works and the subject song is neither side, output nothing for it.
4. Never invent names or titles that are not written in the fact. If the artist is named but the song is not, use null for "song"; if a PERSON's voice is sampled with no song named, that is {"song": null, "artist": person} — never put a person's name into "song".
5. Writers, co-writers, featured artists, engineers, DJs and remixers are NOT producers. Only explicit producing counts ("produced by X", "X produced/co-produced", "production by X", "X handled production").
6. A person appears in "producers" at most once; use the fullest form of the name written in the fact.
7. "used the same sample as" another song is not a link of the subject song.
8. When nothing qualifies, return empty arrays. Output must be valid JSON."""

# The classifier labels; anything else the model invents is dropped on parse.
_RELATIONS = frozenset({
    "sample", "interpolation", "cover", "remix",
    "lyrical_reference", "inspiration", "other",
})
_DIRECTIONS = frozenset({"source", "usage"})


def mark_fact(fact, candidates):
    """Insert ``[Type: span]`` markers for candidate spans into ``fact``.

    ``candidates`` is a list of ``(text, type)`` pairs, e.g.
    ``[("Rick Rubin", "Person"), ("Amen Brother", "Song")]``. Longer spans are
    marked first, each span only at its first occurrence, and a span is never
    marked if it would nest inside an already-inserted marker.
    """
    marked = fact
    done = set()
    for text, typ in sorted(candidates, key=lambda s: -len(s[0])):
        if not text or text.lower() in done:
            continue
        done.add(text.lower())
        idx = marked.find(text)
        if idx < 0:
            continue
        before = marked[max(0, idx - 30):idx]
        if before.count("[") <= before.count("]"):
            marked = marked[:idx] + f"[{typ}: {text}]" + marked[idx + len(text):]
    return marked


def build_llm_messages(subject_title, subject_artist, marked_fact):
    """Build the chat messages for the LLM-RE call over one marked fact."""
    artist_display = (subject_artist or "").replace("-", " ")
    subject_line = f'"{subject_title or ""}" (artist: {artist_display})'
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Subject song: {subject_line}\n\nFact:\n{marked_fact}\n\nJSON:"},
    ]


def _clean_str(v):
    return v if isinstance(v, str) and v.strip() else None


def _norm_links(items):
    """Validate the classifier's link list; None when the shape is wrong.

    An unknown ``relation`` becomes ``"other"`` rather than invalidating the
    whole reply — a 12B model that invents a label is still telling us the
    link is not a sample, which is the only thing that matters downstream.
    """
    if not isinstance(items, list):
        return None
    out = []
    for it in items:
        if not isinstance(it, dict):
            return None
        song = _clean_str(it.get("song"))
        artist = _clean_str(it.get("artist"))
        if song is None and artist is None:
            continue
        relation = str(it.get("relation") or "").strip().lower()
        direction = str(it.get("direction") or "").strip().lower()
        out.append({
            "song": song,
            "artist": artist,
            "relation": relation if relation in _RELATIONS else "other",
            "direction": direction if direction in _DIRECTIONS else "source",
        })
    return out


def parse_llm_re(raw):
    """Parse a raw LLM-RE reply (``dict`` or ``str``) into a normalized dict.

    Returns ``{"producers": [str], "links": [{"song", "artist", "relation",
    "direction"}, ...]}`` or ``None`` if the reply is not valid JSON or does
    not match the expected shape.
    """
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        s = raw.strip()
        s = re.sub(r"^```(?:json)?", "", s).strip().rstrip("`").strip()
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except (ValueError, TypeError):
            return None
    else:
        return None

    if not isinstance(data, dict):
        return None

    producers = data.get("producers", [])
    if not isinstance(producers, list) or not all(isinstance(p, str) for p in producers):
        return None

    links = _norm_links(data.get("links", []))
    if links is None:
        return None

    return {
        "producers": [p.strip() for p in producers if p and p.strip()],
        "links": links,
    }


def _link_key(link):
    return (
        norm(link.get("song") or ""),
        norm(link.get("artist") or ""),
        link.get("direction") or "source",
    )


def _merge_links(a_list, b_list):
    """Union of two link lists, keyed on (song, artist, direction).

    The first occurrence wins. That ordering matters: callers pass the AS_IS
    (explicit-wording) links first, so a claim already backed by "sampled" /
    "interpolated" in the text is never downgraded by a later model guess on
    the same pair.
    """
    result = []
    seen = set()
    for it in [*(a_list or []), *(b_list or [])]:
        if not isinstance(it, dict):
            continue
        key = _link_key(it)
        if (key[0], key[1]) == ("", "") or key in seen:
            continue
        seen.add(key)
        result.append(dict(it))
    return result


def merge_results(a, b):
    """Merge two ``{"producers": [...], "links": [...]}`` dicts.

    Used both to fold the AS_IS triage into the LLM leg for one fact and to
    accumulate across a song's facts.
    """
    a, b = a or {}, b or {}

    producers = list(a.get("producers", []) or [])
    seen_prod = {norm(p) for p in producers}
    for p in (b.get("producers", []) or []):
        if not p or norm(p) in seen_prod:
            continue
        seen_prod.add(norm(p))
        producers.append(p)

    return {
        "producers": producers,
        "links": _merge_links(a.get("links"), b.get("links")),
    }
