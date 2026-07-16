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

SYSTEM = """You extract music metadata from ONE fact about a KNOWN subject song. Entity candidates are pre-marked inline as [Person: name], [Song: title], [Artist: name]. Markers come from an automatic NER system: they can be wrong or incomplete. Trust the sentence meaning, not the markers.

Return ONLY a JSON object, no explanations, exactly this shape:
{"producers": ["name", ...], "samples": [{"song": "title or null", "artist": "name or null"}, ...], "sampled_by": [{"song": "title or null", "artist": "name or null"}, ...]}

Definitions:
- "producers": people or teams the fact EXPLICITLY credits as producer or co-producer of the subject song (or of the album containing it).
- "samples": OLDER songs/artists whose material the subject song uses (a sample or an interpolation). X in "this song samples X" goes here.
- "sampled_by": NEWER songs/artists that sampled or interpolated the subject song. In "Y sampled this song in his track Z" -> {"song": "Z", "artist": "Y"}.

Traps you MUST avoid:
1. Writers, co-writers, featured artists, engineers, DJs, remixers are NOT producers. "X wrote the song" or "co-wrote with X" does not make X a producer. Only explicit producing counts ("produced by X", "X produced/co-produced", "production by X", "X handled production").
2. Direction: "this song sampled the song from [Artist: X]" means the SUBJECT uses X's material -> it goes to "samples", NOT "sampled_by". "was sampled by Y" / "Y sampled it" -> "sampled_by".
3. The fact may describe a sample relation between two OTHER songs that are not the subject song. If the subject song is not one of the two sides, output nothing for that relation.
4. Unconfirmed lawsuits or accusations ("accused of ripping off", "sued claiming...") are NOT samples. Skip them.
5. Never invent names or titles that are not written in the fact. If the artist is stated but the song title is not, use null for "song".
6. A person can appear in "producers" only once; use the fullest form of the name written in the fact.
7. When nothing qualifies, return empty arrays. Output must be valid JSON.
8. If a PERSON's voice/vocals are sampled (no song named), put {"song": null, "artist": person} — never put a person's name into "song".
9. "Inspired by", "tribute to", "cover of", "remix" are NOT samples.
10. "used the same sample" about another song is NOT a relation of the subject song.
11. Facts starting with "Lyrics string:" are lyric annotations: extract a sample ONLY if source song or artist is explicitly named."""


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


def _norm_relations(items):
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
        out.append({"song": song, "artist": artist})
    return out


def parse_llm_re(raw):
    """Parse a raw LLM-RE reply (``dict`` or ``str``) into a normalized dict.

    Returns ``{"producers": [str], "samples": [{"song", "artist"}, ...],
    "sampled_by": [...]}`` or ``None`` if the reply is not valid JSON or does
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

    samples = _norm_relations(data.get("samples", []))
    if samples is None:
        return None
    sampled_by = _norm_relations(data.get("sampled_by", []))
    if sampled_by is None:
        return None

    return {
        "producers": [p.strip() for p in producers if p and p.strip()],
        "samples": samples,
        "sampled_by": sampled_by,
    }


def _merge_relations(a_list, b_list):
    result = list(a_list or [])
    seen = {(norm(it.get("song") or ""), norm(it.get("artist") or "")) for it in result}
    for it in (b_list or []):
        if not isinstance(it, dict):
            continue
        song, artist = it.get("song"), it.get("artist")
        key = (norm(song or ""), norm(artist or ""))
        if key == ("", "") or key in seen:
            continue
        seen.add(key)
        result.append({"song": song, "artist": artist})
    return result


def merge_results(as_is, llm):
    """Merge ``AS_IS`` triage results with the (possibly ``None``) LLM leg.

    Dedup is case/whitespace-insensitive via ``triage.norm``. Result shape:
    ``{"producers": [str], "samples": [{"song","artist"}], "sampled_by": [...]}``.
    """
    as_is = as_is or {}
    llm = llm or {}

    producers = list(as_is.get("producers", []) or [])
    seen_prod = {norm(p) for p in producers}
    for p in (llm.get("producers", []) or []):
        if not p or norm(p) in seen_prod:
            continue
        seen_prod.add(norm(p))
        producers.append(p)

    samples = _merge_relations(as_is.get("samples", []), llm.get("samples", []))
    sampled_by = _merge_relations(as_is.get("sampled_by", []), llm.get("sampled_by", []))

    return {"producers": producers, "samples": samples, "sampled_by": sampled_by}
