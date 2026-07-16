"""Pure-Python triage over GLiNER2 raw extraction output for one fact.

Buckets each candidate producer/sample relation into:

- ``AS_IS``: confident enough to write straight to the DB
- ``LLM``: needs LLM confirmation before it's trusted

This is a 1-to-1 port of the validated triage rules from the offline
validation script (``.superpowers/sdd/reference_gliner_triage.py``), adapted
to take the subject song's title/artist as parameters instead of looking them
up from a global ``song_meta`` dict keyed by slug. No torch/gliner2/sqlite
imports here -- pure ``re`` + stdlib.
"""
import re

NAME = r"[A-Z][\w.'&-]*(?:\s+[A-Z(][\w().'&-]*){0,4}"
RX_PROD = [re.compile(rf"\b(?:co-)?produced (?:the (?:track|song|album) )?by ({NAME})"),
           re.compile(rf"\bproducer[s]?,? ({NAME})"),
           re.compile(rf"\b({NAME}) (?:co-)?produced\b")]
SUBJ_WORDS = {"song", "track", "album", "it", "this", "this song", "the song", "the track", "the album"}
WRITER_WORDS = re.compile(r"\b(co-?wr(?:ote|itten)|wr(?:ote|itten)|penned|writer|featuring|features|feat\.)\b", re.I)
USAGE_CUE = re.compile(
    r"(?:\b(?:was|been|got)\s+(?:later\s+)?sampl)"          # was/been sampled
    r"|(?:\bsampl\w+\s+(?:this|it)\b)"                        # sampled this / sampled it
    r"|(?:\bsampl\w+\s+the\s+(?:famous\s+)?(?:open(?:ing)?|riff|hook|vocals?|beat|drums?)\b)",
    re.I)
SOURCE_CUE = re.compile(
    r"(?:\b(?:this|song|track|it)\s+(?:also\s+)?(?:sampl|interpolat|borrow|contains|features?\s+a\s+(?:vocal\s+)?sampl))"
    r"|(?:\ba\s+sample\s+(?:of|from)\b)", re.I)


def norm(s):
    return re.sub(r"[^\w]+", " ", (s or "").lower()).strip()


def head_is_subject(head_text, subject_title):
    h = norm(head_text)
    if h in SUBJ_WORDS:
        return True
    title = subject_title or ""
    return bool(title) and len(h) > 2 and (h in norm(title) or norm(title) in h)


def overlap(a, b):
    na, nb = norm(a), norm(b)
    return bool(na) and bool(nb) and (na in nb or nb in na)


def looks_like_name(s):
    s = (s or "").strip()
    return len(s) >= 3 and any(c.isupper() for c in s) and norm(s) not in SUBJ_WORDS


def get_conf(item):
    return item.get("confidence", 1.0) if isinstance(item, dict) else 1.0


def get_text(item):
    if isinstance(item, dict):
        return item.get("text", "") or ""
    return item or ""


def triage_producer(fact, gliner_out, subject_title):
    ents = gliner_out.get("entities", {})
    rels = gliner_out.get("relation_extraction", {})
    ent_names = [get_text(e) for e in ents.get("producer", [])]
    rx_names = [mm.group(1).rstrip(". ") for rx in RX_PROD for mm in rx.finditer(fact)]
    results, seen = [], set()
    for r in rels.get("produced_by", []):
        head, tail = r["head"], r["tail"]
        tname = get_text(tail)
        if not looks_like_name(tname) or norm(tname) in seen:
            continue
        seen.add(norm(tname))
        subj = head_is_subject(get_text(head), subject_title)
        ent_ok = any(overlap(tname, e) for e in ent_names)
        conf_ok = get_conf(tail) >= 0.7 and get_conf(head) >= 0.6
        pos = fact.find(tname)
        near = fact[max(0, pos - 70):pos + len(tname) + 70] if pos >= 0 else fact
        writer_risk = bool(WRITER_WORDS.search(near))
        if subj and ent_ok and conf_ok and not writer_risk:
            results.append(("AS_IS", tname))
        elif subj or ent_ok:
            results.append(("LLM", tname))
    for e in ent_names:
        if norm(e) not in seen and looks_like_name(e) and any(overlap(e, rx) for rx in rx_names):
            seen.add(norm(e))
            results.append(("LLM", e))
    return results


def triage_samples(fact, gliner_out, subject_title, subject_artist):
    ents = gliner_out.get("entities", {})
    rels = gliner_out.get("relation_extraction", {})
    my_title = subject_title or ""
    my_artist = (subject_artist or "").replace("-", " ")
    ent_titles = [get_text(e) for e in ents.get("sampled_song", [])]
    src_structs = gliner_out.get("sample_source", []) or []
    use_structs = gliner_out.get("sample_usage", []) or []
    if isinstance(src_structs, dict):
        src_structs = [src_structs]
    if isinstance(use_structs, dict):
        use_structs = [use_structs]
    samples_tails = [r["tail"] for r in rels.get("samples", [])
                      if head_is_subject(get_text(r["head"]), my_title)]
    sampledby_tails = [r["tail"] for r in rels.get("sampled_by", [])
                        if head_is_subject(get_text(r["head"]), my_title)]
    conflict = {norm(get_text(t)) for t in samples_tails} & {norm(get_text(t)) for t in sampledby_tails}

    source, seen = [], set()
    for t in samples_tails:
        tname = get_text(t)
        if not looks_like_name(tname) or norm(tname) in seen or overlap(tname, my_title):
            continue
        seen.add(norm(tname))
        ent_ok = any(overlap(tname, e) for e in ent_titles)
        st_artist = ""
        for s in src_structs:
            if isinstance(s, dict) and overlap(get_text(s.get("song", "")), tname):
                st_artist = get_text(s.get("artist", ""))
                break
        bucket = "AS_IS" if (ent_ok and get_conf(t) >= 0.7 and norm(tname) not in conflict
                             and SOURCE_CUE.search(fact)) else "LLM"
        source.append((bucket, tname, st_artist))
    for e in ent_titles:
        if norm(e) not in seen and looks_like_name(e) and not overlap(e, my_title):
            seen.add(norm(e))
            source.append(("LLM", e, ""))

    source_keys = {norm(x[1]) for x in source} | {norm(x[2]) for x in source if x[2]}
    usage, useen = [], set()
    for s in use_structs:
        if not isinstance(s, dict):
            continue
        song_v, artist_v = get_text(s.get("song", "")), get_text(s.get("artist", ""))
        key = norm(song_v) or norm(artist_v)
        if not key or key in useen:
            continue
        useen.add(key)
        rel_ok = any(overlap(artist_v, get_text(t)) or overlap(song_v, get_text(t)) for t in sampledby_tails)
        cue_ok = bool(USAGE_CUE.search(fact))
        not_self = (not overlap(song_v, my_title)) and (not overlap(artist_v, my_artist)) \
            and (not overlap(artist_v, my_title))
        not_source_echo = norm(song_v) not in source_keys and norm(artist_v) not in source_keys
        if rel_ok and cue_ok and not_self and not_source_echo \
                and (looks_like_name(song_v) or looks_like_name(artist_v)):
            usage.append(("AS_IS", song_v, artist_v))
        elif (rel_ok or cue_ok) and not_self:
            usage.append(("LLM", song_v, artist_v))
    for t in sampledby_tails:
        tname = get_text(t)
        if norm(tname) not in useen and looks_like_name(tname) and not overlap(tname, my_artist) \
                and norm(tname) not in source_keys:
            useen.add(norm(tname))
            usage.append(("LLM", "", tname))
    return source, usage
