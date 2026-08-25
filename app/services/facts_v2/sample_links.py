"""Clean and verify extracted sampling links.

An LLM reading prose produces `dst_artist` + `dst_title` pairs that are mostly
right and sometimes are a TV show, an album, a misspelling, or the same link
written two ways. Production shows all four in 22 rows:

    Pete Rock & CL Smooth  — Mecca and the Soul Brother   ← an ALBUM
    Pete Rock and CL Smooth — The Basement                ← same artist, other spelling
    Rogers and Hammerstein — The Carousel Waltz           ← Rodgers, misspelt
    Dr. Hans Jenny         — None                         ← no title at all
    The Jamie Foxx Show    — The Jamie Foxx Show          ← a TV show

The checks run cheapest-first and each one is allowed to be the last word:

  0. shape     — free, deterministic, rejects the impossible
  1. evidence  — free, both names must occur in the fact the link came from
  2. library   — free, and PROOF when it hits: the recording is in the user's files
  3. corroboration — free, the same link extracted for several different tracks
  4. musicbrainz  — one HTTP call, the only check that can tell a song from an
                    album for a recording nobody in the library owns

Verdicts are cached in ``sample_link_verdicts`` rather than in the process:
MusicBrainz answers about one request a second (their limit is a per-IP budget,
`x-ratelimit-limit: 1200`, not a cadence), so a full pass over the corpus costs
30-60 minutes and must not be paid twice — and an incremental run after new
tracks arrive should cost only the new links.
"""

from __future__ import annotations

import difflib
import os
import re
import time

MB_MIN_INTERVAL = 0.4          # musicbrainzngs paces the calls itself
MB_GOOD_SCORE = 88             # their own 0-100 match score
MB_TIMEOUT = 15.0              # urllib has none by default; a stall hangs forever

# Not recordings. A sampling link whose "song" is one of these is a parse error,
# not a discovery.
_NOT_A_SONG = re.compile(
    r"(?i)\b(the .* show|tv series|television|soundtrack|episode|podcast|"
    r"documentary|commercial|advert|trailer|movie|film)\b")

_STRIP = " \t\n\r\"'“”«»‘’.,;:!?()[]"
_FEAT = re.compile(r"(?i)\s*[\(\[]?\s*(feat\.?|ft\.?|featuring|with)\s+.*$")
_PAREN = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]\s*$")


def norm(text: str, *, drop_feat: bool = True) -> str:
    """Comparison key. Display keeps the original spelling."""
    t = (text or "").strip(_STRIP).lower()
    if drop_feat:
        t = _FEAT.sub("", t)
    t = _PAREN.sub("", t)
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9а-яё]+", " ", t)
    t = re.sub(r"^(the|a|an)\s+", "", t).strip()
    return re.sub(r"\s+", " ", t)


def db_key(artist: str, title: str) -> str:
    """Identity of the other side, as stored in ``sample_links.dst_key``."""
    return f"{norm(artist)}|{norm(title)}"


def to_db_row(link: dict) -> dict:
    """One cleaned link in the shape ``replace_sample_links`` inserts.

    Shared by both writers — the extraction and the verification lane — so the
    same link cannot end up stored under two different ``dst_key`` spellings
    depending on which of them wrote it last.
    """
    return {
        "direction": link["direction"],
        "dst_key": db_key(link.get("artist") or "", link.get("title") or ""),
        "dst_title": link.get("title"),
        "dst_artist": link.get("artist"),
        "dst_slug": link.get("dst_slug"),
        "relation": link.get("relation") or "sample",
        "src_year": link.get("src_year"),
        "dst_year": link.get("dst_year"),
        "evidence": (link.get("fact") or "")[:400] or None,
        "confidence": link.get("confidence"),
    }


def shape_reject(link: dict, src_artist: str, src_title: str) -> str | None:
    """Reasons a link cannot be right, whatever the world says."""
    a, t = (link.get("artist") or "").strip(), (link.get("title") or "").strip()
    if not a or not t:
        return "empty_side"
    na, nt = norm(a), norm(t)
    if not na or not nt:
        return "empty_after_norm"
    if na == nt:
        return "artist_equals_title"          # "The Jamie Foxx Show — The Jamie Foxx Show"
    if _NOT_A_SONG.search(a) or _NOT_A_SONG.search(t):
        return "not_a_recording"
    if nt == norm(src_title) and na == norm(src_artist):
        return "self_reference"
    if len(nt) < 2 or len(na) < 2:
        return "too_short"
    if link.get("direction") not in ("source", "usage"):
        return "bad_direction"
    if link.get("relation") not in ("sample", "interpolation"):
        return "bad_relation"
    return None


def evidence_ok(link: dict, fact_text: str) -> bool:
    """Both sides must actually occur in the prose the link was read from.

    Cheap anti-invention, the same idea as the fact pipeline's name check —
    and it is the only check that catches a plausible-looking pair the model
    assembled out of thin air.
    """
    hay = norm(fact_text, drop_feat=False)
    for side in ("artist", "title"):
        needle = norm(link.get(side) or "")
        if not needle:
            return False
        if needle in hay:
            continue
        # a long title may be quoted with different punctuation; allow a close
        # match against any window of the same length
        if difflib.SequenceMatcher(None, needle, hay).find_longest_match(
                0, len(needle), 0, len(hay)).size >= max(4, len(needle) * 0.8):
            continue
        return False
    return True


def _better_form(a: dict, b: dict) -> bool:
    """Which spelling of the same link survives the merge.

    What decides is EVIDENCE, not shape: a link the library or MusicBrainz
    confirmed beats one nobody could confirm. Length was tried first and got it
    backwards — "The Supreme — You Can't Hurry Love" is longer than
    "The Supremes — Can't Hurry Love" and is the misspelling.
    """
    ka = (RANK.get(a.get("reason", ""), 0), a.get("verbatim", False),
          -len(a["artist"]))
    kb = (RANK.get(b.get("reason", ""), 0), b.get("verbatim", False),
          -len(b["artist"]))
    return ka >= kb


def exact_dedupe(links: list) -> list:
    """Collapse identical normalised keys — "Pete Rock & CL Smooth" and
    "Pete Rock and CL Smooth" are one link. Free, and it cuts MB calls."""
    groups: dict = {}
    for lk in links:
        groups.setdefault((norm(lk["artist"]), norm(lk["title"]),
                           lk["direction"]), []).append(lk)
    out = []
    for items in groups.values():
        best = max(items, key=lambda x: (x.get("verbatim", False),
                                         len(x["artist"]), len(x["title"])))
        best["n_sources"] = len(items)
        out.append(best)
    return out


def fuzzy_merge(links: list) -> list:
    """Second pass: near-identical links that survived the exact pass."""
    out: list = []
    for lk in sorted(links, key=lambda x: -x.get("n_sources", 1)):
        twin = None
        for kept in out:
            if kept["direction"] != lk["direction"]:
                continue
            # Fuzzy on BOTH sides, and one of them has to be nearly exact. The
            # first run left "The Supreme — You Can't Hurry Love" beside
            # "The Supremes — Can't Hurry Love": requiring an identical title key
            # missed it, because a dropped "You" changes the key.
            ar = difflib.SequenceMatcher(None, norm(kept["artist"]),
                                         norm(lk["artist"])).ratio()
            tr = difflib.SequenceMatcher(None, norm(kept["title"]),
                                         norm(lk["title"])).ratio()
            if ar >= 0.85 and tr >= 0.80 and max(ar, tr) >= 0.9:
                twin = kept
                break
        if twin is None:
            out.append(lk)
            continue
        # Which spelling survives matters: the first run merged the correct
        # "The Supremes" INTO the misspelt "The Supreme", because the survivor
        # was whichever happened to be seen first. Pick deliberately.
        keep, drop = (twin, lk) if _better_form(twin, lk) else (lk, twin)
        if keep is lk:
            out[out.index(twin)] = lk
        keep["n_sources"] = twin.get("n_sources", 1) + lk.get("n_sources", 1)
        keep.setdefault("merged_from", []).append(
            f'{drop["artist"]} — {drop["title"]}')
    return out


# ── the one external check ───────────────────────────────────────────────────

def _side_matches(asked: str, got: str) -> float:
    """How well one side of the pair lines up with what MusicBrainz returned.

    A plain string ratio is too strict on both sides, and the first run showed
    exactly how:

      asked  Julius Fučík
      got    Julius Fučík Arturo Rodríguez Budapest Art Orchestra   ratio 0.35
      asked  What Will Santa Say?
      got    What Will Santa Say? (When He Finds Everyone Swingin')  ratio 0.55

    Both are the right recording. MusicBrainz credits every performer on a
    recording and disambiguates titles with a parenthetical, so what we want is
    CONTAINMENT — did the thing we asked for survive inside the answer.
    """
    a, b = norm(asked), norm(got)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _mb_verified(cand: dict) -> bool:
    """MusicBrainz scores the QUERY, not the pair — a query naming an album still
    matches some recording — so the score never decides alone.

    Two ways to pass: a high score with both sides lining up, or both sides
    lining up almost exactly at a lower score. The second is what admits
    "Entrance of the Gladiators" → "Entry of the Gladiators" (score 72, the
    right recording under its variant title)."""
    a, t, sc = (cand.get("artist_ratio", 0), cand.get("title_ratio", 0),
                cand.get("score", 0))
    return bool((sc >= MB_GOOD_SCORE and a >= 0.75 and t >= 0.75)
                or (a >= 0.9 and t >= 0.9 and sc >= 65))


class MusicBrainz:
    """Recording lookup through musicbrainzngs, rate-limited and cached.

    `recording` is the entity that matters: it is a SONG. Searching it is what
    tells an album apart from a track for something nobody in the library owns,
    which no local check can do — a query for "Mecca and the Soul Brother"
    finds no recording, because it is a record.
    """

    def __init__(self, enabled: bool = True, interval: float = MB_MIN_INTERVAL,
                 cache_get=None, cache_put=None):
        self.enabled = enabled
        self.cache: dict = {}
        self.cache_get = cache_get
        self.cache_put = cache_put
        self.calls = 0
        self.errors: dict = {}
        self.mb = None
        if not enabled:
            return
        try:
            import musicbrainzngs as mbngs
        except ImportError:
            self.enabled = False
            return
        # musicbrainzngs goes through urllib, which reads the proxy from the
        # environment — so the proxy is SET here rather than passed. Reading it
        # still comes from .env via proxy_config, per the proxy contract.
        try:
            from app.services.proxy_config import get_proxy_url
            proxy = get_proxy_url()
            if proxy:
                os.environ.setdefault("http_proxy", proxy)
                os.environ.setdefault("https_proxy", proxy)
        except Exception:                          # noqa: BLE001
            pass
        # Importable is not the same as usable: the test suite stubs this
        # module with an empty one, and a future release could rename either
        # call. Since the lane now starts inside indexing, a failure here would
        # abort a fact run rather than merely skip a check — so it disables the
        # client instead of raising.
        try:
            # urllib has no default timeout, so a stalled read waits forever —
            # that is what hung the first production run: the process sat at
            # 0:07 of CPU for minutes with the network idle.
            import socket
            socket.setdefaulttimeout(MB_TIMEOUT)
            # MusicBrainz requires "app/version ( contact )" and answers 503 to
            # a bare agent string.
            mbngs.set_useragent("MusiX", "1.0", "https://musixai.ru")
            mbngs.set_rate_limit(limit_or_interval=interval)
        except Exception:                          # noqa: BLE001
            self.enabled = False
            return
        self.mb = mbngs

    def verify(self, artist: str, title: str) -> dict:
        if not self.enabled or self.mb is None:
            return {"checked": False, "error": "musicbrainzngs unavailable"}
        key = (norm(artist), norm(title))
        if key in self.cache:
            return self.cache[key]
        if self.cache_get is not None:
            stored = self.cache_get(*key)
            if stored is not None:
                stored["checked"] = True
                self.cache[key] = stored
                return stored
        # The limit is a BUDGET, not a cadence: the response carries
        # x-ratelimit-limit 1200 with x-ratelimit-remaining counting down, shared
        # per IP. Measured at a 1.1 s interval the answers still alternated
        # 503 / 200 / 503, so pacing alone does not get a clean pass — retrying
        # does. Each attempt is cheap; a 503 comes back in 0.3 s.
        recs = None
        last_err = ""
        for attempt, pause in enumerate((0, 2.0, 5.0)):
            if pause:
                time.sleep(pause)
            self.calls += 1
            try:
                res = self.mb.search_recordings(artist=artist, recording=title,
                                                limit=3)
                recs = res.get("recording-list") or []
                break
            except Exception as exc:               # noqa: BLE001
                name = type(exc).__name__
                last_err = f"{name}: {exc}"[:120]
                self.errors[name] = self.errors.get(name, 0) + 1
                if "503" not in str(exc) and "rate" not in str(exc).lower():
                    break                          # not throttling — do not retry
        if recs is None:
            out = {"checked": False, "error": last_err or "no answer"}
            self.cache[key] = out
            return out

        best = None
        for rec in recs:
            score = int(rec.get("ext:score") or rec.get("score") or 0)
            credit = rec.get("artist-credit") or []
            mb_artist = " ".join(
                (c.get("artist", {}).get("name", "") if isinstance(c, dict) else str(c))
                for c in credit).strip()
            cand = {"checked": True, "score": score,
                    "artist_ratio": round(_side_matches(artist, mb_artist), 2),
                    "title_ratio": round(_side_matches(title, rec.get("title") or ""), 2),
                    "mb_artist": mb_artist, "mb_title": rec.get("title"),
                    "mbid": rec.get("id")}
            rank = (_mb_verified(cand),
                    cand["artist_ratio"] + cand["title_ratio"], cand["score"])
            if best is None or rank > (_mb_verified(best),
                                       best["artist_ratio"] + best["title_ratio"],
                                       best["score"]):
                best = cand
        out = best or {"checked": True, "score": 0, "artist_ratio": 0,
                       "title_ratio": 0}
        # A high MB score alone is not enough: it scores the QUERY, and a query
        # naming an album still matches some recording. Both sides must line up.
        out["verified"] = _mb_verified(out)
        self.cache[key] = out
        if self.cache_put is not None:
            self.cache_put(*key, out)
        return out


# ── library resolution ───────────────────────────────────────────────────────

def library_resolver_from_db(collection_name: str):
    """`artist|title` → songs.slug for tracks the user actually owns.

    When this hits, the recording is PROVEN to exist: the user has the file.
    It is the only tier that needs no external opinion, and the only one that
    can also fill ``dst_slug`` so the player can link straight to the track.
    """
    from app.resources.metadata_db import MetadataDB

    index: dict = {}
    for artist, title, slug in MetadataDB.get_library_song_index(collection_name):
        index[(norm(artist), norm(title))] = slug

    def resolve(artist: str, title: str):
        return index.get((norm(artist), norm(title)))
    return resolve, len(index)


# ── driver ───────────────────────────────────────────────────────────────────

RANK = {"in_library": 4, "musicbrainz": 3, "corroborated": 2, "": 0}


def clean(links: list, *, resolve=None, mb: MusicBrainz | None = None,
          fact_cap: int = 400) -> list:
    """links: [{artist,title,direction,relation,src_slug,src_artist,src_title,fact}]

    Order matters and it changed once already. Verifying BEFORE the fuzzy merge
    is what lets the merge keep the right spelling: length is not a guide to
    correctness — it kept the misspelt "The Supreme" over "The Supremes" —
    whereas "which of the two does MusicBrainz know" is.
    """
    staged = []
    for lk in links:
        why = shape_reject(lk, lk.get("src_artist", ""), lk.get("src_title", ""))
        if why:
            lk["verdict"], lk["reason"] = "reject", why
            staged.append(lk)
            continue
        # The evidence check is a hard reject ONLY against a complete fact. A
        # stored fact truncated at the column cap routinely omits the second
        # link it produced — production rejected the real Rick James, Bill
        # Withers and Beethoven samples that way, because the sentence naming
        # them sat past the 400th character.
        fact = lk.get("fact") or ""
        truncated = len(fact) >= fact_cap
        if fact and not truncated and not evidence_ok(lk, fact):
            lk["verdict"], lk["reason"] = "reject", "not_in_source_text"
            staged.append(lk)
            continue
        lk["evidence_weak"] = bool(fact) and not evidence_ok(lk, fact)
        lk["verbatim"] = bool(fact) and norm(lk["title"]) in norm(
            fact, drop_feat=False)
        lk["verdict"] = "pending"
        staged.append(lk)

    pending = [lk for lk in staged if lk["verdict"] == "pending"]
    rejected = [lk for lk in staged if lk["verdict"] == "reject"]

    # Exact-key collapse first: free, and it cuts the number of MB calls.
    pending = exact_dedupe(pending)

    for lk in pending:
        slug = resolve(lk["artist"], lk["title"]) if resolve else None
        if slug:
            lk.update(verdict="verified", reason="in_library", dst_slug=slug,
                      confidence=1.0)
            continue
        if mb is not None:
            print(f"  … MB: {lk['artist']} — {lk['title']}", flush=True)
            res = mb.verify(lk["artist"], lk["title"])
            lk["mb"] = res
            if res.get("verified"):
                lk.update(verdict="verified", reason="musicbrainz",
                          confidence=0.9)
                continue
            if res.get("checked"):
                lk.update(verdict="unverified", reason="mb_no_match",
                          confidence=0.3)
                continue
            lk.update(verdict="unverified",
                      reason=f"mb_error:{res.get('error','?')[:40]}",
                      confidence=0.5)
            continue
        lk.update(verdict="unverified", reason="unchecked", confidence=0.5)

    merged = fuzzy_merge(pending)
    for lk in merged:
        if lk["verdict"] != "verified" and lk.get("n_sources", 1) >= 2:
            lk.update(verdict="verified", reason="corroborated", confidence=0.8)
    return merged + rejected
