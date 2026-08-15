"""Hosts that are never an answer to a question about music.

A broken engine does not fail cleanly. It returns a page of adult-cam or casino
affiliate spam, SearXNG fuses that with the good engines' results by rank, and
the two interleave — one real link, one brothel, one real link. From inside the
pipeline the spam looks exactly like a search result: it has a title, a snippet
and a rank, so it survives to the cross-encoder, sometimes past it, and
occasionally into an answer.

Two rules, and the split matters:

**Hosts only, never titles or snippets.** "Casino" is an album, "Sex" is in a
hundred song titles, and a filter reading the text would delete real results
while a domain named ``stripchat.global`` sails past. The host is the part that
cannot be a coincidence.

**Distinctive substrings, exact labels for the rest.** ``porn`` and ``casino``
are safe to match anywhere in a host. ``bet`` and ``sex`` are not — they live
inside ``betterhelp.com``, ``arbeit.de`` and ``middlesex.gov.uk`` — so those
match only as a whole domain label.

The list is not meant to be complete. It is the cheap first cut; the per-engine
spam counter in ``web_sources`` is what actually identifies the engine to remove.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

# Safe anywhere in a hostname: no ordinary site contains these by accident.
_SUBSTRINGS = (
    # adult
    "stripchat", "chaturbate", "bongacams", "livejasmin", "camsoda",
    "myfreecams", "onlyfans", "pornhub", "xhamster", "xvideos", "youporn",
    "redtube", "brazzers", "camgirl", "sexcam", "livesex", "adultwork",
    "porn", "xxx", "escort", "hentai", "nsfw",
    # gambling
    "casino", "1xbet", "melbet", "mostbet", "parimatch", "betway", "bet365",
    "pokerstars", "playtech", "slotomania", "gambling", "bookmaker",
)

# Only as a whole domain label — as substrings these hit real sites.
_LABELS = frozenset({
    "bet", "bets", "betting", "poker", "slots", "slot", "sex", "cams", "cam",
    "adult", "casinos", "gamble", "bahis", "vulkan", "pinup",
})

_LABEL_SPLIT = re.compile(r"[.\-_]")

# "cams" is adult when it is a suffix — lemoncams, freecams, hdcams — but the
# bare substring also sits inside "scams", which a real site can be about. One
# lookbehind is cheaper and clearer than an exception list.
_CAMS_RE = re.compile(r"(?<!s)cams")


def is_spam_host(url: str) -> bool:
    """True when the host is an adult or gambling domain."""
    host = (urlsplit(url or "").netloc or "").lower()
    if not host:
        return False
    host = host.split("@")[-1].split(":")[0]

    # Separators removed before matching, so a name split across a dot or a
    # hyphen still reads as one word: "strip.chat" and "chatur-bate.com" are the
    # same trick as "stripchat.com".
    collapsed = _LABEL_SPLIT.sub("", host)
    if any(token in collapsed for token in _SUBSTRINGS):
        return True
    if _CAMS_RE.search(collapsed):
        return True
    return any(label in _LABELS for label in _LABEL_SPLIT.split(host) if label)


def spam_report(urls) -> dict:
    """How many of these URLs are spam, grouped by host. For diagnostics."""
    counts: dict = {}
    for url in urls:
        if not is_spam_host(url):
            continue
        host = (urlsplit(url).netloc or "").lower()
        counts[host] = counts.get(host, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
