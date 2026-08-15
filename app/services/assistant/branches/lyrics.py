"""Finding a song in the user's own library from words they remember.

Deterministic, shaped like the playlist branch: the planner produces two queries,
Qdrant produces candidates, the cross-encoder orders them, and the model gets one
job — say which of these is the song, and quote the line.

What is deliberately NOT reproduced from the previous engine: its four-attempt
loop with a validator that re-asks. That architecture is what this work exists to
leave. What IS reproduced, verbatim, are the parts of it that were right:

* resolving the model's free-text title back to a concrete hit through three
  passes, so "(Remastered)" does not lose the track;
* re-anchoring the highlight on the line the ANSWER quoted rather than on the
  executed query, because the two regularly disagree;
* shipping an empty hit list when the model named no song, so near-misses are
  never presented as an answer.

**Filtering happens here, not in Qdrant.** The vector store's artist filter is an
exact ``MatchValue`` on the payload, so a request for "Sade" would silently drop
every track tagged "Sade feat. …" — and «Канье» would match nothing at all. The
library catalog already knows how to compare names across spellings and
alphabets, so the pool is pulled deeper and narrowed here.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from app.services.assistant.contracts import LyricsResult, Plan, Subject
from app.services.assistant.llm import as_str
from app.services.assistant.prompts import LYRICS_ANSWER_SYSTEM
from app.services.library_catalog import filter_by_era

logger = logging.getLogger(__name__)


class LyricsBranch:
    def __init__(self, agent):
        self.agent = agent
        self.cfg = agent.cfg
        self.sink = agent.sink
        self.timings = agent.timings

    async def run(self, message: str, plan: Plan,
                  subject: Optional[Subject] = None) -> LyricsResult:
        service = self.agent.search_service
        if service is None:
            return LyricsResult(message="", notes=["search service unavailable"])

        query = plan.lyrics_query or plan.ce_query or message
        ce_query = plan.ce_query or message
        artist = self.agent.library_artist(plan.filters.artist)

        # Deeper than the pack, because the artist and era filters are applied
        # here rather than in the query: a pool cut to size first would come back
        # empty the moment a filter is set.
        pool = self.cfg.lyrics_pool * (2 if (artist or plan.filters.era) else 1)
        self.sink.put("search", source="library", query=query, found=0)
        with self.timings.span("search.qdrant"):
            hits = await service.search(query, mode="text", limit=pool,
                                        collection_name=self.agent.collection_name)
        logger.info("[lyrics] %r -> %d candidates", query, len(hits))

        hits = self._narrow(hits, artist=artist, era=plan.filters.era)
        self.sink.put("matched", claims=len(hits), resolved=len(hits),
                      target=self.cfg.lyrics_ctx_hits)
        if not hits:
            return LyricsResult(message=_nothing(self.cfg.lang),
                                notes=["nothing in the library matched"])

        with self.timings.span("rerank.ce"):
            ranked = await asyncio.to_thread(self._rerank, hits, ce_query)
        if not ranked:
            # No cross-encoder. The RRF order is still an order — worse, but the
            # alternative is refusing to answer over a missing model.
            logger.info("[lyrics] no cross-encoder — keeping the retrieval order")
            ranked = [(h, 0.0) for h in hits]

        top = ranked[:self.cfg.lyrics_ctx_hits]
        self.sink.put("chunks", selected=len(top),
                      threshold=self.cfg.lyrics_min_prob,
                      best=round(top[0][1], 3) if top else None)

        with self.timings.span("llm.answer"):
            answer = await self._ask(message, top)

        song, artist_name = answer.get("song"), answer.get("artist")
        best = _match_best_hit([h for h, _ in top], song, artist_name)

        # Re-anchor the highlight on the line the answer actually quoted. The
        # window that ranked highest was chosen against the planner's ce_query,
        # which regularly disagrees with the line the model went on to cite.
        if best is not None:
            lyrics = (best.track.lyrics or best.lyrics or "")
            for fragment in _quoted_fragments(answer.get("message", "")):
                line = _pick_matched_line(lyrics, fragment)
                if line:
                    best.matched_line = line
                    break

        # The model named no song = nothing was found. Ship an empty hit list
        # rather than the retrieval tail, so near-misses are not presented as an
        # answer.
        found = bool(song and best is not None)
        self.sink.put("result", tracks=1 if found else 0, missing=0)
        return LyricsResult(
            message=answer.get("message") or _nothing(self.cfg.lang),
            song=(best.track.title if best is not None else None),
            artist=(best.track.artist if best is not None else None),
            confidence=answer.get("confidence", "low"),
            best_hit=best,
            hits=[h for h, _ in top[:10]] if found else [])

    # ── narrowing ─────────────────────────────────────────────────────────

    def _narrow(self, hits: list, *, artist: Optional[str],
                era: Optional[tuple]) -> list:
        """Apply the planner's filters to the candidate pool."""
        before = len(hits)
        if artist:
            hits = [h for h in hits if _artist_matches(artist, h.track.artist)]
            if before != len(hits):
                logger.info("[lyrics] artist %r kept %d/%d", artist, len(hits),
                            before)
        if era:
            kept = filter_by_era(hits, era, year=lambda h: h.track.year)
            if len(kept) != len(hits):
                logger.info("[lyrics] era %s kept %d/%d", era, len(kept), len(hits))
            hits = kept
        return hits

    # ── reranking over lyric windows ──────────────────────────────────────

    def _rerank(self, hits: list, ce_query: str) -> list:
        """``[(hit, probability), ...]``, best first, filtered by the threshold.

        The cross-encoder reads (question, window) pairs rather than whole lyrics.
        A 3000-character text truncated at 512 tokens loses the second half of the
        song, and the line being asked about is as likely to be there as anywhere.
        Scoring windows also hands back, for free, the one that matched — which is
        what the UI highlights.
        """
        pairs: list = []       # (hit index, window text)
        for i, hit in enumerate(hits):
            lyrics = hit.track.lyrics or hit.lyrics or ""
            for window in _windows(lyrics, self.cfg.lyrics_window_words,
                                   self.cfg.lyrics_window_stride):
                pairs.append((i, window))
        if not pairs:
            return []

        probs = self.agent.hub.ce_probabilities(ce_query, [w for _, w in pairs])
        if probs is None:
            return []

        best: dict = {}
        for (i, window), prob in zip(pairs, probs):
            if i not in best or prob > best[i][0]:
                best[i] = (prob, window)

        out: list = []
        for i, (prob, window) in best.items():
            if prob < self.cfg.lyrics_min_prob:
                continue
            hit = hits[i]
            hit.score = float(prob)
            hit.matched_line = window
            out.append((hit, float(prob)))
        out.sort(key=lambda pair: -pair[1])
        logger.info("[lyrics] reranked %d windows over %d tracks -> %d above "
                    "p>=%.2f", len(pairs), len(hits), len(out),
                    self.cfg.lyrics_min_prob)
        return out

    # ── the one model call ────────────────────────────────────────────────

    async def _ask(self, message: str, ranked: list) -> dict:
        lang = ("Russian" if (self.cfg.lang or "").lower().startswith("ru")
                else "English")
        listing = []
        for i, (hit, prob) in enumerate(ranked, start=1):
            line = hit.matched_line or ""
            listing.append(f"{i}. {hit.track.artist} — {hit.track.title}\n"
                           f"   …{line}…")
        raw = await self.agent.llm.ask_json([
            {"role": "system", "content": LYRICS_ANSWER_SYSTEM.format(lang=lang)},
            {"role": "user",
             "content": f"The listener said: {message}\n\nTracks:\n"
                        + "\n".join(listing)},
        ], required=("message",))
        return raw or {}


# ── helpers, ported from the previous engine ─────────────────────────────────


def _windows(lyrics: str, size: int, stride: int):
    """Overlapping word windows over a lyric, line breaks preserved as spaces.

    Bounded at 40 windows per track: a 6-minute song is long, but a page of
    repeated chorus does not become more informative by being scored forty more
    times, and the batch is quadratic in the reranker.
    """
    words = (lyrics or "").split()
    if not words:
        return
    size = max(4, size)
    stride = max(1, stride)
    produced = 0
    for start in range(0, len(words), stride):
        chunk = words[start:start + size]
        if not chunk:
            break
        yield " ".join(chunk)
        produced += 1
        if produced >= 40 or start + size >= len(words):
            break


def _artist_matches(query: str, value: str) -> bool:
    """Loose containment both ways, on the folded forms.

    "Sade" matches a track tagged "Sade feat. …", and a query for "Kanye West"
    matches "Kanye West, Jay-Z". This is the comparison the exact payload filter
    in Qdrant cannot make.
    """
    from app.services.text_normalize import fold

    a, b = fold(query or ""), fold(value or "")
    return bool(a) and bool(b) and (a in b or b in a)


def _match_best_hit(hits: list, song: Optional[str],
                    artist: Optional[str]):
    """Resolve the model's named song back to a concrete hit.

    Three passes, loosening in order: title AND artist, title alone (the artist
    may be formatted differently — feat. credits, "&" vs "and", a different
    romanisation), then containment either way, which tolerates a "(Remastered)"
    or "- Live" suffix the model dropped or added.
    """
    if not song:
        return None

    def _norm(s: Optional[str]) -> str:
        return " ".join((s or "").strip().casefold().split())

    target_song = _norm(song)
    target_artist = _norm(artist)
    if not target_song:
        return None

    for h in hits:
        if _norm(h.track.title) == target_song and (
                not target_artist or _norm(h.track.artist) == target_artist):
            return h
    for h in hits:
        if _norm(h.track.title) == target_song:
            return h
    for h in hits:
        ht = _norm(h.track.title)
        if ht and (ht in target_song or target_song in ht):
            return h
    return None


def _quoted_fragments(message: str) -> list:
    """Lyric quotes the model embedded in its answer, longest first.

    The answer cites the matching line in «…», “…” or "…", and that citation is
    the ground truth for the highlight. Fragments shorter than three words are
    dropped — those are song titles and single-word emphasis.
    """
    frags = re.findall(r"[«“\"]([^«»“”\"]{10,200})[»”\"]", message or "")
    frags = [f.strip() for f in frags if len(f.split()) >= 3]
    frags.sort(key=len, reverse=True)
    return frags


def _pick_matched_line(lyrics: str, query: str) -> Optional[str]:
    """The lyric line with the most word overlap with ``query``.

    Pure python, no model call. Returns None when nothing overlaps.
    """
    terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}
    if not terms or not lyrics:
        return None

    def candidates():
        for line in lyrics.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) <= 160:
                yield stripped
            else:
                # An excerpt with mangled or absent line breaks — a whole-blob
                # "line" would be useless for highlighting, so score sliding
                # ~12-word windows instead.
                words = stripped.split()
                for i in range(0, len(words), 6):
                    chunk = " ".join(words[i:i + 12])
                    if chunk:
                        yield chunk

    best, best_score = None, 0
    for cand in candidates():
        words = set(re.findall(r"\w+", cand.lower()))
        score = len(terms & words)
        if score > best_score:
            best, best_score = cand, score
    return best


def _nothing(lang: str) -> str:
    return ("Не нашёл в твоей библиотеке ничего похожего. Попробуй вспомнить "
            "ещё пару слов из строчки."
            if (lang or "").lower().startswith("ru") else
            "I couldn't find anything like that in your library. Try recalling a "
            "couple more words from the line.")
