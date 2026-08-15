"""Expanding an abbreviation before it reaches a search box.

"музыка из GTA 5" has to become "Grand Theft Auto V" or the search returns forum
threads. Three sources of truth, in increasing cost and decreasing availability:

1. **The model's own guess.** Free, usually right for famous titles, and
   occasionally confidently wrong ("TDU 2" → "Touring Drive Unlimited 2").
2. **The user.** Authoritative, but costs a round trip — so it is a policy, not a
   default, and the streaming route is the only place it can even be offered.
3. **Wikipedia's article title.** The article for "GTA 5" is titled "Grand Theft
   Auto V". That is a curated redirect graph doing the disambiguation work, and
   it is one search away.

When all three come up empty the run stops and says so, rather than searching for
the abbreviation and pretending the results mean anything.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from app.services.assistant.config import AgentConfig
from app.services.assistant.contracts import Abbreviation, ClarifyRequest
from app.services.text_normalize import fold

logger = logging.getLogger(__name__)

ClarifyCallback = Callable[[ClarifyRequest], Awaitable[Optional[str]]]

# Below this the model is guessing rather than recalling, so its expansion is not
# offered as a fact even under the "auto" policy.
CONFIDENT = 0.8


class AbbreviationResolver:
    def __init__(self, sources, config: Optional[AgentConfig] = None, sink=None,
                 on_clarify: Optional[ClarifyCallback] = None):
        self.sources = sources
        self.cfg = config or AgentConfig()
        self.sink = sink
        self.on_clarify = on_clarify

    async def resolve(self, abbr: Optional[Abbreviation]) -> tuple:
        """``(expansion, clarify_request)``.

        A clarify request comes back only when the run genuinely cannot continue —
        the caller surfaces it and stops. An expansion of ``None`` with no request
        means there was nothing to expand.
        """
        if abbr is None:
            return None, None

        policy = self.cfg.clarify_policy
        self._emit("clarify_start", raw=abbr.raw, guess=abbr.expansion,
                   confidence=abbr.confidence, policy=policy)

        if policy == "auto" and abbr.confidence >= CONFIDENT:
            abbr.resolved_by = "llm"
            self._emit("clarify_done", expansion=abbr.expansion, by="llm")
            return abbr.expansion, None

        if policy == "ask" and self.on_clarify is not None:
            answer = await self._ask(abbr)
            if answer:
                abbr.expansion, abbr.resolved_by = answer, "user"
                self._emit("clarify_done", expansion=answer, by="user")
                return answer, None

        title = await self._wikipedia(abbr.raw)
        if title:
            abbr.expansion, abbr.resolved_by = title, "wikipedia"
            self._emit("clarify_done", expansion=title, by="wikipedia")
            return title, None

        # An unconfident guess is still better than the bare abbreviation, but
        # only when nothing else was available. Say where it came from.
        if abbr.expansion:
            abbr.resolved_by = "llm"
            logger.info("[clarify] falling back to the model's guess %r (p=%.2f)",
                        abbr.expansion, abbr.confidence)
            self._emit("clarify_done", expansion=abbr.expansion, by="llm-weak")
            return abbr.expansion, None

        abbr.resolved_by = "unresolved"
        request = ClarifyRequest(
            kind="abbreviation",
            question=(f"Не удалось понять, что такое «{abbr.raw}». "
                      "Напиши полное название, пожалуйста."),
            suggestion=None)
        self._emit("clarify_failed", raw=abbr.raw)
        return None, request

    async def _ask(self, abbr: Abbreviation) -> Optional[str]:
        request = ClarifyRequest(kind="abbreviation",
                                 question=f"«{abbr.raw}» — это {abbr.expansion}?",
                                 suggestion=abbr.expansion)
        try:
            answer = await self.on_clarify(request)
        except Exception:  # noqa: BLE001 — a broken UI must not end the run
            logger.warning("[clarify] callback failed", exc_info=True)
            return None
        if not answer:
            return None
        answer = answer.strip()
        # "yes" in either language means "take your own suggestion".
        if fold(answer) in ("da", "yes", "y", "ok", "да", "ага", "верно"):
            return abbr.expansion
        return answer

    async def _wikipedia(self, term: str) -> Optional[str]:
        try:
            return await asyncio.to_thread(self.sources.wikipedia_title, term)
        except Exception:  # noqa: BLE001
            logger.warning("[clarify] wikipedia lookup failed", exc_info=True)
            return None

    def _emit(self, stage: str, **fields) -> None:
        if self.sink is not None:
            self.sink.put(stage, **fields)
