"""Context-aware chunking of a markdown page.

Three rules, in order of importance:

1. **A chunk never loses where it came from.** The heading path is prepended to
   the text that gets embedded, so a paragraph under
   "Kanye West > Controversies > 2009 VMA incident" carries that meaning into
   the vector. A bare paragraph about "the interruption" matches nothing.
2. **A block is never split.** A paragraph, a list, a table and a fenced code
   block are indivisible; a table cut in half is worse than useless, because
   the half without the header row looks like prose.
3. **Micro-sections are glued to their neighbour.** A two-line section under
   the same parent produces a chunk that is mostly heading; merged, it produces
   context.

Only when a single block is longer than the budget does it get cut, and then on
sentence boundaries.
"""

from __future__ import annotations

import re
from typing import Optional

from lab.agent.models import Chunk, Page

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def split_sections(md_text: str) -> list[dict]:
    """Cut the document at headings, keeping the hierarchy in ``path``."""
    sections: list[dict] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence, fence = False, None

    def flush() -> None:
        text = "\n".join(buf).strip()
        buf.clear()
        if text:
            sections.append({"path": [t for _, t in stack], "text": text})

    for line in (md_text or "").split("\n"):
        m_fence = FENCE_RE.match(line)
        if m_fence:
            if not in_fence:
                in_fence, fence = True, m_fence.group(1)
            elif line.strip().startswith(fence):
                in_fence = False
            buf.append(line)
            continue

        m_head = HEADING_RE.match(line) if not in_fence else None
        if m_head:
            flush()
            level, title = len(m_head.group(1)), m_head.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue
        buf.append(line)

    flush()
    return sections


def split_blocks(text: str) -> list[str]:
    """Paragraph / list / table / fenced code — one indivisible block each."""
    blocks: list[str] = []
    cur: list[str] = []
    in_fence, fence = False, None

    for line in text.split("\n"):
        m_fence = FENCE_RE.match(line)
        if m_fence and not in_fence:
            in_fence, fence = True, m_fence.group(1)
            cur.append(line)
            continue
        if in_fence:
            cur.append(line)
            if line.strip().startswith(fence):
                in_fence = False
                blocks.append("\n".join(cur))
                cur = []
            continue
        if not line.strip():
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)

    if cur:
        blocks.append("\n".join(cur))
    return [b for b in blocks if b.strip()]


def _size(parts: list[str]) -> int:
    """Length of ``parts`` once joined with a blank line between each."""
    return sum(len(p) for p in parts) + 2 * max(0, len(parts) - 1)


def pack_blocks(blocks: list[str], *, max_chars: int,
                overlap: int = 1) -> list[str]:
    out: list[str] = []
    cur: list[str] = []

    for block in blocks:
        if len(block) > max_chars:
            if cur:
                out.append("\n\n".join(cur))
                cur = []
            acc = ""
            for part in re.split(r"(?<=[.!?])\s+", block):
                if len(acc) + len(part) > max_chars and acc:
                    out.append(acc.strip())
                    acc = ""
                acc += part + " "
            if acc.strip():
                out.append(acc.strip())
            continue

        if cur and _size(cur) + 2 + len(block) > max_chars:
            out.append("\n\n".join(cur))
            # Carry a tail into the next chunk, but only if it IS a tail (not
            # the whole chunk again) and only if the new block still fits.
            tail = cur[-overlap:] if overlap and len(cur) > overlap else []
            cur = tail if _size(tail) + 2 + len(block) <= max_chars else []
        cur.append(block)

    if cur:
        out.append("\n\n".join(cur))
    return out


class MarkdownChunker:
    def __init__(self, config=None):
        from lab.agent.config import AgentConfig

        self.cfg = config or AgentConfig()

    def split(self, markdown: str, *, url: str = "", title: str = "",
              source: str = "web", start_id: int = 0) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffered: Optional[dict] = None

        def emit(section: dict) -> None:
            for body in pack_blocks(split_blocks(section["text"]),
                                    max_chars=self.cfg.chunk_max_chars,
                                    overlap=self.cfg.chunk_overlap_blocks):
                chunks.append(Chunk(id=start_id + len(chunks),
                                    path=list(section["path"]), body=body,
                                    url=url, title=title, source=source))

        for section in split_sections(markdown):
            same_parent = (buffered is not None
                           and section["path"][:-1] == buffered["path"][:-1])
            if (buffered is not None and same_parent
                    and len(section["text"]) < self.cfg.chunk_min_chars):
                heading = section["path"][-1] if section["path"] else ""
                buffered["text"] += f"\n\n**{heading}**\n\n" + section["text"]
                continue
            if buffered is not None:
                emit(buffered)
            buffered = dict(section)

        if buffered is not None:
            emit(buffered)
        return chunks

    def split_page(self, page: Page, *, start_id: int = 0) -> list[Chunk]:
        if not page.ok:
            return []
        return self.split(page.markdown, url=page.url, title=page.title,
                          source=page.source, start_id=start_id)
