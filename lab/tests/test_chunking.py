"""Chunking: does a chunk still know where it came from, and does it keep the
things that must not be cut?"""

from lab.agent.chunking import (MarkdownChunker, pack_blocks, split_blocks,
                                split_sections)
from lab.agent.config import AgentConfig

DOC = """# Kanye West

An American rapper.

## Career

### Early years

He produced for Roc-A-Fella.

### 2009 VMA incident

He interrupted Taylor Swift.

## Discography

| Album | Year |
| --- | --- |
| The College Dropout | 2004 |
| Graduation | 2007 |
"""


class TestSections:
    def test_the_heading_path_is_kept(self):
        sections = split_sections(DOC)
        paths = [s["path"] for s in sections]
        assert ["Kanye West", "Career", "2009 VMA incident"] in paths

    def test_a_heading_with_no_body_still_parents_its_children(self):
        """"Career" has no text of its own; its children must still carry it."""
        sections = split_sections(DOC)
        early = next(s for s in sections if "Early years" in s["path"])
        assert early["path"] == ["Kanye West", "Career", "Early years"]

    def test_a_hash_inside_a_fence_is_not_a_heading(self):
        doc = "# Real\n\n```\n# not a heading\n```\n\ntext\n"
        assert len(split_sections(doc)) == 1


class TestBlocks:
    def test_paragraphs_split_on_blank_lines(self):
        assert len(split_blocks("one\n\ntwo\n\nthree")) == 3

    def test_a_table_is_one_block(self):
        table = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        assert split_blocks(table) == [table]

    def test_a_fenced_block_survives_its_blank_lines(self):
        doc = "```\nline\n\nline\n```"
        assert split_blocks(doc) == [doc]


class TestPacking:
    def test_blocks_are_grouped_up_to_the_budget(self):
        blocks = ["x" * 100 for _ in range(10)]
        packed = pack_blocks(blocks, max_chars=250, overlap=0)
        assert all(len(p) <= 250 for p in packed)
        assert len(packed) >= 4

    def test_an_oversized_block_is_cut_on_sentences(self):
        block = " ".join(f"Sentence number {i}." for i in range(200))
        packed = pack_blocks([block], max_chars=300)
        assert len(packed) > 1
        assert all(len(p) <= 320 for p in packed)

    def test_overlap_does_not_duplicate_the_whole_chunk(self):
        """A tail carried forward must be a TAIL — carrying everything would
        make each chunk a copy of the previous one."""
        blocks = ["a" * 200, "b" * 200]
        packed = pack_blocks(blocks, max_chars=250, overlap=1)
        assert packed[0] != packed[1]


class TestChunker:
    def test_the_path_is_prepended_to_the_embedded_text(self):
        """A paragraph about "the interruption" matches nothing on its own."""
        cfg = AgentConfig(chunk_min_chars=0)     # no merging, one section per chunk
        chunks = MarkdownChunker(cfg).split(DOC, url="https://ex/kw")
        incident = next(c for c in chunks if "Taylor Swift" in c.body)
        assert incident.text.startswith("Kanye West > Career > 2009 VMA incident")
        assert incident.url == "https://ex/kw"

    def test_micro_sections_are_glued_to_their_neighbour(self):
        """Two-line sections would otherwise produce chunks that are mostly
        heading."""
        chunks = MarkdownChunker(AgentConfig()).split(DOC)
        assert len(chunks) < len(split_sections(DOC))

    def test_a_glued_section_keeps_its_own_heading_in_the_body(self):
        """Merging must not cost the reader the subheading — without it a
        chunk reads as if everything in it happened under the first one."""
        merged = next(c for c in MarkdownChunker(AgentConfig()).split(DOC)
                      if "Taylor Swift" in c.body)
        assert "**2009 VMA incident**" in merged.body

    def test_ids_are_sequential_from_the_offset(self):
        chunker = MarkdownChunker(AgentConfig())
        first = chunker.split(DOC)
        second = chunker.split(DOC, start_id=len(first))
        assert second[0].id == len(first)

    def test_an_empty_document_yields_nothing(self):
        assert MarkdownChunker(AgentConfig()).split("") == []
