"""Cutting the appendix off a MediaWiki page.

Wrong in either direction costs something real: leaving the references in
spends two model passes on citation lines and lets them out-rank prose on a
name query, while cutting too eagerly silently loses the end of an article.
So the match is exact-heading only, and there is a floor on how much may go.
"""

from lab.agent.cleanup import strip_appendix

ARTICLE = """# Kanye West

An American rapper.

## Career

He produced for Roc-A-Fella and then went solo.

## 2009 VMA incident

He interrupted Taylor Swift's acceptance speech.

## References

1. ^ Smith, John (2010). "Kanye". Rolling Stone.
2. ^ Jones, Ann (2011). "Yeezus". Pitchfork.

## External links

- Official site
"""


class TestStrip:
    def test_everything_from_references_on_is_gone(self):
        kept, removed = strip_appendix(ARTICLE)
        assert "Rolling Stone" not in kept
        assert "Official site" not in kept
        assert removed > 0

    def test_the_heading_itself_goes_too(self):
        """A lone "References" trailing the last real chunk is noise in the
        embedded text."""
        kept, _ = strip_appendix(ARTICLE)
        assert "References" not in kept

    def test_the_article_body_survives_intact(self):
        kept, _ = strip_appendix(ARTICLE)
        assert "interrupted Taylor Swift" in kept
        assert "Roc-A-Fella" in kept

    def test_it_cuts_at_the_first_appendix_heading(self):
        """"See also" comes before "References" and everything after it is
        appendix too."""
        doc = ARTICLE.replace("## References", "## See also\n\n- Yeezus\n\n## References")
        kept, _ = strip_appendix(doc)
        assert "Yeezus" not in kept

    def test_a_page_without_an_appendix_is_untouched(self):
        doc = "# Song\n\n## Meaning\n\nIt is about a car.\n"
        kept, removed = strip_appendix(doc)
        assert kept == doc and removed == 0

    def test_a_heading_that_merely_contains_the_word_is_left_alone(self):
        """"References to earlier work" is a section someone wrote on purpose;
        a substring rule would eat it and the rest of the article with it."""
        doc = ("# A\n\n## References to earlier work\n\n"
               "The riff quotes a 1974 record.\n")
        kept, removed = strip_appendix(doc)
        assert removed == 0
        assert "1974 record" in kept

    def test_numbering_and_markup_in_the_heading_do_not_hide_it(self):
        doc = ARTICLE.replace("## References", "## 7. **References**")
        kept, _ = strip_appendix(doc)
        assert "Rolling Stone" not in kept

    def test_russian_headings_are_recognised(self):
        doc = ("# Песня\n\n## Смысл\n\nО машине, и довольно длинно, чтобы "
               "остаток страницы был больше пятой части.\n\n"
               "## Примечания\n\n1. Источник\n")
        kept, removed = strip_appendix(doc)
        assert removed > 0 and "Источник" not in kept

    def test_an_appendix_at_the_very_top_is_refused(self):
        """If the "article" is 90% references, the headings were misread —
        keeping it whole is the safer error."""
        doc = "# A\n\n## References\n\n" + "1. ^ a citation line\n" * 50
        kept, removed = strip_appendix(doc)
        assert removed == 0 and kept == doc

    def test_empty_input_is_safe(self):
        assert strip_appendix("") == ("", 0)
        assert strip_appendix(None) == (None, 0)
