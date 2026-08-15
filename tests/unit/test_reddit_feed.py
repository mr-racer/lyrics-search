"""Reading a thread through the Atom feed.

Built after every other route was measured shut from a blocked IP: the HTML
page renders in JavaScript (an 8 KB shell), a browser-fingerprinted client gets
a 167 KB "Prove your humanity" page with status 200, and every `.json`
endpoint answers 403. The feed is the same public thread in the format Reddit
publishes for readers, and it answers about once a minute.
"""

import pytest

from app.resources import reddit_feed as reddit_rss

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>All tracks and radio stations in GTA V : GrandTheftAutoV</title>
  <entry>
    <id>t3_1mvobd</id>
    <title>All tracks and radio stations in GTA V</title>
    <content type="html">&lt;!-- SC_OFF --&gt;&lt;div class="md"&gt;&lt;p&gt;Here is the full list.&lt;/p&gt;&lt;/div&gt;&lt;!-- SC_ON --&gt;</content>
  </entry>
  <entry>
    <author><name>/u/DaddySasquatch</name></author>
    <id>t1_ccd12i4</id>
    <content type="html">&lt;div class="md"&gt;&lt;p&gt;&lt;strong&gt;East Los FM&lt;/strong&gt;&lt;/p&gt;&lt;p&gt;&lt;a href="http://x"&gt;Los Buitres - El Cocaino&lt;/a&gt;&lt;/p&gt;&lt;/div&gt;</content>
  </entry>
  <entry>
    <author><name>/u/someone</name></author>
    <id>t1_ccd7xz3</id>
    <content type="html">&lt;div class="md"&gt;&lt;p&gt;Can&amp;#39;t find these anywhere&lt;/p&gt;&lt;/div&gt;</content>
  </entry>
</feed>
"""


class TestFeedUrl:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.reddit.com/r/x/comments/1/t/",
         "https://www.reddit.com/r/x/comments/1/t/.rss"),
        ("https://www.reddit.com/r/x/comments/1/t",
         "https://www.reddit.com/r/x/comments/1/t/.rss"),
    ])
    def test_the_suffix_is_appended_once(self, url, expected):
        assert reddit_rss.feed_url(url) == expected

    def test_a_feed_url_is_left_alone(self):
        url = "https://www.reddit.com/r/x/comments/1/t/.rss"
        assert reddit_rss.feed_url(url) == url

    def test_query_and_fragment_go(self):
        """They mean nothing to the feed and would land after the suffix."""
        assert reddit_rss.feed_url(
            "https://www.reddit.com/r/x/comments/1/t/?utm_source=share#top"
        ) == "https://www.reddit.com/r/x/comments/1/t/.rss"


class TestParsing:
    def test_the_post_and_every_comment_come_through(self):
        title, markdown = reddit_rss.parse_feed(FEED)
        assert title == "All tracks and radio stations in GTA V"
        assert "Here is the full list." in markdown
        assert "Los Buitres - El Cocaino" in markdown
        assert "Can't find these anywhere" in markdown

    def test_the_subreddit_is_not_part_of_the_title(self):
        title, _ = reddit_rss.parse_feed(FEED)
        assert " : GrandTheftAutoV" not in title

    def test_each_comment_gets_a_heading_to_cut_on(self):
        """A thread is a pile of independent answers. Splitting on length
        instead would staple one person's tracklist to the next one's."""
        _, markdown = reddit_rss.parse_feed(FEED)
        assert "## /u/DaddySasquatch" in markdown
        assert "## /u/someone" in markdown

    def test_link_text_survives_the_tag(self):
        """In a tracklist thread the songs ARE the link labels, so dropping
        anchors whole throws away what the thread was read for."""
        _, markdown = reddit_rss.parse_feed(FEED)
        assert "Los Buitres - El Cocaino" in markdown
        assert "http://x" not in markdown

    def test_reddits_own_wrappers_are_not_content(self):
        _, markdown = reddit_rss.parse_feed(FEED)
        assert "SC_OFF" not in markdown and "SC_ON" not in markdown
        assert "<div" not in markdown and "&#39;" not in markdown

    @pytest.mark.parametrize("xml", [
        "", "not xml at all", "<html><body>nope</body></html>",
        '<feed xmlns="http://www.w3.org/2005/Atom"><title>x</title></feed>',
    ])
    def test_anything_that_is_not_a_thread_feed_is_none(self, xml):
        assert reddit_rss.parse_feed(xml) is None


class _Resp:
    headers = {}

    def read(self):
        return FEED.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestCooldown:
    """The cooldown is a GATE, not a wait.

    In the lab it was a ``time.sleep`` in front of the request, which is right
    for a notebook and wrong for a server: a user's question would sit for a
    minute inside a worker thread on the chance a comment thread helps. Reddit
    is a parachute that only opens when nothing else answered, so a skip means
    "the parachute did not open", never "the request hung".
    """

    def test_the_first_call_goes_through(self, monkeypatch):
        monkeypatch.setattr(reddit_rss, "_next_call_at", 0.0)
        monkeypatch.setattr(reddit_rss.urllib.request, "build_opener",
                            lambda *a, **kw: _Opener())
        assert reddit_rss.fetch_thread(
            "https://www.reddit.com/r/x/comments/1/t/", cooldown=60) is not None

    def test_a_second_call_inside_the_cooldown_is_skipped_not_delayed(
            self, monkeypatch):
        slept = []
        monkeypatch.setattr(reddit_rss.time, "sleep", slept.append)
        monkeypatch.setattr(reddit_rss, "_next_call_at", 0.0)
        monkeypatch.setattr(reddit_rss.urllib.request, "build_opener",
                            lambda *a, **kw: _Opener())
        url = "https://www.reddit.com/r/x/comments/1/t/"
        assert reddit_rss.fetch_thread(url, cooldown=60) is not None
        assert reddit_rss.fetch_thread(url, cooldown=60) is None
        assert slept == []                      # and nothing waited

    def test_the_cooldown_is_claimed_before_the_request(self, monkeypatch):
        """A slow read must not let a second caller through behind it."""
        monkeypatch.setattr(reddit_rss, "_next_call_at", 0.0)
        monkeypatch.setattr(reddit_rss.urllib.request, "build_opener",
                            lambda *a, **kw: _Opener())
        reddit_rss.fetch_thread("https://www.reddit.com/r/x/comments/1/t/",
                                cooldown=60)
        assert reddit_rss.cooldown_remaining() > 0

    def test_an_http_error_costs_one_page_and_no_exception(self, monkeypatch):
        import urllib.error

        monkeypatch.setattr(reddit_rss, "_next_call_at", 0.0)

        class _Boom:
            @staticmethod
            def open(*a, **kw):
                raise urllib.error.HTTPError("u", 403, "Blocked", {}, None)

        monkeypatch.setattr(reddit_rss.urllib.request, "build_opener",
                            lambda *a, **kw: _Boom())
        assert reddit_rss.fetch_thread(
            "https://www.reddit.com/r/x/comments/1/t/", cooldown=0) is None


class _Opener:
    @staticmethod
    def open(*a, **kw):
        return _Resp()
