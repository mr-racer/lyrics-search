"""Characterise the shared ``scroll_all`` pagination helper.

Locks the termination semantics the call sites relied on: stop when the
returned offset is ``None`` OR a page comes back empty, forwarding scroll
kwargs verbatim and advancing the offset between pages.
"""

from app.resources.qdrant_utils import scroll_all


class _FakeQdrant:
    """Returns canned ``(points, next_offset)`` pages in call order."""

    def __init__(self, pages):
        self._pages = pages
        self.calls = []

    def scroll(self, *, collection_name, offset, limit, **kw):
        self.calls.append(
            {"collection_name": collection_name, "offset": offset, "limit": limit, **kw}
        )
        return self._pages[len(self.calls) - 1]


def test_yields_every_point_and_terminates_on_none_offset():
    fake = _FakeQdrant([(["a", "b"], "off1"), (["c"], None)])

    out = list(scroll_all(fake, "col", batch_size=2, with_payload=True))

    assert out == ["a", "b", "c"]
    assert len(fake.calls) == 2
    # first call starts at offset=None with the requested batch size + kwargs
    assert fake.calls[0]["offset"] is None
    assert fake.calls[0]["limit"] == 2
    assert fake.calls[0]["with_payload"] is True
    # offset is advanced from the previous page's next_offset
    assert fake.calls[1]["offset"] == "off1"


def test_terminates_on_empty_batch_even_if_offset_not_none():
    fake = _FakeQdrant([(["a"], "off1"), ([], "off2")])

    out = list(scroll_all(fake, "col"))

    assert out == ["a"]
    assert len(fake.calls) == 2
