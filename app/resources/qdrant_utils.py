"""Small Qdrant helpers shared across services.

`scroll_all` centralises the offset-following pagination loop that was copied
across indexing/library/similarity code. It yields **every** point in a
collection, following ``next_page_offset`` until it comes back ``None`` (or a
batch comes back empty). It does not impose a page/count cap and propagates any
client exception to the caller — callers that need a cap, an early break, or
partial-result-on-error must keep their own loop.
"""

from __future__ import annotations

from typing import Iterator


def scroll_all(client, collection_name: str, *, batch_size: int = 256, **scroll_kwargs) -> Iterator:
    """Yield each point of ``collection_name``, paging via Qdrant scroll.

    Parameters mirror ``client.scroll`` — extra keyword args (``with_payload``,
    ``with_vectors``, ``scroll_filter`` …) are forwarded verbatim. Stops when
    the returned offset is ``None`` or a page is empty, matching the
    ``if offset is None or not points: break`` idiom the call sites used.
    """
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=batch_size,
            **scroll_kwargs,
        )
        for point in points:
            yield point
        if offset is None or not points:
            break
