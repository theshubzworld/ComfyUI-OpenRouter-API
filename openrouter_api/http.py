from __future__ import annotations

from typing import Any


class ResponseTooLarge(RuntimeError):
    pass


async def read_bounded(content: Any, limit: int, *, label: str) -> bytes:
    """Read a chunked HTTP body to EOF without ever retaining more than limit bytes."""

    body = bytearray()
    async for chunk in content.iter_chunked(min(64 * 1024, limit + 1)):
        body.extend(chunk)
        if len(body) > limit:
            raise ResponseTooLarge(f"{label} exceeded the {limit} byte safety limit")
    return bytes(body)
