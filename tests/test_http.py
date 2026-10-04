import unittest

from openrouter_simple.http import ResponseTooLarge, read_bounded


class ChunkedContent:
    def __init__(self, chunks):
        self.chunks = chunks

    async def iter_chunked(self, _size):
        for chunk in self.chunks:
            yield chunk


class BoundedReadTests(unittest.IsolatedAsyncioTestCase):
    async def test_accumulates_all_network_chunks_to_eof(self):
        content = ChunkedContent([b'{"data":', b'[{"id":"model"}', b']}' ])
        body = await read_bounded(content, 1000, label="catalog")
        self.assertEqual(body, b'{"data":[{"id":"model"}]}')

    async def test_rejects_as_soon_as_limit_is_crossed(self):
        content = ChunkedContent([b"1234", b"5678"])
        with self.assertRaisesRegex(ResponseTooLarge, "safety limit"):
            await read_bounded(content, 7, label="response")


if __name__ == "__main__":
    unittest.main()
