import asyncio
import json
import os
import time
import unittest
from unittest import mock

from aiohttp import web

from openrouter_api.cancellation import NodeDeadline, NodeTimeoutError
from openrouter_api.client import OpenRouterRequestError, create_chat, lookup_credits


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def start_server(self, routes):
        app = web.Application()
        for method, path, handler in routes:
            app.router.add_route(method, path, handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.addAsyncCleanup(runner.cleanup)
        return f"http://127.0.0.1:{port}"

    async def test_stalled_chat_obeys_total_deadline(self):
        release = asyncio.Event()

        async def stalled(_request):
            await release.wait()
            return web.json_response({"choices": [{"message": {"content": "late"}}]})

        base = await self.start_server([("POST", "/chat/completions", stalled)])
        started = time.monotonic()
        try:
            with mock.patch.dict(os.environ, {"OPENROUTER_BASE_URL": base}):
                with self.assertRaises(NodeTimeoutError):
                    await create_chat(NodeDeadline(0.2, poll_interval=0.02), {"model": "x"}, "secret")
        finally:
            release.set()
        self.assertLess(time.monotonic() - started, 1.0)

    async def test_rejection_is_friendly_and_redacted(self):
        secret = "sk-or-v1-SUPERSECRET"

        async def rejected(_request):
            return web.json_response({"error": {"message": f"bad Bearer {secret}"}}, status=401)

        base = await self.start_server([("POST", "/chat/completions", rejected)])
        with mock.patch.dict(os.environ, {"OPENROUTER_BASE_URL": base}):
            with self.assertRaises(OpenRouterRequestError) as raised:
                await create_chat(NodeDeadline(2), {"model": "x"}, secret)
        message = str(raised.exception)
        self.assertIn("authentication failed", message)
        self.assertNotIn(secret, message)
        self.assertIn("[REDACTED]", message)

    async def test_rejection_surfaces_nested_provider_diagnostic(self):
        secret = "sk-or-v1-SUPERSECRET"
        saw_metadata_header = False

        async def rejected(request):
            nonlocal saw_metadata_header
            saw_metadata_header = request.headers.get("X-OpenRouter-Metadata") == "enabled"
            return web.json_response(
                {
                    "error": {
                        "message": "Provider returned error",
                        "code": 400,
                        "metadata": {
                            "provider_name": "Google AI Studio",
                            "provider_error_code": "INVALID_ARGUMENT",
                            "raw": json.dumps(
                                {
                                    "error": {
                                        "message": (
                                            "inline video data:video/mp4;base64,AAAAAA was rejected; "
                                            f"Bearer {secret}"
                                        )
                                    }
                                }
                            ),
                        },
                    },
                    "openrouter_metadata": {
                        "summary": "available=1, attempts exhausted",
                        "attempts": [{"provider": "Google AI Studio", "status": 400}],
                    },
                },
                status=400,
            )

        base = await self.start_server([("POST", "/chat/completions", rejected)])
        with mock.patch.dict(os.environ, {"OPENROUTER_BASE_URL": base}):
            with self.assertRaises(OpenRouterRequestError) as raised:
                await create_chat(NodeDeadline(2), {"model": "x"}, secret)
        message = str(raised.exception)
        self.assertTrue(saw_metadata_header)
        self.assertIn("Google AI Studio", message)
        self.assertIn("INVALID_ARGUMENT", message)
        self.assertIn("inline video [REDACTED MEDIA] was rejected", message)
        self.assertIn("available=1, attempts exhausted", message)
        self.assertNotIn("AAAAAA", message)
        self.assertNotIn(secret, message)

    async def test_credits_failure_is_non_fatal(self):
        async def failed(_request):
            return web.json_response({}, status=500)

        base = await self.start_server([("GET", "/key", failed)])
        with mock.patch.dict(os.environ, {"OPENROUTER_BASE_URL": base}, clear=False):
            with mock.patch.dict(os.environ, {"OPENROUTER_MANAGEMENT_KEY": ""}, clear=False):
                result = await lookup_credits(NodeDeadline(1), "generation-key")
        self.assertTrue(result.startswith("Credits unavailable"))

    async def test_thinking_tags_are_cropped_from_output(self):
        async def thinking_response(_request):
            return web.json_response({
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "<think>\nThinking about Punjabi words...\n</think>\nsubject_definitions:\n<Picture 1> celestial palace",
                    },
                }],
                "id": "gen-test-1",
                "usage": {},
            })

        base = await self.start_server([("POST", "/chat/completions", thinking_response)])
        with mock.patch.dict(os.environ, {"OPENROUTER_BASE_URL": base}):
            result = await create_chat(NodeDeadline(2), {"model": "x"}, "secret")
        self.assertEqual(result.text, "subject_definitions:\n<Picture 1> celestial palace")

    async def test_empty_content_with_only_reasoning_raises_clear_error(self):
        async def reasoning_only_response(_request):
            return web.json_response({
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "The detailed prompt thoughts and analysis",
                    },
                }],
                "id": "gen-test-2",
                "usage": {},
            })

        base = await self.start_server([("POST", "/chat/completions", reasoning_only_response)])
        with mock.patch.dict(os.environ, {"OPENROUTER_BASE_URL": base}):
            with self.assertRaises(OpenRouterRequestError) as raised:
                await create_chat(NodeDeadline(2), {"model": "x"}, "secret")
        self.assertIn("reasoning thoughts", str(raised.exception))

    async def test_max_tokens_length_diagnostic(self):
        async def length_exhausted(_request):
            return web.json_response({
                "choices": [{
                    "index": 0,
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": None,
                    },
                }],
                "id": "gen-test-2",
                "usage": {},
            })

        base = await self.start_server([("POST", "/chat/completions", length_exhausted)])
        with mock.patch.dict(os.environ, {"OPENROUTER_BASE_URL": base}):
            with self.assertRaises(OpenRouterRequestError) as raised:
                await create_chat(NodeDeadline(2), {"model": "x"}, "secret")
        self.assertIn("max_tokens", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
