import asyncio
import importlib.util
import json
import math
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from openrouter_api.cancellation import NodeDeadline
from openrouter_api.client import ChatResult
from openrouter_api.media import PreparedMedia
from openrouter_api.models import ModelInfo, ModelSnapshot


class NodeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "comfyui_openrouter_api",
            root / "__init__.py",
            submodule_search_locations=[str(root)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.module = module
        cls.node_module = sys.modules[f"{spec.name}.node"]

    def test_registration_and_text_only_outputs(self):
        node = self.module.NODE_CLASS_MAPPINGS["OpenRouterSimple"]
        self.assertEqual(node.RETURN_TYPES, ("STRING", "STRING", "STRING"))
        self.assertEqual(node.RETURN_NAMES, ("text", "info", "credits"))
        self.assertEqual(self.module.WEB_DIRECTORY, "./web")

    def test_exact_approved_input_surface(self):
        node = self.module.NODE_CLASS_MAPPINGS["OpenRouterSimple"]
        inputs = node.INPUT_TYPES()
        self.assertEqual(
            set(inputs["required"]),
            {
                "system_preset",
                "system_prompt",
                "user_prompt",
                "model",
                "reasoning_effort",
                "timeout_seconds",
                "temperature",
                "max_tokens",
                "response_format",
                "zdr",
                "regenerate",
            },
        )
        self.assertEqual(inputs["required"]["regenerate"], ("BOOLEAN", {"default": True}))
        self.assertEqual(inputs["required"]["system_preset"][1]["default"], "Ref2VA")
        self.assertIn("Ref2VA", inputs["required"]["system_preset"][0])
        self.assertIn("I2VA", inputs["required"]["system_preset"][0])
        self.assertIn("RefAud2VA", inputs["required"]["system_preset"][0])
        self.assertIn("T2VA", inputs["required"]["system_preset"][0])
        self.assertIn("FL2VA", inputs["required"]["system_preset"][0])
        self.assertIn("None", inputs["required"]["system_preset"][0])
        self.assertEqual(
            list(inputs["optional"]),
            [
                "image",
                "image_2",
                "image_3",
                "video",
                "video_2",
                "video_3",
                "audio",
                "audio_2",
                "audio_3",
                "api_key",
            ],
        )

    def test_optional_key_keeps_existing_widget_order(self):
        inputs = self.node_module.OpenRouterSimple.INPUT_TYPES()
        self.assertEqual(list(inputs["required"]), [
            "model", "reasoning_effort", "timeout_seconds", "temperature",
            "max_tokens", "response_format", "zdr", "regenerate",
            "system_preset", "system_prompt", "user_prompt",
        ])
        self.assertEqual(inputs["optional"]["api_key"][0], "STRING")
        self.assertEqual(inputs["optional"]["api_key"][1]["default"], "")

    def test_key_selection_reaches_chat_and_credits_without_leaking(self):
        selected = ModelInfo("vendor/text", "Text", ("text",), ("text",), ("max_tokens",), False)
        kwargs = dict(system_prompt="", user_prompt="Hello", model="vendor/text",
                      reasoning_effort="auto", timeout_seconds=5, temperature=1,
                      max_tokens=32, response_format="text", zdr=False, regenerate=False)
        cases = [
            ({"api_key": "  explicit-key  "}, {"OPENROUTER_API_KEY": "env-key", "LLM_KEY": "legacy-key"}, "explicit-key"),
            ({"api_key": "explicit-key"}, {}, "explicit-key"),
            ({"api_key": "  "}, {"OPENROUTER_API_KEY": " env-key "}, "env-key"),
            ({}, {"OPENROUTER_API_KEY": "  ", "LLM_KEY": " legacy-key "}, "legacy-key"),
            ({}, {"LLM_KEY": "legacy-key"}, "legacy-key"),
        ]
        for key_arg, env, expected in cases:
            with self.subTest(key_arg=key_arg, env=env), mock.patch.dict(os.environ, env, clear=True), \
                 mock.patch.object(self.node_module.CATALOG, "get", mock.AsyncMock(return_value=ModelSnapshot((selected,), 1))), \
                 mock.patch.object(self.node_module, "create_chat", mock.AsyncMock(return_value=ChatResult("ok", "test", {}))) as chat, \
                 mock.patch.object(self.node_module, "lookup_credits", mock.AsyncMock(return_value="credits")) as credits:
                result = asyncio.run(self.node_module.OpenRouterSimple().run(**kwargs, **key_arg))
                self.assertEqual(chat.call_args.args[2], expected)
                self.assertEqual(credits.call_args.args[1], expected)
                self.assertNotIn(expected, json.dumps(chat.call_args.args[1]))
                self.assertNotIn(expected, json.dumps(result))
                self.assertEqual(dict(os.environ), env)
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.node_module.CATALOG, "get", mock.AsyncMock()) as catalog:
            with self.assertRaisesRegex(ValueError, "api_key"):
                asyncio.run(self.node_module.OpenRouterSimple().run(**kwargs))
            catalog.assert_not_called()

    def test_regenerate_controls_comfy_cache_fingerprint(self):
        node = self.module.NODE_CLASS_MAPPINGS["OpenRouterSimple"]
        self.assertTrue(math.isnan(node.IS_CHANGED(regenerate=True)))
        self.assertIs(node.IS_CHANGED(regenerate=False), False)

    def test_all_populated_media_slots_are_prepared_in_stable_order(self):
        values = [(name, object()) for name in (
            "image",
            "image_2",
            "image_3",
            "video",
            "video_2",
            "video_3",
            "audio",
            "audio_2",
            "audio_3",
        )]

        async def prepare(modality):
            return PreparedMedia(modality, "application/octet-stream", modality.encode(), 1, {})

        async def prepare_image_mock(_deadline, _value):
            return await prepare("image")

        async def prepare_video_mock(_deadline, _value):
            return await prepare("video")

        async def prepare_audio_mock(_deadline, _value):
            return await prepare("audio")

        with (
            mock.patch.object(
                self.node_module,
                "prepare_image",
                side_effect=prepare_image_mock,
            ),
            mock.patch.object(
                self.node_module,
                "prepare_video",
                side_effect=prepare_video_mock,
            ),
            mock.patch.object(
                self.node_module,
                "prepare_audio",
                side_effect=prepare_audio_mock,
            ),
        ):
            prepared = asyncio.run(
                self.node_module._prepare_media(NodeDeadline(2), media_inputs=values)
            )

        self.assertEqual([name for name, _item in prepared], [name for name, _value in values])
        self.assertEqual(
            [item.modality for _name, item in prepared],
            ["image"] * 3 + ["video"] * 3 + ["audio"] * 3,
        )

    def test_sparse_additional_slots_reach_payload_and_named_info(self):
        selected = ModelInfo(
            id="vendor/omni",
            name="Omni",
            input_modalities=("text", "image", "video", "audio"),
            output_modalities=("text",),
            supported_parameters=("max_tokens",),
            reasoning=False,
        )
        snapshot = ModelSnapshot(models=(selected,), fetched_at=1.0)
        captured_payload = None

        async def catalog_get(**_kwargs):
            return snapshot

        async def prepare_image_mock(_deadline, value):
            return PreparedMedia("image", "image/webp", str(value).encode(), 1, {})

        async def prepare_video_mock(_deadline, value):
            return PreparedMedia("video", "video/mp4", str(value).encode(), 1, {})

        async def prepare_audio_mock(_deadline, value):
            return PreparedMedia("audio", "audio/mpeg", str(value).encode(), 1, {})

        async def create_chat_mock(_deadline, payload, _api_key):
            nonlocal captured_payload
            captured_payload = payload
            return ChatResult(text="ok", response_id="gen-test", usage={})

        async def credits_mock(_deadline, _api_key):
            return "credits"

        with (
            mock.patch.object(self.node_module.CATALOG, "get", side_effect=catalog_get),
            mock.patch.object(self.node_module, "resolve_generation_key", return_value="key"),
            mock.patch.object(self.node_module, "prepare_image", side_effect=prepare_image_mock),
            mock.patch.object(self.node_module, "prepare_video", side_effect=prepare_video_mock),
            mock.patch.object(self.node_module, "prepare_audio", side_effect=prepare_audio_mock),
            mock.patch.object(self.node_module, "create_chat", side_effect=create_chat_mock),
            mock.patch.object(self.node_module, "lookup_credits", side_effect=credits_mock),
        ):
            text, info_json, credits = asyncio.run(
                self.node_module.OpenRouterSimple().run(
                    system_prompt="",
                    user_prompt="enumerate",
                    model="vendor/omni",
                    reasoning_effort="auto",
                    timeout_seconds=5,
                    temperature=1,
                    max_tokens=64,
                    response_format="text",
                    zdr=False,
                    regenerate=True,
                    image="first-image",
                    image_3="third-image",
                    video_2="second-video",
                    audio_3={"clip": "third-audio"},
                )
            )

        self.assertEqual((text, credits), ("ok", "credits"))
        self.assertIsNotNone(captured_payload)
        self.assertNotIn("regenerate", captured_payload)
        user_message = [m for m in captured_payload["messages"] if m["role"] == "user"][0]
        content = user_message["content"]
        self.assertEqual(
            [part["type"] for part in content],
            ["text", "image_url", "image_url", "video_url", "input_audio"],
        )
        info = json.loads(info_json)
        self.assertEqual(info["required_modalities"], ["audio", "image", "text", "video"])
        self.assertEqual(list(info["media"]), ["image", "image_3", "video_2", "audio_3"])

    def test_system_presets_injection(self):
        selected = ModelInfo("vendor/text", "Text", ("text",), ("text",), ("max_tokens",), False)
        snapshot = ModelSnapshot(models=(selected,), fetched_at=1.0)
        captured_payload = None

        async def create_chat_mock(_deadline, payload, _api_key):
            nonlocal captured_payload
            captured_payload = payload
            return ChatResult(text="ok", response_id="gen-test", usage={})

        async def credits_mock(_deadline, _api_key):
            return "credits"

        node = self.node_module.OpenRouterSimple()
        base_kwargs = dict(
            user_prompt="Hello",
            model="vendor/text",
            reasoning_effort="auto",
            timeout_seconds=5,
            temperature=1.0,
            max_tokens=64,
            response_format="text",
            zdr=False,
            regenerate=True,
            api_key="test-key",
        )

        with (
            mock.patch.object(self.node_module.CATALOG, "get", mock.AsyncMock(return_value=snapshot)),
            mock.patch.object(self.node_module, "create_chat", side_effect=create_chat_mock),
            mock.patch.object(self.node_module, "lookup_credits", side_effect=credits_mock),
        ):
            # 1. Default preset Ref2VA with empty system_prompt
            asyncio.run(node.run(**base_kwargs, system_preset="Ref2VA", system_prompt=""))
            sys_msg = [m for m in captured_payload["messages"] if m["role"] == "system"]
            self.assertEqual(len(sys_msg), 1)
            self.assertIn("Ref2VA (Full Reference Mode)", sys_msg[0]["content"])
            self.assertIn("subject_definitions", sys_msg[0]["content"])
            self.assertIn("retention_analysis", sys_msg[0]["content"])

            # 2. Ref2VA with custom user instructions appended
            asyncio.run(node.run(**base_kwargs, system_preset="Ref2VA", system_prompt="Keep it punchy."))
            sys_msg = [m for m in captured_payload["messages"] if m["role"] == "system"]
            self.assertIn("Ref2VA (Full Reference Mode)", sys_msg[0]["content"])
            self.assertIn("[Additional Instructions]\nKeep it punchy.", sys_msg[0]["content"])

            # 3. I2VA preset
            asyncio.run(node.run(**base_kwargs, system_preset="I2VA", system_prompt=""))
            sys_msg = [m for m in captured_payload["messages"] if m["role"] == "system"]
            self.assertIn("Task: I2VA", sys_msg[0]["content"])
            self.assertIn("<Picture 1> (from [Shot 1]) is fully referenced", sys_msg[0]["content"])

            # 4. RefAud2VA preset
            asyncio.run(node.run(**base_kwargs, system_preset="RefAud2VA", system_prompt=""))
            sys_msg = [m for m in captured_payload["messages"] if m["role"] == "system"]
            self.assertIn("Task: RefAud2VA", sys_msg[0]["content"])
            self.assertIn("<Audio 1> is the synchronized audio track", sys_msg[0]["content"])

            # 5. T2VA preset
            asyncio.run(node.run(**base_kwargs, system_preset="T2VA", system_prompt=""))
            sys_msg = [m for m in captured_payload["messages"] if m["role"] == "system"]
            self.assertIn("Task: T2VA", sys_msg[0]["content"])

            # 6. FL2VA preset
            asyncio.run(node.run(**base_kwargs, system_preset="FL2VA", system_prompt=""))
            sys_msg = [m for m in captured_payload["messages"] if m["role"] == "system"]
            self.assertIn("Task: FL2VA", sys_msg[0]["content"])
            self.assertIn("Picture 1 (from Shot 1) aligns with the 0.00-second mark", sys_msg[0]["content"])

            # 7. None preset with empty system_prompt -> no system message
            asyncio.run(node.run(**base_kwargs, system_preset="None", system_prompt=""))
            sys_msg = [m for m in captured_payload["messages"] if m["role"] == "system"]
            self.assertEqual(len(sys_msg), 0)

            # 8. None preset with custom prompt -> only custom prompt
            asyncio.run(node.run(**base_kwargs, system_preset="None", system_prompt="Custom solo prompt"))
            sys_msg = [m for m in captured_payload["messages"] if m["role"] == "system"]
            self.assertEqual(len(sys_msg), 1)
            self.assertEqual(sys_msg[0]["content"], "Custom solo prompt")


if __name__ == "__main__":
    unittest.main()
