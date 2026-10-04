import unittest

from openrouter_simple.media import PreparedMedia
from openrouter_simple.models import ModelInfo
from openrouter_simple.payload import build_payload


def model(*parameters: str, reasoning: bool = True) -> ModelInfo:
    return ModelInfo(
        id="vendor/model",
        name="Model",
        input_modalities=("text", "image", "video", "audio"),
        output_modalities=("text",),
        supported_parameters=tuple(parameters),
        reasoning=reasoning,
    )


class PayloadTests(unittest.TestCase):
    def test_text_only_output_and_approved_controls(self):
        media = [PreparedMedia("image", "image/webp", b"abc", 9, {})]
        payload, info = build_payload(
            model=model("temperature", "max_completion_tokens", "seed", "reasoning", "response_format"),
            system_prompt="system",
            user_prompt="user",
            media=media,
            reasoning_effort="high",
            temperature=0.5,
            max_tokens=1234,
            response_format="json_object",
            zdr=True,
        )
        self.assertEqual(payload["modalities"], ["text"])
        self.assertEqual(payload["max_completion_tokens"], 1234)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["provider"], {"zdr": True})
        self.assertNotIn("seed", payload)
        self.assertNotIn("seed", info["applied"])
        self.assertNotIn("seed", info["omitted"])
        self.assertEqual(payload["messages"][1]["content"][1]["type"], "image_url")
        self.assertEqual(info["applied"]["max_tokens"]["wire_parameter"], "max_completion_tokens")
        for excluded in ("tools", "web_search", "top_p", "plugins", "transforms"):
            self.assertNotIn(excluded, payload)

    def test_legacy_budget_and_unsupported_optional_parameters(self):
        payload, info = build_payload(
            model=model("max_tokens", reasoning=False),
            system_prompt="",
            user_prompt="hello",
            media=[],
            reasoning_effort="auto",
            temperature=1.0,
            max_tokens=99,
            response_format="text",
            zdr=False,
        )
        self.assertEqual(payload["max_tokens"], 99)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("seed", payload)
        self.assertIn("temperature", info["omitted"])
        self.assertNotIn("seed", info["omitted"])

    def test_all_nine_media_items_are_serialized_in_order(self):
        media = [
            *(PreparedMedia("image", "image/webp", bytes([index]), 1, {}) for index in range(3)),
            *(PreparedMedia("video", "video/mp4", bytes([index]), 1, {}) for index in range(3)),
            *(PreparedMedia("audio", "audio/mpeg", bytes([index]), 1, {}) for index in range(3)),
        ]
        payload, _info = build_payload(
            model=model("max_tokens"),
            system_prompt="",
            user_prompt="enumerate every attachment",
            media=media,
            reasoning_effort="auto",
            temperature=1,
            max_tokens=256,
            response_format="text",
            zdr=False,
        )
        content = payload["messages"][0]["content"]
        self.assertEqual(
            [part["type"] for part in content],
            ["text"] + ["image_url"] * 3 + ["video_url"] * 3 + ["input_audio"] * 3,
        )

    def test_json_format_requires_model_support(self):
        with self.assertRaisesRegex(ValueError, "JSON response"):
            build_payload(
                model=model(),
                system_prompt="",
                user_prompt="hello",
                media=[],
                reasoning_effort="auto",
                temperature=1,
                max_tokens=10,
                response_format="json_object",
                zdr=False,
            )


if __name__ == "__main__":
    unittest.main()
