import unittest

from openrouter_api.models import normalize_model


class ModelTests(unittest.TestCase):
    def test_normalizes_and_checks_modality_intersection(self):
        model = normalize_model(
            {
                "id": "vendor/omni",
                "name": "Omni",
                "architecture": {
                    "input_modalities": ["text", "image", "video", "audio"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["temperature", "seed", "reasoning"],
                "reasoning": {"supported": True},
            }
        )
        self.assertIsNotNone(model)
        self.assertTrue(model.accepts({"text", "image", "video", "audio"}))
        self.assertTrue(model.supports("seed"))
        self.assertTrue(model.reasoning)

    def test_rejects_non_text_output_models(self):
        model = normalize_model(
            {
                "id": "vendor/image-only",
                "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
            }
        )
        self.assertIsNone(model)

    def test_requires_every_connected_modality(self):
        model = normalize_model(
            {
                "id": "vendor/vision",
                "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            }
        )
        self.assertTrue(model.accepts({"text", "image"}))
        self.assertFalse(model.accepts({"text", "image", "audio"}))

    def test_find_model_supports_alias_and_case_insensitive_lookup(self):
        from openrouter_api.models import ModelInfo, ModelSnapshot

        m1 = ModelInfo("qwen/qwen3.8-27b:free", "Qwen 27B Free", ("text", "image"), ("text",), (), True)
        m2 = ModelInfo("google/gemini-2.5-flash", "Gemini 2.5 Flash", ("text", "image"), ("text",), (), True)
        snapshot = ModelSnapshot(models=(m1, m2), fetched_at=100.0)

        # Exact
        self.assertEqual(snapshot.find_model("qwen/qwen3.8-27b:free"), m1)
        # Case insensitive
        self.assertEqual(snapshot.find_model("QWEN/QWEN3.8-27B:FREE"), m1)
        # Token alias
        self.assertEqual(snapshot.find_model("qwen 27b"), m1)
        self.assertEqual(snapshot.find_model("qwen 27b free"), m1)
        self.assertEqual(snapshot.find_model("gemini flash"), m2)
        # None for unknown
        self.assertIsNone(snapshot.find_model("unknown model"))


if __name__ == "__main__":
    unittest.main()
