"""Current capture/geometry suggestions must not depend on the archived model."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from tools import annotation_assistant as assistant


class LegacyModelConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="model-independent-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "sample.png"
        Image.new("RGB", (300, 400), "#123456").save(self.image)

    def suggest(self, platform="youtube"):
        return assistant.suggest(self.image, {}, [], self.root, platform, "short_video")

    def test_default_does_not_call_or_load_legacy_model(self):
        with patch.dict(os.environ, {assistant.LEGACY_MODEL_FLAG: ""}), \
                patch.object(assistant, "model_proposals") as model, \
                patch.object(assistant, "reference") as reference, \
                patch.object(assistant, "template_proposals", return_value=([], {})):
            result = self.suggest()
        model.assert_not_called()
        reference.assert_not_called()
        self.assertEqual(result["model_status"]["mode"], "disabled")
        self.assertFalse(result["model_status"]["used"])
        self.assertFalse(result["model_status"]["replacement_configured"])
        self.assertEqual(result["model_classes"], [])

    def test_cached_sessions_do_not_enable_or_advertise_disabled_model(self):
        with patch.dict(os.environ, {assistant.LEGACY_MODEL_FLAG: "0"}), \
                patch.object(assistant, "_RUNTIMES", (object(), object())), \
                patch.object(assistant, "model_proposals") as model, \
                patch.object(assistant, "reference") as reference, \
                patch.object(assistant, "template_proposals", return_value=([], {})):
            result = self.suggest()
        model.assert_not_called()
        reference.assert_not_called()
        self.assertEqual(result["model_classes"], [])
        self.assertEqual(result["model_status"]["mode"], "disabled")

    def test_legacy_inference_requires_explicit_opt_in(self):
        image = Image.new("RGB", (300, 400), "black")
        ImageDraw.Draw(image).polygon([(26, 25), (26, 55), (55, 40)], fill="white")
        image.save(self.image)
        prediction = {"class": "play", "xyxy": [10, 10, 70, 70],
                      "source": "studio_v11_onnx", "review_required": True}
        with patch.dict(os.environ, {assistant.LEGACY_MODEL_FLAG: "1"}), \
                patch.object(assistant, "_RUNTIMES", None), \
                patch.object(assistant, "model_proposals", return_value=[prediction]) as model, \
                patch.object(assistant, "template_proposals", return_value=([], {})):
            result = self.suggest()
        model.assert_called_once()
        self.assertEqual(result["model_status"]["mode"], "legacy_opt_in")
        self.assertTrue(result["model_status"]["used"])
        self.assertEqual(result["proposals"][0]["class"], "play")

    def test_legacy_opt_in_does_not_run_youtube_model_on_other_platforms(self):
        with patch.dict(os.environ, {assistant.LEGACY_MODEL_FLAG: "1"}), \
                patch.object(assistant, "model_proposals") as model, \
                patch.object(assistant, "template_proposals", return_value=([], {})):
            result = self.suggest("tiktok")
        model.assert_not_called()
        self.assertEqual(result["model_status"]["mode"], "unsupported_platform")
        self.assertEqual(result["model_classes"], [])


if __name__ == "__main__":
    unittest.main()
