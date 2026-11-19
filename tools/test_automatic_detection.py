"""Pixel timeline and automatic recovery regressions using isolated captures."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageDraw

from tools import annotation_assistant as assistant
from tools.annotation_geometry import inside_track, localize_timeline, track_box
from tools.dashboard import server


def timeline_fixture(*, thumb=True, color=(255, 0, 51), played=260):
    image = Image.new("RGB", (640, 400), (40, 70, 95))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 304, 639, 399), fill="black")
    draw.rectangle((20, 300, 619, 303), fill=(120, 120, 120))
    draw.rectangle((20, 300, played, 303), fill=color)
    if thumb:
        draw.ellipse((played-11, 291, played+11, 313), fill=color)
    return image


SLIDER = {"class": "progress_bar", "xyxy": [20, 265, 620, 305],
          "source": "studio_accessibility_matcher", "value": "50 percent"}


class TimelineLocalizationTests(unittest.TestCase):
    def test_partial_red_and_gray_track_and_thumb_are_localized_from_pixels(self):
        original = deepcopy(SLIDER)
        found, reason = localize_timeline(np.asarray(timeline_fixture()), SLIDER)
        self.assertEqual([item["class"] for item in found], ["progress_bar", "playhead"], reason)
        self.assertEqual(found[0]["xyxy"], [20, 299.5, 620, 304.5])
        self.assertEqual(found[0]["visible_content_xyxy"], [20, 300, 620, 304])
        self.assertEqual(found[1]["xyxy"], [249, 291, 272, 314])
        self.assertTrue(inside_track(found[1]["xyxy"], found[0]["xyxy"]))
        self.assertGreater(found[1]["xyxy"][3], SLIDER["xyxy"][3])
        self.assertTrue(all(item["review_required"] for item in found))
        self.assertEqual(SLIDER, original)

    def test_track_without_a_visible_thumb_does_not_invent_a_playhead(self):
        for played in (100, 500):
            found, reason = localize_timeline(np.asarray(timeline_fixture(thumb=False, played=played)), SLIDER)
            self.assertEqual([item["class"] for item in found], ["progress_bar"], reason)
            self.assertEqual(found[0]["xyxy"], [20, 299.5, 620, 304.5])

    def test_centered_track_height_scales_with_original_image_height(self):
        track = [20, 300, 620, 304]
        for height in (720, 1668, 2388):
            box = track_box(track, height)
            self.assertEqual(box[::2], track[::2])
            self.assertEqual((box[1]+box[3])/2, 302)
            self.assertTrue(.012 <= (box[3]-box[1])/height <= .015)

    def test_white_thumb_is_supported_without_using_slider_percentage(self):
        found, reason = localize_timeline(np.asarray(timeline_fixture(color="white", played=430)), SLIDER)
        self.assertEqual([item["class"] for item in found], ["progress_bar", "playhead"], reason)
        self.assertEqual(found[1]["xyxy"], [419, 291, 442, 314])

    def test_timestamp_or_layout_metadata_without_track_pixels_produces_nothing(self):
        blank = np.zeros((400, 640, 3), dtype=np.uint8)
        found, _ = localize_timeline(blank, {**SLIDER, "value": "36 seconds of 1 minute 26 seconds"})
        self.assertFalse(found)

    def test_unattached_circle_and_a_square_protrusion_are_not_playheads(self):
        for shape in ("circle", "rectangle"):
            image = timeline_fixture(thumb=False)
            draw = ImageDraw.Draw(image)
            if shape == "circle":
                draw.ellipse((249, 270, 271, 292), fill=(255, 0, 51))
            else:
                draw.rectangle((249, 291, 271, 313), fill=(255, 0, 51))
            found, reason = localize_timeline(np.asarray(image), SLIDER)
            self.assertEqual([item["class"] for item in found], ["progress_bar"], reason)

    def test_multiple_parallel_stripes_require_manual_review(self):
        image = timeline_fixture(thumb=False)
        ImageDraw.Draw(image).rectangle((20, 280, 619, 283), fill=(200, 200, 200))
        found, reason = localize_timeline(np.asarray(image), SLIDER)
        self.assertFalse(found)
        self.assertIn("Multiple", reason)


class AutomaticRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="automatic-detection-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.collected = self.root / "youtube/tablet/short_video/collected"
        (self.collected / "images").mkdir(parents=True)
        (self.collected / "accessibility").mkdir()
        image = timeline_fixture()
        draw = ImageDraw.Draw(image)
        draw.polygon([(36, 34), (45, 34), (55, 25), (55, 60), (45, 51), (36, 51)], outline="white", width=3)
        draw.line((62, 36, 72, 48), fill="white", width=3)
        draw.line((72, 36, 62, 48), fill="white", width=3)
        draw.ellipse((105, 25, 128, 48), outline="white", width=3)
        draw.line((124, 46, 136, 58), fill="white", width=3)
        self.image_path = self.collected / "images/recover.png"
        image.save(self.image_path)
        self.snapshot = {"image_sha256": hashlib.sha256(self.image_path.read_bytes()).hexdigest(),
                         "coordinate_space": "image_pixels", "elements": [
                             {"label": "Unmute video", "frame": [25, 15, 60, 60], "visible": True},
                             {"label": "Search", "frame": [95, 15, 50, 50], "visible": True},
                             {"identifier": "id.player.scrubber.slider", "frame": [20, 265, 600, 40],
                              "visible": True, "value": "50 percent"}]}
        self.snapshot_path = self.collected / "accessibility/recover.json"
        self.snapshot_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
        self.raw_path = self.collected / "annotations.jsonl"
        self.record = {"record_id": "recover", "raw_annotations": [], "context": {"image_size": [640, 400]}}
        self.raw_path.write_text(json.dumps(self.record)+"\n", encoding="utf-8")
        self.original = self.raw_path.read_bytes()
        self.data_patch = patch.object(server, "DATA_ROOT", self.root)
        self.data_patch.start()
        self.addCleanup(self.data_patch.stop)
        self.args = ("youtube", "tablet", "short_video", "recover")

    def detect(self):
        return server.api_detect(*self.args, server.SuggestPayload(annotations=[]))

    def test_automatic_recovery_supplies_missing_classes_without_model_or_templates(self):
        with patch.dict(os.environ, {assistant.LEGACY_MODEL_FLAG: "1"}), \
                patch.object(assistant, "model_proposals", side_effect=AssertionError("No model needed")), \
                patch.object(assistant, "template_proposals", side_effect=AssertionError("No template needed")):
            result = self.detect()
        self.assertEqual({item["class"] for item in result["proposals"]},
                         {"volume_muted", "search", "progress_bar", "playhead"})
        self.assertTrue(all(item["review_required"] for item in result["proposals"]))
        self.assertEqual(self.raw_path.read_bytes(), self.original)
        self.assertFalse((self.collected / "curation/annotation_overrides.json").exists())
        self.assertTrue((self.collected / "curation/detections/recover.json").exists())

    def test_repeated_accessibility_observations_do_not_duplicate_recovered_controls(self):
        self.snapshot["elements"] *= 2
        self.snapshot_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
        result = self.detect()
        self.assertEqual(len(result["proposals"]), 4)

    def test_reviewed_timeline_requires_a_scaled_track_and_thumb_center_inside_it(self):
        short = {"class": "progress_bar", "xyxy": [20, 300, 620, 304]}
        track = {"class": "progress_bar", "xyxy": [20, 299.5, 620, 304.5]}
        thumb = {"class": "playhead", "xyxy": [249, 291, 272, 314]}
        misplaced = {"class": "playhead", "xyxy": [249, 315, 272, 338]}
        for annotations in ([short, thumb], [track, misplaced], [thumb]):
            with self.assertRaises(HTTPException) as error:
                server.api_curate(*self.args, server.CuratePayload(annotations=annotations, status="approved"))
            self.assertEqual(error.exception.status_code, 422)
        result = server.api_curate(*self.args, server.CuratePayload(annotations=[track, thumb], status="approved"))
        self.assertEqual(result["saved_boxes"], 2)

    def test_mismatched_or_missing_snapshot_cannot_create_missing_controls(self):
        self.snapshot["image_sha256"] = "wrong"
        self.snapshot_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
        self.assertFalse(self.detect()["proposals"])
        self.snapshot_path.unlink()
        self.assertFalse(self.detect()["proposals"])

    def test_padded_template_does_not_duplicate_an_existing_tight_icon(self):
        old = {"class": "more_options", "xyxy": [500, 80, 530, 88], "geometry_reviewed": True}
        padded = {"class": "more_options", "xyxy": [485, 60, 545, 108], "visual_xyxy": [500, 80, 530, 88], "source": "studio_reviewed_template"}
        distinct = {"class": "more_options", "xyxy": [400, 80, 430, 88], "visual_xyxy": [400, 80, 430, 88], "source": "studio_reviewed_template"}
        with patch.dict(os.environ, {assistant.LEGACY_MODEL_FLAG: "0"}), \
                patch.object(assistant, "template_proposals", return_value=([padded, distinct], {})):
            result = assistant.suggest(self.image_path, self.record, [old], self.root, "youtube", "short_video")
        self.assertEqual([item["visible_content_xyxy"] for item in result["proposals"]], [distinct["xyxy"]])


if __name__ == "__main__":
    unittest.main()
