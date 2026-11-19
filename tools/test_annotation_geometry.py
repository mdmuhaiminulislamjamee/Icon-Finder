"""Icon/text separation regressions; fixtures have known rendered geometry."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageDraw, ImageFont

from tools import annotation_assistant as assistant
from tools.annotation_geometry import geometry_policy, geometry_proposals, localize_control
from tools.annotation_quality import accessibility_proposals, clean_annotations, match_accessibility
from tools.dashboard import server


def icon_fixture(theme="dark", icon=True):
    background, foreground = ("black", "white") if theme == "dark" else ("white", "black")
    if theme == "muted":
        background, foreground = (84, 100, 42), (158, 170, 136)
    image = Image.new("RGB", (300, 400), background)
    draw = ImageDraw.Draw(image)
    if icon:
        draw.line([(90, 80), (128, 80), (128, 126), (109, 114), (90, 126), (90, 80)], fill=foreground, width=3)
    font = ImageFont.load_default(size=18)
    draw.text((86, 152), "Save", fill=foreground, font=font)
    draw.text((87, 180), "40K", fill=foreground, font=font)
    return image


class VisualGeometryTests(unittest.TestCase):
    def test_pixel_box_excludes_label_and_count_in_both_themes(self):
        for theme in ("dark", "light", "muted"):
            with self.subTest(theme=theme):
                raw = {"class": "save", "xyxy": [60, 60, 160, 210], "source": "ios_xctest_accessibility"}
                before = deepcopy(raw)
                result, reason = localize_control(np.asarray(icon_fixture(theme)), raw)
                self.assertIsNotNone(result, reason)
                self.assertLess(result["xyxy"][3], 140)
                self.assertGreaterEqual(result["xyxy"][2]-result["xyxy"][0], 35)
                self.assertGreaterEqual(result["xyxy"][3]-result["xyxy"][1], 43)
                self.assertEqual(result["interaction_xyxy"], raw["xyxy"])
                self.assertEqual(raw, before)
                self.assertTrue(result["review_required"])

    def test_text_without_an_icon_does_not_become_an_icon_box(self):
        result, _ = localize_control(np.asarray(icon_fixture(icon=False)), {"class": "save", "xyxy": [60, 60, 160, 210]})
        self.assertIsNone(result)

    def test_ambiguous_shapes_are_left_for_manual_review(self):
        image = Image.new("RGB", (300, 400), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 80, 110, 115), outline="white", width=3)
        draw.rectangle((130, 80, 160, 115), outline="white", width=3)
        result, reason = localize_control(np.asarray(image), {"class": "save", "xyxy": [60, 60, 180, 160]})
        self.assertIsNone(result)
        self.assertIn("Multiple", reason)

    def test_pause_keeps_both_bars_and_ellipsis_keeps_three_dots(self):
        image = Image.new("RGB", (300, 400), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 80, 89, 120), fill="white")
        draw.rectangle((105, 80, 114, 120), fill="white")
        result, reason = localize_control(np.asarray(image), {"class": "pause", "xyxy": [60, 60, 150, 160]})
        self.assertIsNotNone(result, reason)
        self.assertLessEqual(result["xyxy"][0], 80)
        self.assertGreaterEqual(result["xyxy"][2], 115)
        image = Image.new("RGB", (300, 400), "black")
        draw = ImageDraw.Draw(image)
        for x in (80, 96, 112):
            draw.ellipse((x, 80, x+5, 85), fill="white")
        result, reason = localize_control(np.asarray(image), {"class": "more_options", "xyxy": [60, 60, 150, 130]})
        self.assertIsNotNone(result, reason)
        self.assertLessEqual(result["xyxy"][0], 80)
        self.assertGreaterEqual(result["xyxy"][2], 118)

    def test_subscribe_and_skip_ad_keep_the_users_explicit_policies(self):
        self.assertEqual(geometry_policy("subscribe"), "visible_button")
        self.assertEqual(geometry_policy("skip_ad"), "text_and_icon")
        self.assertEqual(match_accessibility({"label": "Subscribe"})[0], "subscribe")
        annotations = [{"class": "skip_ad", "xyxy": [60, 60, 160, 210]}]
        proposals, _ = geometry_proposals(np.asarray(icon_fixture()), annotations)
        self.assertFalse(proposals)

    def test_subscribe_includes_the_button_background_in_both_themes(self):
        for background, button, label in (("black", "white", "black"), ("white", "#303030", "white")):
            with self.subTest(background=background):
                image = Image.new("RGB", (300, 400), background)
                draw = ImageDraw.Draw(image)
                draw.rounded_rectangle((64, 90, 191, 129), radius=20, fill=button)
                draw.text((83, 98), "Subscribe", fill=label, font=ImageFont.load_default(size=18))
                raw = {"class": "subscribe", "xyxy": [83, 98, 164, 120], "visual_xyxy": [83, 98, 164, 120],
                       "geometry_source": "native_ocr", "interaction_xyxy": [64, 90, 192, 130]}
                result, reason = localize_control(np.asarray(image), raw)
                self.assertIsNotNone(result, reason)
                self.assertEqual(result["xyxy"], [64, 90, 192, 130])
                self.assertEqual(result["geometry_policy"], "visible_button")

    def test_subscribe_does_not_generate_a_button_from_text_alone(self):
        image = Image.new("RGB", (300, 400), "black")
        draw = ImageDraw.Draw(image)
        draw.text((80, 100), "Subscribe", fill="white", font=ImageFont.load_default(size=18))
        result, _ = localize_control(np.asarray(image), {"class": "subscribe", "xyxy": [60, 80, 210, 150]})
        self.assertIsNone(result)

    def test_play_and_next_follow_complete_glyphs_on_textured_backgrounds(self):
        for cls in ("play", "next", "previous"):
            with self.subTest(cls=cls):
                image = Image.new("RGB", (240, 200), "#cb9874")
                draw = ImageDraw.Draw(image)
                draw.ellipse((82, 62, 157, 137), fill="#766f65")
                draw.rectangle((71, 52, 96, 65), fill="white")
                triangle = [(103, 84), (103, 115), (132, 100)]
                if cls == "previous":
                    triangle = [(239-x, y) for x, y in triangle]
                draw.polygon(triangle, fill="#beb7ad")
                if cls in ("previous", "next"):
                    bar = (99, 84, 102, 115) if cls == "previous" else (137, 84, 140, 115)
                    draw.rectangle(bar, fill="#beb7ad")
                raw = {"class": cls, "xyxy": [72, 52, 97, 66], "visual_xyxy": [72, 52, 97, 66],
                       "interaction_xyxy": [72, 52, 168, 148], "source": "ios_xctest_accessibility"}
                result, reason = localize_control(np.asarray(image), raw)
                self.assertIsNotNone(result, reason)
                expected = {"play": [103, 84, 133, 116], "next": [103, 84, 141, 116], "previous": [99, 84, 137, 116]}
                self.assertEqual(result["visible_content_xyxy"], expected[cls])
                x1, y1, x2, y2 = result["xyxy"]
                ex1, ey1, ex2, ey2 = expected[cls]
                self.assertTrue(x1 < ex1 and 66 < y1 < ey1 and x2 > ex2 and y2 > ey2)
                self.assertLess(x2-x1, 60)

    def test_padding_is_trimmed_at_a_nearby_control_and_is_not_applied_twice(self):
        image = np.asarray(Image.new("RGB", (160, 220)))
        icon = {"class": "search", "xyxy": [30, 40, 60, 70], "visual_xyxy": [30, 40, 60, 70]}
        neighbor = [64, 40, 85, 70]
        result, _ = localize_control(image, icon, exclusions=[neighbor])
        self.assertEqual(result["xyxy"], [24, 34, 64, 76])
        self.assertEqual(result["visible_content_xyxy"], icon["xyxy"])
        reapplied, _ = localize_control(image, result, exclusions=[neighbor])
        self.assertEqual(reapplied["xyxy"], result["xyxy"])

    def test_padding_excludes_a_single_count_below_an_existing_glyph(self):
        image = Image.new("RGB", (160, 220), "black")
        draw = ImageDraw.Draw(image)
        draw.ellipse((40, 50, 69, 79), outline="white", width=3)
        draw.rectangle((54, 84, 56, 94), fill="white")
        icon = {"class": "like", "xyxy": [40, 50, 70, 80], "visual_xyxy": [40, 50, 70, 80],
                "interaction_xyxy": [25, 35, 85, 110]}
        result, _ = localize_control(np.asarray(image), icon)
        self.assertGreater(result["xyxy"][3], 80)
        self.assertLessEqual(result["xyxy"][3], 84)

    def test_multipart_remix_keeps_arrows_and_plus_excluding_its_label(self):
        image = Image.new("RGB", (300, 400), (84, 100, 42))
        draw = ImageDraw.Draw(image)
        color = (158, 170, 136)
        draw.arc((80, 80, 122, 120), 190, 355, fill=color, width=3)
        draw.arc((80, 80, 122, 120), 10, 175, fill=color, width=3)
        draw.line((94, 100, 108, 100), fill=color, width=3)
        draw.line((101, 93, 101, 107), fill=color, width=3)
        draw.text((78, 152), "Remix", fill=color, font=ImageFont.load_default(size=18))
        result, reason = localize_control(np.asarray(image), {"class": "remix", "xyxy": [60, 60, 160, 210]})
        self.assertIsNotNone(result, reason)
        self.assertLessEqual(result["xyxy"][0], 82)
        self.assertGreaterEqual(result["xyxy"][2], 121)
        self.assertLess(result["xyxy"][3], 140)

    def test_explicit_visual_geometry_projects_without_rewriting_raw(self):
        raw = {"class": "save", "xyxy": [60, 60, 160, 210], "visual_xyxy": [89, 79, 130, 128], "source": "ios_xctest_accessibility"}
        before = deepcopy(raw)
        clean, quality = clean_annotations([raw], (300, 400))
        self.assertEqual(clean[0]["xyxy"], raw["visual_xyxy"])
        self.assertEqual(clean[0]["interaction_xyxy"], raw["xyxy"])
        self.assertEqual(quality["unverified_icon_regions"], 0)
        self.assertEqual(raw, before)

    def test_refining_a_visual_box_does_not_invent_a_touch_target(self):
        result, reason = localize_control(np.asarray(icon_fixture()), {"class": "save", "xyxy": [60, 60, 160, 210], "source": "human_curated"})
        self.assertIsNotNone(result, reason)
        self.assertNotIn("interaction_xyxy", result)
        self.assertEqual(result["localization_region_xyxy"], [60, 60, 160, 210])

    def test_disabled_skip_ad_is_not_an_available_skip_button(self):
        snapshot = {"elements": [{"label": "Skip Ad", "enabled": False, "visible": True, "xyxy": [10, 10, 80, 40]}]}
        proposals, diagnostics = accessibility_proposals(snapshot, (300, 400), allow_unconfirmed_regions=True)
        self.assertFalse(proposals)
        self.assertFalse(diagnostics[0]["proposed"])

    def test_visual_pixels_do_not_get_scaled_twice_with_point_frames(self):
        snapshot = {"coordinate_space": "screen_points", "screen_frame": [0, 0, 150, 200],
                    "elements": [{"label": "Save", "frame": [30, 30, 50, 75], "visible": True,
                                  "visual_xyxy": [89, 79, 130, 128]}]}
        proposals, _ = accessibility_proposals(snapshot, (300, 400))
        self.assertEqual(proposals[0]["xyxy"], [89, 79, 130, 128])
        self.assertEqual(proposals[0]["interaction_xyxy"], [60, 60, 160, 210])


class GeometryApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="icon-geometry-api-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.collected = self.root / "youtube/tablet/short_video/collected"
        (self.collected / "images").mkdir(parents=True)
        self.image = self.collected / "images/geometry_fixture.png"
        icon_fixture().save(self.image)
        self.annotation = {"class": "save", "xyxy": [60, 60, 160, 210], "source": "ios_xctest_accessibility"}
        self.raw = self.collected / "annotations.jsonl"
        self.raw.write_text(json.dumps({"record_id": "geometry_fixture", "context": {"image_size": [300, 400]},
                                       "raw_annotations": [self.annotation]})+"\n")
        self.original = self.raw.read_bytes()
        self.patcher = patch.object(server, "DATA_ROOT", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.args = ("youtube", "tablet", "short_video", "geometry_fixture")

    def test_refinement_is_a_replacement_proposal_and_preserves_source_record(self):
        result = server.api_refine_geometry(*self.args, server.SuggestPayload())
        self.assertEqual(len(result["proposals"]), 1)
        proposal = result["proposals"][0]
        self.assertEqual(proposal["action"], "replace_geometry")
        self.assertEqual(proposal["replaces"]["xyxy"], self.annotation["xyxy"])
        self.assertLess(proposal["xyxy"][3], 140)
        self.assertEqual(self.raw.read_bytes(), self.original)
        self.assertFalse((self.collected / "curation/annotation_overrides.json").exists())

    def test_unreviewed_accessibility_hit_regions_cannot_be_approved_as_icon_truth(self):
        with self.assertRaises(HTTPException) as raised:
            server.api_curate(*self.args, server.CuratePayload(annotations=[self.annotation], status="approved"))
        self.assertEqual(raised.exception.status_code, 422)
        result = server.api_refine_geometry(*self.args, server.SuggestPayload())
        accepted = dict(result["proposals"][0], review_required=False, geometry_reviewed=True, reviewed_by="human")
        server.api_curate(*self.args, server.CuratePayload(annotations=[accepted], status="approved"))
        effective = server.api_records()[0]["effective_annotations"]
        self.assertEqual(len(effective), 1)
        self.assertEqual(effective[0]["geometry_policy"], "icon_only")
        self.assertLess(effective[0]["xyxy"][3], 140)
        self.assertEqual(self.raw.read_bytes(), self.original)

    def test_accessibility_label_alone_is_not_an_icon_localization(self):
        Image.new("RGB", (300, 400), "black").save(self.image)
        snapshot = {"image_sha256": hashlib.sha256(self.image.read_bytes()).hexdigest(),
                    "elements": [{"label": "Share", "visible": True, "xyxy": [60, 60, 160, 210]}]}
        record = {"record_id": "geometry_fixture", "raw_annotations": []}
        with patch.object(assistant, "model_proposals", return_value=[]), patch.object(assistant, "template_proposals", return_value=([], {})):
            result = assistant.suggest(self.image, record, [], self.root, "youtube", "short_video", snapshot)
        self.assertFalse(result["proposals"])
        self.assertTrue(result["geometry_diagnostics"])

    def test_non_hittable_but_visible_icon_can_be_proposed_from_actual_pixels(self):
        snapshot = {"image_sha256": hashlib.sha256(self.image.read_bytes()).hexdigest(),
                    "elements": [{"label": "Save", "visible": False, "xyxy": [60, 60, 160, 210]}]}
        with patch.object(assistant, "model_proposals", return_value=[]), patch.object(assistant, "template_proposals", return_value=([], {})):
            result = assistant.suggest(self.image, {}, [], self.root, "youtube", "short_video", snapshot)
        self.assertEqual([item["class"] for item in result["proposals"]], ["save"])
        self.assertTrue(result["proposals"][0]["visibility_unconfirmed"])
        self.assertTrue(result["proposals"][0]["review_required"])
        self.assertLess(result["proposals"][0]["xyxy"][3], 140)


if __name__ == "__main__":
    unittest.main()
