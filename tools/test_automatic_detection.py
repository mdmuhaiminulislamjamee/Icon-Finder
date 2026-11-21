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


def embedded_timeline_fixture(*, track=True, thumb=True, color=(255, 0, 51)):
    width, height = 640, 540
    image = Image.new("RGB", (width, height), (43, 62, 60))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 324, width-1, height-1), fill="white")
    if track:
        draw.rectangle((10, 320, 629, 323), fill=(112, 112, 112))
        draw.rectangle((135, 320, 429, 322), fill=(170, 160, 160))
        draw.rectangle((10, 320, 134, 323), fill=color)
    if track and thumb:
        draw.ellipse((123, 310, 145, 333), fill=color)
    return image


SLIDER = {"class": "progress_bar", "xyxy": [20, 265, 620, 305],
          "source": "studio_accessibility_matcher", "value": "50 percent"}
EMBEDDED_SLIDER = {"class": "progress_bar", "xyxy": [0, 280, 640, 324], "value": "95 percent"}
EMBEDDED_PLAYER = [0, 24, 640, 324]


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

    def test_translucent_track_at_the_embedded_player_edge_is_found(self):
        rgb = np.asarray(embedded_timeline_fixture())
        self.assertFalse(localize_timeline(rgb, EMBEDDED_SLIDER)[0])
        found, reason = localize_timeline(rgb, EMBEDDED_SLIDER, EMBEDDED_PLAYER)
        self.assertEqual([ann["class"] for ann in found], ["progress_bar", "playhead"], reason)
        self.assertEqual(found[0]["visible_content_xyxy"], [10, 320, 630, 324])
        self.assertEqual(found[0]["xyxy"], [10, 318.5, 630, 325.5])
        self.assertEqual(found[1]["xyxy"], [123, 310, 146, 334])
        self.assertTrue(inside_track(found[1]["xyxy"], found[0]["xyxy"]))

    def test_embedded_page_boundary_and_slider_value_do_not_invent_geometry(self):
        for track in (False, True):
            image = embedded_timeline_fixture(track=track, thumb=False)
            if not track:
                ImageDraw.Draw(image).ellipse((123, 310, 145, 333), fill=(255, 0, 51))
            found, _ = localize_timeline(np.asarray(image), EMBEDDED_SLIDER, EMBEDDED_PLAYER)
            self.assertEqual([ann["class"] for ann in found], ["progress_bar"] if track else [])

    def test_ad_timeline_requires_gold_and_does_not_use_the_timer_as_a_playhead(self):
        ad_slider = {**EMBEDDED_SLIDER, "class": "ad_progress_bar", "xyxy": [0, 154, 640, 324]}
        for color in ((251, 192, 45), (255, 0, 51)):
            rgb = np.asarray(embedded_timeline_fixture(thumb=False, color=color))
            found, reason = localize_timeline(rgb, ad_slider, EMBEDDED_PLAYER)
            if color == (255, 0, 51):
                self.assertFalse(found)
            else:
                self.assertEqual([ann["class"] for ann in found], ["ad_progress_bar"], reason)
                self.assertEqual(found[0]["visible_content_xyxy"], [10, 320, 630, 324])
                self.assertEqual(found[0]["xyxy"], [10, 318.5, 630, 325.5])


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

    def test_embedded_youtube_detection_uses_player_geometry_even_in_shorts_folder(self):
        embedded_timeline_fixture().save(self.image_path)
        self.record["context"]["image_size"] = [640, 540]
        self.raw_path.write_text(json.dumps(self.record)+"\n", encoding="utf-8")
        self.snapshot.update(image_sha256=hashlib.sha256(self.image_path.read_bytes()).hexdigest(),
                             coordinate_space="screen_points", screen_frame=[0, 0, 320, 270], elements=[
                                 {"identifier": "id.player", "frame": [0, 12, 320, 150]},
                                 {"identifier": "id.player.scrubber.slider", "frame": [0, 140, 320, 22],
                                  "visible": True, "value": "95 percent"}])
        self.snapshot_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
        found = {ann["class"]: ann for ann in self.detect()["proposals"]}
        self.assertEqual(set(found), {"progress_bar", "playhead"})
        self.assertEqual(found["progress_bar"]["visible_content_xyxy"], [10, 320, 630, 324])
        self.assertEqual(found["playhead"]["xyxy"], [123, 310, 146, 334])

    def test_visible_ad_player_supersedes_the_wrong_fullscreen_and_companion_menu_labels(self):
        image = embedded_timeline_fixture(color=(251, 192, 45), thumb=False)
        draw = ImageDraw.Draw(image)
        draw.ellipse((547, 244, 626, 323), fill="#212121")
        for left, top, tip, end in ((568, 265, 2, 14), (588, 285, 14, 2)):
            for endpoint in ((end, tip), (tip, end), (end, end)):
                draw.line((left+tip, top+tip, left+endpoint[0], top+endpoint[1]), fill="white", width=4)
        for arm in (0, 1, 2):
            draw.ellipse((602, 43+12*arm, 609, 50+12*arm), fill="white")
            draw.ellipse((549+12*arm, 358, 556+12*arm, 365), fill="black")
        image.save(self.image_path)
        self.record["context"]["image_size"] = list(image.size)
        self.raw_path.write_text(json.dumps(self.record)+"\n", encoding="utf-8")
        self.snapshot.update(image_sha256=hashlib.sha256(self.image_path.read_bytes()).hexdigest(), elements=[
            {"identifier": "id.player", "xyxy": EMBEDDED_PLAYER},
            {"identifier": "id.player.ad_info.button", "frame": [14, 264, 70, 56], "visible": True},
            {"identifier": "id.player.scrubber.slider", "xyxy": [0, 154, 640, 324], "visible": True},
            {"identifier": "id.player.watch.fullscreen.button", "frame": [542, 235, 90, 89], "visible": True},
            {"identifier": "id.player.watch.exit_fullscreen.button", "frame": [542, 235, 90, 89], "visible": False},
            {"identifier": "id.ui.generic.button", "label": "overflow vertical 24pt", "frame": [579, 28, 54, 62], "visible": True},
            {"identifier": "eml.ad_engagement_header", "frame": [0, 335, 640, 55]},
            {"identifier": "eml.overflow_button", "label": "More options", "frame": [535, 339, 65, 45], "visible": True},
        ])
        self.snapshot_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
        old = [
            {"class": "fullscreen", "xyxy": [552, 248, 620, 317], "interaction_xyxy": [542, 235, 632, 324]},
            {"class": "exit_fullscreen", "xyxy": [552, 248, 620, 317], "interaction_xyxy": [542, 235, 632, 324]},
            {"class": "more_options", "xyxy": [549, 358, 581, 366], "interaction_xyxy": [535, 339, 600, 384]},
        ]
        result = server.api_detect(*self.args, server.SuggestPayload(annotations=[dict(ann, review_required=True) for ann in old]))
        self.assertEqual({item["class"] for item in result["proposals"]},
                         {"fullscreen", "more_options", "ad_more_options", "ad_progress_bar"})
        self.assertEqual(result["superseded_proposals"], [
            {"class": "exit_fullscreen", "xyxy": old[1]["xyxy"]},
            {"class": "more_options", "xyxy": old[2]["xyxy"]},
        ])
        self.assertEqual(next(item["xyxy"] for item in result["proposals"] if item["class"] == "more_options"),
                         [600.0, 37.0, 612.0, 81.0])

    def test_video_overflow_requires_dots_in_the_action_row_and_back_requires_a_chevron(self):
        image = Image.new("RGB", (640, 400))
        draw = ImageDraw.Draw(image)
        draw.line(((32, 25), (21, 40), (32, 55)), fill="white", width=4, joint="curve")
        for left in (139, 151, 163, 380, 392, 404):
            draw.ellipse((left, 348, left+7, 355), fill="white")
        image.save(self.image_path)
        elements = [
            {"identifier": "id.ui.browse.back.button", "frame": [4, 11, 50, 59], "visible": True},
            {"identifier": "id.video.add_to.button", "frame": [53, 320, 68, 64], "visible": True},
            {"identifier": "id.elements.button", "frame": [121, 320, 64, 64], "visible": True},
            {"identifier": "id.elements.button", "frame": [354, 320, 80, 64], "visible": True},
        ]
        self.snapshot["image_sha256"] = hashlib.sha256(self.image_path.read_bytes()).hexdigest()
        for nodes in (elements, [elements[0], elements[1], elements[3]]):
            self.snapshot["elements"] = nodes
            self.snapshot_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
            result = self.detect()
            found = {ann["class"]: ann["visible_content_xyxy"] for ann in result["proposals"]}
            self.assertEqual(set(found), {"back", "more_options"})
            self.assertEqual(found["more_options"], [139.0, 348.0, 171.0, 356.0])
            self.assertLess(found["back"][2], 36)

    def test_ad_panel_close_follows_the_visible_x_when_its_accessibility_frame_is_stale(self):
        header = {"identifier": "eml.ad_engagement_header", "frame": [195, 21, 116, 24], "visible": True}
        menu = {"identifier": "eml.overflow_button", "label": "More options", "frame": [275, 27, 12, 12], "visible": True}
        close = {"identifier": "eml.ad_panel_header_close_button", "label": "Close ad panel",
                 "frame": [203, 27, 12, 12], "visible": False, "enabled": True}
        for visible_x, with_menu in ((True, True), (False, True), (True, False)):
            with self.subTest(visible_x=visible_x, with_menu=with_menu):
                image = Image.new("RGB", (640, 400), "#303030")
                draw = ImageDraw.Draw(image)
                draw.rectangle((380, 31, 639, 399), fill="white")
                draw.line((92, 59, 106, 73), fill="white", width=3)
                draw.line((92, 73, 106, 59), fill="white", width=3)
                if with_menu:
                    for left in (554, 560, 566):
                        draw.ellipse((left, 64, left+3, 67), fill="black")
                if visible_x:
                    draw.line((604, 60, 616, 72), fill="black", width=2)
                    draw.line((604, 72, 616, 60), fill="black", width=2)
                image.save(self.image_path)
                self.snapshot.update(image_sha256=hashlib.sha256(self.image_path.read_bytes()).hexdigest(),
                                     coordinate_space="screen_points", screen_frame=[0, 0, 320, 200],
                                     elements=[header, close] + ([menu, menu] if with_menu else []))
                self.snapshot_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
                boxes = self.detect()["proposals"]
                self.assertEqual(sum(ann["class"] == "ad_more_options" for ann in boxes), int(with_menu))
                hide = [ann for ann in boxes if ann["class"] == "hide_content"]
                self.assertEqual(len(hide), int(visible_x and with_menu))
                if hide:
                    self.assertEqual(hide[0]["interaction_xyxy"], [406, 54, 430, 78])
                    left, top, right, bottom = hide[0]["xyxy"]
                    self.assertTrue(598 <= left < 604 < 616 < right <= 622)
                    self.assertTrue(54 <= top < 60 < 72 < bottom <= 78)

    def test_shorts_toolbar_recovers_stale_volume_and_menu_without_the_manual_template(self):
        elements = [
            {"identifier": "eml.shorts-video-title-new", "visible": True},
            {"identifier": "eml.top_bar", "xyxy": [36, 22, 632, 86], "visible": True},
            {"label": "Search", "xyxy": [519, 36, 553, 72]},
            {"label": "Mute video", "identifier": "id.elements.button", "xyxy": [36, 22, 100, 86], "visible": False},
            {"identifier": "id.reel_more_button", "xyxy": [36, 22, 68, 54], "visible": False},
        ]
        wrong_menu = {"class": "more_options", "xyxy": [524, 43, 550, 68], "visual_xyxy": [524, 43, 550, 68],
                      "resource_id": "id.reel_more_button", "review_required": True, "source": "ios_xctest_accessibility"}
        wide_menu = {"class": "more_options", "xyxy": [572, 26, 628, 82], "visual_xyxy": [572, 26, 628, 82],
                     "source": "studio_reviewed_template", "geometry_source": "reviewed_visual_template", "review_required": True}
        for volume in ("volume_on", "volume_muted", None):
            with self.subTest(volume=volume):
                image = Image.new("RGB", (640, 400), "#333940")
                draw = ImageDraw.Draw(image)
                draw.ellipse((524, 43, 544, 63), outline="white", width=2)
                draw.line((542, 61, 549, 68), fill="white", width=2)
                if volume:
                    draw.polygon(((458, 50), (462, 48), (473, 41), (473, 67), (462, 61), (458, 59)),
                                 outline="white", width=2)
                    if volume == "volume_on":
                        draw.arc((463, 47, 482, 61), -59, 59, fill="white", width=2)
                        draw.arc((463, 43, 488, 65), -57, 57, fill="white", width=2)
                    else:
                        draw.line((477, 49, 487, 59), fill="white", width=2)
                        draw.line((477, 59, 487, 49), fill="white", width=2)
                    for x in (590, 598, 606):
                        draw.ellipse((x, 52, x+4, 56), fill="white")
                image.save(self.image_path)
                self.snapshot.update(image_sha256=hashlib.sha256(self.image_path.read_bytes()).hexdigest(), elements=elements)
                self.snapshot_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
                result = self.detect()
                found = {ann["class"]: ann for ann in result["proposals"]}
                expected = {"search", "more_options", volume} if volume else {"search"}
                self.assertEqual(set(found), expected)
                if volume == "volume_on":
                    self.assertEqual(found["more_options"]["visible_content_xyxy"], [590, 52, 611, 57])
                    self.assertEqual(found["more_options"]["xyxy"], [586, 50, 615, 59])
                    with patch.object(assistant, "template_proposals", return_value=([wide_menu], {})):
                        manual = server.api_suggest(*self.args, server.SuggestPayload(annotations=[wrong_menu, wide_menu]))
                    menu, = [ann for ann in manual["proposals"] if ann["class"] == "more_options"]
                    self.assertEqual(menu["xyxy"], found["more_options"]["xyxy"])
                    self.assertEqual(manual["superseded_proposals"], [
                        {"class": "more_options", "xyxy": old["xyxy"]} for old in (wrong_menu, wide_menu)
                    ])

    def test_shorts_playback_follows_the_visible_overlay_without_an_accessibility_button(self):
        self.record["context"].update(image_size=[540, 760], capture_state="paused_controls_visible")
        self.raw_path.write_text(json.dumps(self.record)+"\n", encoding="utf-8")
        player = {"identifier": "id.player", "frame": [0, 40, 540, 660], "visible": False}
        reel = {"identifier": "eml.shorts-video-title-new", "label": "Short title", "visible": True}
        for kind, backing, nodes, expected in (
                ("pause", True, [player, reel], "pause"), ("play", True, [player, reel], "play"),
                ("play", False, [player, reel], None), (None, True, [player, reel], None),
                ("pause", True, [player], None)):
            with self.subTest(kind=kind, backing=backing, shorts=len(nodes) == 2):
                image = Image.new("RGB", (540, 760), (95, 125, 145))
                draw = ImageDraw.Draw(image)
                draw.rectangle((270, 0, 539, 759), fill=(160, 120, 90))
                if backing:
                    draw.ellipse((234, 334, 305, 405), fill=(49, 60, 64))
                if kind == "pause":
                    draw.rectangle((262, 360, 267, 379), fill=(150, 150, 150))
                    draw.rectangle((272, 360, 277, 379), fill=(150, 150, 150))
                    draw.rectangle((263, 361, 266, 378), fill="white")
                    draw.rectangle((273, 361, 276, 378), fill="white")
                elif kind == "play":
                    draw.polygon(((262, 361), (262, 378), (280, 369)), fill="white")
                    bounds = [262, 361, 281, 379]
                image.save(self.image_path)
                self.snapshot.update(image_sha256=hashlib.sha256(self.image_path.read_bytes()).hexdigest(), elements=nodes)
                self.snapshot_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
                found = self.detect()["proposals"]
                self.assertEqual([ann["class"] for ann in found], [expected] if expected else [])
                if expected == "pause":
                    self.assertEqual(found[0]["visible_content_xyxy"], [263, 361, 277, 379])
                    self.assertEqual(found[0]["xyxy"], [260, 357, 280, 383])
                    old_box = [220, 320, 320, 420]
                    old = dict(found[0], xyxy=old_box, visual_xyxy=old_box,
                               visible_content_xyxy=[234, 334, 306, 406])
                    refreshed = server.api_detect(*self.args, server.SuggestPayload(annotations=[old]))["proposals"]
                    self.assertEqual(len(refreshed), 1)
                    self.assertEqual(refreshed[0]["replaces"], {"class": "pause", "xyxy": old_box})
                    self.assertEqual(refreshed[0]["xyxy"], found[0]["xyxy"])
                    reviewed = dict(old, geometry_reviewed=True, review_required=False)
                    self.assertEqual(server.api_detect(*self.args, server.SuggestPayload(annotations=[reviewed]))["proposals"], [])
                    refined = server.api_refine_geometry(*self.args, server.SuggestPayload(annotations=[reviewed]))["proposals"]
                    self.assertEqual(len(refined), 1)
                    self.assertEqual(refined[0]["xyxy"], found[0]["xyxy"])
                elif expected == "play":
                    self.assertEqual(found[0]["visible_content_xyxy"], bounds)
                if expected:
                    self.assertTrue(found[0]["review_required"])
                    self.assertNotIn("interaction_xyxy", found[0])

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
