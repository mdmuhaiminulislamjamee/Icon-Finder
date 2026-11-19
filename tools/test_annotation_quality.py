"""Annotation regressions using isolated data; never repopulates the gallery."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

from tools import annotation_assistant as assistant
from tools.annotation_quality import accessibility_proposals, clean_annotations, match_accessibility
from tools.dashboard import server


def box(cls="save", xyxy=(10, 20, 40, 60), **extra):
    return {"class": cls, "xyxy": list(xyxy), **extra}


class AnnotationQualityTests(unittest.TestCase):
    def test_duplicate_capture_boxes_are_hidden_without_changing_raw(self):
        raw = [box(cls) for cls in ("search", "search", "save", "save", "remix", "remix")]
        before = deepcopy(raw)
        clean, report = clean_annotations(raw, (1668, 2388))
        self.assertEqual([item["class"] for item in clean], ["search", "save", "remix"])
        self.assertEqual(report["duplicates_removed"], 3)
        self.assertEqual(raw, before)

    def test_near_duplicate_geometry_is_removed_but_distinct_controls_survive(self):
        raw = [box(), box(xyxy=(10.2, 20.1, 40.2, 60.1)), box(xyxy=(60, 20, 90, 60)), box("share")]
        clean, report = clean_annotations(raw, (100, 100))
        self.assertEqual(len(clean), 3)
        self.assertEqual(report["duplicates_removed"], 1)

    def test_invalid_and_outside_boxes_are_reported(self):
        raw = [box(xyxy=(40, 20, 10, 50)), box(xyxy=(0, 0, float("nan"), 50)),
               box(xyxy=(101, 101, 120, 120)), box(xyxy=(-3, 20, 40, 60))]
        clean, report = clean_annotations(raw, (100, 100))
        self.assertEqual(report["invalid_boxes"], 3)
        self.assertEqual(clean[0]["xyxy"], [0, 20, 40, 60])

    def test_identifier_precedes_label_and_dislike_does_not_match_like(self):
        self.assertEqual(match_accessibility({"identifier": "id.reel_remix_button", "label": "Something else"})[0], "remix")
        self.assertEqual(match_accessibility({"identifier": "dislike_button", "label": "Like"})[0], "dislike")
        self.assertEqual(match_accessibility({"label": "Dislike"})[0], "dislike")

    def test_counts_case_and_button_suffix_are_normalized(self):
        for label, expected in (("Like, 28K likes", "like"), ("Comments, 346", "comments"),
                                ("SEARCH button", "search"), ("Share video", "share"),
                                ("like this video along with 503 other people", "like"),
                                ("পছন্দ", "like")):
            with self.subTest(label=label):
                self.assertEqual(match_accessibility({"label": label})[0], expected)

    def test_audio_actions_resolve_to_current_state(self):
        self.assertEqual(match_accessibility({"label": "Mute"})[0], "volume_on")
        self.assertEqual(match_accessibility({"label": "Unmute"})[0], "volume_muted")

    def test_unmatched_text_never_creates_an_invisible_control(self):
        for label in ("I like this song", "Advertisement", "Next up in 10 seconds", "Subscribe to my channel for more"):
            self.assertIsNone(match_accessibility({"label": label})[0])

    def test_full_snapshot_retains_unmatched_diagnostics_and_scales_points(self):
        snapshot = {"coordinate_space": "screen_points", "screen_frame": [0, 0, 100, 200], "elements": [
            {"identifier": "share_button", "frame": [10, 20, 30, 40], "visible": True},
            {"label": "Channel name", "frame": [0, 0, 10, 10], "visible": True},
            {"label": "Play", "frame": [0, 0, 10, 10], "visible": False},
        ]}
        proposals, diagnostics = accessibility_proposals(snapshot, (200, 400))
        self.assertEqual(len(diagnostics), 3)
        self.assertEqual([item["class"] for item in proposals], ["share"])
        self.assertEqual(proposals[0]["xyxy"], [20, 40, 80, 120])
        self.assertTrue(proposals[0]["review_required"])
        self.assertEqual(snapshot["elements"][1]["label"], "Channel name")

    def test_snapshot_with_unknown_coordinate_system_is_rejected(self):
        with self.assertRaises(ValueError):
            accessibility_proposals({"coordinate_space": "window_guess", "elements": []}, (100, 100))


class AnnotationApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="studio-annotation-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.collected = self.root / "youtube/tablet/short_video/collected"
        (self.collected / "images").mkdir(parents=True)
        self.image_path = self.collected / "images/capture_test.png"
        Image.new("RGB", (100, 200), "#123456").save(self.image_path)
        self.record = {"record_id": "capture_test", "context": {"image_size": [100, 200]},
                       "raw_annotations": [box(), box()]}
        self.raw_path = self.collected / "annotations.jsonl"
        self.raw_path.write_text(json.dumps(self.record) + "\n", encoding="utf-8")
        self.raw_bytes = self.raw_path.read_bytes()
        self.data_patch = patch.object(server, "DATA_ROOT", self.root)
        self.data_patch.start()
        self.addCleanup(self.data_patch.stop)
        self.args = ("youtube", "tablet", "short_video", "capture_test")

    def test_records_preserve_contract_and_offer_clean_display_copy(self):
        record = server.api_records()[0]
        self.assertEqual(len(record["raw_annotations"]), 2)
        self.assertEqual(len(record["effective_annotations"]), 2)
        self.assertEqual(len(record["display_annotations"]), 1)
        self.assertEqual(record["annotation_quality"]["duplicates_removed"], 1)
        self.assertEqual(self.raw_path.read_bytes(), self.raw_bytes)

    def test_curation_deduplicates_and_audits_exact_changes(self):
        payload = server.CuratePayload(annotations=[box("share"), box("share")], status="approved")
        result = server.api_curate(*self.args, payload)
        self.assertEqual(result["saved_boxes"], 1)
        audit = json.loads((self.collected / "curation/annotation_audit.jsonl").read_text())
        self.assertEqual(audit["saved_override"]["curated_annotations"], [box("share")])
        self.assertEqual(audit["duplicates_removed"], 1)
        self.assertEqual(self.raw_path.read_bytes(), self.raw_bytes)
        records = server.api_records()
        self.assertEqual(records[0]["effective_annotations"], [box("share")])

    def test_empty_curated_annotation_set_overrides_raw(self):
        server.api_curate(*self.args, server.CuratePayload(annotations=[]))
        self.assertEqual(server.api_records()[0]["effective_annotations"], [])

    def test_pending_suggestions_survive_save_without_becoming_effective(self):
        pending = box("share", review_required=True, proposal_id="pending", decision="pending")
        server.api_curate(*self.args, server.CuratePayload(annotations=[box()], reviewed_proposals=[pending]))
        record = server.api_records()[0]
        self.assertEqual(record["proposal_review"], [pending])
        self.assertEqual(record["effective_annotations"], [box()])

    def test_new_local_approvals_are_tried_before_bundled_template_styles(self):
        server.api_curate(*self.args, server.CuratePayload(annotations=[box("comments")], status="approved"))
        first = next(assistant._reviewed_examples(self.root, "youtube", "short_video"))
        self.assertEqual(first[2], "approved:capture_test")

    def test_bad_status_coordinates_and_unreviewed_predictions_are_rejected(self):
        cases = [server.CuratePayload(annotations=[box()], status="raw"),
                 server.CuratePayload(annotations=[box(xyxy=(-2, 0, 30, 40))]),
                 server.CuratePayload(annotations=[box(review_required=True)]),
                 server.CuratePayload(annotations=[box()], status="approved", reviewed_proposals=[{"decision": "pending"}])]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(HTTPException) as raised:
                    server.api_curate(*self.args, payload)
                self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(self.raw_path.read_bytes(), self.raw_bytes)

    def test_corrupt_existing_overrides_are_not_replaced(self):
        path = self.collected / "curation/annotation_overrides.json"
        path.parent.mkdir()
        path.write_text("not json")
        with self.assertRaises(HTTPException) as raised:
            server.api_curate(*self.args, server.CuratePayload(annotations=[]))
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(path.read_text(), "not json")

    def test_suggestions_do_not_modify_raw_or_curated_annotations(self):
        proposal = {"proposals": [box("share", review_required=True)], "warnings": []}
        with patch.object(assistant, "suggest", return_value=proposal) as mock:
            result = server.api_suggest(*self.args, server.SuggestPayload(annotations=[box("like")]))
        self.assertEqual(mock.call_args.args[2], [box("like")])
        self.assertEqual(len(result["proposals"]), 1)
        self.assertTrue((self.collected / "curation/proposals/capture_test.json").exists())
        self.assertFalse((self.collected / "curation/annotation_overrides.json").exists())
        self.assertEqual(self.raw_path.read_bytes(), self.raw_bytes)

    def test_diagnostics_returns_complete_snapshot_including_unmatched(self):
        path = self.collected / "accessibility/capture_test.json"
        path.parent.mkdir()
        snapshot = {"elements": [{"label": "unmatched channel title"}]}
        path.write_text(json.dumps(snapshot))
        result = server.api_annotation_diagnostics(*self.args)
        self.assertEqual(result["accessibility_snapshot"], snapshot)
        self.assertTrue(server.api_records()[0]["has_accessibility_snapshot"])

    def test_mismatched_snapshot_hash_is_not_used(self):
        with patch.object(assistant, "model_proposals", return_value=[]), patch.object(assistant, "template_proposals", return_value=([], {})):
            result = assistant.suggest(self.image_path, self.record, [], self.root, "youtube", "short_video",
                                       snapshot={"image_sha256": "wrong", "elements": [{"label": "Share", "visible": True, "xyxy": [1, 1, 20, 20]}]})
        self.assertFalse(result["proposals"])
        self.assertIn("does not match", result["warnings"][0])

    def test_matching_snapshot_can_propose_missing_class(self):
        snapshot = {"image_sha256": hashlib.sha256(self.image_path.read_bytes()).hexdigest(),
                    "elements": [{"label": "Share", "visible": True, "xyxy": [1, 1, 40, 60],
                                  "visual_xyxy": [10, 10, 30, 30], "geometry_source": "native_pixel_localization"}]}
        with patch.object(assistant, "model_proposals", return_value=[]), patch.object(assistant, "template_proposals", return_value=([], {})):
            result = assistant.suggest(self.image_path, self.record, [], self.root, "youtube", "short_video", snapshot=snapshot)
        self.assertEqual([item["class"] for item in result["proposals"]], ["share"])
        self.assertTrue(result["proposals"][0]["proposal_id"])
        self.assertEqual(result["proposals"][0]["xyxy"], [6, 6, 34, 34])
        self.assertEqual(result["proposals"][0]["visible_content_xyxy"], [10, 10, 30, 30])
        self.assertEqual(result["proposals"][0]["interaction_xyxy"], [1, 1, 40, 60])

    def test_routes_reject_path_escape(self):
        with self.assertRaises(HTTPException):
            server.collection_path("..", "tablet", "short_video", "capture_test")


class ReviewedTemplateTests(unittest.TestCase):
    def test_template_fallback_finds_a_reviewed_icon_and_rejects_noise(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory(prefix="template-test-") as directory:
            root = Path(directory)
            icon = np.zeros((40, 40, 3), dtype=np.uint8)
            cv2.rectangle(icon, (4, 5), (34, 27), (255, 255, 255), 2)
            cv2.line(icon, (12, 28), (8, 36), (255, 255, 255), 2)
            cv2.line(icon, (8, 36), (24, 28), (255, 255, 255), 2)
            reference = np.zeros((300, 300, 3), dtype=np.uint8)
            reference[50:90, 50:90] = icon
            path = root / "reviewed.png"
            Image.fromarray(reference).save(path)
            target = np.zeros((300, 300, 3), dtype=np.uint8)
            target[150:190, 200:240] = icon
            examples = [(path, [box("comments", (50, 50, 90, 90))], "approved:example")]
            proposals, counts = assistant.template_proposals(target, examples)
            self.assertTrue(proposals)
            self.assertEqual(proposals[0]["class"], "comments")
            self.assertTrue(all(abs(a-b) <= 1 for a, b in zip(proposals[0]["xyxy"], [200, 150, 240, 190])))
            self.assertEqual(counts, {"comments": 1})
            noise = np.random.default_rng(42).integers(0, 256, size=(300, 300, 3), dtype=np.uint8)
            self.assertFalse(assistant.template_proposals(noise, examples)[0])


if __name__ == "__main__":
    unittest.main()
