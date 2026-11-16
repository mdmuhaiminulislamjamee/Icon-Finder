from dataclasses import replace
import cv2
import numpy as np
import pytest
from PIL import Image
from detector import detect
from icon_manager import ProfileManager
from icon_manager.icon_registry import IconRegistry
from io import BytesIO
from conftest import scene


@pytest.mark.parametrize("detector_name", ["SIFT", "ORB", "AKAZE"])
def test_simple_icons_gracefully_handle_few_features(manager, settings, detector_name):
    report = detect(scene([("play", 40, 50, 1)]), manager.load("generic"), ["play"],
                    replace(settings, methods=("feature_matching",), feature_detector=detector_name))
    assert report.timing_ms["feature_matching"] >= 0
    assert report.warnings


def test_feature_homography_finds_repeated_textured_references(tmp_path, settings):
    rng = np.random.default_rng(45)
    gray = cv2.GaussianBlur(rng.integers(0, 256, (112, 112), np.uint8), (3, 3), 0)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    manager = ProfileManager(tmp_path)
    manager.create("texture")
    IconRegistry(manager).add("texture", "detail", buffer.getvalue())
    screen = np.full((280, 420, 3), 238, np.uint8)
    screen[30:142, 20:132] = rgb
    screen[145:257, 280:392] = rgb
    report = detect(screen, manager.load("texture"), settings=replace(settings, methods=("feature_matching",),
                    threshold=0.75, scale_min=0.9, scale_max=1.1))
    assert len(report.results) == 2
    centers = sorted((r["center"]["x"], r["center"]["y"]) for r in report.results)
    assert centers[0] == pytest.approx((76, 86), abs=2)
    assert centers[1] == pytest.approx((336, 201), abs=2)
