from dataclasses import replace
import numpy as np
import pytest
from config import DetectionSettings
from detector import detect
from detector.template_matching import response_peaks
from conftest import scene


def test_repeated_instances_and_original_coordinates(manager, settings):
    rgb = scene([("play", 13, 17, 1), ("play", 270, 160, 1), ("pause", 140, 95, 1)])
    report = detect(rgb, manager.load("generic"), ["play", "pause"], settings)
    assert len(report.results) == 3
    positions = {(r["icon"], r["bounding_box"]["x1"], r["bounding_box"]["y1"]) for r in report.results}
    assert positions == {("play", 13, 17), ("play", 270, 160), ("pause", 140, 95)}
    for result in report.results:
        assert result["image_size"] == {"width": 360, "height": 240}
        box = result["bounding_box"]
        assert result["center"] == {"x": box["x1"]+12, "y": box["y1"]+12}


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("method", ["CCOEFF_NORMED", "CCORR_NORMED", "SQDIFF_NORMED"])
def test_themes_and_normalized_methods(manager, settings, theme, method):
    report = detect(scene([("play", 81, 57, 1)], theme), manager.load("generic"), ["play"],
                    replace(settings, methods=("template_matching",), template_method=method, theme=theme))
    assert len(report.results) == 1
    assert report.results[0]["bounding_box"] == {"x1": 81, "y1": 57, "x2": 105, "y2": 81}


def test_multiscale_and_larger_than_screen(manager, settings):
    report = detect(scene([("play", 167, 119, 1.5)]), manager.load("generic"), ["play"],
                    replace(settings, scale_min=0.5, scale_max=2, scale_step=0.25))
    assert len(report.results) == 1
    assert report.results[0]["scale"] == 1.5
    assert report.results[0]["bounding_box"]["x1"] == 167
    tiny = np.full((10, 10, 3), 238, dtype=np.uint8)
    assert not detect(tiny, manager.load("generic"), ["play"], settings).results


def test_slight_color_change_and_no_match(manager, settings):
    rgb = scene([("play", 32, 65, 1)])
    changed = np.clip(rgb.astype(float) * 0.8 + 18, 0, 255).astype(np.uint8)
    assert len(detect(changed, manager.load("generic"), ["play"], settings).results) == 1
    blank = np.full((240, 360, 3), 238, dtype=np.uint8)
    assert detect(blank, manager.load("generic"), ["play", "pause"], settings).results == []


def test_partial_occlusion_reduces_similarity(manager, settings):
    rgb = scene([("play", 50, 50, 1)])
    full = detect(rgb, manager.load("generic"), ["play"], settings).results
    rgb[55:62, 57:66] = 238
    occluded = detect(rgb, manager.load("generic"), ["play"], replace(settings, threshold=0.6)).results
    assert full and occluded
    assert max(r["confidence"] for r in occluded) < full[0]["confidence"] - 0.01


def test_reject_nan_and_inf_response_peaks():
    response = np.array([[np.nan, np.inf], [-np.inf, 0.95]], dtype=np.float32)
    assert response_peaks(response, 0.8, 4, 4) == [(1, 1, pytest.approx(0.95))]
