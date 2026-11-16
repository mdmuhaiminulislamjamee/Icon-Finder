from dataclasses import replace
import pytest
from detector import detect
from conftest import scene


def test_edges_survive_theme_inversion(manager, settings):
    report = detect(scene([("play", 83, 51, 1)], "dark"), manager.load("generic"), ["play"],
                    replace(settings, methods=("edge_matching",), threshold=0.8))
    assert len(report.results) == 1
    assert report.results[0]["method"] == "edge_matching"
    assert abs(report.results[0]["bounding_box"]["x1"] - 83) <= 1


def test_contour_shape_method(manager, settings):
    report = detect(scene([("play", 60, 40, 1)]), manager.load("generic"), ["play"],
                    replace(settings, methods=("shape_matching",), threshold=0.8, scale_min=0.9, scale_max=1.1))
    assert report.results
    assert report.results[0]["bounding_box"] == {"x1": 60, "y1": 40, "x2": 84, "y2": 64}


@pytest.mark.parametrize("bar", ["play_bar_01", "play_bar_02", "play_bar_03", "play_bar_04"])
def test_play_bar_styles_are_distinct(manager, settings, bar):
    all_bars = [f"play_bar_{i:02}" for i in range(1, 5)]
    report = detect(scene([(bar, 50, 90, 1)]), manager.load("generic"), all_bars, settings)
    assert {r["icon"] for r in report.results} == {bar}
