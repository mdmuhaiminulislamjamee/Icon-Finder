from dataclasses import replace
from detector import detect
from detector.coordinates import clip_box
from conftest import scene


def test_large_resolution_border_coordinates(manager, settings):
    rgb = scene([("play", 1880, 1030, 1), ("play", 0, 0, 1)], size=(1920, 1080))
    report = detect(rgb, manager.load("generic"), ["play"], settings)
    assert len(report.results) == 2
    assert {(r["bounding_box"]["x1"], r["bounding_box"]["y1"]) for r in report.results} == {(1880, 1030), (0, 0)}
    assert all(r["image_size"] == {"width": 1920, "height": 1080} for r in report.results)
    assert clip_box((-1, -2, 2000, 1100), 1920, 1080) == (0, 0, 1920, 1080)


def test_priority_region_never_restricts_search(manager, settings):
    rgb = scene([("play", 20, 20, 1), ("play", 280, 180, 1)])
    report = detect(rgb, manager.load("generic"), ["play"], replace(settings, priority_region=(250, 150, 100, 80)))
    assert len(report.results) == 2
    assert report.results[0]["bounding_box"]["x1"] == 280
