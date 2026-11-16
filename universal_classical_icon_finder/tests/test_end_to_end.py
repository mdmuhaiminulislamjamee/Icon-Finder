from dataclasses import replace
import json
from pathlib import Path
import pytest
from config import DetectionSettings
from detector import detect
from detector.output import annotate, results_json, results_csv, png_bytes
from detector.preprocessing import load_rgb

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("sample,theme", [("desktop_light", "light"), ("mobile_dark", "dark")])
def test_sample_screenshots_match_ground_truth(manager, sample, theme):
    truth = json.loads((ROOT / "sample_data" / f"{sample}_truth.json").read_text())
    source = ROOT / "sample_data" / f"{sample}.png"
    report = detect(source, manager.load("generic"), list({t["icon"] for t in truth}),
                    DetectionSettings(threshold=0.9, scale_min=1, scale_max=1.5, scale_step=0.5, theme=theme))
    assert len(report.results) == len(truth)
    for expected in truth:
        assert any(r["icon"] == expected["icon"] and r["bounding_box"] == expected["bounding_box"] for r in report.results)
    assert len(json.loads(results_json(report.results))) == len(truth)
    assert len(results_csv(report.results).splitlines()) == len(truth)+1
    rgb = load_rgb(source)
    annotated = annotate(rgb, report.results)
    assert annotated.size == (rgb.shape[1], rgb.shape[0])
    assert png_bytes(annotated).startswith(b"\x89PNG")


@pytest.mark.parametrize("kwargs", [{"threshold": 0}, {"scale_min": 2, "scale_max": 1}, {"scale_step": 0},
                                   {"methods": ()}, {"iou_threshold": 1}, {"threshold": float("nan")},
                                   {"theme": "invalid"}, {"method_weights": {"template_matching": 2}}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        DetectionSettings(**kwargs)


def test_scale_endpoints_and_empty_csv():
    scales = DetectionSettings(scale_min=0.75, scale_max=1.1, scale_step=0.2).scales()
    assert scales == [0.75, 0.95, 1.1]
    assert results_csv([]).startswith("icon,platform,confidence")
