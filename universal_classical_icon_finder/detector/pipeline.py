from dataclasses import dataclass
from time import perf_counter
import cv2
import numpy as np

from config import DetectionSettings
from .preprocessing import load_rgb, load_template, preprocess
from .template_matching import match_templates
from .edge_matching import match_edges, match_shapes
from .feature_matching import match_features, create_feature_detector
from .candidate_fusion import fuse_candidates
from .ranking import rank_candidates
from .nms import non_max_suppression
from .coordinates import in_priority_region


@dataclass
class DetectionReport:
    results: list[dict]
    timing_ms: dict[str, float]
    warnings: list[str]
    image_size: tuple[int, int]
    evaluated_variants: int


def detect(screenshot, profile, icons=None, settings=None):
    # Imports stay here to avoid a profile/preprocessing import cycle.
    from icon_manager.profile_manager import safe_template_path
    from icon_manager.variant_manager import select_variants
    settings = settings or DetectionSettings()
    start = perf_counter()
    image = preprocess(load_rgb(screenshot))
    height, width = image.gray.shape
    timings = {"preprocessing": (perf_counter() - start) * 1000}
    for method in settings.methods:
        timings[method] = 0.0
    selected = profile.icons if icons is None else [i for i in profile.icons if i["name"] in icons or
                                                   set(i.get("aliases", [])) & set(icons)]
    if icons is not None and not selected:
        raise ValueError("No selected icons exist in this profile.")
    warnings, candidates = [], []
    screen_features, screen_contours = None, None
    if "feature_matching" in settings.methods:
        tick = perf_counter()
        feature_detector, _ = create_feature_detector(settings.feature_detector)
        screen_features = feature_detector.detectAndCompute(image.gray, None)
        timings["feature_matching"] += (perf_counter() - tick) * 1000
    if "shape_matching" in settings.methods:
        tick = perf_counter()
        screen_contours, _ = cv2.findContours(image.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        timings["shape_matching"] += (perf_counter() - tick) * 1000
    functions = {"template_matching": match_templates, "edge_matching": match_edges,
                 "shape_matching": match_shapes, "feature_matching": match_features}
    evaluated = 0
    for icon in selected:
        variants = list(select_variants(icon, settings.theme))
        if not variants:
            warnings.append(f"{icon['name']}: no variants for the selected theme.")
        for relative in variants:
            tick = perf_counter()
            try:
                template = load_template(safe_template_path(profile.directory, relative), settings.theme)
            except (ValueError, OSError) as exc:
                warnings.append(f"{icon['name']} / {relative}: {exc}")
                continue
            timings["template_preparation"] = timings.get("template_preparation", 0.0) + (perf_counter() - tick) * 1000
            if float(template.gray.std()) < 1:
                warnings.append(f"{icon['name']} / {relative}: blank reference skipped.")
                continue
            evaluated += 1
            for method in settings.methods:
                if method not in icon.get("enabled_methods", settings.methods):
                    continue
                tick = perf_counter()
                extras = {"color_sensitive": icon.get("color_sensitive", False)}
                if method == "feature_matching":
                    extras["screen_features"] = screen_features
                elif method == "shape_matching":
                    extras["screen_contours"] = screen_contours
                found = functions[method](image, template, icon["name"], profile.name, relative, settings, **extras)
                elapsed = (perf_counter() - tick) * 1000
                timings[method] += elapsed
                for candidate in found:
                    candidate.processing_time_ms = elapsed
                candidates.extend(found)
            template.scaled.clear()
    tick = perf_counter()
    ranked = rank_candidates(candidates, settings)
    fused = fuse_candidates(ranked, settings.iou_threshold)
    kept = non_max_suppression(fused, settings.iou_threshold)
    kept.sort(key=lambda c: (in_priority_region(c, settings.priority_region), c.confidence), reverse=True)
    timings["fusion"] = (perf_counter() - tick) * 1000
    timings["total"] = (perf_counter() - start) * 1000
    if "feature_matching" in settings.methods:
        warnings.append("Feature matching can return no candidates for tiny or simple icons; template and edge methods are usually stronger there.")
    return DetectionReport([c.to_dict(width, height) for c in kept],
                           {k: round(v, 3) for k, v in timings.items()}, warnings, (width, height), evaluated)
