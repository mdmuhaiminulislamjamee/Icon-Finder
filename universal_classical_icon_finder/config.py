"""Validated, platform-independent detector settings."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILE_ROOT = ROOT / "platform_profiles"
METHODS = ("template_matching", "edge_matching", "shape_matching", "feature_matching")


@dataclass(frozen=True)
class DetectionSettings:
    methods: tuple[str, ...] = ("template_matching", "edge_matching")
    threshold: float = 0.86
    scale_min: float = 0.75
    scale_max: float = 1.5
    scale_step: float = 0.125
    iou_threshold: float = 0.35
    theme: str = "auto"
    template_method: str = "CCOEFF_NORMED"
    feature_detector: str = "SIFT"
    ratio_test: float = 0.72
    max_candidates_per_scale: int = 100
    priority_region: tuple[int, int, int, int] | None = None
    method_weights: dict[str, float] = field(default_factory=lambda: {
        "template_matching": 1.0, "edge_matching": 0.97,
        "shape_matching": 0.94, "feature_matching": 0.98,
    })

    def __post_init__(self):
        import math
        values = (self.threshold, self.scale_min, self.scale_max, self.scale_step,
                  self.iou_threshold, self.ratio_test, *self.method_weights.values())
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Settings must contain finite numbers.")
        if not self.methods or not set(self.methods) <= set(METHODS):
            raise ValueError("Select at least one supported detection method.")
        if not 0 < self.threshold <= 1 or not 0 <= self.iou_threshold < 1:
            raise ValueError("Confidence must be in (0, 1]; IoU must be in [0, 1).")
        if not 0.1 <= self.scale_min <= self.scale_max <= 5 or self.scale_step < 0.02:
            raise ValueError("Scales must be 0.1–5, with an ordered range and step >= 0.02.")
        if (self.scale_max - self.scale_min) / self.scale_step > 200:
            raise ValueError("Use no more than 201 scales.")
        if self.theme not in {"auto", "light", "dark"}:
            raise ValueError("Theme must be auto, light, or dark.")
        if self.template_method not in {"CCOEFF_NORMED", "CCORR_NORMED", "SQDIFF_NORMED"}:
            raise ValueError("Unsupported normalized matching method.")
        if self.feature_detector not in {"SIFT", "ORB", "AKAZE"}:
            raise ValueError("Unsupported feature detector.")
        if not 0 < self.ratio_test < 1 or not 1 <= self.max_candidates_per_scale <= 10000:
            raise ValueError("Invalid feature ratio or candidate limit.")
        if any(not 0 <= w <= 1 for w in self.method_weights.values()):
            raise ValueError("Method weights must be between zero and one.")
        if self.priority_region and (len(self.priority_region) != 4 or
                self.priority_region[2] <= 0 or self.priority_region[3] <= 0):
            raise ValueError("Priority region must be (x, y, positive width, positive height).")

    def scales(self):
        count = int((self.scale_max - self.scale_min) / self.scale_step + 1e-8)
        result = [round(self.scale_min + i * self.scale_step, 6) for i in range(count + 1)]
        if result[-1] < self.scale_max - 1e-6:
            result.append(self.scale_max)
        return result
