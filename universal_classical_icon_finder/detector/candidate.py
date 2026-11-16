from dataclasses import dataclass, field


@dataclass
class Candidate:
    icon: str
    platform: str
    method: str
    raw_score: float
    confidence: float
    bounding_box: tuple[int, int, int, int]
    scale: float
    template: str
    processing_time_ms: float = 0.0
    supporting_methods: list[str] = field(default_factory=list)

    @property
    def center(self):
        x1, y1, x2, y2 = self.bounding_box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def to_dict(self, width: int, height: int):
        return {
            "icon": self.icon, "platform": self.platform, "method": self.method,
            "raw_score": round(float(self.raw_score), 6),
            "confidence": round(float(self.confidence), 6),
            "bounding_box": dict(zip(("x1", "y1", "x2", "y2"), self.bounding_box)),
            "center": dict(zip(("x", "y"), self.center)),
            "scale": round(float(self.scale), 6), "template": self.template,
            "processing_time_ms": round(self.processing_time_ms, 3),
            "supporting_methods": self.supporting_methods or [self.method],
            "image_size": {"width": width, "height": height},
        }
