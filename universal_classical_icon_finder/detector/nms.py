def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def non_max_suppression(candidates, threshold=0.35):
    kept = []
    for candidate in sorted(candidates, key=lambda c: c.confidence, reverse=True):
        if all(candidate.icon != other.icon or iou(candidate.bounding_box, other.bounding_box) <= threshold
               for other in kept):
            kept.append(candidate)
    return kept
