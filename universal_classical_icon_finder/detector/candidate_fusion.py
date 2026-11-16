from .nms import iou


def fuse_candidates(candidates, threshold=0.35):
    """Greedy IoU clustering, per identity; retain the strongest box, no score inflation."""
    fused = []
    for candidate in sorted(candidates, key=lambda c: c.confidence, reverse=True):
        match = next((other for other in fused if candidate.icon == other.icon and
                      iou(candidate.bounding_box, other.bounding_box) > threshold), None)
        if match is None:
            candidate.supporting_methods = [candidate.method]
            fused.append(candidate)
        else:
            match.supporting_methods = sorted(set(match.supporting_methods + [candidate.method]))
    return fused
