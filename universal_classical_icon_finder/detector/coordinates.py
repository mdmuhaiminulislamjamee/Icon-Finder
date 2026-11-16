"""Boxes use half-open original pixels: [x1, x2), [y1, y2)."""


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    return (max(0, min(width, int(x1))), max(0, min(height, int(y1))),
            max(0, min(width, int(x2))), max(0, min(height, int(y2))))


def in_priority_region(candidate, region):
    if region is None:
        return False
    x, y, w, h = region
    cx, cy = candidate.center
    return x <= cx < x + w and y <= cy < y + h
