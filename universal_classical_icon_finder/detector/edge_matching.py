import cv2
import numpy as np
from .candidate import Candidate
from .template_matching import response_peaks
from .ranking import color_agreement
from .preprocessing import scaled_reference


def match_edges(screen, template, identity, platform, variant, settings, color_sensitive=False):
    results = []
    sh, sw = screen.gray.shape
    th, tw = template.gray.shape
    for scale in settings.scales():
        w, h = max(2, round(tw * scale)), max(2, round(th * scale))
        if w > sw or h > sh:
            continue
        rgb, gray, edges = scaled_reference(template, w, h)
        if np.count_nonzero(edges) < 4:
            continue
        response = cv2.matchTemplate(screen.edges, edges, cv2.TM_CCORR_NORMED)
        for x, y, score in response_peaks(response, settings.threshold, w, h, settings.max_candidates_per_scale):
            score *= color_agreement(rgb, screen.rgb[y:y+h, x:x+w], color_sensitive)
            results.append(Candidate(identity, platform, "edge_matching", float(response[y, x]), score,
                                     (x, y, x+w, y+h), scale, variant))
    return results


def match_shapes(screen, template, identity, platform, variant, settings, screen_contours=None, color_sensitive=False):
    """Hu moments plus contour occupancy. Suited to isolated, connected silhouettes."""
    references, _ = cv2.findContours(template.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    references = [c for c in references if cv2.contourArea(c) >= 4]
    if not references:
        return []
    ref = max(references, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(ref)
    ref_area = cv2.contourArea(ref) / (rw * rh)
    if screen_contours is None:
        screen_contours, _ = cv2.findContours(screen.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    sh, sw = screen.gray.shape
    th, tw = template.gray.shape
    for contour in screen_contours:
        if cv2.contourArea(contour) < 4:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        scale = (w / rw + h / rh) / 2
        if not settings.scale_min <= scale <= settings.scale_max:
            continue
        aspect = min((w / h) / (rw / rh), (rw / rh) / (w / h))
        distance = cv2.matchShapes(ref, contour, cv2.CONTOURS_MATCH_I1, 0)
        occupancy = cv2.contourArea(contour) / (w * h)
        score = float(np.exp(-5 * distance) * aspect * np.exp(-3 * abs(occupancy - ref_area)))
        x1, y1 = round(x - rx * scale), round(y - ry * scale)
        bw, bh = max(2, round(tw * scale)), max(2, round(th * scale))
        if x1 < 0 or y1 < 0 or x1 + bw > sw or y1 + bh > sh:
            continue
        rgb = cv2.resize(template.rgb, (bw, bh))
        score *= color_agreement(rgb, screen.rgb[y1:y1+bh, x1:x1+bw], color_sensitive)
        if score >= settings.threshold:
            results.append(Candidate(identity, platform, "shape_matching", float(distance), score,
                                     (x1, y1, x1+bw, y1+bh), scale, variant))
    return sorted(results, key=lambda c: c.confidence, reverse=True)[:settings.max_candidates_per_scale]
