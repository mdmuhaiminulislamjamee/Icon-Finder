import cv2
import numpy as np
from .candidate import Candidate
from .ranking import color_agreement
from .preprocessing import scaled_reference


def response_peaks(response, threshold, width, height, limit=100):
    """Finite local maxima, with local suppression; avoid allocating all passing pixels."""
    safe = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
    kernel = np.ones((3, 3), np.uint8)
    maxima = (safe >= threshold) & (safe >= cv2.dilate(safe, kernel))
    ys, xs = np.nonzero(maxima)
    if not len(xs):
        return []
    scores = safe[ys, xs]
    # Bound work even for repetitive screenshots or a permissive threshold.
    if len(scores) > limit * 20:
        indices = np.argpartition(scores, -limit * 20)[-limit * 20:]
        xs, ys, scores = xs[indices], ys[indices], scores[indices]
    peaks = []
    for index in np.argsort(-scores, kind="stable"):
        x, y, score = int(xs[index]), int(ys[index]), float(scores[index])
        if all(abs(x - px) >= max(2, width // 3) or abs(y - py) >= max(2, height // 3)
               for px, py, _ in peaks):
            peaks.append((x, y, score))
            if len(peaks) >= limit:
                break
    return peaks


def match_templates(screen, template, identity, platform, variant, settings, color_sensitive=False):
    candidates = []
    sh, sw = screen.gray.shape
    th, tw = template.gray.shape
    method = getattr(cv2, "TM_" + settings.template_method)
    floor = settings.threshold  # Final ranking can only reduce scores.
    for scale in settings.scales():
        w, h = max(2, round(tw * scale)), max(2, round(th * scale))
        if w > sw or h > sh:
            continue
        rgb, gray, _ = scaled_reference(template, w, h)
        # Constant or nearly empty references cannot discriminate a UI icon.
        if float(gray.std()) < 1.0:
            continue
        variants = [gray]
        if settings.theme == "auto" and method != cv2.TM_CCOEFF_NORMED:
            variants.append(255 - gray)
        for query in variants:
            raw = cv2.matchTemplate(screen.gray, query, method)
            if method == cv2.TM_SQDIFF_NORMED:
                response = 1 - raw
            elif method == cv2.TM_CCOEFF_NORMED and settings.theme == "auto":
                response = np.abs(raw)
            else:
                response = raw
            for x, y, score in response_peaks(response, floor, w, h, settings.max_candidates_per_scale):
                if method != cv2.TM_CCOEFF_NORMED:
                    # Bright background alone produces very high CCORR/SQDIFF
                    # scores. Require centered structural agreement as well.
                    patch = screen.gray[y:y+h, x:x+w]
                    structural = float(cv2.matchTemplate(patch, query, cv2.TM_CCOEFF_NORMED)[0, 0])
                    structural = abs(structural) if settings.theme == "auto" else max(0, structural)
                    score = min(score, structural)
                color_score = color_agreement(rgb, screen.rgb[y:y+h, x:x+w], color_sensitive)
                candidates.append(Candidate(identity, platform, "template_matching", float(raw[y, x]),
                                            score * color_score, (x, y, x+w, y+h), scale, variant))
    return candidates
