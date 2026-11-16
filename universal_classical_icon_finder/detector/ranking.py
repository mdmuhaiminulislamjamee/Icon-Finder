import cv2
import numpy as np
from .coordinates import in_priority_region


def color_agreement(template_rgb, patch_rgb, strict=False):
    """Preserve colored bar identities, while letting monochrome icons change theme."""
    if not strict:
        return 1.0
    template_hsv = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2HSV)
    mask = (template_hsv[:, :, 1] > 90) & (template_hsv[:, :, 2] > 65)
    if np.count_nonzero(mask) < 3:
        if strict:
            border = np.concatenate((template_rgb[0], template_rgb[-1], template_rgb[:, 0], template_rgb[:, -1]))
            background = np.median(border, axis=0)
            foreground = np.max(np.abs(template_rgb.astype(float) - background), axis=2) > 14
            if np.any(foreground):
                patch_hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
                delta = np.abs(patch_hsv[:, :, 1][foreground].astype(float) -
                               template_hsv[:, :, 1][foreground].astype(float))
                return float(np.mean(np.exp(-delta / 60)))
        return 1.0
    patch_hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    delta = np.abs(template_hsv[:, :, 0][mask].astype(float) - patch_hsv[:, :, 0][mask])
    hue_distance = np.minimum(delta, 180 - delta)
    delta_saturation = np.abs(patch_hsv[:, :, 1][mask].astype(float) - template_hsv[:, :, 1][mask].astype(float))
    saturation = np.exp(-delta_saturation / 60)
    return float(np.mean(np.exp(-hue_distance / 18) * saturation))


def rank_candidates(candidates, settings):
    for candidate in candidates:
        candidate.confidence = max(0.0, min(1.0, candidate.confidence *
                                          settings.method_weights.get(candidate.method, 1.0)))
    accepted = [c for c in candidates if c.confidence >= settings.threshold]
    return sorted(accepted, key=lambda c: (in_priority_region(c, settings.priority_region), c.confidence), reverse=True)
