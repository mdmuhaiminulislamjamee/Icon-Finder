import cv2
import numpy as np
from .candidate import Candidate
from .ranking import color_agreement


def create_feature_detector(name):
    if name == "SIFT":
        return cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02), cv2.NORM_L2
    if name == "ORB":
        return cv2.ORB_create(nfeatures=4000, edgeThreshold=5, patchSize=15), cv2.NORM_HAMMING
    return cv2.AKAZE_create(), cv2.NORM_HAMMING


def match_features(screen, template, identity, platform, variant, settings, screen_features=None, color_sensitive=False):
    detector, norm = create_feature_detector(settings.feature_detector)
    tk, td = detector.detectAndCompute(template.gray, None)
    if td is None or len(tk) < 4:
        return []
    sk, sd = screen_features if screen_features is not None else detector.detectAndCompute(screen.gray, None)
    if sd is None or len(sk) < 4:
        return []
    # Match screen -> template so identical repeated icons do not fail a two-neighbor tie.
    pairs = cv2.BFMatcher(norm).knnMatch(sd, td, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair]
            if m.distance < settings.ratio_test * n.distance]
    results = []
    th, tw = template.gray.shape
    sh, sw = screen.gray.shape
    while len(good) >= 4 and len(results) < min(30, settings.max_candidates_per_scale):
        src = np.float32([tk[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([sk[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if homography is None or mask is None:
            break
        inliers = mask.ravel().astype(bool)
        if np.count_nonzero(inliers) < 4:
            break
        corners = cv2.perspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2), homography)
        if not np.all(np.isfinite(corners)):
            break
        points = corners.reshape(4, 2)
        area = abs(cv2.contourArea(points))
        scale = (area / (tw * th)) ** 0.5
        x1, y1 = np.floor(points.min(axis=0)).astype(int)
        x2, y2 = np.ceil(points.max(axis=0)).astype(int)
        valid = (abs(np.linalg.det(homography)) > 1e-12 and cv2.isContourConvex(points) and area >= 4 and x1 >= 0 and y1 >= 0 and
                 x2 <= sw and y2 <= sh and x2 > x1 and y2 > y1 and
                 settings.scale_min <= scale <= settings.scale_max)
        if valid:
            # Verify appearance after geometric alignment, not only the number of matches.
            aligned = cv2.warpPerspective(screen.rgb, np.linalg.inv(homography), (tw, th))
            gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
            correlation = float(cv2.matchTemplate(gray, template.gray, cv2.TM_CCOEFF_NORMED)[0, 0])
            appearance = abs(correlation) if settings.theme == "auto" else max(0.0, correlation)
            projected = cv2.perspectiveTransform(src[inliers], homography)
            error = float(np.mean(np.linalg.norm(projected - dst[inliers], axis=2)))
            score = appearance * np.exp(-error / 8) * color_agreement(template.rgb, aligned, color_sensitive)
            if score >= settings.threshold:
                results.append(Candidate(identity, platform, "feature_matching", appearance, float(score),
                                         tuple(map(int, (x1, y1, x2, y2))), scale, variant))
        # Remove all matched keypoints in the fitted instance, then search for another.
        if valid:
            good = [m for m, used in zip(good, inliers) if not used and not
                    (x1 <= sk[m.queryIdx].pt[0] <= x2 and y1 <= sk[m.queryIdx].pt[1] <= y2)]
        else:
            good = [m for m, used in zip(good, inliers) if not used]
    return results
