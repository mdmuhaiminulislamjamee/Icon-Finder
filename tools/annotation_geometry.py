"""Visible geometry policies and conservative pixel proposals inside known controls.

This does not classify icons or alter the V11 full-frame model input. A supplied
semantic control region is localized only when its pixels support a clear glyph.
All resulting proposals still require human review.
"""
from copy import deepcopy
from itertools import combinations
import math


POLICY_VERSION = "media_control_visual_geometry_v3"
ICON_PADDING = .18
SHORTS_PLAYBACK_SEARCH_RATIO = .052
TRACK_HEIGHT_RATIO = .0135
MIN_TRACK_HEIGHT_RATIO = .012
TRANSPORT_GLYPHS = {"play", "previous", "next"}
PLAYBACK_GLYPHS = {"play", "pause"}
FULLSCREEN_GLYPHS = {"fullscreen", "exit_fullscreen"}
ELLIPSIS_GLYPHS = {"more_options", "ad_more_options"}
TRACK_CLASSES = {"progress_bar", "ad_progress_bar"}
CAPTIONED_ICONS = {"like", "dislike", "comments", "save", "share", "remix", "ask", "download", "report_problem"}
SPECIAL_POLICIES = {
    **dict.fromkeys(TRACK_CLASSES, "visible_track"), "playhead": "visible_marker",
    "sound": "artwork", "channel": "artwork", "autoplay": "visible_toggle",
    "subscribe": "visible_button", "ad_cta": "visible_button",
    "visit_advertiser": "text_button", "skip_ad": "text_and_icon",
    "time_display": "text_readout", "more_videos": "composite_control",
}
INTERACTION_SOURCES = {"ios_xctest_accessibility", "studio_accessibility_matcher",
                       "xctest", "uiautomator", "android_accessibility"}


def geometry_policy(cls):
    return SPECIAL_POLICIES.get(cls, "icon_only")


def valid_box(value, image_size=None):
    try:
        if len(value) != 4 or any(isinstance(item, bool) for item in value):
            return False
        box = [float(item) for item in value]
        if not all(math.isfinite(item) for item in box) or box[2] <= box[0] or box[3] <= box[1]:
            return False
        if image_size and image_size[0] > 0 and image_size[1] > 0:
            return box[0] >= 0 and box[1] >= 0 and box[2] <= image_size[0] and box[3] <= image_size[1]
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def needs_visual_geometry(annotation):
    if annotation.get("class") == "subscribe" and annotation.get("geometry_policy") == "text_control":
        return not annotation.get("geometry_reviewed")
    return (geometry_policy(annotation.get("class")) in {"icon_only", "visible_button", "visible_toggle", "text_and_icon", "text_button"}
            and annotation.get("source") in INTERACTION_SOURCES
            and not annotation.get("geometry_reviewed")
            and not valid_box(annotation.get("visual_xyxy")))


def project_visual_geometry(annotation):
    """Project explicit visual pixels without losing the original interaction box."""
    result = deepcopy(annotation)
    if "visual_xyxy" in result:
        # Invalid explicit visual data must fail normal validation, not silently
        # fall back to the larger interaction region and become an icon label.
        if result.get("source") in INTERACTION_SOURCES or result.get("geometry_status") == "interaction_region":
            result.setdefault("interaction_xyxy", deepcopy(result.get("xyxy")))
        result["xyxy"] = deepcopy(result["visual_xyxy"])
    return result


def _components(mask):
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    result = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w*h < 4:
            continue
        touches = sum((x <= 0, y <= 0, x+w >= width, y+h >= height))
        if touches >= 3 or (w >= .95*width and h >= .95*height):
            continue  # Uniform background, not an icon.
        result.append((x, y, x+w, y+h))
    return result


def _text_rows(boxes):
    """Reject lines of small aligned characters; this is not OCR or classification."""
    rows = []
    for anchor in boxes:
        ah = anchor[3]-anchor[1]
        group = []
        for item in boxes:
            ih = item[3]-item[1]
            overlap = min(anchor[3], item[3])-max(anchor[1], item[1])
            if .55 <= ih/max(ah, 1) <= 1.8 and overlap >= .7*min(ah, ih):
                group.append(item)
        group.sort(key=lambda item: item[0])
        runs, run = [], []
        for item in group:
            if run and item[0]-run[-1][2] > 1.8*ah:
                runs.append(run)
                run = []
            run.append(item)
        runs.append(run)
        for run in runs:
            if len(run) >= 3:
                bounds = _union(run)
                if bounds[2]-bounds[0] >= 2*(bounds[3]-bounds[1]):
                    rows.append(bounds)
    return rows


def _union(boxes):
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _intersection_area(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def _dots(boxes, mask, orientation):
    height, width = mask.shape
    dots = []
    for item in boxes:
        left, top, right, bottom = item
        dot_width, dot_height = right-left, bottom-top
        if (max(3, .055*min(height, width)) <= min(dot_width, dot_height)
                and max(dot_width, dot_height) <= .24*min(height, width)
                and .65 <= dot_width/dot_height <= 1.55
                and .54 <= (mask[top:bottom, left:right] > 0).mean() <= .96):
            dots.append(item)
    result = []
    axes = {"horizontal": (0,), "vertical": (1,)}.get(orientation, (0, 1))
    for group in combinations(dots[:30], 3):
        sizes = [max(item[2]-item[0], item[3]-item[1]) for item in group]
        if max(sizes) > 1.45*min(sizes):
            continue
        centers = [((item[0]+item[2])/2, (item[1]+item[3])/2) for item in group]
        for axis in axes:
            ordered = sorted(centers, key=lambda item: item[axis])
            a, b = ordered[1][axis]-ordered[0][axis], ordered[2][axis]-ordered[1][axis]
            if (1.1*max(sizes) <= a <= 3*max(sizes) and .76 <= b/max(a, 1) <= 1.32
                    and max(item[1-axis] for item in centers)-min(item[1-axis] for item in centers) <= .42*max(sizes)
                    and abs(ordered[1][1-axis]-(height, width)[axis]/2) <= .3*(height, width)[axis]):
                result.append(_union(group))
    return result


def _pause_pairs(boxes):
    result = []
    for a, b in combinations(boxes, 2):
        aw, ah, bw, bh = a[2]-a[0], a[3]-a[1], b[2]-b[0], b[3]-b[1]
        overlap = min(a[3], b[3])-max(a[1], b[1])
        gap = max(a[0], b[0])-min(a[2], b[2])
        if (ah >= 2*aw and bh >= 2*bw and min(ah, bh) >= .8*max(ah, bh)
                and overlap >= .8*max(ah, bh) and 0 < gap < .85*max(ah, bh)):
            result.append(_union((a, b)))
    return result


def _control_region(annotation, size, *, prefer_localization=False):
    keys = ("localization_region_xyxy", "interaction_xyxy") if prefer_localization else ("interaction_xyxy", "localization_region_xyxy")
    for key in (*keys, "xyxy"):
        if valid_box(annotation.get(key), size):
            box = list(map(float, annotation[key]))
            return [math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])]
    return None


def _foreground_masks(roi):
    import cv2
    import numpy as np

    maximum, minimum = roi.max(axis=2), roi.min(axis=2)
    neutral = maximum-minimum <= 55
    yield np.uint8(neutral & (minimum >= 190))*255
    yield np.uint8(neutral & (maximum <= 65))*255
    # Local contrast also captures disabled strokes and text over a video.
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    diameter = max(5, min(21, round(min(gray.shape)/8))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    for operation in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
        contrast = cv2.morphologyEx(gray, operation, kernel)
        yield np.uint8(contrast >= 18)*255


def _short_captions(parts, glyph, cls):
    if cls not in CAPTIONED_ICONS:
        return []
    x1, y1, x2, y2 = glyph
    height = y2-y1
    result = []
    for left, top, right, bottom in parts:
        if not .2*height <= bottom-top <= .72*height:
            continue
        # A single count may be too short to be identified as a line of text.
        below = 0 <= top-y2 <= .25*height and min(x2, right) > max(x1, left)
        beside = 0 <= left-x2 <= .25*height and min(y2, bottom) > max(y1, top)
        if below or beside:
            result.append([left, top, right, bottom])
    return result


def _padded_icon(glyph, size, exclusions):
    left, top, right, bottom = glyph
    px = max(2, (right-left)*ICON_PADDING)
    py = max(2, (bottom-top)*ICON_PADDING)
    padded = [max(0, math.floor(left-px)), max(0, math.floor(top-py)),
              min(size[0], math.ceil(right+px)), min(size[1], math.ceil(bottom+py))]
    for other in exclusions:
        if not _intersection_area(padded, other) or _intersection_area(glyph, other):
            continue
        # Trim the padding at the closest obstruction without cutting the glyph.
        cuts = []
        if math.ceil(other[2]) < left:
            cuts.append((max(0, (math.ceil(other[2])-padded[0])/px), 0, math.ceil(other[2])))
        if math.ceil(other[3]) < top:
            cuts.append((max(0, (math.ceil(other[3])-padded[1])/py), 1, math.ceil(other[3])))
        if math.floor(other[0]) > right:
            cuts.append((max(0, (padded[2]-math.floor(other[0]))/px), 2, math.floor(other[0])))
        if math.floor(other[1]) > bottom:
            cuts.append((max(0, (padded[3]-math.floor(other[1]))/py), 3, math.floor(other[1])))
        if cuts:
            _, edge, value = min(cuts)
            padded[edge] = value
    return padded


def _horizontal_triangle(contour, rect, *, points_left=False):
    import cv2

    _, top, width, height = rect
    triangle = cv2.approxPolyDP(contour, .065*cv2.arcLength(contour, True), True)
    if len(triangle) != 3 or not .35 <= cv2.contourArea(contour)/(width*height) <= .68:
        return False
    base_top, base_bottom, tip = sorted(triangle[:, 0].tolist(), reverse=points_left)
    return abs(base_top[0]-base_bottom[0]) <= .3*width and top+.28*height <= tip[1] <= top+.72*height


def _transport_glyph(rgb, region, cls):
    import cv2
    import numpy as np

    left, top, right, bottom = region
    gray = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    size = min(height, width)
    if size < 16 or max(height, width) > 1.8*size:
        return None
    yy, xx = np.ogrid[:height, :width]
    radius = np.hypot(xx+.5-width/2, yy+.5-height/2)/size
    ring = gray[(radius >= .30) & (radius <= .37)]
    low, background, high = np.percentile(ring, [25, 50, 75])
    delta = gray.astype(np.float32)-background
    threshold = max(10, 1.25*(high-low)+3)
    matches = []
    for polarity in (1, -1):
        for gain in (1, 1.5, 2):
            mask = np.uint8((polarity*delta >= gain*threshold) & (radius <= .42))
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            parts = [(cv2.boundingRect(c), c) for i, c in enumerate(contours)
                     if hierarchy[0, i, 3] < 0 and cv2.contourArea(c) >= size*.11]
            for (x, y, w, h), contour in parts:
                if (not .18*size <= h <= .73*size or not .55 <= w/h <= 1.45
                        or abs(x+w/2-width/2) > .17*size or abs(y+h/2-height/2) > .12*size):
                    continue
                if not _horizontal_triangle(contour, (x, y, w, h), points_left=cls == "previous"):
                    continue
                glyph = [x, y, x+w, y+h]
                if cls != "play":
                    bars = []
                    for (bx, by, bw, bh), _ in parts:
                        gap = x-bx-bw if cls == "previous" else bx-x-w
                        overlap = min(y+h, by+bh)-max(y, by)
                        if (0 <= gap <= .4*h and .65*h <= bh <= 1.35*h and bw <= .34*bh
                                and overlap >= .75*min(h, bh)):
                            bars.append([bx, by, bx+bw, by+bh])
                    if len(bars) != 1:
                        continue
                    glyph = _union([glyph, bars[0]])
                if not any(_intersection_area(glyph, old) >= .80*min(
                        (glyph[2]-glyph[0])*(glyph[3]-glyph[1]), (old[2]-old[0])*(old[3]-old[1])) for old in matches):
                    matches.append(glyph)
    if len(matches) == 1:
        x1, y1, x2, y2 = matches[0]
        return [left+x1, top+y1, left+x2, top+y2]
    return None


def _has_shorts_overlay_disc(rgb, center_x, center_y, glyph_height):
    import numpy as np

    angles = np.arange(96) * (2*np.pi/96)
    radii = np.arange(math.ceil(1.35*glyph_height), math.floor(2.65*glyph_height)+1)[:, None]
    margin = max(2, round(.08*glyph_height))
    if radii[-1, 0]+margin >= min(center_x, center_y, rgb.shape[1]-center_x, rgb.shape[0]-center_y):
        return False
    brightness = []
    for offset in (-margin, margin):
        ring_x = np.rint(center_x + (radii+offset)*np.cos(angles)).astype(int)
        ring_y = np.rint(center_y + (radii+offset)*np.sin(angles)).astype(int)
        brightness.append(rgb[ring_y, ring_x].mean(axis=-1))
    inside, outside = brightness
    contrast = outside-inside
    darkens = contrast >= np.maximum(9, .13*outside)
    quadrants = darkens.reshape(-1, 4, 24).mean(axis=-1)
    return bool(((quadrants.min(axis=1) >= .29) & (quadrants.mean(axis=1) >= .60)).any())


def _shorts_playback_glyph(rgb, search_region, span):
    """Measure the white glyph, using the round backing to verify the overlay."""
    import cv2
    import numpy as np

    left, top, right, bottom = search_region
    mid_x, mid_y = (right-left)/2, (bottom-top)/2
    search = rgb[top:bottom, left:right]
    low, high = search.min(axis=-1), search.max(axis=-1)
    matches = []
    for threshold in (205, 235):
        mask = np.uint8((low >= threshold) & (high-low <= 32))*255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        triangles, bars = [], []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if not max(10, .012*span) <= height <= .055*span:
                continue
            fill = (mask[y:y+height, x:x+width] > 0).mean()
            box = (x, y, x+width, y+height)
            if .55 <= width/height <= 1.35 and .34 <= fill <= .72 and _horizontal_triangle(contour, (x, y, width, height)):
                triangles.append(box)
            if .10 <= width/height <= .36 and fill >= .78:
                bars.append(box)
        for cls, boxes in (("play", triangles), ("pause", _pause_pairs(bars))):
            for x1, y1, x2, y2 in boxes:
                height = y2-y1
                if abs((x1+x2)/2-mid_x) > .26*height or abs((y1+y2)/2-mid_y) > .15*height:
                    continue
                if cls == "pause" and not .48 <= (x2-x1)/height <= 1.12:
                    continue
                glyph = [left+x1, top+y1, left+x2, top+y2]
                if any(cls == old_cls and _intersection_area(glyph, old) >= .8*(x2-x1)*height for old_cls, old in matches):
                    continue
                if _has_shorts_overlay_disc(rgb, left+mid_x, top+mid_y, height):
                    matches.append((cls, glyph))
    return matches[0] if len(matches) == 1 else None


def localize_shorts_playback(rgb, player_region):
    """Find the visible glyph in the center of a verified Shorts viewport."""
    size = (rgb.shape[1], rgb.shape[0])
    if not valid_box(player_region, size):
        return None
    left, top, right, bottom = player_region
    span = min(right-left, bottom-top)
    center_x, center_y = round((left+right)/2), round((top+bottom)/2)
    radius = max(8, round(SHORTS_PLAYBACK_SEARCH_RATIO*span))
    region = [center_x-radius, center_y-radius, center_x+radius, center_y+radius]
    if not valid_box(region, size):
        return None
    match = _shorts_playback_glyph(rgb, region, span)
    if not match:
        return None
    cls, glyph = match
    return _visual_annotation(
        {"class": cls, "source": "studio_shorts_pixels", "visible_evidence": "Visible central Shorts playback overlay"},
        glyph, region, "pixel_shorts_playback", _padded_icon(glyph, size, []),
    )


def _fullscreen_glyph(rgb, region, cls):
    """Find the complete mirrored pair of diagonal arrows around the tap center."""
    import cv2
    import numpy as np

    left, top, right, bottom = region
    width, height = right-left, bottom-top
    size = min(width, height)
    if size < 16 or max(width, height) > 2*size:
        return None
    template = np.zeros((20, 20), np.uint8)
    for endpoint in ((17, 2), (2, 17), (17, 17)):
        cv2.line(template, (2, 2), endpoint, 1, 4)
    template = template.astype(bool)
    inward = cls == "exit_fullscreen"
    matches = []
    for mask in _foreground_masks(rgb[top:bottom, left:right]):
        corners = {True: [], False: []}
        for part in _components(mask):
            start_x, start_y, end_x, end_y = part
            part_width, part_height = end_x-start_x, end_y-start_y
            if not .07*size <= part_height <= .42*size or not .72 <= part_width/part_height <= 1.38:
                continue
            normalized = cv2.resize(mask[start_y:end_y, start_x:end_x], (20, 20), interpolation=cv2.INTER_AREA) >= 128
            for northwest in corners:
                pixels = normalized if northwest else normalized[::-1, ::-1]
                score = np.count_nonzero(pixels & template)/np.count_nonzero(pixels | template)
                if score >= .62:
                    corners[northwest].append((part, score))
        for first, first_score in corners[not inward]:
            for second, second_score in corners[inward]:
                dx = (second[0]+second[2]-first[0]-first[2])/2
                dy = (second[1]+second[3]-first[1]-first[3])/2
                arrow_size = ((first[3]-first[1])+(second[3]-second[1]))/2
                if not .72*arrow_size <= dx <= 1.9*arrow_size or not .7 <= dy/dx <= 1.4:
                    continue
                if not .73 <= (second[3]-second[1])/(first[3]-first[1]) <= 1.37:
                    continue
                union = _union((first, second))
                offset_x = abs((union[0]+union[2]-width)/2)/width
                offset_y = abs((union[1]+union[3]-height)/2)/height
                if max(offset_x, offset_y) <= .20:
                    matches.append((min(first_score, second_score)-offset_x-offset_y, union))
    if not matches:
        return None
    best_score, best = max(matches)
    for score, other in matches:
        if score >= best_score-.08 and _intersection_area(best, other) < .75*(best[2]-best[0])*(best[3]-best[1]):
            return None
    return [left+best[0], top+best[1], left+best[2], top+best[3]]


def _toggle_glyphs(rgb, region_height):
    import numpy as np

    bright = np.uint8((rgb.min(axis=-1) >= 190) & (rgb.max(axis=-1)-rgb.min(axis=-1) <= 45))*255
    parts = _components(bright)
    for part in parts:
        left, top, right, bottom = part
        knob_width, knob_height = right-left, bottom-top
        if (.45*region_height <= knob_height <= region_height and .8 <= knob_width/knob_height <= 1.22
                and np.mean(bright[top:bottom, left:right] > 0) >= .48
                and _transport_glyph(rgb, part, "play")):
            yield "on", part
    for part in _pause_pairs(parts):
        yield "off", part


def _capsule_edge_scores(gray, lefts, tops, width, height):
    import numpy as np

    radius = height/2
    angles = np.linspace(-np.pi/2, np.pi/2, 11)[1:-1]
    cosine, sine = np.cos(angles), np.sin(angles)
    straight = np.linspace(radius, width-radius, len(angles))
    zeros, ones = np.zeros_like(straight), np.ones_like(straight)
    boundary_x = np.stack((straight, straight, radius-cosine*radius, width-radius+cosine*radius))
    boundary_y = np.stack((zeros, ones*height, radius+sine*radius, radius+sine*radius))
    normal_x = np.stack((zeros, zeros, -cosine, cosine))
    normal_y = np.stack((-ones, ones, sine, sine))
    positions_x, positions_y = np.meshgrid(lefts, tops)
    xs, ys = positions_x.reshape(-1, 1, 1), positions_y.reshape(-1, 1, 1)

    def contrast(distance):
        inner = gray[ys+np.floor(boundary_y-distance*normal_y).astype(int),
                     xs+np.floor(boundary_x-distance*normal_x).astype(int)]
        outer = gray[ys+np.floor(boundary_y+distance*normal_y).astype(int),
                     xs+np.floor(boundary_x+distance*normal_x).astype(int)]
        return inner-outer

    far = contrast(2)
    support = np.mean(far >= 16, axis=2)
    good = np.flatnonzero((support.min(axis=1) >= .55) & (support.mean(axis=1) >= .72))
    if good.size:
        near = contrast(.9)[good]
        scores = np.clip(near, -40, 64).mean(axis=(1, 2)) + .5*np.clip(far[good], -40, 64).mean(axis=(1, 2))
        for index, score in zip(good, scores):
            yield float(score), int(xs[index, 0, 0]), int(ys[index, 0, 0])


def _autoplay_toggle(rgb, region):
    """Trace both arcs and horizontal edges around a verified toggle glyph."""
    import cv2
    import numpy as np

    left, top, right, bottom = region
    width, height = right-left, bottom-top
    if height < 12 or not .95 <= width/height <= 3:
        return None
    margin = max(3, round(.30*height))
    origin_x, origin_y = max(0, left-margin), max(0, top-margin)
    crop = rgb[origin_y:bottom+margin, origin_x:right+margin]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.int16)
    center_x, center_y = (left+right)/2-origin_x, (top+bottom)/2-origin_y
    matches = []
    for state, glyph in _toggle_glyphs(crop, height):
        gx, gy = (glyph[0]+glyph[2])/2, (glyph[1]+glyph[3])/2
        glyph_width, glyph_height = glyph[2]-glyph[0], glyph[3]-glyph[1]
        low, high = (.66, .97) if state == "on" else (.27, .59)
        min_height = max(12, math.ceil(.53*height), math.ceil(glyph_height/high))
        max_height = min(math.floor(1.18*height), math.floor(glyph_height/low))
        for ph in range(min_height, max_height+1):
            if state == "off" and not .18*ph <= glyph_width <= .50*ph:
                continue
            radius = ph/2
            offset = max(2, round(.06*ph))
            origin = round(gy-radius)
            tops = np.arange(max(2, origin-offset), min(gray.shape[0]-ph-3, origin+offset)+1)
            tops = tops[np.abs(tops+radius-center_y) <= .24*height]
            min_width = math.ceil(max(.68*width, 1.42*ph))
            max_width = math.floor(min(1.38*width, 2.35*ph))
            for pw in range(min_width, max_width+1):
                origin = round(gx-pw+radius if state == "on" else gx-radius)
                lefts = np.arange(max(2, origin-offset), min(gray.shape[1]-pw-3, origin+offset)+1)
                lefts = lefts[np.abs(lefts+pw/2-center_x) <= .24*width]
                if lefts.size and tops.size:
                    for score, px, py in _capsule_edge_scores(gray, lefts, tops, pw, ph):
                        matches.append((score, [origin_x+px, origin_y+py, origin_x+px+pw, origin_y+py+ph], state))
    if not matches:
        return None
    best, box, state = max(matches)
    if any(score >= best-6 and (other_state != state or _intersection_area(box, other) < .75*(box[2]-box[0])*(box[3]-box[1]))
           for score, other, other_state in matches):
        return None
    return box, state


def _button_background(rgb, region, label_box, *, ad_button=False):
    import cv2
    import numpy as np

    left, top, right, bottom = region
    rw, rh = right-left, bottom-top
    if not 1.3 <= rw/rh <= (16 if ad_button else 10) or rh < 10:
        return None
    margin = max(3, round(rh*.13))
    ox, oy = max(0, left-margin), max(0, top-margin)
    crop = rgb[oy:bottom+margin, ox:right+margin]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    matches = []
    for bright in (True, False):
        levels = (40, 80, 120, 160, 200, 232, 244, 250) if ad_button else (40, 80, 120, 160, 200, 232)
        for level in levels:
            mask = np.uint8(gray >= level if bright else gray <= level)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if (not .66*rw <= w <= 1.18*rw or not .55*rh <= h <= 1.22*rh or w < 1.3*h
                        or x == 0 or y == 0 or x+w == gray.shape[1] or y+h == gray.shape[0]
                        or abs(ox+x+w/2-(left+right)/2) > .15*rw
                        or abs(oy+y+h/2-(top+bottom)/2) > .19*rh
                        or cv2.contourArea(contour) < .69*w*h):
                    continue
                button = [ox+x, oy+y, ox+x+w, oy+y+h]
                if label_box and not (button[0] <= label_box[0] and button[1] <= label_box[1]
                                      and button[2] >= label_box[2] and button[3] >= label_box[3]):
                    continue
                # The button interior must have a field of contrasting label strokes.
                field = gray[y+h//5:y+h-h//5, x+w//6:x+w-w//6].astype(np.float32)
                tone = np.median(field)
                contrast = tone-field if bright else field-tone
                label = np.uint8(contrast >= 25)
                rows = _text_rows(_components(label))
                if ad_button and not rows:
                    rows = _text_rows(_components(np.uint8(-contrast >= 25)))
                if rows or (label_box and .02 < np.mean(label) < .48):
                    matches.append(button)
    if not matches:
        return None
    outer = _union(matches)
    if any(_intersection_area(box, outer) < .75*(outer[2]-outer[0])*(outer[3]-outer[1]) for box in matches):
        return None
    return outer


def _text_button_label(rgb, region):
    left, top, right, bottom = region
    width, height = right-left, bottom-top
    if not 2 <= width/height <= 18:
        return None
    rows = [row for mask in _foreground_masks(rgb[top:bottom, left:right])
            for row in _text_rows(_components(mask))
            if row[2]-row[0] >= .48*width and abs((row[1]+row[3]-height)/2) <= .22*height]
    if not rows:
        return None
    label = max(rows, key=lambda row: row[2]-row[0])
    bounds = _union([row for row in rows if _intersection_area(label, row) >= .70*(label[2]-label[0])*(label[3]-label[1])])
    return [left+bounds[0], top+bounds[1], left+bounds[2], top+bounds[3]]


def _small_symbol(rgb, region, cls):
    import cv2
    import numpy as np

    left, top, right, bottom = region
    width, height = right-left, bottom-top
    if min(width, height) < 12 or max(width, height) > 2*min(width, height):
        return None
    template = np.zeros((28, 28), np.uint8)
    if cls == "back":
        cv2.polylines(template, [np.array([(25, 2), (2, 14), (25, 25)])], False, 1, 4)
        aspect, min_size = (.40, .86), .23
    elif cls == "hide_content":
        cv2.line(template, (2, 2), (25, 25), 1, 4)
        cv2.line(template, (2, 25), (25, 2), 1, 4)
        aspect, min_size = (.78, 1.28), .24
    else:
        cv2.circle(template, (14, 14), 12, 1, 3)
        cv2.circle(template, (14, 8), 2, 1, cv2.FILLED)
        cv2.line(template, (14, 12), (14, 20), 1, 2)
        aspect, min_size = (.85, 1.18), .13
    template = template.astype(bool)
    matches = []
    for mask in _foreground_masks(rgb[top:bottom, left:right]):
        for x1, y1, x2, y2 in _components(mask):
            pw, ph = x2-x1, y2-y1
            offset = max(abs((x1+x2-width)/2)/width, abs((y1+y2-height)/2)/height)
            if not aspect[0] <= pw/ph <= aspect[1] or not min_size*height <= ph <= .90*height or offset > .25:
                continue
            shape = cv2.resize(mask[y1:y2, x1:x2], (28, 28), interpolation=cv2.INTER_AREA) >= 128
            if cls == "ad_info":
                dot = shape[5:11, 10:18].mean()
                stem = shape[13:20, 10:18].mean()
                gap = shape[11:13, 11:17].mean()
                if not (.04 <= dot <= .55 and .14 <= stem <= .65 and gap < .45):
                    continue
            score = np.count_nonzero(shape & template)/np.count_nonzero(shape | template)
            if score >= (.49 if cls == "ad_info" else .56):
                matches.append((score-offset, [left+x1, top+y1, left+x2, top+y2]))
    if not matches:
        return None
    best_score, best = max(matches)
    if any(score >= best_score-.08 and _intersection_area(best, other) < .72*(best[2]-best[0])*(best[3]-best[1])
           for score, other in matches):
        return None
    return best


def localize_shorts_speaker(rgb, region):
    """Identify the speaker and its waves or mute mark in a Shorts toolbar slot."""
    import cv2

    left, top, right, bottom = region
    width, height = right-left, bottom-top
    mask = next(_foreground_masks(rgb[top:bottom, left:right]))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    shapes = _components(mask)
    matches = []
    for contour, (_, _, hole, parent) in zip(contours, hierarchy[0]):
        if parent >= 0 or hole < 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if (not .34*height <= h <= .84*height or not .42 <= w/h <= .82
                or abs(x+w-width/2) > .20*h or abs(y+h/2-height/2) > .18*h):
            continue
        if not .36 <= cv2.contourArea(contours[hole])/max(1, cv2.contourArea(contour)) <= .80:
            continue
        body = mask[y:y+h, x:x+w]
        edge_width = max(1, round(.2*w))
        if body[:, :edge_width].any(axis=1).mean() >= .78 or body[:, -edge_width:].any(axis=1).mean() < .82:
            continue
        marks = sorted(box for box in shapes if 0 <= box[0]-x-w <= .38*h
                       and box[2]-x-w <= .74*h and .30*h <= box[3]-box[1] <= .95*h
                       and abs((box[1]+box[3])/2-y-h/2) <= .14*h)
        if len(marks) == 1 and _small_symbol(rgb, [left+x+w, top+y, right, top+y+h], "hide_content"):
            cls = "volume_muted"
        elif (len(marks) == 2 and marks[1][2] > marks[0][2]
              and marks[1][3]-marks[1][1] >= 1.22*(marks[0][3]-marks[0][1])
              and all(.18 <= (b[2]-b[0])/(b[3]-b[1]) <= .58 for b in marks)):
            cls = "volume_on"
        else:
            continue
        l, t, r, b = _union([(x, y, x+w, y+h), *marks])
        matches.append((cls, [left+l, top+t, left+r, top+b]))
    if len(matches) != 1:
        return None
    cls, glyph = matches[0]
    evidence = "Visible speaker with sound waves" if cls == "volume_on" else "Visible speaker with mute X"
    return _visual_annotation({"class": cls, "source": "studio_shorts_toolbar", "visible_evidence": evidence},
                              glyph, region, "pixel_shorts_toolbar", _padded_icon(glyph, (rgb.shape[1], rgb.shape[0]), []))


def track_box(track, height):
    left, top, right, bottom = track
    center = (top+bottom)/2
    half_height = max((bottom-top)/2, round(height*TRACK_HEIGHT_RATIO)/2)
    half_height = min(half_height, center, height-center)
    return [left, center-half_height, right, center+half_height]


def inside_track(marker, track):
    cx, cy = (marker[0]+marker[2])/2, (marker[1]+marker[3])/2
    return track[0] <= cx <= track[2] and track[1] <= cy <= track[3]


def _gold_pixels(image):
    red, green, blue = (image[..., channel] for channel in range(3))
    return (red >= 160) & (green >= 115) & (green >= .58*red) & (red >= .90*green) & (green-blue >= 65)


def _has_left_color_run(mask, minimum_fraction):
    import numpy as np

    transitions = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return any(start <= .04*mask.size and stop-start >= max(6, minimum_fraction*mask.size)
               for start, stop in zip(transitions[::2], transitions[1::2]))


def _embedded_seek_stripe(pixels, slider_region, player, ad_track=False):
    """Follow a real played/unplayed line across a translucent player/page seam."""
    import numpy as np

    height, width = pixels.shape[:2]
    if not valid_box(player, (width, height)):
        return None
    left, top, right, bottom = slider_region
    edge = round(player[3])
    band = max(3, min(12, round(.08*(bottom-top))))
    if height-edge <= max(12, .035*height) or abs(bottom-edge) > band:
        return None
    left, right = max(left, math.ceil(player[0])), min(right, math.floor(player[2]))
    span = right-left
    if span < .75*(slider_region[2]-slider_region[0]):
        return None
    max_thickness = max(3, min(12, round(.008*span)))
    scan_top = max(1, top, edge-band-max_thickness)
    scan_bottom = min(height-1, bottom+band, edge+band)
    crop = pixels[scan_top-1:scan_bottom+1, left:right]
    played = (_gold_pixels(crop) if ad_track else
              (crop[..., 0] >= 165) & (crop[..., 0]-crop[..., 1:].max(axis=-1) >= 75) & (crop[..., 1] <= 105))
    matches = []
    for end in range(max(edge-band, scan_top+2), scan_bottom+1):
        after = end-scan_top+1
        for start in range(max(scan_top, end-max_thickness), end-1):
            first = start-scan_top+1
            anchor = played[first:after].all(axis=0) & ~played[first-1] & ~played[after]
            if not _has_left_color_run(anchor, .01 if ad_track else .025):
                continue
            middle = (first+after-1)//2
            row = crop[middle]
            neutral = (row.max(axis=1)-row.min(axis=1) <= 70) & (row.max(axis=1) >= 52)
            upper = np.max(np.abs(crop[first]-crop[first-1]), axis=1) >= 14
            lower = np.max(np.abs(row-crop[after]), axis=1) >= 24
            columns = np.flatnonzero((played[middle] | neutral) & upper & lower)
            if len(columns) >= .65*span and columns[-1]-columns[0]+1 >= .80*span:
                matches.append([left+int(columns[0]), start, left+int(columns[-1])+1, end])
    return matches[0] if len(matches) == 1 else None


def localize_timeline(rgb, annotation, player_region=None):
    """Find an observed seek stripe and optional protruding thumb in an AX region.

    A verified embedded player frame can limit the search to its lower edge,
    but the slider value and playback time never supply geometry.
    A thumb can extend outside the AX hit region, so inspect its attached caps
    in the original image after independently finding the thin track.
    """
    import cv2
    import numpy as np

    height, width = rgb.shape[:2]
    region = _control_region(annotation, (width, height))
    if region is None:
        return [], "Invalid slider region"
    left, top, right, bottom = region
    ad_track = annotation.get("class") == "ad_progress_bar"
    if ad_track:
        top = max(top, bottom-max(12, round(.05*(right-left))))
    rw, rh = right-left, bottom-top
    if rw < 5*rh or rw < 40 or rh < 3:
        return [], "Slider region is not a horizontal seek control"
    pixels = np.asarray(rgb, dtype=np.int16)

    def colors(rows):
        maximum, minimum = rows.max(axis=-1), rows.min(axis=-1)
        red = (rows[..., 0] >= 150) & (rows[..., 0] >= 1.4*rows[..., 1]) & (rows[..., 0] >= 1.4*rows[..., 2])
        neutral = (maximum-minimum <= 45) & (maximum >= 60)
        return red, neutral

    # Require a long, thin, internally stable stripe with an independently
    # observed color boundary above AND below. Red and gray/white segments
    # together form the track; neither segment must span the whole control.
    stripes = []
    for y in range(max(1, top), min(bottom, height-1)):
        row = pixels[y, left:right]
        red, neutral = colors(row)
        played = _gold_pixels(row) if ad_track else red
        start = (played | neutral) & (np.max(np.abs(row-pixels[y-1, left:right]), axis=1) >= 20)
        if ad_track and not _has_left_color_run(played & start, .01):
            continue
        if np.count_nonzero(start) < .70*rw:
            continue
        stable = start.copy()
        for end in range(y+1, min(height, bottom+1, y+min(13, max(3, rh//4)))):
            if end > y+1:
                stable &= np.max(np.abs(pixels[end-1, left:right]-row), axis=1) <= 16
            supported = stable & (np.max(np.abs(pixels[end-1, left:right]-pixels[end, left:right]), axis=1) >= 20)
            xs = np.flatnonzero(supported)
            if len(xs) >= .70*rw and xs[-1]-xs[0]+1 >= .80*rw:
                stripes.append((len(xs), [left+int(xs[0]), y, left+int(xs[-1])+1, end]))
    unique = []
    for score, stripe in sorted(stripes, key=lambda item: item[0], reverse=True):
        if not any(_intersection_area(stripe, old) > 0 for _, old in unique):
            unique.append((score, stripe))
    if len(unique) > 1:
        return [], "Multiple horizontal stripes in the slider region; review manually"
    track = unique[0][1] if unique else _embedded_seek_stripe(pixels, region, player_region, ad_track)
    if track is None:
        return [], "No thin seek track with visible upper and lower boundaries"
    # The rounded endpoints get shorter vertically, so they may not appear in
    # the full-height boundary mask. Follow the visible stroke at its center.
    middle = pixels[(track[1]+track[3]-1)//2, left:right]
    ends = np.minimum(np.max(np.abs(middle-pixels[track[1]-1, left:right]), axis=1),
                      np.max(np.abs(middle-pixels[track[3], left:right]), axis=1)) >= 16
    for edge, step in ((0, -1), (2, 1)):
        for _ in range(track[3]-track[1]):
            x = track[edge]-1 if step < 0 else track[edge]
            if not left <= x < right or not ends[x-left]:
                break
            track[edge] += step

    def candidate(cls, content, evidence):
        box = track_box(content, height) if cls in TRACK_CLASSES else content
        result = _visual_annotation(annotation, content, region, "pixel_timeline", box)
        result.update({"class": cls, "geometry_policy": geometry_policy(cls),
                       "source": "studio_timeline_pixels", "visible_evidence": evidence})
        return result

    if ad_track:
        return [candidate("ad_progress_bar", track, "Observed gold and neutral ad progress stripe")], "Pixel-localized ad progress track"
    evidence = ("Observed thin red/neutral seek track with two visible boundaries" if unique else
                "Observed played/unplayed stripe at the embedded player edge")
    result = [candidate("progress_bar", track, evidence)]
    radius = max(8, min(64, round(rh*.8)))
    scan_top, scan_bottom = max(0, track[1]-radius), min(height, track[3]+radius)
    scan = pixels[scan_top:scan_bottom, track[0]:track[2]]
    red, neutral = colors(scan)
    if not unique:
        stripe_row = scan[(track[1]+track[3]-1)//2-scan_top]
        red &= np.max(np.abs(scan-stripe_row), axis=-1) <= 65
    white = neutral & (scan.min(axis=-1) >= 180)
    markers = []
    for mask in (red, white):
        split_top, split_bottom = track[1]-scan_top, track[3]-scan_top

        def caps(part, offset, above):
            if not part.size:
                return []
            _, _, stats, _ = cv2.connectedComponentsWithStats(part.astype(np.uint8), connectivity=8)
            caps_found = []
            for x, y, w, h, area in stats[1:]:
                if (h >= 2 and 6 <= w <= 2*radius and
                        (y+h == part.shape[0] if above else y == 0)):
                    caps_found.append([int(x), int(y)+offset, int(x+w), int(y+h)+offset])
            return caps_found

        above = caps(mask[:split_top], 0, True)
        below = caps(mask[split_bottom:], split_bottom, False)
        for upper in above:
            for lower in below:
                uw, lw = upper[2]-upper[0], lower[2]-lower[0]
                if abs(sum(upper[::2])-sum(lower[::2]))/2 > max(3, .15*max(uw, lw)) or not .6 <= uw/lw <= 1.7:
                    continue
                thumb = _union((upper, lower))
                tw, th = thumb[2]-thumb[0], thumb[3]-thumb[1]
                fill = np.count_nonzero(mask[thumb[1]:thumb[3], thumb[0]:thumb[2]])/(tw*th)
                if not .55 <= tw/th <= 1.8 or not .50 <= fill <= .93:
                    continue
                observed = [thumb[0]+track[0], thumb[1]+scan_top, thumb[2]+track[0], thumb[3]+scan_top]
                if observed not in markers:
                    markers.append(observed)
    if len(markers) == 1 and inside_track(markers[0], result[0]["xyxy"]):
        result.append(candidate("playhead", markers[0], "Observed round thumb protruding above and below the seek track; no time/percentage estimate"))
    return result, "Pixel-localized seek track" + (" and visible thumb" if len(result) == 2 else "; no unambiguous visible thumb")


def _visual_annotation(annotation, content, region, source, box=None):
    result = deepcopy(annotation)
    if region is not None:
        result.setdefault("localization_region_xyxy", list(region))
    if annotation.get("source") in INTERACTION_SOURCES or annotation.get("geometry_status") == "interaction_region":
        result.setdefault("interaction_xyxy", deepcopy(annotation["xyxy"]))
    result.setdefault("geometry_parent_source", annotation.get("source", "unknown"))
    result.update(xyxy=box or content, visual_xyxy=box or content, visible_content_xyxy=content,
                  geometry_policy=geometry_policy(annotation["class"]), geometry_source=source,
                  geometry_policy_version=POLICY_VERSION, geometry_reviewed=False, review_required=True)
    return result


def localize_control(rgb, annotation, *, exclusions=()):
    """Locate measured visible pixels, then apply the class's annotation box rules."""
    import cv2
    import numpy as np

    height, width = rgb.shape[:2]
    size = (width, height)
    cls = annotation.get("class")
    policy = geometry_policy(cls)
    region = _control_region(annotation, size, prefer_localization=cls in {"visit_advertiser", "ad_cta"})
    explicit = annotation.get("visual_xyxy")
    if "visual_xyxy" in annotation and not valid_box(explicit, size):
        return None, "Invalid explicit visual geometry"
    if explicit:
        explicit = list(map(float, explicit))
    if cls == "autoplay":
        toggle = _autoplay_toggle(rgb, region) if region else None
        if toggle:
            box, state = toggle
            result = _visual_annotation(annotation, box, region, "pixel_autoplay_toggle")
            result.update(autoplay_state=state, visible_evidence=f"Visible Autoplay capsule with {state} glyph")
            return result, f"Visible Autoplay toggle ({state})"
        return None, "No verified Autoplay toggle and state glyph in this region"
    if cls in FULLSCREEN_GLYPHS and region:
        matches = [(kind, glyph) for kind in sorted(FULLSCREEN_GLYPHS)
                   if (glyph := _fullscreen_glyph(rgb, region, kind))]
        if len(matches) == 1:
            kind, glyph = matches[0]
            visual = _visual_annotation({**annotation, "class": kind}, glyph, region, "pixel_fullscreen_arrows",
                                        _padded_icon(glyph, size, exclusions))
            visual["visible_evidence"] = "Visible pair of inward arrows" if kind == "exit_fullscreen" else "Visible pair of outward arrows"
            return visual, "Verified the direction of the two fullscreen arrows"
        if annotation.get("source") in INTERACTION_SOURCES or annotation.get("interaction_xyxy"):
            return None, "No unambiguous pair of fullscreen arrows in the control region"
    if (cls == "pause" and annotation.get("geometry_source") == "pixel_shorts_playback"
            and valid_box(annotation.get("localization_region_xyxy"), size)):
        # Older Shorts proposals measured the disc. Recheck the saved search area
        # so automatic recovery and explicit refinement can both correct them.
        search_region = _control_region(annotation, size, prefer_localization=True)
        left, top, right, bottom = search_region
        span = min(right-left, bottom-top)/(2*SHORTS_PLAYBACK_SEARCH_RATIO)
        match = _shorts_playback_glyph(rgb, search_region, span)
        if match and match[0] == cls:
            glyph = match[1]
            visual = _visual_annotation(annotation, glyph, search_region, "pixel_shorts_playback", _padded_icon(glyph, size, exclusions))
            return visual, "Verified both visible pause bars"
    if explicit and annotation.get("geometry_policy_version") == POLICY_VERSION and annotation.get("geometry_policy") == policy:
        return project_visual_geometry(annotation), "Current annotation geometry"
    if policy == "visible_track":
        timeline, reason = localize_timeline(rgb, annotation)
        if timeline:
            return timeline[0], reason
        if (explicit and annotation.get("geometry_source") == "native_visible_track"
                and explicit[2]-explicit[0] >= 5*(explicit[3]-explicit[1])):
            return _visual_annotation(annotation, explicit, region, "native_visible_track", track_box(explicit, height)), "Centered on the observed native track"
        return None, reason
    if policy == "visible_button":
        source = annotation.get("geometry_source") or ""
        label = explicit if "ocr" in source else None
        button = _button_background(rgb, region, label, ad_button=cls == "ad_cta") if region else None
        if button:
            return _visual_annotation(annotation, button, region, "pixel_button_background"), "Visible button background and label"
        return None, f"The full {cls} button background could not be verified"
    if policy == "text_button":
        label = _text_button_label(rgb, region) if region else None
        if label:
            tap = _control_region(annotation, size)
            button = _button_background(rgb, tap, label, ad_button=True) if tap != region else None
            return _visual_annotation(annotation, button or label, region, "pixel_action_text"), "Visible Visit advertiser action"
        return None, "Could not verify the Visit advertiser text"
    if cls in {"ad_info", "back", "hide_content"} and region:
        glyph = _small_symbol(rgb, region, cls)
        if glyph is None and cls == "hide_content":
            # The ad panel can render its close button at the other end of the header.
            alternate = _control_region(annotation, size, prefer_localization=True)
            if alternate and alternate != region:
                glyph = _small_symbol(rgb, alternate, cls)
                if glyph:
                    region = alternate
        if glyph:
            return _visual_annotation(annotation, glyph, region, "pixel_control_symbol",
                                      _padded_icon(glyph, size, exclusions)), f"Verified the {cls} glyph"
        return None, f"No distinct {cls} glyph in the control region"
    if cls in TRANSPORT_GLYPHS and region:
        glyph = _transport_glyph(rgb, region, cls)
        if glyph:
            return _visual_annotation(annotation, glyph, region, "pixel_transport_shape", _padded_icon(glyph, size, exclusions)), "Verified centered playback glyph"
        if annotation.get("source") in INTERACTION_SOURCES or annotation.get("interaction_xyxy"):
            return None, "No unambiguous playback glyph in the control region"
    if explicit and annotation.get("geometry_source") != "human_drawn":
        if policy in {"icon_only", "artwork"}:
            blockers = list(exclusions)
            if cls in CAPTIONED_ICONS and region:
                x1, y1, x2, y2 = region
                if (x2-x1)*(y2-y1) <= .15*width*height:
                    for mask in _foreground_masks(rgb[y1:y2, x1:x2]):
                        parts = _components(mask)
                        absolute = [[x1+l, y1+t, x1+r, y1+b] for l, t, r, b in parts + _text_rows(parts)]
                        blockers.extend(_short_captions(absolute, explicit, cls))
            return _visual_annotation(annotation, explicit, region, annotation.get("geometry_source", "observed_glyph"),
                                      _padded_icon(explicit, size, blockers)), "Padded the visible pixels"
        return project_visual_geometry(annotation), "Explicit visual geometry"
    if region is None:
        return None, "Invalid control region"
    if policy != "icon_only":
        return None, f"{policy}: preserve the class-specific target; review its visible bounds manually"
    x1, y1, x2, y2 = region
    roi = rgb[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if min(rh, rw) < 6 or max(rh, rw) > .6*max(width, height) or rw*rh > .15*width*height:
        return None, "Control region is too small or broad for reliable glyph localization"

    candidates = []
    nearby_text = set()
    nearby_parts = set()
    for mask in _foreground_masks(roi):
        components = _components(mask)
        nearby_parts.update(components)
        rows = _text_rows(components)
        if cls not in ELLIPSIS_GLYPHS:
            nearby_text.update(tuple(row) for row in rows)
        if cls in ELLIPSIS_GLYPHS:
            groups = _dots(components, mask, annotation.get("ellipsis_orientation"))
        elif cls == "pause":
            groups = _pause_pairs(components)
        else:
            join_size = max(3, round(min(rw, rh)*.06) | 1) if cls == "remix" else 3
            joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((join_size, join_size), np.uint8))
            groups = _components(joined)
            if cls == "remix":
                # The two observed arrows and center plus can have overlapping
                # bounds while their strokes remain disconnected. Merge those
                # fragments, after excluding the separate "Remix" text row.
                fragments = [item for item in groups if not any(
                    _intersection_area(item, row) >= .45*(item[2]-item[0])*(item[3]-item[1]) for row in rows)]
                clusters = []
                for item in fragments:
                    merging = [item]
                    remaining = []
                    for cluster in clusters:
                        if (max(0, item[0]-cluster[2], cluster[0]-item[2]) <= 2 and
                                max(0, item[1]-cluster[3], cluster[1]-item[3]) <= 2):
                            merging.append(cluster)
                        else:
                            remaining.append(cluster)
                    clusters = remaining + [_union(merging)]
                groups.extend(clusters)
            # Speaker waves and similar pieces may be disconnected. Join only
            # neighbors aligned with the glyph; a lower text row stays separate.
            if cls in {"volume_on", "volume_muted", "cast"}:
                fragments = []
                for item in groups:
                    ih = item[3]-item[1]
                    aligned = [other for other in groups if
                               abs((other[1]+other[3]-item[1]-item[3])/2) <= .4*max(ih, other[3]-other[1])
                               and max(0, other[0]-item[2], item[0]-other[2]) <= .65*max(ih, other[3]-other[1])]
                    if len(aligned) > 1:
                        fragments.append(_union(aligned))
                groups.extend(fragments)
        for item in groups:
            iw, ih = item[2]-item[0], item[3]-item[1]
            if cls not in ELLIPSIS_GLYPHS:
                if min(iw, ih) < max(6, .12*min(rw, rh)) or not .35 <= iw/max(ih, 1) <= 2.5:
                    continue
                # Reject both individual letters and merged text words.
                if any(_intersection_area(item, row) >= .45*iw*ih for row in rows):
                    continue
            if iw*ih >= .95*rw*rh:
                continue
            score = max(iw, ih)
            candidates.append((score, item))
    # Collapse candidates that describe the same connected glyph in two masks.
    unique = []
    for score, item in sorted(candidates, key=lambda entry: (
            entry[0], (entry[1][2]-entry[1][0])*(entry[1][3]-entry[1][1]), tuple(entry[1])), reverse=True):
        if not any(max(abs(a-b) for a, b in zip(item, old)) <= 3 or
                   _intersection_area(item, old) >= .8*min((item[2]-item[0])*(item[3]-item[1]),
                                                         (old[2]-old[0])*(old[3]-old[1])) for _, old in unique):
            unique.append((score, item))
    if not unique:
        return None, "No isolated visible glyph; label/count text was not used as an icon"
    if len(unique) > 1 and unique[1][0] >= .78*unique[0][0]:
        return None, "Multiple similarly sized shapes; manual icon bounds are required"
    _, item = unique[0]
    glyph = [x1+item[0], y1+item[1], x1+item[2], y1+item[3]]
    text = [[x1+l, y1+t, x1+r, y1+b] for l, t, r, b in sorted(nearby_text)]
    text.extend(_short_captions([[x1+l, y1+t, x1+r, y1+b] for l, t, r, b in sorted(nearby_parts)], glyph, cls))
    visual = _padded_icon(glyph, size, [*exclusions, *text])
    return _visual_annotation(annotation, glyph, region, "pixel_components", visual), "Pixel-localized glyph with padding"


def visible_neighbors(annotations, target, size):
    result = []
    for annotation in annotations:
        if annotation is target:
            continue
        if geometry_policy(annotation.get("class")) in {"visible_track", "visible_marker"}:
            continue
        box = annotation.get("visible_content_xyxy", annotation.get("visual_xyxy"))
        if box is None and annotation.get("geometry_reviewed"):
            box = annotation.get("xyxy")
        if valid_box(box, size):
            result.append(list(map(float, box)))
    return result


def geometry_proposals(rgb, annotations, *, preserve_reviewed=False):
    proposals, diagnostics = [], []
    size = (rgb.shape[1], rgb.shape[0])
    for index, annotation in enumerate(annotations):
        if preserve_reviewed and annotation.get("geometry_reviewed"):
            continue
        policy = geometry_policy(annotation.get("class"))
        if policy not in {"icon_only", "visible_track", "visible_button", "visible_toggle", "text_button"} and not (
                policy == "artwork" and annotation.get("visual_xyxy")):
            continue
        localized, reason = localize_control(rgb, annotation, exclusions=visible_neighbors(annotations, annotation, size))
        diagnostic = {"annotation_index": index, "class": localized["class"] if localized else annotation.get("class"),
                      "reason": reason}
        if localized is None and (policy in {"visible_button", "visible_toggle", "text_button"}
                                  or annotation.get("class") in TRANSPORT_GLYPHS | FULLSCREEN_GLYPHS):
            diagnostic["manual_geometry"] = {"class": annotation["class"], "xyxy": annotation["xyxy"]}
        diagnostics.append(diagnostic)
        if localized is None:
            continue
        original = annotation["xyxy"]
        if (localized["class"] == annotation["class"] and max(abs(a-b) for a, b in zip(original, localized["xyxy"])) <= 1
                and localized.get("autoplay_state") == annotation.get("autoplay_state")):
            continue
        localized.update(action="replace_geometry", replaces={"class": annotation["class"], "xyxy": deepcopy(original)},
                         source="studio_icon_geometry", descriptor=reason)
        proposals.append(localized)
    return proposals, diagnostics
