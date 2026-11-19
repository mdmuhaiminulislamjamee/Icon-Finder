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
TRACK_HEIGHT_RATIO = .0135
MIN_TRACK_HEIGHT_RATIO = .012
TRANSPORT_GLYPHS = {"play", "previous", "next"}
CAPTIONED_ICONS = {"like", "dislike", "comments", "save", "share", "remix", "ask", "download", "report_problem"}
SPECIAL_POLICIES = {
    "progress_bar": "visible_track", "playhead": "visible_marker",
    "sound": "artwork", "channel": "artwork", "autoplay": "visible_toggle",
    "subscribe": "visible_button", "skip_ad": "text_and_icon",
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
    return (geometry_policy(annotation.get("class")) in {"icon_only", "visible_button", "text_and_icon"}
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


def _dots(boxes):
    dots = [item for item in boxes if .55 <= (item[2]-item[0])/max(1, item[3]-item[1]) <= 1.8]
    result = []
    for group in combinations(dots[:30], 3):
        sizes = [max(item[2]-item[0], item[3]-item[1]) for item in group]
        if max(sizes) > 1.7*min(sizes):
            continue
        centers = [((item[0]+item[2])/2, (item[1]+item[3])/2) for item in group]
        for axis in (0, 1):
            ordered = sorted(centers, key=lambda item: item[axis])
            a, b = ordered[1][axis]-ordered[0][axis], ordered[2][axis]-ordered[1][axis]
            if (a >= .7*max(sizes) and .65 <= b/max(a, 1) <= 1.55
                    and max(item[1-axis] for item in centers)-min(item[1-axis] for item in centers) <= .6*max(sizes)):
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


def _control_region(annotation, size):
    for key in ("interaction_xyxy", "localization_region_xyxy", "xyxy"):
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
                triangle = cv2.approxPolyDP(contour, .065*cv2.arcLength(contour, True), True)
                if len(triangle) != 3 or not .35 <= cv2.contourArea(contour)/(w*h) <= .68:
                    continue
                points = sorted(triangle[:, 0].tolist(), reverse=cls == "previous")
                base, tip = points[:2], points[2]
                if abs(base[0][0]-base[1][0]) > .3*w or not y+.28*h <= tip[1] <= y+.72*h:
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


def _subscribe_button(rgb, region, label_box):
    import cv2
    import numpy as np

    left, top, right, bottom = region
    rw, rh = right-left, bottom-top
    if not 1.3 <= rw/rh <= 10 or rh < 10:
        return None
    margin = max(3, round(rh*.13))
    ox, oy = max(0, left-margin), max(0, top-margin)
    crop = rgb[oy:bottom+margin, ox:right+margin]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    matches = []
    for bright in (True, False):
        for level in (40, 80, 120, 160, 200, 232):
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
                label = np.uint8((tone-field if bright else field-tone) >= 25)
                rows = _text_rows(_components(label))
                if rows or (label_box and .02 < np.mean(label) < .48):
                    matches.append(button)
    if not matches:
        return None
    outer = _union(matches)
    if any(_intersection_area(box, outer) < .75*(outer[2]-outer[0])*(outer[3]-outer[1]) for box in matches):
        return None
    return outer


def track_box(track, height):
    left, top, right, bottom = track
    center = (top+bottom)/2
    half_height = max((bottom-top)/2, round(height*TRACK_HEIGHT_RATIO)/2)
    half_height = min(half_height, center, height-center)
    return [left, center-half_height, right, center+half_height]


def inside_track(marker, track):
    cx, cy = (marker[0]+marker[2])/2, (marker[1]+marker[3])/2
    return track[0] <= cx <= track[2] and track[1] <= cy <= track[3]


def localize_timeline(rgb, annotation):
    """Find an observed seek stripe and optional protruding thumb in an AX region.

    The slider value, playback time and screen layout never supply geometry.
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
        start = (red | neutral) & (np.max(np.abs(row-pixels[y-1, left:right]), axis=1) >= 20)
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
    if not unique:
        return [], "No thin seek track with visible upper and lower boundaries"
    if len(unique) > 1:
        return [], "Multiple horizontal stripes in the slider region; review manually"
    _, track = unique[0]
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
        box = track_box(content, height) if cls == "progress_bar" else content
        result = _visual_annotation(annotation, content, region, "pixel_timeline", box)
        result.update({"class": cls, "geometry_policy": geometry_policy(cls),
                       "source": "studio_timeline_pixels", "visible_evidence": evidence})
        return result

    result = [candidate("progress_bar", track, "Observed thin red/neutral seek track with two visible boundaries")]
    radius = max(8, min(64, round(rh*.8)))
    scan_top, scan_bottom = max(0, track[1]-radius), min(height, track[3]+radius)
    scan = pixels[scan_top:scan_bottom, track[0]:track[2]]
    red, neutral = colors(scan)
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
    region = _control_region(annotation, size)
    explicit = annotation.get("visual_xyxy")
    if "visual_xyxy" in annotation and not valid_box(explicit, size):
        return None, "Invalid explicit visual geometry"
    if explicit:
        explicit = list(map(float, explicit))
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
        button = _subscribe_button(rgb, region, label) if region else None
        if button:
            return _visual_annotation(annotation, button, region, "pixel_button_background"), "Visible Subscribe button and label"
        return None, "Outline the full Subscribe button; its background was ambiguous"
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
        if cls != "more_options":
            nearby_text.update(tuple(row) for row in rows)
        if cls == "more_options":
            groups = _dots(components)
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
            if cls in {"volume_on", "volume_muted", "cast", "fullscreen", "exit_fullscreen"}:
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
            if cls != "more_options":
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
        if policy not in {"icon_only", "visible_track", "visible_button"} and not (
                policy == "artwork" and annotation.get("visual_xyxy")):
            continue
        localized, reason = localize_control(rgb, annotation, exclusions=visible_neighbors(annotations, annotation, size))
        diagnostic = {"annotation_index": index, "class": annotation.get("class"), "reason": reason}
        if localized is None and (policy == "visible_button" or annotation.get("class") in TRANSPORT_GLYPHS):
            diagnostic["manual_geometry"] = {"class": annotation["class"], "xyxy": annotation["xyxy"]}
        diagnostics.append(diagnostic)
        if localized is None:
            continue
        original = annotation["xyxy"]
        if max(abs(a-b) for a, b in zip(original, localized["xyxy"])) <= 1:
            continue
        localized.update(action="replace_geometry", replaces={"class": annotation["class"], "xyxy": deepcopy(original)},
                         source="studio_icon_geometry", descriptor=reason)
        proposals.append(localized)
    return proposals, diagnostics
