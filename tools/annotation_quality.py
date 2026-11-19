"""Annotation validation and accessibility matching, without changing raw records."""
from collections import Counter
from copy import deepcopy
import math
import re
import unicodedata

from tools.annotation_geometry import POLICY_VERSION, geometry_policy, needs_visual_geometry, project_visual_geometry, valid_box


CLASSES = (
    "skip_ad", "progress_bar", "playhead", "play", "pause", "volume_on",
    "volume_muted", "fullscreen", "exit_fullscreen", "previous", "next",
    "seek_backward", "seek_forward", "like", "dislike", "comments", "share",
    "remix", "sound", "save", "search", "more_options", "settings", "captions",
    "autoplay", "time_display", "more_videos", "show_cards", "ask",
    "minimize_player", "cast", "hide_controls", "subscribe", "channel", "download", "report_problem",
)


def normalized(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^\w]+|_", " ", value).split())


# Identifier aliases are exact after normalization, so e.g. "dislike" cannot
# accidentally match "like". Extend these with identifiers from real snapshots.
_IDENTIFIERS = {}
for _class in CLASSES:
    for _alias in (_class, f"{_class}_button", f"reel_{_class}_button",
                   f"id.{_class}_button", f"id.reel_{_class}_button"):
        _IDENTIFIERS[normalized(_alias)] = _class
_IDENTIFIERS.update({normalized(key): value for key, value in {
    "id.reel_comment_button": "comments", "comment_button": "comments",
    "id.reel_pivot_button": "sound", "decorated-avatar-id": "channel",
    "id.player.watch.captions.button": "captions", "id.player.overflow.button": "settings",
    "id.timestamp.toggle.view": "time_display", "eml.compact_subscribe_button": "subscribe",
    "eml.animated_subscribe_button": "subscribe", "id.player.scrubber.slider": "progress_bar",
    "skip_ad_button": "skip_ad",
    "player_seek_bar": "progress_bar", "player_scrubber": "playhead",
}.items()})

# These describe the action on the button. A button labelled "Unmute" means
# audio is currently muted; "Mute" means audio is currently on.
_LABELS = {
    "skip_ad": ("skip", "skip ad", "skip ads", "skip advertisement", "বিজ্ঞাপন এড়িয়ে যান"),
    "exit_fullscreen": ("exit full screen", "exit fullscreen"),
    "fullscreen": ("full screen", "fullscreen", "enter full screen"),
    "volume_muted": ("unmute", "unmute video", "sound off", "volume muted"),
    "volume_on": ("mute", "mute video", "sound on", "volume on"),
    "dislike": ("dislike", "remove dislike", "dislike this video"),
    "like": ("like", "unlike", "like this video", "unlike this video", "remove like", "পছন্দ"),
    "comments": ("comments", "comment", "view comments", "open comments", "মন্তব্য"),
    "share": ("share", "share video", "শেয়ার"),
    "remix": ("remix", "remix this video"),
    "sound": ("original sound", "use this sound", "sound", "audio track"),
    "save": ("save", "save video", "save to playlist", "saved", "সংরক্ষণ"),
    "search": ("search", "search youtube", "অনুসন্ধান"),
    "more_options": ("more options", "more actions", "additional options"),
    "previous": ("previous", "previous video"),
    "next": ("next", "next video"),
    "seek_backward": ("rewind", "seek backward", "rewind 10 seconds"),
    "seek_forward": ("fast forward", "seek forward", "forward 10 seconds"),
    "pause": ("pause", "pause video"),
    "play": ("play", "play video", "resume video"),
    "progress_bar": ("video progress", "seek bar", "progress bar"),
    "playhead": ("playhead", "scrubber", "seek handle"),
    "settings": ("settings",), "captions": ("captions", "subtitles"),
    "autoplay": ("autoplay",),
    "minimize_player": ("minimize player", "collapse player", "collapse video"),
    "cast": ("cast", "cast to device"),
    "hide_controls": ("hide controls", "close controls"),
    "subscribe": ("subscribe", "subscribed"),
    "download": ("download", "download video"),
    "report_problem": ("report", "report problem"),
}


def match_accessibility(element):
    """Return (class, evidence) or (None, reason); never guess from position."""
    identifier = element.get("identifier") or element.get("resource_id", "")
    action = normalized(element.get("label") or element.get("source_label") or element.get("caption"))
    if (re.search(r"\bskip(?: ads?)? in\b", action) or
            (element.get("enabled") is False and ("skip" in action or "skip" in normalized(identifier)))):
        return None, "Skip control is unavailable"
    match = _IDENTIFIERS.get(normalized(identifier))
    if match:
        return match, f"identifier: {identifier}"
    label = normalized(element.get("label") or element.get("source_label") or element.get("caption"))
    # Remove common UI state and numeric count suffixes, preserving the action.
    label = re.sub(r"\s+(?:button|selected|not selected)$", "", label)
    label = re.sub(r"\s+along with\s+\d[\d\s.,]*(?:k|m|b)?\s+other people$", "", label)
    label = re.sub(r"\s+\d[\d\s.,]*(?:k|m|b|thousand|million)?(?:\s+(?:likes?|comments?|views?))?$", "", label)
    for cls, aliases in _LABELS.items():
        if label in {normalized(alias) for alias in aliases}:
            return cls, f"label: {element.get('label') or element.get('source_label') or element.get('caption')}"
    return None, "No supported identifier or action label"


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def clean_annotations(annotations, image_size=None):
    """Clean a display/curation copy; callers must retain original raw data."""
    result = []
    report = {"input_count": len(annotations), "duplicates_removed": 0,
              "invalid_boxes": 0, "clipped_boxes": 0, "issues": []}
    width, height = image_size or (0, 0)
    for index, annotation in enumerate(annotations):
        try:
            if not isinstance(annotation, dict) or not isinstance(annotation.get("class"), str) or not annotation["class"].strip():
                raise ValueError("missing class")
            annotation = project_visual_geometry(annotation)
            values = annotation["xyxy"]
            if len(values) != 4 or any(isinstance(value, bool) for value in values):
                raise ValueError("expected four pixel coordinates")
            box = [float(value) for value in values]
            if not all(math.isfinite(value) for value in box):
                raise ValueError("non-finite coordinate")
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("empty or inverted box")
            if width > 0 and height > 0:
                clipped = [max(0, min(width, box[0])), max(0, min(height, box[1])),
                           max(0, min(width, box[2])), max(0, min(height, box[3]))]
                if clipped != box:
                    report["clipped_boxes"] += 1
                box = clipped
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError("box outside image")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            report["invalid_boxes"] += 1
            report["issues"].append(f"Box {index + 1}: {exc}")
            continue
        duplicate = next((item for item in result if item["class"] == annotation["class"]
                          and (max(abs(a-b) for a, b in zip(item["xyxy"], box)) <= 1
                               or box_iou(item["xyxy"], box) >= .98)), None)
        if duplicate is not None:
            report["duplicates_removed"] += 1
            continue
        item = deepcopy(annotation)
        item["xyxy"] = [round(value, 3) for value in box]
        result.append(item)
    report["display_count"] = len(result)
    report["class_counts"] = dict(Counter(item["class"] for item in result))
    report["unverified_icon_regions"] = sum(needs_visual_geometry(item) and geometry_policy(item["class"]) == "icon_only" for item in result)
    report["unverified_text_regions"] = sum(needs_visual_geometry(item) and geometry_policy(item["class"]) == "visible_button" for item in result)
    report["explicit_visual_boxes"] = sum("visual_xyxy" in item for item in result)
    report["geometry_policies"] = dict(Counter(geometry_policy(item["class"]) for item in result))
    return result, report


def accessibility_proposals(snapshot, image_size, allow_unconfirmed_regions=False):
    """Match a complete stored snapshot, retaining diagnostics for every element."""
    proposals, diagnostics = [], []
    width, height = image_size
    coordinate_space = snapshot.get("coordinate_space", "image_pixels")
    if coordinate_space not in {"image_pixels", "screen_points"}:
        raise ValueError("Unsupported accessibility coordinate space")
    screen_frame = snapshot.get("screen_frame")
    if coordinate_space == "screen_points" and (not screen_frame or len(screen_frame) != 4
                                                or screen_frame[2] <= 0 or screen_frame[3] <= 0):
        raise ValueError("screen_points snapshot requires [x, y, width, height] screen_frame")
    for index, element in enumerate(snapshot.get("elements", [])):
        cls, reason = match_accessibility(element)
        visible = element.get("visible", element.get("is_hittable"))
        box = element.get("xyxy")
        if box is None and element.get("frame") is not None:
            x, y, w, h = element["frame"]
            box = [x, y, x+w, y+h]
        if box is not None and coordinate_space == "screen_points":
            sx, sy, sw, sh = screen_frame
            box = [(box[0]-sx)*width/sw, (box[1]-sy)*height/sh,
                   (box[2]-sx)*width/sw, (box[3]-sy)*height/sh]
        candidate, _ = clean_annotations([{"class": cls or "unmatched", "xyxy": box}], image_size)
        accepted = cls is not None and (visible is True or allow_unconfirmed_regions) and bool(candidate)
        unavailable_skip = cls == "skip_ad" and (element.get("enabled") is False or visible is not True or not element.get("visual_xyxy"))
        if unavailable_skip:
            accepted = False
        diagnostics.append({"element_index": index, "matched_class": cls,
                            "proposed": accepted, "reason": reason if accepted else
                            ("Skip Ad is not enabled/available" if unavailable_skip else
                             "Element is not confirmed visible" if visible is not True else reason)})
        if accepted:
            proposal = {"class": cls, "xyxy": candidate[0]["xyxy"],
                              "interaction_xyxy": candidate[0]["xyxy"],
                              "geometry_policy": geometry_policy(cls),
                              "geometry_status": "interaction_region",
                              "visibility_unconfirmed": visible is not True,
                              "accessibility_element_index": index,
                              "source": "studio_accessibility_matcher", "source_label": element.get("label", ""),
                              "resource_id": element.get("identifier", element.get("resource_id", "")),
                              "descriptor": reason, "review_required": True,
                              "visible_evidence": "Accessibility supplies semantics and interaction bounds; verify visible geometry"}
            # The optional visual box is ALWAYS in original screenshot pixels,
            # even when the accessibility frame above is in screen points.
            if "visual_xyxy" in element:
                proposal["visual_xyxy"] = element["visual_xyxy"]
                proposal["geometry_source"] = element.get("geometry_source", "native_visual_localization")
                proposal["geometry_status"] = "visual_candidate"
                if (element.get("geometry_policy_version", snapshot.get("geometry_policy_version")) == POLICY_VERSION
                        and valid_box(element.get("visible_content_xyxy"), image_size)):
                    proposal["visible_content_xyxy"] = element["visible_content_xyxy"]
                    proposal["geometry_policy_version"] = POLICY_VERSION
            proposals.append(proposal)
    return clean_annotations(proposals, image_size)[0], diagnostics
