"""Reviewable proposals from accessibility, visible geometry and reviewed icons.

Predictions never become raw annotations or curated ground truth automatically.
The historical V11 adapter is retained for explicit opt-in comparisons only.
"""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading

from tools.annotation_quality import accessibility_proposals, box_iou, clean_annotations
from tools.annotation_geometry import (POLICY_VERSION, geometry_policy, geometry_proposals,
                                       localize_control, localize_timeline, needs_visual_geometry,
                                       valid_box, visible_neighbors)

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "youtube_media_control_models_v11_768"
_MODEL_LOCK = threading.RLock()
_REFERENCE = None
_RUNTIMES = None
LEGACY_MODEL_FLAG = "MEDIACOLLECTOR_ENABLE_LEGACY_MODEL"
TEMPLATE_CLASSES = {"skip_ad", "comments", "share", "remix", "save", "search",
                    "more_options", "settings", "captions", "like", "dislike",
                    "play", "pause", "volume_on", "volume_muted", "fullscreen", "exit_fullscreen",
                    "minimize_player", "cast", "hide_controls", "download", "report_problem", "subscribe"}


def reference():
    global _REFERENCE
    with _MODEL_LOCK:
        if _REFERENCE is None:
            spec = importlib.util.spec_from_file_location("_studio_v11_reference", DELIVERY / "scripts/infer.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _REFERENCE = module
        return _REFERENCE


def model_proposals(rgb):
    """Legacy opt-in adapter: unchanged historical two-model postprocessing.

Independent still captures have no temporal relationship: live-track smoothing
would mix unrelated images and is deliberately not applied here.
"""
    global _RUNTIMES
    ref = reference()
    with _MODEL_LOCK:
        if _RUNTIMES is None:
            paths = ref.model_paths(DELIVERY, "onnx", "float32")
            _RUNTIMES = tuple(ref.Runtime(path) for path in paths)
        padded, gain, left, top = ref.letterbox(rgb)
        controls = ref.decode(_RUNTIMES[0].invoke(padded), ref.CONTROL_NAMES,
                              .25, .70, gain, left, top, rgb.shape[:2])
        heads = ref.decode(_RUNTIMES[1].invoke(padded), ref.PLAYHEAD_NAMES,
                           .20, .70, gain, left, top, rgb.shape[:2])
        detections = controls + ref.select_one_playhead(controls, heads, rgb.shape[:2])
    return [dict(item.as_dict(), source="studio_v11_onnx", review_required=True,
                 geometry_policy=geometry_policy(item.label), geometry_source="full_frame_visual_model",
                 descriptor="V11 full-frame controls/progress + playhead detector",
                 visible_evidence="Model prediction; check icon and box against screenshot")
            for item in detections]


def legacy_model_enabled():
    return os.environ.get(LEGACY_MODEL_FLAG, "").strip() == "1"


def _reviewed_examples(data_root, platform, content_format):
    # Local approvals take precedence over reference styles. Otherwise the
    # per-class template limit could prevent a newly reviewed iPad style from
    # ever being tried after the three bundled examples fill that limit.
    paths = data_root.glob(f"{platform}/*/{content_format}/collected/curation/annotation_overrides.json")
    for overrides_file in sorted(paths, key=lambda path: path.stat().st_mtime_ns, reverse=True):
        overrides = json.loads(overrides_file.read_text(encoding="utf-8"))
        ordered = sorted(overrides.items(), key=lambda pair: pair[1].get("updated_at", ""), reverse=True)
        for rid, override in ordered:
            if override.get("status") not in {"approved", "resolved"}:
                continue
            images = overrides_file.parent.parent / "images"
            for extension in (".png", ".jpg", ".jpeg"):
                path = (images / f"{rid}{extension}").resolve()
                if path.is_relative_to(images.resolve()) and path.is_file():
                    yield path, override.get("curated_annotations", []), f"approved:{rid}"
                    break
    # The handover is a reference source for suggestions only, never an active
    # collection or a replacement for canonical source annotations.
    handover = ROOT / "annotation_format_handover(1)"
    examples = handover / "CURATED_EXAMPLES.json"
    if examples.exists():
        for example in json.loads(examples.read_text(encoding="utf-8"))["examples"]:
            context = example.get("context", {})
            if context.get("platform") == platform and example.get("curation", {}).get("status") in {"approved", "resolved"}:
                path = (handover / example["packaged_image"]).resolve()
                if path.is_relative_to(handover.resolve()):
                    yield path, example["curated_annotations"], f"handover:{example['record_id']}"


def template_proposals(rgb, examples, exclude_image=None):
    """Conservative multi-scale edge matching against reviewed visual examples.

Search the complete frame. A template similarity is not model probability.
    Artwork (e.g. sound thumbnails), seek tracks and playheads are excluded.
    Subscribe and Skip Ad retain their explicit text/control geometry policies.
"""
    import cv2
    import numpy as np
    from PIL import Image

    height, width = rgb.shape[:2]
    gain = min(1.0, 1280 / max(width, height))
    size = (max(1, round(width*gain)), max(1, round(height*gain)))
    gray = cv2.cvtColor(cv2.resize(rgb, size), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    counts, proposals = Counter(), []
    for path, annotations, origin in examples:
        if exclude_image is not None and path.resolve() == exclude_image.resolve():
            continue
        with Image.open(path) as image:
            source = np.asarray(image.convert("RGB"))
        sh, sw = source.shape[:2]
        for ann in annotations:
            cls = ann.get("class")
            if cls not in TEMPLATE_CLASSES or counts[cls] >= 3:
                continue
            valid, _ = clean_annotations([ann], (sw, sh))
            if not valid or needs_visual_geometry(valid[0]):
                continue
            x1, y1, x2, y2 = map(round, valid[0]["xyxy"])
            if min(x2-x1, y2-y1) < 8 or (x2-x1)*(y2-y1) > sw*sh*.025:
                continue
            crop = cv2.cvtColor(source[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
            counts[cls] += 1
            for scale in (.5, .65, .8, 1.0, 1.25, 1.5):
                tw, th = round(crop.shape[1]*scale*gain), round(crop.shape[0]*scale*gain)
                if min(tw, th) < 8 or tw > gray.shape[1] or th > gray.shape[0]:
                    continue
                template = cv2.Canny(cv2.resize(crop, (tw, th)), 60, 140)
                if np.count_nonzero(template) < 12:
                    continue
                scores = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)
                for _ in range(2):
                    _, score, _, (x, y) = cv2.minMaxLoc(scores)
                    if not np.isfinite(score) or score < .90:
                        break
                    box = [x/gain, y/gain, (x+tw)/gain, (y+th)/gain]
                    proposals.append({"class": cls, "xyxy": box, "similarity": round(score, 5),
                                      "visual_xyxy": box, "geometry_policy": geometry_policy(cls),
                                      "geometry_source": "reviewed_visual_template",
                                      "source": "studio_reviewed_template", "template_origin": origin,
                                      "review_required": True,
                                      "descriptor": "Multi-scale edge match against a reviewed example",
                                      "visible_evidence": "Template similarity; verify class, state and geometry"})
                    scores[max(0, y-th//2):y+th//2+1, max(0, x-tw//2):x+tw//2+1] = -1
    # Per-class suppression also combines overlapping matches from scale variants.
    selected = []
    for item in sorted(proposals, key=lambda item: item["similarity"], reverse=True):
        if not any(old["class"] == item["class"] and box_iou(old["xyxy"], item["xyxy"]) > .5 for old in selected):
            selected.append(item)
    return selected[:60], dict(counts)


def _proposal_ids(proposals):
    for item in proposals:
        fingerprint = json.dumps([item["class"], item["xyxy"], item["source"], item.get("replaces")], sort_keys=True)
        item["proposal_id"] = hashlib.sha256(fingerprint.encode()).hexdigest()[:20]
    return proposals


def refine_geometry(image_path, annotations):
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    proposals, diagnostics = geometry_proposals(rgb, annotations)
    return {"schema": "studio_annotation_proposals_v1", "geometry_policy_version": POLICY_VERSION,
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "proposals": _proposal_ids(proposals), "geometry_diagnostics": diagnostics,
            "warnings": ["Zoom in to review each replacement. Draw any ambiguous bounds manually."]}


def _same_visible_control(a, b):
    if a["class"] != b["class"]:
        return False
    x, y = a["xyxy"], b["xyxy"]
    if box_iou(x, y) >= .5:
        return True
    intersection = max(0, min(x[2], y[2])-max(x[0], y[0])) * max(0, min(x[3], y[3])-max(x[1], y[1]))
    smaller = min((x[2]-x[0])*(x[3]-x[1]), (y[2]-y[0])*(y[3]-y[1]))
    tolerance = max(2, .15*min(max(x[2]-x[0], x[3]-x[1]), max(y[2]-y[0], y[3]-y[1])))
    # A padded template around a tiny ellipsis is the same control even when
    # its IoU is low. Distinct same-class controls keep their separate centers.
    return (intersection >= .85*smaller and abs(x[0]+x[2]-y[0]-y[2])/2 <= tolerance
            and abs(x[1]+x[3]-y[1]-y[3])/2 <= tolerance)


def _same_control(a, b):
    if a["class"] != b["class"]:
        return False
    first, second = a.get("interaction_xyxy"), b.get("interaction_xyxy")
    if valid_box(first) and valid_box(second) and box_iou(list(map(float, first)), list(map(float, second))) >= .95:
        return True
    return _same_visible_control(a, b)


def suggest(image_path, raw_record, existing, data_root, platform, content_format, snapshot=None, *, include_fallbacks=True):
    import numpy as np
    from PIL import Image

    image_bytes = image_path.read_bytes()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
        image_size = image.size
    warnings, diagnostics = [], []
    proposals, geometry_diagnostics = geometry_proposals(rgb, existing, preserve_reviewed=not include_fallbacks)
    template_counts = {}
    if snapshot is None:
        snapshot = raw_record.get("accessibility_snapshot")
    if snapshot is not None:
        if snapshot.get("image_sha256") != image_hash:
            warnings.append("Accessibility snapshot does not match this image's SHA256; it was not used.")
        else:
            try:
                matched, diagnostics = accessibility_proposals(snapshot, image_size, allow_unconfirmed_regions=True)
                for candidate in matched:
                    policy = geometry_policy(candidate["class"])
                    if candidate["class"] == "progress_bar":
                        timeline, reason = localize_timeline(rgb, candidate)
                        if not timeline and candidate.get("visual_xyxy"):
                            track, reason = localize_control(rgb, candidate)
                            if track:
                                timeline = [track]
                        proposals.extend(timeline)
                        geometry_diagnostics.append({"class": "progress_bar", "reason": reason,
                                                     "interaction_xyxy": candidate["interaction_xyxy"]})
                        for diagnostic in diagnostics:
                            if diagnostic["element_index"] == candidate["accessibility_element_index"]:
                                diagnostic.update(proposed=bool(timeline), reason=reason)
                    elif candidate["class"] == "playhead":
                        # Associate the visible thumb with its actual track above.
                        continue
                    elif (policy in {"icon_only", "visible_button", "text_and_icon"}
                          or (policy == "artwork" and candidate.get("visual_xyxy")) or candidate.get("visibility_unconfirmed")):
                        visual, reason = localize_control(rgb, candidate, exclusions=visible_neighbors(existing + matched, candidate, image_size))
                        if visual is None:
                            for diagnostic in diagnostics:
                                if diagnostic["element_index"] == candidate["accessibility_element_index"]:
                                    diagnostic.update(proposed=False, reason=f"Semantic match only: {reason}")
                            geometry_diagnostics.append({"class": candidate["class"], "reason": reason,
                                                         "interaction_xyxy": candidate["xyxy"]})
                            continue
                        proposals.append(visual)
                    else:
                        proposals.append(candidate)
            except (TypeError, ValueError, KeyError) as exc:
                warnings.append(f"Could not match accessibility snapshot: {exc}")
    else:
        warnings.append("No complete accessibility snapshot was saved with this capture.")
    # Current capture/geometry work is independent of the backdated model.
    # Even cached model sessions must not enable it without an explicit opt-in.
    legacy_enabled = include_fallbacks and legacy_model_enabled()
    legacy_used = False
    model_status = {"mode": "disabled", "used": False, "replacement_configured": False}
    fallback_proposals = []
    if not include_fallbacks:
        model_status["mode"] = "not_requested"
    elif legacy_enabled and platform == "youtube":
        try:
            fallback_proposals.extend(model_proposals(rgb))
            legacy_used = True
            model_status.update(mode="legacy_opt_in", used=True)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            model_status["mode"] = "legacy_unavailable"
            warnings.append(f"Optional legacy visual detector unavailable: {exc}")
    elif legacy_enabled:
        model_status["mode"] = "unsupported_platform"
        warnings.append("The optional legacy YouTube model is not used for other platforms.")
    else:
        warnings.append("Legacy neural model disabled. Suggestions use accessibility, pixel geometry and reviewed examples; no replacement neural model is configured.")
    if include_fallbacks:
        try:
            matched, template_counts = template_proposals(rgb, _reviewed_examples(data_root, platform, content_format), image_path)
            fallback_proposals.extend(matched)
        except (ImportError, OSError, ValueError, KeyError) as exc:
            warnings.append(f"Reviewed-icon fallback unavailable: {exc}")
    for item in fallback_proposals:
        policy = geometry_policy(item["class"])
        if policy in {"visible_track", "visible_marker", "text_and_icon"}:
            continue  # Track, thumb and Skip Ad need the control evidence above.
        if policy in {"icon_only", "visible_button"}:
            exclusions = visible_neighbors(existing + proposals, item, image_size)
            visual, _ = localize_control(rgb, item, exclusions=exclusions)
            if visual:
                proposals.append(visual)
        else:
            proposals.append(item)
    proposals, _ = clean_annotations(proposals, image_size)
    filtered, replacements = [], set()
    for item in proposals:
        if item.get("action") != "replace_geometry":
            old = next((ann for ann in existing if _same_control(ann, item)), None)
            if old is not None:
                if not needs_visual_geometry(old) or old.get("geometry_reviewed"):
                    continue
                item.update(action="replace_geometry", replaces={"class": old["class"], "xyxy": old["xyxy"]})
                item.setdefault("interaction_xyxy", old.get("interaction_xyxy", old["xyxy"]))
        if item.get("action") == "replace_geometry":
            key = json.dumps(item["replaces"], sort_keys=True)
            if key in replacements:
                continue
            replacements.add(key)
        if not any(old.get("replaces") == item.get("replaces") and _same_control(old, item) for old in filtered):
            filtered.append(item)
    proposals = _proposal_ids(filtered)
    warnings.append("Suggestions can miss visible controls or misidentify icons. Review the entire image before approving.")
    return {"schema": "studio_annotation_proposals_v1", "image_sha256": image_hash,
            "proposals": proposals, "warnings": warnings, "accessibility_diagnostics": diagnostics,
            "geometry_diagnostics": geometry_diagnostics, "geometry_policy_version": POLICY_VERSION,
            "template_classes": template_counts, "model_status": model_status,
            "model_classes": list(reference().CONTROL_NAMES) + ["playhead"]
            if legacy_used and _RUNTIMES is not None else [],
            "temporal_smoothing": "not applied to independent still captures"}
