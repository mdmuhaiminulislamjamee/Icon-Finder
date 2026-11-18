"""
Visualizes raw and curated bounding boxes onto the screenshots for quick visual audit and curation.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Color palette for classes (R, G, B)
CLASS_COLORS = {
    "play": (0, 230, 118),
    "pause": (255, 171, 0),
    "volume_on": (0, 176, 255),
    "volume_muted": (255, 23, 68),
    "fullscreen": (213, 0, 249),
    "exit_fullscreen": (156, 39, 176),
    "progress_bar": (255, 87, 34),
    "playhead": (255, 235, 59),
    "skip_ad": (233, 30, 99),
    "like": (76, 175, 80),
    "dislike": (244, 67, 54),
    "comments": (33, 150, 243),
    "share": (0, 188, 212),
    "more_options": (120, 144, 156),
}
DEFAULT_COLOR = (255, 255, 255)


def visualize_collected_folder(collected_path: str, output_dir: str = None):
    collected = Path(collected_path)
    images_dir = collected / "images"
    annotations_file = collected / "annotations.jsonl"
    overrides_file = collected / "curation" / "annotation_overrides.json"

    if not annotations_file.exists():
        print(f"No annotations.jsonl found at {annotations_file}")
        return

    overrides = {}
    if overrides_file.exists():
        try:
            with open(overrides_file, "r", encoding="utf-8") as f:
                overrides = json.load(f)
        except Exception:
            overrides = {}

    out_path = Path(output_dir) if output_dir else collected / "curation" / "visual_audit_previews"
    out_path.mkdir(parents=True, exist_ok=True)

    with open(annotations_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Found {len(lines)} records to visualize.")

    for line in lines:
        record = json.loads(line)
        record_id = record.get("record_id", "unknown")
        
        # Determine effective annotations (curated if present, else raw)
        is_curated = False
        if record_id in overrides and "curated_annotations" in overrides[record_id]:
            annotations = overrides[record_id]["curated_annotations"]
            is_curated = True
        else:
            annotations = record.get("raw_annotations", [])

        # Find image
        img_file = images_dir / f"{record_id}.png"
        if not img_file.exists():
            img_file = Path(record.get("canonical_source", ""))
            if not img_file.exists():
                print(f"Skipping {record_id}: image not found.")
                continue

        img = Image.open(img_file).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw boxes
        for ann in annotations:
            cls = ann.get("class", "unknown")
            xyxy = ann.get("xyxy", [0, 0, 0, 0])
            color = CLASS_COLORS.get(cls, DEFAULT_COLOR)

            draw.rectangle(xyxy, outline=color, width=4)
            label_text = f"{cls}"
            draw.text((xyxy[0] + 4, max(0, xyxy[1] - 18)), label_text, fill=color)

        tag = "CURATED" if is_curated else "RAW"
        out_file = out_path / f"{record_id}_{tag}_annotated.png"
        img.save(out_file)
        print(f"Saved audit preview: {out_file.name} ({len(annotations)} boxes, {tag})")

    print(f"\nAll visual previews saved to: {out_path}")


if __name__ == "__main__":
    import sys
    target = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/platform_specialist_yolo/youtube/tablet/short_video/collected"
    )
    visualize_collected_folder(target)
