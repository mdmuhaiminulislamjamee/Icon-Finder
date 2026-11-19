"""
Packages a collected dataset folder into a clean, audited handover ZIP with SHA256 checksums
ready to deliver to the ML training engineer.
"""

import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def package_handover(collected_path: str, output_zip: str = None):
    collected = Path(collected_path)
    if not collected.exists():
        print(f"Error: {collected} does not exist.")
        return

    images_dir = collected / "images"
    annotations_file = collected / "annotations.jsonl"
    curation_dir = collected / "curation"
    overrides_file = curation_dir / "annotation_overrides.json"
    audit_file = curation_dir / "annotation_audit.jsonl"

    if not annotations_file.exists():
        print(f"Error: No annotations.jsonl found in {collected}")
        return

    # Count statistics
    total_records = 0
    class_counts = {}
    with open(annotations_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                total_records += 1
                for k, v in rec.get("raw_annotation_summary", {}).items():
                    class_counts[k] = class_counts.get(k, 0) + v

    curated_count = 0
    if overrides_file.exists():
        try:
            with open(overrides_file, "r", encoding="utf-8") as f:
                curated_dict = json.load(f)
                curated_count = len(curated_dict)
        except Exception:
            pass

    images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))

    print("\n" + "=" * 50)
    print("DATASET HANDOVER SUMMARY")
    print("=" * 50)
    print(f"Source Folder:     {collected}")
    print(f"Total Screenshots: {len(images)}")
    print(f"Total Records:     {total_records}")
    print(f"Curated Overrides: {curated_count}")
    print(f"Class Breakdown:   {json.dumps(class_counts, indent=2)}")
    print("=" * 50 + "\n")

    # Generate SHA256SUMS.txt inside the collected folder
    checksum_lines = []
    files_to_hash = []
    if annotations_file.exists():
        files_to_hash.append(annotations_file)
    if overrides_file.exists():
        files_to_hash.append(overrides_file)
    if audit_file.exists():
        files_to_hash.append(audit_file)
    # Diagnostics and unaccepted proposals travel with the handover, while
    # remaining separate from raw records and effective curated annotations.
    files_to_hash.extend(sorted((collected / "accessibility").glob("*.json")))
    files_to_hash.extend(sorted((curation_dir / "proposals").glob("*.json")))
    for img in sorted(images):
        files_to_hash.append(img)

    for fpath in files_to_hash:
        rel_path = fpath.relative_to(collected)
        sha = compute_sha256(fpath)
        checksum_lines.append(f"{sha}  {rel_path.as_posix()}")

    checksum_file = collected / "SHA256SUMS.txt"
    checksum_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"Generated checksums: {checksum_file}")

    # Build ZIP package
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = collected.parts
    platform_name = parts[-4] if len(parts) >= 4 else "platform"
    device_name = parts[-3] if len(parts) >= 4 else "device"
    format_name = parts[-2] if len(parts) >= 4 else "format"

    default_zip_name = f"handover_{platform_name}_{device_name}_{format_name}_{timestamp}.zip"
    zip_path = Path(output_zip) if output_zip else collected.parent / default_zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fpath in files_to_hash + [checksum_file]:
            arcname = Path(platform_name) / device_name / format_name / "collected" / fpath.relative_to(collected)
            z.write(fpath, arcname=str(arcname))

    print(f"Successfully created handover package: {zip_path.resolve()}\n")


if __name__ == "__main__":
    target = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/platform_specialist_yolo/youtube/tablet/short_video/collected"
    )
    package_handover(target)
