"""
Sync dataset files from the iOS device (MediaCollector app sandbox) to local data directory.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_ROOT = PROJECT_ROOT / "data" / "platform_specialist_yolo"
APP_BUNDLE_ID = "com.grozziie.media-collector"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def list_device_files():
    """Queries core-device to list all files in the app container."""
    try:
        res = subprocess.run(
            [
                "pymobiledevice3",
                "developer",
                "core-device",
                "list-directory",
                "appDataContainer",
                "/",
                "--identifier",
                APP_BUNDLE_ID,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
        )
        if res.returncode == 0:
            return json.loads(res.stdout)
    except Exception as e:
        logging.error("Failed to list device files: %s", e)
    return []


def download_device_file(remote_path: str, local_path: Path):
    """Downloads a file from the device container to local path."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        res = subprocess.run(
            [
                "pymobiledevice3",
                "developer",
                "core-device",
                "read-file",
                "appDataContainer",
                remote_path,
                "--identifier",
                APP_BUNDLE_ID,
                "--output",
                str(local_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        return res.returncode == 0 and local_path.exists() and local_path.stat().st_size > 0
    except Exception as e:
        logging.error("Failed to download %s: %s", remote_path, e)
        return False


def merge_jsonl(local_path: Path, remote_jsonl_content: str):
    """Merges remote JSONL lines into local file without duplicate record_ids."""
    existing_records = {}
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rid = obj.get("record_id")
                    if rid:
                        existing_records[rid] = line
                except Exception:
                    pass

    new_count = 0
    for line in remote_jsonl_content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            rid = obj.get("record_id")
            if rid and rid not in existing_records:
                existing_records[rid] = line
                new_count += 1
        except Exception:
            pass

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
        for line in existing_records.values():
            f.write(line + "\n")

    return new_count


def sync_all_from_device():
    """Full sync pipeline from device to PC."""
    logging.info("Listing files on connected iPad...")
    all_files = list_device_files()
    dataset_files = [f for f in all_files if f.startswith("Documents/data/")]

    if not dataset_files:
        logging.info("No dataset files found on iPad.")
        return {"synced_images": 0, "synced_annotations": 0, "total_remote": 0}

    logging.info("Found %d dataset files on iPad. Syncing to %s...", len(dataset_files), LOCAL_DATA_ROOT)

    synced_images = 0
    synced_annotations = 0

    # Separate annotations from images
    jsonl_files = [f for f in dataset_files if f.endswith(".jsonl")]
    image_files = [f for f in dataset_files if f.endswith(".png") or f.endswith(".jpg")]

    # 1. Sync Annotations
    for remote_path in jsonl_files:
        rel_path = remote_path.replace("Documents/data/platform_specialist_yolo/", "")
        local_target = LOCAL_DATA_ROOT / rel_path

        # Read remote content to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            tmp_path = Path(tmp.name)
        try:
            if download_device_file(remote_path, tmp_path):
                content = tmp_path.read_text(encoding="utf-8")
                added = merge_jsonl(local_target, content)
                synced_annotations += added
                logging.info("Merged %s: +%d new records", rel_path, added)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # 2. Sync Images
    for remote_path in image_files:
        rel_path = remote_path.replace("Documents/data/platform_specialist_yolo/", "")
        local_target = LOCAL_DATA_ROOT / rel_path

        if local_target.exists() and local_target.stat().st_size > 0:
            continue  # Already downloaded

        logging.info("Downloading image: %s", local_target.name)
        if download_device_file(remote_path, local_target):
            synced_images += 1

    logging.info("Sync complete! Synced %d new images, %d new annotation records.", synced_images, synced_annotations)
    return {
        "synced_images": synced_images,
        "synced_annotations": synced_annotations,
        "total_remote": len(dataset_files),
    }


if __name__ == "__main__":
    result = sync_all_from_device()
    print(json.dumps(result, indent=2))
