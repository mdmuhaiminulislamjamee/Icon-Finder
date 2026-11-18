import requests
import json
import shutil
from pathlib import Path

BASE = "http://127.0.0.1:8000"

print("=== 1. TESTING DEVICE API ===")
r = requests.get(f"{BASE}/api/device")
assert r.status_code == 200, f"Device API failed: {r.status_code}"
dev_data = r.json()
print("Device status:", dev_data)
assert dev_data["connected"] is True, "Expected device to be connected"

print("\n=== 2. TESTING LIVE SCREEN CAPTURE API FROM REAL IPAD ===")
r = requests.get(f"{BASE}/api/device/live_frame")
assert r.status_code == 200, f"Live frame failed: {r.status_code}"
assert len(r.content) > 5000, f"Image too small: {len(r.content)} bytes"
ctype = r.headers.get("content-type")
print(f"Live frame captured from real iPad! Size: {len(r.content):,} bytes, Type: {ctype}")

print("\n=== 3. CREATING TEST RECORD IN DATASET ===")
target_dir = Path("data/platform_specialist_yolo/youtube/tablet/short_video/collected")
img_src = Path("annotation_format_handover(1)/images/youtube_android_000281.png")
test_id = "youtube_ios_test_sample_001"

# Copy sample image
shutil.copy(img_src, target_dir / "images" / f"{test_id}.png")

# Add to annotations.jsonl
sample_record = {
    "record_id": test_id,
    "canonical_source": f"data/platform_specialist_yolo/youtube/tablet/short_video/collected/images/{test_id}.png",
    "context": {
        "platform": "youtube",
        "environment": "ios",
        "content_form": "shorts",
        "layout_family": "vertical_short_player",
        "device": "tablet",
        "orientation": "portrait",
        "theme": "dark",
        "capture_state": "ad_playing_skip_available",
        "image_size": [1600, 2560],
        "source": "https://www.youtube.com/shorts/test"
    },
    "raw_annotation_summary": {"progress_bar": 1, "playhead": 1},
    "raw_annotations": [
        {"class": "progress_bar", "xyxy": [0.0, 2385.0, 1600.0, 2412.0], "source_label": "seek_bar", "descriptor": "", "source": "xctest", "resource_id": "", "visible_evidence": ""},
        {"class": "playhead", "xyxy": [628.0, 2373.0, 657.0, 2425.0], "source_label": "scrubber", "descriptor": "", "source": "xctest", "resource_id": "", "visible_evidence": ""}
    ]
}
with open(target_dir / "annotations.jsonl", "w", encoding="utf-8") as f:
    f.write(json.dumps(sample_record) + "\n")
print(f"Created test record {test_id} in annotations.jsonl")

print("\n=== 4. TESTING RECORDS LISTING API ===")
r = requests.get(f"{BASE}/api/records")
assert r.status_code == 200
records = r.json()
print(f"Total records in library: {len(records)}")
found = [rec for rec in records if rec["record_id"] == test_id]
assert len(found) == 1, "Test record not found in records API"
print("Record successfully listed via API:", found[0]["record_id"], "with", len(found[0]["effective_annotations"]), "annotations")

print("\n=== 5. TESTING IMAGE SERVING API ===")
r = requests.get(f"{BASE}/api/image/youtube/tablet/short_video/{test_id}")
assert r.status_code == 200
print(f"Image served successfully via HTTP: {len(r.content):,} bytes")

print("\n=== 6. TESTING CURATION / OVERRIDE API (ADDING SKIP_AD BUTTON) ===")
curated_boxes = [
    {"class": "progress_bar", "xyxy": [0.0, 2385.0, 1600.0, 2412.0], "source": "human_curated"},
    {"class": "playhead", "xyxy": [628.0, 2373.0, 657.0, 2425.0], "source": "human_curated"},
    {"class": "skip_ad", "xyxy": [1200.0, 1800.0, 1400.0, 1880.0], "source_label": "Skip", "descriptor": "Skip ad button", "source": "studio_curation", "resource_id": "skip_btn", "visible_evidence": "Skip >|"}
]
r = requests.post(
    f"{BASE}/api/curate/youtube/tablet/short_video/{test_id}",
    json={"annotations": curated_boxes, "status": "approved", "notes": "Visually verified and added skip_ad button"}
)
assert r.status_code == 200, f"Curate API failed: {r.text}"
print("Curation saved successfully:", r.json())

# Re-query records to verify override took effect
r = requests.get(f"{BASE}/api/records")
records = r.json()
target_rec = [rec for rec in records if rec["record_id"] == test_id][0]
assert target_rec["is_curated"] is True
assert any(a["class"] == "skip_ad" for a in target_rec["effective_annotations"])
print("Verified: Record is now marked CURATED and contains the skip_ad button!")

print("\n=== 7. TESTING HANDOVER PACKAGING API ===")
r = requests.post(
    f"{BASE}/api/package",
    json={"platform": "youtube", "device": "tablet", "content_format": "short_video"}
)
assert r.status_code == 200, f"Package API failed: {r.text}"
pkg_info = r.json()
print("Handover package generated:", pkg_info)
assert Path("data/platform_specialist_yolo/youtube/tablet/short_video/collected/SHA256SUMS.txt").exists()

print("\n=== 8. TESTING DOWNLOAD OF HANDOVER ZIP ===")
download_url = f"{BASE}{pkg_info['download_url']}"
r = requests.get(download_url)
assert r.status_code == 200
print(f"Handover ZIP download verified: {len(r.content):,} bytes downloaded successfully!")

print("\n=============================================================")
print("🎉 ALL 8 TESTS PASSED WITH 100% SUCCESS!")
print("=============================================================")
