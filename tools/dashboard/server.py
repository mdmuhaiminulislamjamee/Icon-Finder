"""
MediaControl Studio Dashboard Backend
Powered by FastAPI & pymobiledevice3
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from pymobiledevice3.remote.userspace_tunnel import establish_userspace_rsd
from pymobiledevice3.services.dvt.instruments.dvt_provider import DvtProvider
from pymobiledevice3.services.dvt.instruments.screenshot import Screenshot

app = FastAPI(title="MediaControl Collector Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "platform_specialist_yolo"
STATIC_DIR = Path(__file__).resolve().parent / "static"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.annotation_quality import CLASSES, clean_annotations
from tools.annotation_geometry import MIN_TRACK_HEIGHT_RATIO, POLICY_VERSION, TRACK_CLASSES, geometry_policy, inside_track, needs_visual_geometry
from tools.sync_from_device import _atomic_write, _safe_name

CURATION_LOCK = threading.RLock()
CAPTURE_LOCK = asyncio.Lock()


@app.get("/api/health")
def api_health():
    return {"service": "media_collector_dashboard", "status": "ok", "process_id": os.getpid()}


def capture_timeout_seconds():
    try:
        timeout = int(os.environ.get("CAPTURE_TIMEOUT_SECONDS", "180"))
        if timeout <= 0:
            raise ValueError
        return timeout
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="CAPTURE_TIMEOUT_SECONDS must be a positive integer") from exc


def xctest_failure(return_code, log):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log)
    issue = re.search(r"(?m)^\[case\][ \t]+FAIL[ \t]+([^\r\n]+)", text)
    if issue:
        reason = issue[1].strip()
        if "Failed to determine hittability of" in reason and "Activation point invalid" in reason:
            return ("The iPad Capture Runner stopped because a control had no valid tap point. "
                    "Install the updated Capture Runner on the iPad.")
        return f"The Capture Runner reported a test failure. {reason}"
    if return_code != 0:
        return f"The Capture Runner exited with code {return_code}. See View Log for details."
    if re.search(r"\[case\]\s+FAIL\b|\bfail=[1-9]\d*", text):
        return "The Capture Runner reported a failed test. See View Log for details."
    if "[plan] finish" not in text or not re.search(r"\brun=[1-9]\d*\s+fail=0\b", text):
        return "The Capture Runner did not confirm successful completion. See View Log for details."
    return None


def collection_path(platform, device, content_format, record_id=None):
    try:
        for value in (platform, device, content_format, *([record_id] if record_id is not None else [])):
            _safe_name(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = (DATA_ROOT / platform / device / content_format / "collected").resolve()
    if not path.is_relative_to(DATA_ROOT.resolve()):
        raise HTTPException(status_code=400, detail="Invalid collection path")
    return path


def load_record(collected, record_id):
    annotations = collected / "annotations.jsonl"
    if annotations.exists():
        for line in annotations.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("record_id") == record_id:
                    return record
    raise HTTPException(status_code=404, detail="Capture record not found")


def image_path(collected, record_id):
    for suffix in (".png", ".jpg", ".jpeg"):
        path = (collected / "images" / f"{record_id}{suffix}").resolve()
        if path.is_relative_to(collected) and path.is_file():
            return path
    raise HTTPException(status_code=404, detail="Image not found")


def read_json(path, default=None):
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=f"Cannot read {path.name}; existing data was preserved") from exc


def write_json(path, value):
    _atomic_write(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def get_connected_device():
    """Queries pymobiledevice3 for connected USB devices."""
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pymobiledevice3", "usbmux", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
        )
        if res.returncode == 0:
            devices = json.loads(res.stdout)
            if devices and isinstance(devices, list) and len(devices) > 0:
                dev = devices[0]
                return {
                    "connected": True,
                    "name": dev.get("DeviceName", "iOS Device"),
                    "model": dev.get("ProductType", "Unknown"),
                    "device_class": dev.get("DeviceClass", "iPad/iPhone"),
                    "ios_version": dev.get("ProductVersion", "Unknown"),
                    "udid": dev.get("UniqueDeviceID", ""),
                    "connection": dev.get("ConnectionType", "USB"),
                }
    except Exception as e:
        pass
    return {"connected": False, "name": "None", "model": "None"}


# ---------------------------------------------------------
# High-Performance In-Memory iOS Stream Manager
# ---------------------------------------------------------

class StreamManager:
    def __init__(self):
        self.is_active = False
        self.is_paused = False
        self.latest_frame: Optional[bytes] = None
        self.latest_fps: float = 0.0
        self.subscribers = 0
        self.lock = asyncio.Lock()
        self.new_frame_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        async with self.lock:
            if not self.is_active:
                self.is_active = True
                self._task = asyncio.create_task(self._capture_worker())

    async def stop(self):
        async with self.lock:
            self.is_active = False
            if self._task and not self._task.done():
                self._task.cancel()
                self._task = None

    async def pause(self):
        self.is_paused = True

    async def resume(self):
        self.is_paused = False

    async def _capture_worker(self):
        while self.is_active:
            try:
                rsd = await establish_userspace_rsd()
                async with DvtProvider(rsd) as dvt, Screenshot(dvt) as screenshot:
                    while self.is_active:
                        if self.is_paused:
                            await asyncio.sleep(0.4)
                            continue

                        t0 = time.time()
                        try:
                            raw_png = await asyncio.wait_for(screenshot.get_screenshot(), timeout=5.0)
                        except (asyncio.TimeoutError, Exception):
                            await asyncio.sleep(0.5)
                            break

                        if not raw_png:
                            await asyncio.sleep(0.2)
                            continue

                        # Downscale in-memory using PIL for low-latency live preview
                        try:
                            im = Image.open(io.BytesIO(raw_png))
                            max_dim = 1080
                            if im.width > max_dim or im.height > max_dim:
                                ratio = max_dim / max(im.width, im.height)
                                new_size = (int(im.width * ratio), int(im.height * ratio))
                                im = im.resize(new_size, Image.Resampling.BILINEAR)

                            buf = io.BytesIO()
                            im.convert("RGB").save(buf, format="JPEG", quality=75)
                            self.latest_frame = buf.getvalue()

                            dt = time.time() - t0
                            self.latest_fps = round(1.0 / max(dt, 0.01), 1)
                            self.new_frame_event.set()
                            self.new_frame_event.clear()
                        except Exception:
                            pass

                        await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1.0)


stream_mgr = StreamManager()


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/api/device")
def api_device():
    """Returns connected device status."""
    return get_connected_device()


@app.get("/api/annotation-classes")
def api_annotation_classes():
    return {"classes": CLASSES, "geometry_policy_version": POLICY_VERSION,
            "geometry_policies": {cls: geometry_policy(cls) for cls in CLASSES}}


@app.get("/api/device/stream")
async def api_device_stream():
    """Streams live MJPEG frames directly from connected iOS device."""
    await stream_mgr.start()
    stream_mgr.subscribers += 1

    async def frame_generator():
        try:
            while stream_mgr.is_active:
                if stream_mgr.latest_frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + stream_mgr.latest_frame
                        + b"\r\n"
                    )
                try:
                    await asyncio.wait_for(stream_mgr.new_frame_event.wait(), timeout=1.2)
                except asyncio.TimeoutError:
                    pass
        finally:
            stream_mgr.subscribers = max(0, stream_mgr.subscribers - 1)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/device/stream/start")
async def api_stream_start():
    """Starts the background live frame capture worker."""
    await stream_mgr.start()
    return {"status": "started", "active": True}


@app.post("/api/device/stream/stop")
async def api_stream_stop():
    """Stops the background live frame capture worker."""
    await stream_mgr.stop()
    return {"status": "stopped", "active": False}


@app.get("/api/device/stream/status")
def api_stream_status():
    """Returns streaming health, FPS, and status."""
    return {
        "active": stream_mgr.is_active,
        "fps": stream_mgr.latest_fps,
        "subscribers": stream_mgr.subscribers,
        "has_frame": stream_mgr.latest_frame is not None,
    }


@app.get("/api/device/live_frame")
def api_live_frame():
    """Returns the latest screenshot frame as JPEG."""
    if stream_mgr.latest_frame:
        return Response(content=stream_mgr.latest_frame, media_type="image/jpeg")

    temp_dir = tempfile.gettempdir()
    temp_img = Path(temp_dir) / f"live_{int(time.time()*1000)}.png"
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pymobiledevice3", "developer", "dvt", "screenshot", str(temp_img)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=8,
        )
        if temp_img.exists() and temp_img.stat().st_size > 0:
            with open(temp_img, "rb") as f:
                img_data = f.read()
            try:
                temp_img.unlink()
            except Exception:
                pass
            return Response(content=img_data, media_type="image/png")
        raise HTTPException(status_code=500, detail=f"Failed to capture frame: {res.stderr}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Device screenshot timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CaptureRequest(BaseModel):
    platform: str = "youtube"
    device: str = "tablet"
    content_format: str = "short_video"
    capture_state: str = "playing_controls_visible"
    target_bundle: Optional[str] = None
    udid: Optional[str] = None


@app.post("/api/capture")
async def api_capture(req: CaptureRequest):
    """Run one verified XCTest capture and import only its newly published ID."""
    if CAPTURE_LOCK.locked():
        raise HTTPException(status_code=409, detail="A capture is already running")
    async with CAPTURE_LOCK:
        return await _capture_one(req)


async def _capture_one(req):
    from tools.sync_from_device import snapshot_device_captures, sync_all_from_device

    timeout = capture_timeout_seconds()
    target_bundle = req.target_bundle
    if not target_bundle:
        bundle_map = {
            "youtube": "com.google.ios.youtube",
            "tiktok": "com.zhiliaoapp.musically",
            "instagram": "com.burbn.instagram",
            "netflix": "com.netflix.Netflix",
            "spotify": "com.spotify.client",
            "twitch": "tv.twitch",
        }
        target_bundle = bundle_map.get(req.platform.lower(), "com.google.ios.youtube")

    await stream_mgr.pause()
    try:
        loop = asyncio.get_running_loop()
        try:
            before = await loop.run_in_executor(None, partial(snapshot_device_captures, udid=req.udid))
        except Exception as exc:
            return {"success": False, "capture_success": False, "stage": "prepare",
                    "message": "Could not read the iPad capture index. No capture was started.", "log": str(exc)}
        udid = before["udid"]
        with tempfile.TemporaryDirectory(prefix="studio-capture-") as directory:
            log_path = Path(directory) / "capture-log.txt"
            cmd = [
                sys.executable, "-m", "pymobiledevice3", "developer", "dvt", "xcuitest",
                "com.grozziie.media-collector.capture.xctrunner",
                "--target-bundle-id", "com.grozziie.media-collector",
                "--udid", udid, "--timeout", str(timeout), "--output-log", str(log_path),
                "--env", f"PLATFORM_NAME_INPUT={req.platform}",
                "--env", f"CONTENT_FORMAT={req.content_format}",
                "--env", f"CAPTURE_STATE={req.capture_state}",
                "--env", f"TARGET_BUNDLE={target_bundle}",
            ]
            try:
                res = await loop.run_in_executor(None, partial(
                    subprocess.run, cmd, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=timeout + 15,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}))
            finally:
                file_log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        log_out = "\n".join((res.stdout or "", res.stderr or "", file_log))
        failure = xctest_failure(res.returncode, log_out)
        if failure:
            return {"success": False, "capture_success": False, "return_code": res.returncode,
                    "stage": "runner", "log": log_out, "synced": {},
                    "message": failure + " No capture was imported."}
        try:
            after = await loop.run_in_executor(None, partial(snapshot_device_captures, udid=udid))
            fresh_ids = set(after["record_ids"]) - set(before["record_ids"])
            if len(fresh_ids) != 1:
                return {"success": False, "capture_success": True, "return_code": res.returncode,
                        "log": log_out, "synced": {}, "new_record_ids": sorted(fresh_ids),
                        "message": f"The test passed, but {len(fresh_ids)} new records were published instead of one. Automatic import was held for review."}
            synced = await loop.run_in_executor(None, partial(sync_all_from_device, udid=udid, record_ids=fresh_ids, require_sidecar=True))
        except Exception as exc:
            return {"success": False, "capture_success": True, "return_code": res.returncode,
                    "log": log_out + f"\nImport error: {exc}", "synced": {},
                    "message": "Capture completed on the iPad, but its import failed. The saved capture remains on the iPad."}
        return {"success": bool(synced.get("success")), "capture_success": True,
                "return_code": res.returncode, "log": log_out, "synced": synced,
                "new_record_ids": sorted(fresh_ids), "message": "Capture completed. " + synced["message"]}
    except subprocess.TimeoutExpired:
        return {"success": False, "capture_success": False,
                "message": f"Capture did not finish within {timeout + 15} seconds (test limit: {timeout} seconds). Check the iPad before retrying."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await stream_mgr.resume()


@app.post("/api/device/sync")
async def api_device_sync():
    """Import unseen captures, preserving the history of deleted/imported IDs."""
    from tools.sync_from_device import sync_all_from_device
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, sync_all_from_device)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Capture import failed: {exc}") from exc


@app.get("/api/records")
def api_records():
    """Scans all collected dataset folders and returns complete records with metadata."""
    records = []
    if not DATA_ROOT.exists():
        return records

    for platform_dir in DATA_ROOT.iterdir():
        if not platform_dir.is_dir():
            continue
        platform = platform_dir.name

        for device_dir in platform_dir.iterdir():
            if not device_dir.is_dir():
                continue
            device = device_dir.name

            for format_dir in device_dir.iterdir():
                if not format_dir.is_dir():
                    continue
                content_format = format_dir.name

                collected_dir = format_dir / "collected"
                if not collected_dir.exists():
                    continue

                annotations_file = collected_dir / "annotations.jsonl"
                overrides_file = collected_dir / "curation" / "annotation_overrides.json"
                images_dir = collected_dir / "images"

                overrides = {}
                if overrides_file.exists():
                    try:
                        with open(overrides_file, "r", encoding="utf-8") as f:
                            overrides = json.load(f)
                    except Exception:
                        overrides = {}

                if annotations_file.exists():
                    with open(annotations_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                rid = rec.get("record_id", "unknown")
                                is_curated = (
                                    rid in overrides
                                    and "curated_annotations" in overrides[rid]
                                )
                                effective_annotations = (
                                    overrides[rid]["curated_annotations"]
                                    if is_curated
                                    else rec.get("raw_annotations", [])
                                )
                                display_annotations, quality = clean_annotations(
                                    effective_annotations, rec.get("context", {}).get("image_size")
                                )

                                img_exists = any((images_dir / f"{rid}{suffix}").is_file()
                                                 for suffix in (".png", ".jpg", ".jpeg"))

                                records.append({
                                    "record_id": rid,
                                    "platform": platform,
                                    "device": device,
                                    "content_format": content_format,
                                    "capture_state": rec.get("context", {}).get(
                                        "capture_state", "unknown"
                                    ),
                                    "orientation": rec.get("context", {}).get(
                                        "orientation", "portrait"
                                    ),
                                    "image_size": rec.get("context", {}).get(
                                        "image_size", [0, 0]
                                    ),
                                    "source": rec.get("context", {}).get(
                                        "source", ""
                                    ),
                                    "image_url": f"/api/image/{platform}/{device}/{content_format}/{rid}",
                                    "image_exists": img_exists,
                                    "raw_annotations": rec.get(
                                        "raw_annotations", []
                                    ),
                                    "curated_annotations": overrides.get(
                                        rid, {}
                                    ).get("curated_annotations", []),
                                    "effective_annotations": effective_annotations,
                                    "display_annotations": display_annotations,
                                    "annotation_quality": quality,
                                    "proposal_review": overrides.get(rid, {}).get("proposal_review", []),
                                    "has_accessibility_snapshot": bool(rec.get("accessibility_snapshot"))
                                        or (collected_dir / "accessibility" / f"{rid}.json").is_file(),
                                    "is_curated": is_curated,
                                    "curation_status": overrides.get(
                                        rid, {}
                                    ).get("status", "raw"),
                                    "curation_notes": overrides.get(
                                        rid, {}
                                    ).get("notes", ""),
                                    "raw_summary": rec.get(
                                        "raw_annotation_summary", {}
                                    ),
                                })
                            except Exception:
                                continue

    # Sort newest first (by record_id if timestamp formatted)
    records.sort(key=lambda x: x["record_id"], reverse=True)
    return records


@app.get("/api/image/{platform}/{device}/{content_format}/{record_id}")
def api_image(platform: str, device: str, content_format: str, record_id: str):
    """Serves the screenshot image."""
    collected = collection_path(platform, device, content_format, record_id)
    return FileResponse(image_path(collected, record_id))


class CuratePayload(BaseModel):
    annotations: List[dict]
    status: str = "needs_review"
    notes: str = ""
    reviewed_proposals: List[dict] = Field(default_factory=list)


@app.post("/api/curate/{platform}/{device}/{content_format}/{record_id}")
def api_curate(
    platform: str,
    device: str,
    content_format: str,
    record_id: str,
    payload: CuratePayload,
):
    """Saves updated bounding boxes to curation/annotation_overrides.json and logs to audit."""
    coll_dir = collection_path(platform, device, content_format, record_id)
    load_record(coll_dir, record_id)
    if payload.status not in {"approved", "resolved", "needs_review"}:
        raise HTTPException(status_code=422, detail="Choose approved, resolved or needs_review")
    with Image.open(image_path(coll_dir, record_id)) as image:
        image_size = image.size
    annotations, quality = clean_annotations(payload.annotations, image_size)
    if quality["invalid_boxes"] or quality["clipped_boxes"]:
        raise HTTPException(status_code=422, detail="Fix invalid or out-of-image boxes before saving")
    if any(ann.get("review_required") for ann in annotations):
        raise HTTPException(status_code=422, detail="Accept or reject suggestions before saving them as annotations")
    if payload.status in {"approved", "resolved"} and any(needs_visual_geometry(ann) for ann in annotations):
        raise HTTPException(status_code=422, detail="Raw accessibility tap regions need visible geometry review. Refine the boxes or use Edit and Apply Box Changes after checking each target.")
    if payload.status in {"approved", "resolved"} and any(
            item.get("decision") == "pending" for item in payload.reviewed_proposals):
        raise HTTPException(status_code=422, detail="Review remaining suggestions or save as needs_review")
    if payload.status in {"approved", "resolved"}:
        tracks = [ann["xyxy"] for ann in annotations if ann["class"] in TRACK_CLASSES]
        min_height = max(1, round(image_size[1]*MIN_TRACK_HEIGHT_RATIO))
        if any(box[3]-box[1] < min_height for box in tracks):
            raise HTTPException(status_code=422, detail=f"Center each progress box on its visible track and make it at least {min_height} px high for this image.")
        seek_tracks = [ann["xyxy"] for ann in annotations if ann["class"] == "progress_bar"]
        if any(ann["class"] == "playhead" and not any(inside_track(ann["xyxy"], track) for track in seek_tracks)
               for ann in annotations):
            raise HTTPException(status_code=422, detail="Place each visible playhead so its center is inside a progress bar box.")
    curation_dir = coll_dir / "curation"
    curation_dir.mkdir(parents=True, exist_ok=True)

    overrides_file = curation_dir / "annotation_overrides.json"
    audit_file = curation_dir / "annotation_audit.jsonl"

    with CURATION_LOCK:
        overrides = read_json(overrides_file)
        previous = overrides.get(record_id)
        entry = {"status": payload.status, "notes": payload.notes,
                 "updated_at": datetime.now(timezone.utc).isoformat(),
                 "curated_annotations": annotations,
                 "proposal_review": payload.reviewed_proposals}
        # Save exact before/after geometry and review decisions in the audit.
        audit_entry = {"record_id": record_id, "action": "curation_save", "timestamp": entry["updated_at"],
                       "box_count": len(annotations), "status": payload.status, "notes": payload.notes,
                       "previous_override": previous, "saved_override": entry,
                       "reviewed_proposals": payload.reviewed_proposals,
                       "duplicates_removed": quality["duplicates_removed"]}
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        overrides[record_id] = entry
        write_json(overrides_file, overrides)

    return {"success": True, "record_id": record_id, "saved_boxes": len(annotations), "quality": quality}


class SuggestPayload(BaseModel):
    annotations: Optional[List[dict]] = None


@app.post("/api/refine-geometry/{platform}/{device}/{content_format}/{record_id}")
def api_refine_geometry(platform: str, device: str, content_format: str, record_id: str, payload: SuggestPayload):
    from tools.annotation_assistant import refine_geometry

    collected = collection_path(platform, device, content_format, record_id)
    record = load_record(collected, record_id)
    overrides = read_json(collected / "curation" / "annotation_overrides.json")
    existing = payload.annotations if payload.annotations is not None else overrides.get(record_id, {}).get(
        "curated_annotations", record.get("raw_annotations", []))
    existing, _ = clean_annotations(existing, record.get("context", {}).get("image_size"))
    result = refine_geometry(image_path(collected, record_id), existing)
    result.update(record_id=record_id, generated_at=datetime.now(timezone.utc).isoformat())
    write_json(collected / "curation" / "proposals" / f"{record_id}.json", result)
    return result


@app.post("/api/suggest/{platform}/{device}/{content_format}/{record_id}")
def api_suggest(platform: str, device: str, content_format: str, record_id: str, payload: SuggestPayload):
    return _suggest_annotations(platform, device, content_format, record_id, payload, automatic=False)


@app.post("/api/detect/{platform}/{device}/{content_format}/{record_id}")
def api_detect(platform: str, device: str, content_format: str, record_id: str, payload: SuggestPayload):
    """Recover missing boxes automatically from this capture's saved evidence."""
    return _suggest_annotations(platform, device, content_format, record_id, payload, automatic=True)


def _suggest_annotations(platform, device, content_format, record_id, payload, *, automatic):
    from tools.annotation_assistant import suggest

    collected = collection_path(platform, device, content_format, record_id)
    record = load_record(collected, record_id)
    overrides = read_json(collected / "curation" / "annotation_overrides.json")
    existing = payload.annotations if payload.annotations is not None else overrides.get(record_id, {}).get(
        "curated_annotations", record.get("raw_annotations", []))
    existing, _ = clean_annotations(existing, record.get("context", {}).get("image_size"))
    snapshot_file = collected / "accessibility" / f"{record_id}.json"
    snapshot = read_json(snapshot_file) if snapshot_file.exists() else None
    options = {"include_fallbacks": False} if automatic else {}
    result = suggest(image_path(collected, record_id), record, existing, DATA_ROOT, platform, content_format, snapshot, **options)
    result["record_id"] = record_id
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    folder = "detections" if automatic else "proposals"
    write_json(collected / "curation" / folder / f"{record_id}.json", result)
    return result


@app.get("/api/annotation-diagnostics/{platform}/{device}/{content_format}/{record_id}")
def api_annotation_diagnostics(platform: str, device: str, content_format: str, record_id: str):
    collected = collection_path(platform, device, content_format, record_id)
    record = load_record(collected, record_id)
    snapshot_file = collected / "accessibility" / f"{record_id}.json"
    overrides = read_json(collected / "curation" / "annotation_overrides.json")
    effective = overrides.get(record_id, {}).get("curated_annotations", record.get("raw_annotations", []))
    display, quality = clean_annotations(effective, record.get("context", {}).get("image_size"))
    return {"record": record, "override": overrides.get(record_id), "display_annotations": display,
            "quality": quality, "classes": CLASSES,
            "accessibility_snapshot": read_json(snapshot_file) if snapshot_file.exists() else record.get("accessibility_snapshot"),
            "suggestions": read_json(collected / "curation" / "proposals" / f"{record_id}.json")}


@app.delete("/api/record/{platform}/{device}/{content_format}/{record_id}")
def api_delete_record(
    platform: str, device: str, content_format: str, record_id: str
):
    """Delete a frame and remember its ID so it cannot return on the next import."""
    from tools.sync_from_device import remember_deleted_capture

    try:
        remember_deleted_capture(platform, device, content_format, record_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not save capture deletion history: {exc}") from exc
    coll_dir = (
        DATA_ROOT / platform / device / content_format / "collected"
    )
    images_dir = coll_dir / "images"
    annotations_file = coll_dir / "annotations.jsonl"
    overrides_file = coll_dir / "curation" / "annotation_overrides.json"

    # 1. Delete image
    for ext in [".png", ".jpg", ".jpeg"]:
        p = images_dir / f"{record_id}{ext}"
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    # 2. Rewrite annotations.jsonl without this record
    if annotations_file.exists():
        new_lines = []
        with open(annotations_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("record_id") != record_id:
                            new_lines.append(line.strip())
                    except Exception:
                        new_lines.append(line.strip())
        with open(annotations_file, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    # 3. Remove from overrides
    if overrides_file.exists():
        try:
            with open(overrides_file, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            if record_id in overrides:
                del overrides[record_id]
                with open(overrides_file, "w", encoding="utf-8") as f:
                    json.dump(overrides, f, indent=2)
        except Exception:
            pass

    # Capture-bound diagnostics must be removed along with their screenshot.
    for path in (coll_dir / "accessibility" / f"{record_id}.json",
                 coll_dir / "curation" / "proposals" / f"{record_id}.json"):
        if path.is_file():
            path.unlink()

    return {"success": True, "deleted_record_id": record_id}


class PackageRequest(BaseModel):
    platform: str = "youtube"
    device: str = "tablet"
    content_format: str = "short_video"


@app.post("/api/package")
def api_package(req: PackageRequest):
    """Packages the dataset folder into a clean ZIP with SHA256SUMS.txt."""
    from tools.package_handover import package_handover

    target_dir = (
        DATA_ROOT / req.platform / req.device / req.content_format / "collected"
    )
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset folder does not exist")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"handover_{req.platform}_{req.device}_{req.content_format}_{timestamp}.zip"
    zip_out = target_dir.parent / zip_name

    package_handover(str(target_dir), str(zip_out))

    if zip_out.exists():
        return {
            "success": True,
            "zip_name": zip_name,
            "download_url": f"/api/download_package/{req.platform}/{req.device}/{req.content_format}/{zip_name}",
            "file_size": zip_out.stat().st_size,
        }
    raise HTTPException(status_code=500, detail="Failed to create zip package")


@app.get("/api/download_package/{platform}/{device}/{content_format}/{filename}")
def api_download_package(platform: str, device: str, content_format: str, filename: str):
    """Serves the handover zip file."""
    zip_path = (
        DATA_ROOT / platform / device / content_format / filename
    )
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Package file not found")
    return FileResponse(zip_path, filename=filename, media_type="application/zip")


# Mount static frontend files
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("MediaControl Collector Studio starting at http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
