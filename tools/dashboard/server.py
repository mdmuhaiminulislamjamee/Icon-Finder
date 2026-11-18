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
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

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


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def get_connected_device():
    """Queries pymobiledevice3 for connected USB devices."""
    try:
        res = subprocess.run(
            ["pymobiledevice3", "usbmux", "list"],
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
            ["pymobiledevice3", "developer", "dvt", "screenshot", str(temp_img)],
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


@app.post("/api/capture")
async def api_capture(req: CaptureRequest):
    """Triggers the XCTest runner on the connected device with stream bandwidth coordination."""
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

    cmd = [
        "pymobiledevice3",
        "developer",
        "dvt",
        "xcuitest",
        "com.grozziie.media-collector.capture.xctrunner",
        "--target-bundle-id",
        "com.grozziie.media-collector",
        "--env",
        f"PLATFORM_NAME_INPUT={req.platform}",
        "--env",
        f"CONTENT_FORMAT={req.content_format}",
        "--env",
        f"CAPTURE_STATE={req.capture_state}",
        "--env",
        f"TARGET_BUNDLE={target_bundle}",
    ]

    await stream_mgr.pause()
    try:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=25,
            ),
        )
        log_out = res.stdout + "\n" + res.stderr
        success = res.returncode == 0

        # Auto-sync newly captured dataset files from the iPad to PC
        synced_info = {}
        if success:
            try:
                import sys
                sys.path.insert(0, str(PROJECT_ROOT))
                from tools.sync_from_device import sync_all_from_device
                synced_info = await loop.run_in_executor(None, sync_all_from_device)
            except Exception as se:
                log_out += f"\nSync notice: {se}"

        return {
            "success": success,
            "return_code": res.returncode,
            "log": log_out,
            "synced": synced_info,
            "message": "Capture completed & synced from iPad!" if success else "Capture failed. Check log.",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Capture execution timed out after 25 seconds."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await stream_mgr.resume()


@app.post("/api/device/sync")
async def api_device_sync():
    """Manually triggers dataset synchronization from the connected iPad."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from tools.sync_from_device import sync_all_from_device
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, sync_all_from_device)
    return {"success": True, **result}


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

                                img_file = images_dir / f"{rid}.png"
                                img_exists = img_file.exists()

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
    img_path = (
        DATA_ROOT
        / platform
        / device
        / content_format
        / "collected"
        / "images"
        / f"{record_id}.png"
    )
    if not img_path.exists():
        # Try jpg
        img_path = (
            DATA_ROOT
            / platform
            / device
            / content_format
            / "collected"
            / "images"
            / f"{record_id}.jpg"
        )
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img_path)


class CuratePayload(BaseModel):
    annotations: List[dict]
    status: str = "approved"
    notes: str = ""


@app.post("/api/curate/{platform}/{device}/{content_format}/{record_id}")
def api_curate(
    platform: str,
    device: str,
    content_format: str,
    record_id: str,
    payload: CuratePayload,
):
    """Saves updated bounding boxes to curation/annotation_overrides.json and logs to audit."""
    coll_dir = (
        DATA_ROOT / platform / device / content_format / "collected"
    )
    curation_dir = coll_dir / "curation"
    curation_dir.mkdir(parents=True, exist_ok=True)

    overrides_file = curation_dir / "annotation_overrides.json"
    audit_file = curation_dir / "annotation_audit.jsonl"

    overrides = {}
    if overrides_file.exists():
        try:
            with open(overrides_file, "r", encoding="utf-8") as f:
                overrides = json.load(f)
        except Exception:
            overrides = {}

    overrides[record_id] = {
        "status": payload.status,
        "notes": payload.notes,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "curated_annotations": payload.annotations,
    }

    with open(overrides_file, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)

    # Append audit log
    audit_entry = {
        "record_id": record_id,
        "action": "curation_save",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "box_count": len(payload.annotations),
        "status": payload.status,
        "notes": payload.notes,
    }
    with open(audit_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_entry) + "\n")

    return {"success": True, "record_id": record_id, "saved_boxes": len(payload.annotations)}


@app.delete("/api/record/{platform}/{device}/{content_format}/{record_id}")
def api_delete_record(
    platform: str, device: str, content_format: str, record_id: str
):
    """Deletes the screenshot and removes its line from annotations.jsonl and overrides."""
    coll_dir = (
        DATA_ROOT / platform / device / content_format / "collected"
    )
    images_dir = coll_dir / "images"
    annotations_file = coll_dir / "annotations.jsonl"
    overrides_file = coll_dir / "curation" / "annotation_overrides.json"

    # 1. Delete image
    for ext in [".png", ".jpg"]:
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

    return {"success": True, "deleted_record_id": record_id}


class PackageRequest(BaseModel):
    platform: str = "youtube"
    device: str = "tablet"
    content_format: str = "short_video"


@app.post("/api/package")
def api_package(req: PackageRequest):
    """Packages the dataset folder into a clean ZIP with SHA256SUMS.txt."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
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
    print("🚀 MediaControl Collector Studio starting at http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
