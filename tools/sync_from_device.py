"""Import only unseen MediaCollector captures, retaining history after local deletion.

The history lives outside collected folders. Keep it when clearing the gallery.
"""
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import hashlib
import io
import logging
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_ROOT = PROJECT_ROOT / "data" / "platform_specialist_yolo"
STATE_PATH = PROJECT_ROOT / "data" / ".capture_import_history.json"
LOCK_PATH = PROJECT_ROOT / "data" / ".capture_import.lock"
REMOTE_DATA_ROOT = PurePosixPath("/Documents/data/platform_specialist_yolo")
APP_BUNDLE_ID = "com.grozziie.media-collector"
_THREAD_LOCK = threading.RLock()
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
logger = logging.getLogger(__name__)


def _safe_name(value):
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"Invalid capture name: {value!r}")
    return value


def _collection_key(value):
    parts = PurePosixPath(value).parts
    if len(parts) != 4 or parts[-1] != "collected":
        raise ValueError("Invalid capture collection")
    for part in parts:
        _safe_name(part)
    return "/".join(parts)


def _local_path(relative):
    parts = PurePosixPath(relative).parts
    for part in parts:
        _safe_name(part)
    root = LOCAL_DATA_ROOT.resolve()
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("Capture path escapes the local dataset")
    return target


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".import-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@dataclass
class ImportHistory:
    collections: dict = field(default_factory=dict)
    discarded_record_ids: set = field(default_factory=set)

    def seen(self, collection, record_id):
        return record_id in self.discarded_record_ids or record_id in self.collections.get(collection, set())

    def remember(self, collection, record_id):
        self.collections.setdefault(collection, set()).add(record_id)


def _load_history():
    history = ImportHistory()
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if data["version"] != 1 or not isinstance(data["collections"], dict):
                raise ValueError("Unsupported history format")
            for collection, record_ids in data["collections"].items():
                if not isinstance(record_ids, list):
                    raise ValueError("Invalid capture history")
                history.collections[_collection_key(collection)] = {_safe_name(rid) for rid in record_ids}
            discarded = data.get("discarded_record_ids", [])
            if not isinstance(discarded, list):
                raise ValueError("Invalid deleted-capture history")
            history.discarded_record_ids = {_safe_name(rid) for rid in discarded}
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Import history cannot be read; refusing to reimport old captures.") from exc

    # Adopt existing annotations on upgrade. An orphan image alone may be a
    # failed transfer, so it must remain retryable until its annotation exists.
    for annotations in LOCAL_DATA_ROOT.glob("*/*/*/collected/annotations.jsonl"):
        collection = _collection_key(annotations.parent.relative_to(LOCAL_DATA_ROOT).as_posix())
        for line in annotations.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                history.remember(collection, _safe_name(json.loads(line)["record_id"]))
            except (ValueError, TypeError, KeyError):
                logger.warning("Ignoring malformed existing annotation in %s", annotations)
    return history


def _save_history(history):
    data = {
        "version": 1,
        "collections": {key: sorted(value) for key, value in sorted(history.collections.items())},
        "discarded_record_ids": sorted(history.discarded_record_ids),
    }
    _atomic_write(STATE_PATH, (json.dumps(data, indent=2) + "\n").encode("utf-8"))


@contextmanager
def _locked_history():
    """Serialize dashboard/CLI imports and deletion markers across processes."""
    with _THREAD_LOCK:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOCK_PATH.open("a+b") as stream:
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"\0")
                stream.flush()
            deadline = time.monotonic() + 30
            while True:
                try:
                    stream.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Another capture import is still running. Try again shortly.")
                    time.sleep(0.1)
            try:
                history = _load_history()
                try:
                    yield history
                finally:
                    _save_history(history)
            finally:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def remember_deleted_capture(platform, device, content_format, record_id):
    collection = _collection_key(f"{platform}/{device}/{content_format}/collected")
    record_id = _safe_name(record_id)
    with _locked_history() as history:
        history.remember(collection, record_id)
        history.discarded_record_ids.add(record_id)


def remember_discarded_records(record_ids):
    """Keep cleared captures ignored, including copies left in old device inboxes."""
    record_ids = {_safe_name(rid) for rid in record_ids}
    with _locked_history() as history:
        history.discarded_record_ids.update(record_ids)


async def _read_regular_file(files, remote_path):
    info = await files.stat(str(remote_path))
    if info.get("st_ifmt") != "S_IFREG":
        raise ValueError(f"Capture is not a regular file: {remote_path}")
    contents = await files.get_file_contents(str(remote_path))
    if len(contents) != int(info["st_size"]):
        raise ValueError(f"Incomplete transfer: {remote_path}")
    return contents


def _append_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, os.SEEK_END)
            needs_newline = stream.read(1) != b"\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        if needs_newline:
            stream.write("\n")
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


async def _remote_files(files):
    result = set()
    if await files.exists(str(REMOTE_DATA_ROOT)):
        async for folder, directories, names in files.walk(str(REMOTE_DATA_ROOT)):
            result.update(PurePosixPath(folder) / name for name in names)
    return result


async def _capture_ids_from_afc(files):
    """Read the published device index without downloading/importing captures."""
    ids = set()
    for path in sorted(await _remote_files(files)):
        if path.name != "annotations.jsonl":
            continue
        _collection_key(path.parent.relative_to(REMOTE_DATA_ROOT).as_posix())
        raw = (await _read_regular_file(files, path)).decode("utf-8-sig")
        for line in raw.splitlines():
            if line.strip():
                ids.add(_safe_name(json.loads(line)["record_id"]))
    return ids


async def _sync_from_afc(files, record_ids=None, require_sidecar=False):
    selected = None if record_ids is None else {_safe_name(rid) for rid in record_ids}
    result = {
        "success": True, "synced_records": 0, "synced_images": 0,
        "synced_annotations": 0, "synced_accessibility_snapshots": 0,
        "skipped_records": 0, "filtered_records": 0, "total_remote": 0, "errors": [],
    }
    with _locked_history() as history:
        remote_files = await _remote_files(files)
        result["total_remote"] = len(remote_files)

        for annotations in sorted(path for path in remote_files if path.name == "annotations.jsonl"):
            try:
                collection = _collection_key(annotations.parent.relative_to(REMOTE_DATA_ROOT).as_posix())
                contents = (await _read_regular_file(files, annotations)).decode("utf-8-sig")
            except Exception as exc:
                result["errors"].append(f"{annotations}: {exc}")
                continue
            for number, line in enumerate(contents.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    record_id = _safe_name(record["record_id"])
                    if selected is not None and record_id not in selected:
                        result["filtered_records"] += 1
                        continue
                    if history.seen(collection, record_id):
                        result["skipped_records"] += 1
                        continue
                    image = next((annotations.parent / "images" / f"{record_id}{extension}"
                                  for extension in (".png", ".jpg", ".jpeg")
                                  if annotations.parent / "images" / f"{record_id}{extension}" in remote_files), None)
                    if image is None:
                        raise ValueError(f"Screenshot is missing for {record_id}; it will be retried.")
                    image_bytes = await _read_regular_file(files, image)
                    if not image_bytes:
                        raise ValueError(f"Screenshot is empty for {record_id}; it will be retried.")
                    snapshot_path = annotations.parent / "accessibility" / f"{record_id}.json"
                    snapshot_bytes = None
                    if require_sidecar and snapshot_path not in remote_files:
                        raise ValueError("New capture has no accessibility sidecar; verify the updated runner is installed.")
                    if snapshot_path in remote_files:
                        snapshot_bytes = await _read_regular_file(files, snapshot_path)
                        snapshot = json.loads(snapshot_bytes.decode("utf-8-sig"))
                        if not isinstance(snapshot, dict) or snapshot.get("record_id") != record_id:
                            raise ValueError("Accessibility snapshot does not identify this capture; it will be retried.")
                        if require_sidecar:
                            from PIL import Image
                            digest = hashlib.sha256(image_bytes).hexdigest()
                            with Image.open(io.BytesIO(image_bytes)) as image_file:
                                dimensions = list(image_file.size)
                                image_file.verify()
                            if (snapshot.get("schema") != "media_collector_accessibility_v1"
                                    or snapshot.get("image_sha256") != digest
                                    or record.get("image_sha256") != digest
                                    or snapshot.get("image_size") != dimensions
                                    or record.get("context", {}).get("image_size") != dimensions
                                    or not isinstance(snapshot.get("elements"), list)
                                    or not isinstance(snapshot.get("debug_tree"), str)):
                                raise ValueError("New capture image, raw record and complete accessibility snapshot do not agree.")
                        _atomic_write(_local_path(snapshot_path.relative_to(REMOTE_DATA_ROOT).as_posix()), snapshot_bytes)
                    relative_image = image.relative_to(REMOTE_DATA_ROOT).as_posix()
                    _atomic_write(_local_path(relative_image), image_bytes)
                    _append_record(_local_path(f"{collection}/annotations.jsonl"), record)
                    history.remember(collection, record_id)
                    result["synced_records"] += 1
                    result["synced_images"] += 1
                    result["synced_annotations"] += 1
                    result["synced_accessibility_snapshots"] += int(snapshot_bytes is not None)
                except Exception as exc:
                    result["errors"].append(f"{annotations.name}, line {number}: {exc}")

    result["success"] = not result["errors"]
    count = result["synced_records"]
    result["message"] = f"Imported {count} new capture(s)." if count else "No new captures to import."
    if result["errors"]:
        result["message"] += f" {len(result['errors'])} capture import error(s); failed captures remain retryable."
    logger.info("%s Skipped %d previously imported/deleted records.", result["message"], result["skipped_records"])
    return result


async def _sync_connected_device(udid=None, record_ids=None, require_sidecar=False):
    from pymobiledevice3.lockdown import create_using_usbmux
    from pymobiledevice3.services.house_arrest import HouseArrestService

    lockdown = await create_using_usbmux(serial=udid)
    try:
        async with await HouseArrestService.create(lockdown, APP_BUNDLE_ID) as files:
            return await _sync_from_afc(files, record_ids=record_ids, require_sidecar=require_sidecar)
    finally:
        await lockdown.close()


async def _snapshot_connected_device(udid=None):
    from pymobiledevice3.lockdown import create_using_usbmux
    from pymobiledevice3.services.house_arrest import HouseArrestService

    lockdown = await create_using_usbmux(serial=udid)
    try:
        async with await HouseArrestService.create(lockdown, APP_BUNDLE_ID) as files:
            return {"udid": lockdown.udid, "record_ids": await _capture_ids_from_afc(files)}
    finally:
        await lockdown.close()


def snapshot_device_captures(udid=None):
    return asyncio.run(_snapshot_connected_device(udid))


def sync_all_from_device(udid=None, record_ids=None, require_sidecar=False):
    """Import unseen captures; don't restore captures just because they were deleted locally."""
    return asyncio.run(_sync_connected_device(udid, record_ids=record_ids, require_sidecar=require_sidecar))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(json.dumps(sync_all_from_device(), indent=2))
