"""Regression checks for incremental import; uses temporary files and a fake iPad."""
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from tools import sync_from_device as sync


class FakeDevice:
    def __init__(self):
        self.files = {}
        self.reads = Counter()
        self.fail_reads = set()
        self.wrong_sizes = set()

    def add_capture(self, record_id, with_image=True):
        collection = str(sync.REMOTE_DATA_ROOT / "youtube/tablet/short_video/collected")
        annotations = collection + "/annotations.jsonl"
        record = {"record_id": record_id, "raw_annotations": [], "context": {}}
        self.files[annotations] = self.files.get(annotations, b"") + (json.dumps(record) + "\n").encode()
        image = collection + "/images/" + record_id + ".png"
        if with_image:
            self.files[image] = b"screenshot-" + record_id.encode()
        return image

    async def exists(self, path):
        return path in self.files or any(name.startswith(path + "/") for name in self.files)

    async def walk(self, root):
        folders = {}
        for path in self.files:
            if path.startswith(root + "/"):
                item = PurePosixPath(path)
                folders.setdefault(str(item.parent), []).append(item.name)
        for folder, names in sorted(folders.items()):
            yield folder, [], names

    async def stat(self, path):
        return {"st_ifmt": "S_IFREG", "st_size": len(self.files[path]) + (path in self.wrong_sizes)}

    async def get_file_contents(self, path):
        self.reads[path] += 1
        if path in self.fail_reads:
            raise OSError("Simulated USB transfer failure")
        await asyncio.sleep(0)
        return self.files[path]


class IncrementalImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="capture-import-test-")
        root = Path(self.temp.name)
        self.patches = patch.multiple(
            sync, LOCAL_DATA_ROOT=root / "dataset",
            STATE_PATH=root / "history.json", LOCK_PATH=root / "history.lock",
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.addCleanup(self.temp.cleanup)
        self.device = FakeDevice()
        self.collection = sync.LOCAL_DATA_ROOT / "youtube/tablet/short_video/collected"

    def run_import(self):
        return asyncio.run(sync._sync_from_afc(self.device))

    def record_ids(self):
        path = self.collection / "annotations.jsonl"
        if not path.exists():
            return []
        return [json.loads(line)["record_id"] for line in path.read_text().splitlines() if line.strip()]

    def test_repeat_import_only_transfers_new_images(self):
        first = self.device.add_capture("capture_a")
        self.assertEqual(self.run_import()["synced_records"], 1)
        repeat = self.run_import()
        self.assertEqual(repeat["synced_records"], 0)
        self.assertEqual(repeat["skipped_records"], 1)
        second = self.device.add_capture("capture_b")
        self.assertEqual(self.run_import()["synced_records"], 1)
        self.assertEqual(self.device.reads[first], 1)
        self.assertEqual(self.device.reads[second], 1)
        self.assertEqual(self.record_ids(), ["capture_a", "capture_b"])

    def test_device_inventory_reads_ids_without_importing_or_downloading_images(self):
        image = self.device.add_capture("capture_a")
        ids = asyncio.run(sync._capture_ids_from_afc(self.device))
        self.assertEqual(ids, {"capture_a"})
        self.assertEqual(self.device.reads[image], 0)
        self.assertFalse(sync.STATE_PATH.exists())
        self.assertFalse(self.collection.exists())

    def test_scoped_import_does_not_import_older_unseen_captures(self):
        old_image = self.device.add_capture("old_capture")
        new_image = self.device.add_capture("fresh_capture")
        result = asyncio.run(sync._sync_from_afc(self.device, record_ids={"fresh_capture"}))
        self.assertEqual(result["synced_records"], 1)
        self.assertEqual(result["filtered_records"], 1)
        self.assertEqual(self.record_ids(), ["fresh_capture"])
        self.assertEqual(self.device.reads[old_image], 0)
        self.assertEqual(self.device.reads[new_image], 1)

    def test_new_capture_strict_import_requires_accessibility_snapshot(self):
        self.device.add_capture("capture_a")
        result = asyncio.run(sync._sync_from_afc(self.device, record_ids={"capture_a"}, require_sidecar=True))
        self.assertFalse(result["success"])
        self.assertEqual(self.record_ids(), [])

    def strict_capture(self):
        from PIL import Image
        image = self.device.add_capture("strict_capture")
        png = io.BytesIO()
        Image.new("RGB", (20, 30)).save(png, format="PNG")
        self.device.files[image] = png.getvalue()
        digest = hashlib.sha256(png.getvalue()).hexdigest()
        record_path = str(PurePosixPath(image).parent.parent / "annotations.jsonl")
        record = json.loads(self.device.files[record_path])
        record.update(image_sha256=digest, context={"image_size": [20, 30]})
        self.device.files[record_path] = (json.dumps(record) + "\n").encode()
        sidecar = str(PurePosixPath(image).parent.parent / "accessibility/strict_capture.json")
        snapshot = {"schema": "media_collector_accessibility_v1", "record_id": "strict_capture",
                    "image_sha256": digest, "image_size": [20, 30], "elements": [], "debug_tree": "Application"}
        self.device.files[sidecar] = json.dumps(snapshot).encode()
        return sidecar, snapshot

    def test_strict_capture_checks_image_and_snapshot_before_import(self):
        self.strict_capture()
        result = asyncio.run(sync._sync_from_afc(self.device, record_ids={"strict_capture"}, require_sidecar=True))
        self.assertTrue(result["success"], result["errors"])
        self.assertEqual(result["synced_records"], 1)
        self.assertEqual(result["synced_accessibility_snapshots"], 1)

    def test_strict_snapshot_mismatch_is_retryable_and_not_committed(self):
        sidecar, snapshot = self.strict_capture()
        self.device.files[sidecar] = json.dumps({**snapshot, "image_sha256": "wrong"}).encode()
        result = asyncio.run(sync._sync_from_afc(self.device, require_sidecar=True))
        self.assertFalse(result["success"])
        self.assertEqual(self.record_ids(), [])
        self.device.files[sidecar] = json.dumps(snapshot).encode()
        result = asyncio.run(sync._sync_from_afc(self.device, require_sidecar=True))
        self.assertEqual(result["synced_records"], 1)

    def test_history_survives_clearing_local_dataset(self):
        self.device.add_capture("capture_a")
        self.run_import()
        self.assertTrue(sync.LOCAL_DATA_ROOT.resolve().is_relative_to(Path(self.temp.name).resolve()))
        shutil.rmtree(sync.LOCAL_DATA_ROOT)
        self.assertEqual(self.run_import()["synced_records"], 0)
        self.assertFalse(sync.LOCAL_DATA_ROOT.exists())
        self.device.add_capture("capture_b")
        self.assertEqual(self.run_import()["synced_records"], 1)
        self.assertEqual(self.record_ids(), ["capture_b"])

    def test_missing_image_remains_retryable(self):
        image = self.device.add_capture("capture_a", with_image=False)
        first = self.run_import()
        self.assertFalse(first["success"])
        self.assertEqual(first["synced_records"], 0)
        self.assertEqual(self.record_ids(), [])
        self.device.files[image] = b"complete-image"
        self.assertEqual(self.run_import()["synced_records"], 1)

    def test_full_accessibility_snapshot_is_imported_once_with_its_capture(self):
        image = self.device.add_capture("capture_a")
        snapshot_path = str(PurePosixPath(image).parent.parent / "accessibility/capture_a.json")
        snapshot = {"record_id": "capture_a", "elements": [{"label": "unmatched title"}]}
        self.device.files[snapshot_path] = json.dumps(snapshot).encode()
        result = self.run_import()
        self.assertEqual(result["synced_accessibility_snapshots"], 1)
        self.assertEqual(json.loads((self.collection / "accessibility/capture_a.json").read_text()), snapshot)
        self.assertEqual(self.run_import()["synced_accessibility_snapshots"], 0)
        self.assertEqual(self.device.reads[snapshot_path], 1)

    def test_failed_snapshot_transfer_remains_retryable(self):
        image = self.device.add_capture("capture_a")
        snapshot_path = str(PurePosixPath(image).parent.parent / "accessibility/capture_a.json")
        self.device.files[snapshot_path] = b'{"record_id":"wrong_capture"}'
        self.assertFalse(self.run_import()["success"])
        self.assertEqual(self.record_ids(), [])
        self.device.files[snapshot_path] = b'{"record_id":"capture_a","elements":[]}'
        self.assertEqual(self.run_import()["synced_records"], 1)

    def test_failed_or_truncated_transfer_is_not_committed(self):
        image = self.device.add_capture("capture_a")
        self.device.fail_reads.add(image)
        self.assertFalse(self.run_import()["success"])
        self.device.fail_reads.clear()
        self.device.wrong_sizes.add(image)
        self.assertFalse(self.run_import()["success"])
        self.assertEqual(self.record_ids(), [])
        self.device.wrong_sizes.clear()
        self.assertEqual(self.run_import()["synced_records"], 1)
        self.assertEqual(self.record_ids(), ["capture_a"])

    def test_existing_annotations_and_curation_are_preserved(self):
        self.collection.mkdir(parents=True)
        original = '{"record_id":"capture_a","raw_annotations":[{"class":"play"}]}\n'
        (self.collection / "annotations.jsonl").write_text(original)
        curation = self.collection / "curation/annotation_overrides.json"
        curation.parent.mkdir()
        curation.write_text('{"capture_a":{"curated_annotations":[]}}')
        self.device.add_capture("capture_a")
        self.device.add_capture("capture_b")
        self.assertEqual(self.run_import()["synced_records"], 1)
        self.assertTrue((self.collection / "annotations.jsonl").read_text().startswith(original))
        self.assertEqual(curation.read_text(), '{"capture_a":{"curated_annotations":[]}}')

    def test_deleted_frame_does_not_return_via_api_or_sync(self):
        from tools.dashboard import server
        self.device.add_capture("capture_a")
        self.run_import()
        with patch.object(server, "DATA_ROOT", sync.LOCAL_DATA_ROOT):
            server.api_delete_record("youtube", "tablet", "short_video", "capture_a")
        self.assertEqual(self.run_import()["synced_records"], 0)
        self.assertEqual(self.record_ids(), [])
        self.assertFalse((self.collection / "images/capture_a.png").exists())

    def test_cleared_device_records_remain_ignored(self):
        sync.remember_discarded_records(["capture_a"])
        self.device.add_capture("capture_a")
        self.device.add_capture("capture_b")
        self.assertEqual(self.run_import()["synced_records"], 1)
        self.assertEqual(self.record_ids(), ["capture_b"])

    def test_corrupt_history_fails_closed(self):
        sync.STATE_PATH.write_text("not json")
        self.device.add_capture("capture_a")
        with self.assertRaisesRegex(RuntimeError, "refusing to reimport"):
            self.run_import()
        self.assertEqual(self.record_ids(), [])

    def test_unsafe_record_name_is_rejected(self):
        self.device.add_capture("../escape")
        result = self.run_import()
        self.assertFalse(result["success"])
        self.assertEqual(self.record_ids(), [])

    def test_concurrent_imports_do_not_duplicate_records(self):
        self.device.add_capture("capture_a")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.run_import(), range(2)))
        self.assertEqual(sum(result["synced_records"] for result in results), 1)
        self.assertEqual(self.record_ids(), ["capture_a"])

    def test_capture_endpoint_uses_current_python_and_reports_import_result(self):
        from tools.dashboard import server
        result = {"success": True, "synced_records": 1, "message": "Imported 1 new capture(s).", "errors": []}
        snapshots = [{"udid": "test-ipad", "record_ids": {"old"}}, {"udid": "test-ipad", "record_ids": {"old", "fresh"}}]
        with patch.object(server.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="[plan] finish\nrun=1 fail=0", stderr="")) as runner, patch.object(sync, "snapshot_device_captures", side_effect=snapshots):
            with patch.object(sync, "sync_all_from_device", return_value=result):
                with patch.object(server.stream_mgr, "pause", new_callable=AsyncMock), patch.object(server.stream_mgr, "resume", new_callable=AsyncMock):
                    response = asyncio.run(server.api_capture(server.CaptureRequest()))
        self.assertEqual(runner.call_args.args[0][:3], [server.sys.executable, "-m", "pymobiledevice3"])
        self.assertIn("Imported 1 new capture", response["message"])
        self.assertEqual(response["synced"]["synced_records"], 1)


if __name__ == "__main__":
    unittest.main()
