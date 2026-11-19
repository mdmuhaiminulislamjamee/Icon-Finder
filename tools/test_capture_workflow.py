"""Capture orchestration regressions. All device operations are mocked."""
import asyncio
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from tools.dashboard import server
from tools import sync_from_device as sync


class CaptureWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"CAPTURE_TIMEOUT_SECONDS": "180"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.pause = patch.object(server.stream_mgr, "pause", new_callable=AsyncMock)
        self.resume = patch.object(server.stream_mgr, "resume", new_callable=AsyncMock)
        self.pause.start()
        self.resumer = self.resume.start()
        self.addCleanup(self.pause.stop)
        self.addCleanup(self.resume.stop)
        self.lock = patch.object(server, "CAPTURE_LOCK", asyncio.Lock())
        self.lock.start()
        self.addCleanup(self.lock.stop)

    def invoke(self, log="[plan] finish\nrun=1 fail=0", returncode=0, fresh=None, file_log="", runner_error=None):
        snapshots = [{"udid": "test-ipad", "record_ids": {"old"}},
                     {"udid": "test-ipad", "record_ids": {"old", *(fresh if fresh is not None else ["fresh"])}}]
        imported = {"success": True, "synced_records": 1, "message": "Imported 1 new capture(s).", "errors": []}
        def run(args, **kwargs):
            if runner_error:
                raise runner_error
            if file_log:
                Path(args[args.index("--output-log")+1]).write_text(file_log, encoding="utf-8")
            return SimpleNamespace(returncode=returncode, stdout=log, stderr="")
        with patch.object(sync, "snapshot_device_captures", side_effect=snapshots) as inventory, \
                patch.object(sync, "sync_all_from_device", return_value=imported) as importer, \
                patch.object(server.subprocess, "run", side_effect=run) as runner:
            response = asyncio.run(server.api_capture(server.CaptureRequest(udid="test-ipad")))
        return response, runner, importer, inventory

    def test_native_timeout_and_target_are_explicit_and_only_fresh_id_is_imported(self):
        response, runner, importer, _ = self.invoke()
        self.assertTrue(response["success"])
        args = runner.call_args.args[0]
        self.assertEqual(args[args.index("--timeout")+1], "180")
        self.assertEqual(args[args.index("--udid")+1], "test-ipad")
        self.assertEqual(runner.call_args.kwargs["timeout"], 195)
        self.assertEqual(runner.call_args.kwargs["env"]["PYTHONIOENCODING"], "utf-8")
        importer.assert_called_once_with(udid="test-ipad", record_ids={"fresh"}, require_sidecar=True)
        self.resumer.assert_awaited_once()

    def test_failed_xctest_cannot_succeed_with_zero_process_exit(self):
        response, _, importer, _ = self.invoke("[case] FAIL broken capture\n[plan] finish\nrun=1 fail=1")
        self.assertFalse(response["success"])
        self.assertFalse(response["capture_success"])
        importer.assert_not_called()

    def test_missing_finish_or_zero_tests_is_not_success(self):
        for log in ("connected", "[plan] finish\nrun=0 fail=0", "run=1 fail=0"):
            with self.subTest(log=log):
                response, _, importer, _ = self.invoke(log)
                self.assertFalse(response["success"])
                importer.assert_not_called()

    def test_output_log_is_checked_even_if_console_claims_success(self):
        response, _, importer, _ = self.invoke(file_log="[case] FAIL missing snapshot")
        self.assertFalse(response["success"])
        importer.assert_not_called()

    def test_ansi_log_markers_are_supported(self):
        response, _, _, _ = self.invoke("\x1b[32m[plan] finish\x1b[0m\nrun=1 fail=0")
        self.assertTrue(response["success"])

    def test_process_error_wins_over_passed_log(self):
        response, _, importer, _ = self.invoke(returncode=1)
        self.assertFalse(response["success"])
        importer.assert_not_called()

    def test_no_or_multiple_new_ids_do_not_import_the_old_device_collection(self):
        for fresh in ([], ["fresh", "old_pending"]):
            with self.subTest(fresh=fresh):
                response, _, importer, _ = self.invoke(fresh=fresh)
                self.assertFalse(response["success"])
                self.assertTrue(response["capture_success"])
                importer.assert_not_called()

    def test_capture_timeout_resumes_preview_and_does_not_import(self):
        response, _, importer, _ = self.invoke(runner_error=subprocess.TimeoutExpired("capture", 195))
        self.assertFalse(response["success"])
        self.assertIn("195", response["message"])
        importer.assert_not_called()
        self.resumer.assert_awaited_once()

    def test_timeout_configuration_is_used_and_invalid_values_stop_before_capture(self):
        with patch.dict(os.environ, {"CAPTURE_TIMEOUT_SECONDS": "300"}):
            _, runner, _, _ = self.invoke()
        self.assertEqual(runner.call_args.kwargs["timeout"], 315)
        for value in ("0", "-1", "invalid"):
            with patch.dict(os.environ, {"CAPTURE_TIMEOUT_SECONDS": value}), patch.object(sync, "snapshot_device_captures") as inventory:
                with self.assertRaises(HTTPException):
                    asyncio.run(server.api_capture(server.CaptureRequest()))
                inventory.assert_not_called()

    def test_inventory_failure_does_not_start_a_capture(self):
        with patch.object(sync, "snapshot_device_captures", side_effect=OSError("device unavailable")), \
                patch.object(server.subprocess, "run") as runner:
            response = asyncio.run(server.api_capture(server.CaptureRequest()))
        self.assertFalse(response["success"])
        self.assertEqual(response["stage"], "prepare")
        runner.assert_not_called()

    def test_concurrent_capture_is_rejected(self):
        async def run():
            async with server.CAPTURE_LOCK:
                with self.assertRaises(HTTPException) as raised:
                    await server.api_capture(server.CaptureRequest())
                self.assertEqual(raised.exception.status_code, 409)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
