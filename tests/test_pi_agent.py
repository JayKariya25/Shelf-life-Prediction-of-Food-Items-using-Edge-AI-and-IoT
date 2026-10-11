"""The Raspberry Pi agent's gas hook and photo upload, against a live server."""

from __future__ import annotations

import io
import sys
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests.support import AppTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import pi_agent  # noqa: E402


class GasCommand(unittest.TestCase):
    def test_reads_a_number(self):
        self.assertEqual(pi_agent.read_gas("echo 185.5"), 185.5)

    def test_uses_the_last_token(self):
        self.assertEqual(pi_agent.read_gas("echo gas reading: 179"), 179.0)

    def test_rejects_text(self):
        self.assertIsNone(pi_agent.read_gas("echo warm"))

    def test_rejects_nan_and_inf(self):
        self.assertIsNone(pi_agent.read_gas("echo nan"))
        self.assertIsNone(pi_agent.read_gas("echo inf"))

    def test_failing_command_returns_none(self):
        self.assertIsNone(pi_agent.read_gas("false"))

    def test_missing_program_returns_none(self):
        self.assertIsNone(pi_agent.read_gas("definitely-not-a-real-program-xyz"))

    def test_timeout_returns_none(self):
        self.assertIsNone(pi_agent.read_gas("sleep 5", timeout=0.2))


class CaptureCommand(unittest.TestCase):
    def test_a_command_that_writes_nothing_is_reported(self):
        self.assertIsNone(pi_agent.capture_frame("true {path}"))

    def test_a_command_that_writes_a_file(self):
        frame = pi_agent.capture_frame("sh -c 'printf jpeg > {path}'")
        self.assertIsNotNone(frame)
        self.assertEqual(frame.read_bytes(), b"jpeg")
        frame.unlink()


class LiveUpload(AppTestCase):
    """Drive the agent's upload code against a real HTTP server."""

    def setUp(self):
        super().setUp()
        from werkzeug.serving import make_server

        self.register_and_login()
        self.token = self.make_device()
        self.item = self.create_item(label="Agent tomatoes")
        self.server = make_server("127.0.0.1", 0, self.app, threaded=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.uploader = pi_agent.Uploader(f"http://127.0.0.1:{self.server.server_port}", self.token)

    def jpeg_file(self) -> Path:
        from PIL import Image

        path = Path(self._tmp.name) / "frame.jpg"
        buffer = io.BytesIO()
        Image.new("RGB", (320, 240), (90, 140, 60)).save(buffer, "JPEG")
        path.write_bytes(buffer.getvalue())
        return path

    def test_multipart_upload_is_accepted_and_attached(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("pillow not installed")
        stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        result = pi_agent.upload_capture(self.uploader, self.jpeg_file(), str(self.item["id"]), stamp)
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["attached_to_item"], self.item["id"])
        self.assertEqual(result["data"]["captured_at"], stamp)

    def test_reading_with_gas_and_item_is_stored(self):
        self.uploader.send({"temperature_c": 28.5, "humidity_pct": 58.0, "gas_ppm": 185.0,
                            "item_id": self.item["id"],
                            "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat()})
        from shelflife.db import query_one

        with self.app.app_context():
            row = query_one("SELECT gas_ppm, item_id FROM readings ORDER BY id DESC LIMIT 1")
        self.assertEqual(row["gas_ppm"], 185.0)
        self.assertEqual(row["item_id"], self.item["id"])

    def test_config_reports_the_two_minute_cadence(self):
        self.assertEqual(self.uploader.check()["data"]["report_interval_seconds"], 120)


if __name__ == "__main__":
    unittest.main()
