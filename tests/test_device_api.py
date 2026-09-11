"""Device ingest API used by the Raspberry Pi agent."""

from __future__ import annotations

import io
import unittest
from datetime import timedelta

from tests.support import AppTestCase

from shelflife.db import query_one, utcnow

JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"


class DeviceAuth(AppTestCase):
    sensor_source = "auto"

    def setUp(self):
        super().setUp()
        self.register_and_login()
        self.token = self.make_device()

    def headers(self, token=None):
        return {"Authorization": f"Bearer {token or self.token}"}

    def test_token_is_returned_once_and_only_hashed_afterwards(self):
        self.assertTrue(self.token.startswith("slp_"))
        devices = self.client.get("/api/v1/devices").get_json()["data"]["devices"]
        self.assertNotIn("token_hash", devices[0])
        self.assertNotIn(self.token, str(devices))

    def test_reading_without_a_token_is_rejected(self):
        response = self.client.post("/api/device/v1/readings", json={"temperature_c": 22})
        self.assertEqual(response.status_code, 401)

    def test_reading_with_a_bogus_token_is_rejected(self):
        response = self.client.post(
            "/api/device/v1/readings", json={"temperature_c": 22},
            headers=self.headers("slp_deadbeef"),
        )
        self.assertEqual(response.status_code, 401)

    def test_config_endpoint_confirms_the_token(self):
        response = self.client.get("/api/device/v1/config", headers=self.headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["data"]["device"]["name"], "Test Pi")

    def test_deleting_a_device_invalidates_its_token(self):
        device_id = self.client.get("/api/v1/devices").get_json()["data"]["devices"][0]["id"]
        self.client.delete(f"/api/v1/devices/{device_id}",
                           headers={"X-CSRF-Token": self.api_token()})
        response = self.client.get("/api/device/v1/config", headers=self.headers())
        self.assertEqual(response.status_code, 401)


class ReadingIngest(AppTestCase):
    sensor_source = "auto"

    def setUp(self):
        super().setUp()
        self.register_and_login()
        self.token = self.make_device()
        self.auth = {"Authorization": f"Bearer {self.token}"}

    def post(self, payload):
        return self.client.post("/api/device/v1/readings", json=payload, headers=self.auth)

    def test_valid_reading_is_stored_as_live(self):
        response = self.post({"temperature_c": 24.8, "humidity_pct": 61.2, "pressure_hpa": 1008.1})
        self.assertEqual(response.status_code, 201)
        with self.app.app_context():
            row = query_one("SELECT * FROM readings ORDER BY id DESC LIMIT 1")
            self.assertEqual(row["source"], "live")
            self.assertAlmostEqual(row["temperature_c"], 24.8)

    def test_live_reading_flips_the_dashboard_out_of_simulated_mode(self):
        self.post({"temperature_c": 24.8, "humidity_pct": 61.2})
        data = self.client.get("/api/v1/dashboard").get_json()["data"]
        self.assertEqual(data["sensor_mode"], "live")
        self.assertEqual(data["reading"]["temperature_c"], 24.8)
        self.assertEqual(data["live_readings_stored"], 1)

    def test_out_of_range_values_are_rejected(self):
        for payload in ({"temperature_c": 500}, {"humidity_pct": -5},
                        {"pressure_hpa": 5}, {"humidity_pct": 140}):
            response = self.post(payload)
            self.assertEqual(response.status_code, 422, payload)
            self.assertEqual(response.get_json()["error"]["code"], "invalid_reading")

    def test_empty_reading_is_rejected(self):
        self.assertEqual(self.post({}).status_code, 422)

    def test_non_numeric_reading_is_rejected(self):
        self.assertEqual(self.post({"temperature_c": "warm"}).status_code, 422)

    def test_nan_is_rejected(self):
        response = self.client.post(
            "/api/device/v1/readings",
            data='{"temperature_c": NaN}',
            content_type="application/json",
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 422)

    def test_future_timestamp_is_rejected(self):
        future = (utcnow() + timedelta(hours=3)).isoformat()
        response = self.post({"temperature_c": 22, "recorded_at": future})
        self.assertEqual(response.status_code, 422)

    def test_partial_reading_is_accepted(self):
        """A device with only the SHT31 wired up must still be able to report."""
        self.assertEqual(self.post({"temperature_c": 22.5, "humidity_pct": 55}).status_code, 201)

    def test_reading_can_be_attached_to_an_item(self):
        item = self.create_item()
        response = self.post({"temperature_c": 22, "item_id": item["id"]})
        self.assertEqual(response.status_code, 201)

    def test_reading_cannot_be_attached_to_someone_elses_item(self):
        response = self.post({"temperature_c": 22, "item_id": 999999})
        self.assertEqual(response.status_code, 404)

    def test_device_last_seen_is_updated(self):
        self.post({"temperature_c": 22, "firmware": "pi-agent/1.0"})
        devices = self.client.get("/api/v1/devices").get_json()["data"]["devices"]
        self.assertIsNotNone(devices[0]["last_seen_at"])
        self.assertTrue(devices[0]["online"])
        self.assertEqual(devices[0]["firmware"], "pi-agent/1.0")


class BatchIngest(AppTestCase):
    sensor_source = "auto"

    def setUp(self):
        super().setUp()
        self.register_and_login()
        self.auth = {"Authorization": f"Bearer {self.make_device()}"}

    def post(self, payload):
        return self.client.post("/api/device/v1/readings/batch", json=payload, headers=self.auth)

    def test_batch_of_valid_rows_is_stored(self):
        rows = [{"temperature_c": 20 + i, "humidity_pct": 60} for i in range(5)]
        response = self.post({"readings": rows})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["data"]["stored"], 5)

    def test_one_bad_row_does_not_discard_the_whole_buffer(self):
        rows = [{"temperature_c": 20}, {"temperature_c": 9999}, {"temperature_c": 22}]
        response = self.post({"readings": rows})
        self.assertEqual(response.status_code, 201)
        data = response.get_json()["data"]
        self.assertEqual(data["stored"], 2)
        self.assertEqual(len(data["rejected"]), 1)
        self.assertEqual(data["rejected"][0]["index"], 1)

    def test_empty_batch_is_rejected(self):
        self.assertEqual(self.post({"readings": []}).status_code, 400)

    def test_oversized_batch_is_rejected(self):
        rows = [{"temperature_c": 20}] * 501
        self.assertEqual(self.post({"readings": rows}).status_code, 413)

    def test_all_rows_invalid_returns_422(self):
        response = self.post({"readings": [{"temperature_c": 9999}]})
        self.assertEqual(response.status_code, 422)


class CaptureUpload(AppTestCase):
    sensor_source = "auto"

    def setUp(self):
        super().setUp()
        self.register_and_login()
        self.auth = {"Authorization": f"Bearer {self.make_device()}"}

    def test_jpeg_capture_is_accepted_and_attached(self):
        item = self.create_item()
        response = self.client.post(
            "/api/device/v1/captures",
            data={"image": (io.BytesIO(JPEG_BYTES), "frame.jpg"), "item_id": str(item["id"])},
            content_type="multipart/form-data",
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["data"]["attached_to_item"], item["id"])

    def test_non_image_capture_is_rejected(self):
        response = self.client.post(
            "/api/device/v1/captures",
            data={"image": (io.BytesIO(b"not an image"), "frame.jpg")},
            content_type="multipart/form-data",
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 415)


class SensorSourceModes(AppTestCase):
    sensor_source = "live"

    def test_live_only_mode_reports_offline_without_hardware(self):
        self.register_and_login()
        data = self.client.get("/api/v1/dashboard").get_json()["data"]
        self.assertEqual(data["sensor_mode"], "offline")
        self.assertIsNone(data["reading"]["temperature_c"])
        history = self.client.get("/api/v1/readings/history").get_json()["data"]
        self.assertEqual(history["points"], [])


if __name__ == "__main__":
    unittest.main()
