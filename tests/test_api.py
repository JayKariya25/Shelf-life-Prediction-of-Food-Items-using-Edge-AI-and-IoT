"""JSON API: items, predictions, settings, uploads and tenant isolation."""

from __future__ import annotations

import io
import unittest
from datetime import timedelta

from tests.support import AppTestCase

from shelflife.db import query_one, utcnow

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class ItemLifecycle(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register_and_login()

    def test_create_returns_a_prediction_immediately(self):
        item = self.create_item()
        self.assertIsNotNone(item["remaining_hours"])
        self.assertIn(item["freshness_class"], {"Fresh", "Moderately Fresh", "Spoiled"})
        self.assertEqual(item["model_kind"], "heuristic-baseline")

    def test_listing_returns_created_items(self):
        self.create_item(label="Alpha")
        self.create_item(label="Beta")
        data = self.client.get("/api/v1/items").get_json()["data"]
        self.assertEqual({item["label"] for item in data["items"]}, {"Alpha", "Beta"})

    def test_patch_updates_fields_and_recomputes(self):
        item = self.create_item()
        response = self.post_json(
            f"/api/v1/items/{item['id']}",
            {"label": "Renamed", "quantity": "3 kg"},
            method="PATCH",
        )
        self.assertEqual(response.status_code, 200)
        updated = response.get_json()["data"]["item"]
        self.assertEqual(updated["label"], "Renamed")
        self.assertEqual(updated["quantity"], "3 kg")

    def test_marking_consumed_closes_the_item(self):
        item = self.create_item()
        self.post_json(f"/api/v1/items/{item['id']}", {"status": "consumed"}, method="PATCH")
        active = self.client.get("/api/v1/items?status=active").get_json()["data"]["items"]
        consumed = self.client.get("/api/v1/items?status=consumed").get_json()["data"]["items"]
        self.assertEqual(active, [])
        self.assertEqual(len(consumed), 1)
        self.assertIsNotNone(consumed[0]["closed_at"])

    def test_delete_removes_the_item(self):
        item = self.create_item()
        response = self.client.delete(
            f"/api/v1/items/{item['id']}", headers={"X-CSRF-Token": self.api_token()}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/api/v1/items").get_json()["data"]["items"], [])

    def test_deleting_an_item_cascades_to_its_predictions(self):
        item = self.create_item()
        self.client.delete(f"/api/v1/items/{item['id']}",
                           headers={"X-CSRF-Token": self.api_token()})
        with self.app.app_context():
            row = query_one("SELECT COUNT(*) AS n FROM predictions WHERE item_id = ?", (item["id"],))
            self.assertEqual(row["n"], 0)

    def test_predict_endpoint_appends_to_the_history(self):
        item = self.create_item()
        response = self.post_json(f"/api/v1/items/{item['id']}/predict", {})
        self.assertEqual(response.status_code, 200)
        history = self.client.get(f"/api/v1/items/{item['id']}").get_json()["data"]["history"]
        self.assertGreaterEqual(len(history), 2)

    def test_older_items_have_less_life_left_than_fresh_ones(self):
        fresh = self.create_item(label="Fresh batch")
        old_at = (utcnow() - timedelta(hours=100)).isoformat()
        old = self.create_item(label="Old batch", stored_at=old_at)
        self.assertLess(old["remaining_hours"], fresh["remaining_hours"])
        self.assertGreater(old["hours_stored"], 99)


class ItemValidation(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register_and_login()
        with self.app.app_context():
            self.food_id = query_one("SELECT id FROM food_types WHERE key='tomato'")["id"]

    def _post(self, payload):
        return self.post_json("/api/v1/items", payload)

    def test_blank_label_rejected(self):
        response = self._post({"label": "   ", "food_type_id": self.food_id})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "invalid_label")

    def test_unknown_food_type_rejected(self):
        response = self._post({"label": "x", "food_type_id": 99999})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "invalid_food_type")

    def test_non_numeric_food_type_rejected(self):
        response = self._post({"label": "x", "food_type_id": "banana"})
        self.assertEqual(response.status_code, 422)

    def test_unknown_storage_rejected(self):
        response = self._post({"label": "x", "food_type_id": self.food_id, "storage": "spaceship"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "invalid_storage")

    def test_future_stored_at_rejected(self):
        future = (utcnow() + timedelta(days=2)).isoformat()
        response = self._post({"label": "x", "food_type_id": self.food_id, "stored_at": future})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "invalid_stored_at")

    def test_ancient_stored_at_rejected(self):
        ancient = (utcnow() - timedelta(days=500)).isoformat()
        response = self._post({"label": "x", "food_type_id": self.food_id, "stored_at": ancient})
        self.assertEqual(response.status_code, 422)

    def test_unparseable_stored_at_rejected(self):
        response = self._post({"label": "x", "food_type_id": self.food_id, "stored_at": "yesterday"})
        self.assertEqual(response.status_code, 422)

    def test_unknown_image_reference_rejected(self):
        response = self._post({"label": "x", "food_type_id": self.food_id, "image_path": "../../etc/passwd"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "invalid_image")

    def test_long_label_is_truncated_not_rejected(self):
        response = self._post({"label": "T" * 500, "food_type_id": self.food_id})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(len(response.get_json()["data"]["item"]["label"]), 80)

    def test_missing_item_returns_404(self):
        self.assertEqual(self.client.get("/api/v1/items/999999").status_code, 404)


class TenantIsolation(AppTestCase):
    """A signed-in user must never reach another account's data."""

    def setUp(self):
        super().setUp()
        self.register_and_login(email="alice@example.com")
        self.alice_item = self.create_item(label="Alice tomatoes")
        self.client.post("/logout", data={"csrf_token": self.csrf("/dashboard")})

        self.register(email="bob@example.com", password="correct-horse-9")
        self.login(identifier="bob@example.com")

    def test_bob_cannot_list_alice_items(self):
        items = self.client.get("/api/v1/items?status=all").get_json()["data"]["items"]
        self.assertEqual(items, [])

    def test_bob_cannot_read_an_alice_item(self):
        self.assertEqual(self.client.get(f"/api/v1/items/{self.alice_item['id']}").status_code, 404)

    def test_bob_cannot_patch_an_alice_item(self):
        response = self.post_json(
            f"/api/v1/items/{self.alice_item['id']}", {"label": "hijacked"}, method="PATCH"
        )
        self.assertEqual(response.status_code, 404)

    def test_bob_cannot_delete_an_alice_item(self):
        response = self.client.delete(
            f"/api/v1/items/{self.alice_item['id']}", headers={"X-CSRF-Token": self.api_token()}
        )
        self.assertEqual(response.status_code, 404)

    def test_bob_cannot_predict_on_an_alice_item(self):
        response = self.post_json(f"/api/v1/items/{self.alice_item['id']}/predict", {})
        self.assertEqual(response.status_code, 404)

    def test_bob_cannot_open_the_alice_item_page(self):
        self.assertEqual(self.client.get(f"/items/{self.alice_item['id']}").status_code, 404)


class Uploads(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register_and_login()

    def _upload(self, filename, content):
        return self.client.post(
            "/api/v1/uploads",
            data={"image": (io.BytesIO(content), filename)},
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": self.api_token()},
        )

    def test_valid_png_is_accepted(self):
        response = self._upload("photo.png", PNG_BYTES)
        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.get_json()["data"]["image_path"].endswith(".png"))

    def test_disallowed_extension_is_rejected(self):
        response = self._upload("payload.svg", b"<svg></svg>")
        self.assertEqual(response.status_code, 415)

    def test_renamed_non_image_is_rejected_by_content_sniffing(self):
        response = self._upload("payload.png", b"#!/bin/sh\nrm -rf /\n")
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.get_json()["error"]["code"], "unsupported_type")

    def test_missing_file_is_rejected(self):
        response = self.client.post(
            "/api/v1/uploads", data={}, content_type="multipart/form-data",
            headers={"X-CSRF-Token": self.api_token()},
        )
        self.assertEqual(response.status_code, 422)

    def test_stored_filename_does_not_reuse_the_uploaded_name(self):
        """Prevents path tricks and collisions between users."""
        response = self._upload("../../../evil.png", PNG_BYTES)
        self.assertEqual(response.status_code, 201)
        stored = response.get_json()["data"]["image_path"]
        self.assertNotIn("/", stored)
        self.assertNotIn("..", stored)

    def test_uploaded_image_can_be_attached_to_an_item(self):
        stored = self._upload("photo.png", PNG_BYTES).get_json()["data"]["image_path"]
        item = self.create_item(image_path=stored)
        self.assertEqual(item["image_path"], stored)


class SettingsApi(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register_and_login()

    def test_update_and_read_back(self):
        response = self.post_json(
            "/api/v1/settings",
            {"alert_days_before": 5, "email_notifications": True, "sms_notifications": False},
            method="PUT",
        )
        self.assertEqual(response.status_code, 200)
        data = self.client.get("/api/v1/settings").get_json()["data"]
        self.assertEqual(data["alert_days_before"], 5)
        self.assertEqual(data["email_notifications"], 1)
        self.assertEqual(data["sms_notifications"], 0)

    def test_threshold_out_of_range_is_rejected(self):
        for value in (0, 31, -3):
            response = self.post_json("/api/v1/settings", {"alert_days_before": value}, method="PUT")
            self.assertEqual(response.status_code, 422, value)

    def test_non_numeric_threshold_is_rejected(self):
        response = self.post_json("/api/v1/settings", {"alert_days_before": "soon"}, method="PUT")
        self.assertEqual(response.status_code, 422)

    def test_bad_temperature_unit_is_rejected(self):
        response = self.post_json("/api/v1/settings", {"temperature_unit": "K"}, method="PUT")
        self.assertEqual(response.status_code, 422)


class DashboardApi(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register_and_login()

    def test_snapshot_shape(self):
        self.create_item()
        data = self.client.get("/api/v1/dashboard").get_json()["data"]
        for key in ("reading", "items", "counts", "freshness_counts", "at_risk",
                    "settings", "model", "sensor_mode", "live_readings_stored"):
            self.assertIn(key, data)

    def test_simulated_mode_is_labelled_and_not_counted_as_live(self):
        data = self.client.get("/api/v1/dashboard").get_json()["data"]
        self.assertEqual(data["sensor_mode"], "simulated")
        self.assertEqual(data["reading"]["source"], "simulated")
        self.assertEqual(data["live_readings_stored"], 0)

    def test_history_is_stable_between_calls(self):
        """The prototype re-randomised every poll; the same window must not move."""
        first = self.client.get("/api/v1/readings/history?hours=6&points=12").get_json()["data"]
        second = self.client.get("/api/v1/readings/history?hours=6&points=12").get_json()["data"]
        self.assertEqual(
            [round(p["temperature_c"]) for p in first["points"][:-1]],
            [round(p["temperature_c"]) for p in second["points"][:-1]],
        )

    def test_history_bounds_are_clamped(self):
        data = self.client.get("/api/v1/readings/history?hours=99999&points=99999").get_json()["data"]
        self.assertLessEqual(data["hours"], 24 * 30)
        self.assertLessEqual(len(data["points"]), 300)

    def test_garbage_query_parameters_fall_back_to_defaults(self):
        response = self.client.get("/api/v1/readings/history?hours=abc&points=xyz")
        self.assertEqual(response.status_code, 200)

    def test_unauthenticated_api_returns_json_401(self):
        self.client.post("/logout", data={"csrf_token": self.csrf("/dashboard")})
        response = self.client.get("/api/v1/dashboard")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json()["error"]["code"], "unauthenticated")

    def test_unknown_api_route_returns_json_404(self):
        response = self.client.get("/api/v1/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertFalse(response.get_json()["ok"])


if __name__ == "__main__":
    unittest.main()
