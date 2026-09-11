"""Alerting, summaries, sensor determinism and schema migration."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests.support import AppTestCase

from shelflife import db as db_module
from shelflife import sensors, services
from shelflife.db import execute, iso_now, parse_ts, query_one, utcnow


class AlertLevels(unittest.TestCase):
    THRESHOLD = 48.0  # a 2-day reminder threshold

    def test_spoiled_is_always_critical(self):
        self.assertEqual(services.alert_level_for(500, "Spoiled", self.THRESHOLD), "critical")

    def test_inside_a_day_is_critical(self):
        self.assertEqual(services.alert_level_for(12, "Fresh", self.THRESHOLD), "critical")

    def test_inside_the_threshold_is_a_warning(self):
        self.assertEqual(services.alert_level_for(36, "Fresh", self.THRESHOLD), "warning")

    def test_beyond_the_threshold_is_fine(self):
        self.assertEqual(services.alert_level_for(120, "Fresh", self.THRESHOLD), "ok")

    def test_zero_remaining_is_critical(self):
        self.assertEqual(services.alert_level_for(0, "Moderately Fresh", self.THRESHOLD), "critical")

    def test_missing_estimate_does_not_raise_a_false_alarm(self):
        self.assertEqual(services.alert_level_for(None, "Fresh", self.THRESHOLD), "ok")

    def test_urgency_is_independent_of_the_freshness_class(self):
        """A genuinely fresh but very perishable item can still be urgent."""
        self.assertEqual(services.alert_level_for(10, "Fresh", self.THRESHOLD), "critical")


class Formatting(unittest.TestCase):
    def test_duration_wording(self):
        self.assertEqual(services.format_duration(None), "unknown")
        self.assertEqual(services.format_duration(0), "no time")
        self.assertEqual(services.format_duration(0.5), "30 minutes")
        self.assertEqual(services.format_duration(1), "1 hour")
        self.assertEqual(services.format_duration(5), "5 hours")
        self.assertEqual(services.format_duration(72), "3.0 days")


class AlertRaising(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register_and_login()

    def _summary(self, item_id, level="critical", label="Tomatoes", remaining=4.0):
        return {
            "id": item_id, "status": "active", "alert_level": level,
            "remaining_hours": remaining, "label": label, "prediction_id": None,
        }

    def test_alert_is_raised_for_an_at_risk_item(self):
        item = self.create_item()
        with self.app.app_context():
            created = services.evaluate_alerts(1, [self._summary(item["id"])], 48.0)
            self.assertEqual(created, 1)
            alerts = services.list_alerts(1)
            self.assertEqual(alerts[0]["level"], "critical")
            self.assertIn("Tomatoes", alerts[0]["title"])
            # Wording must not read "X is estimated at no time of remaining..."
            self.assertNotIn("no time", alerts[0]["message"])
            self.assertNotIn("Tomatoes is", alerts[0]["message"])

    def test_a_past_estimate_alert_reads_naturally(self):
        item = self.create_item()
        with self.app.app_context():
            services.evaluate_alerts(1, [self._summary(item["id"], remaining=0.0)], 48.0)
            self.assertIn("Past its estimated shelf life", services.list_alerts(1)[0]["message"])

    def test_the_same_alert_is_not_repeated_inside_the_cooldown(self):
        item = self.create_item()
        with self.app.app_context():
            services.evaluate_alerts(1, [self._summary(item["id"])], 48.0)
            again = services.evaluate_alerts(1, [self._summary(item["id"])], 48.0)
            self.assertEqual(again, 0)
            self.assertEqual(len(services.list_alerts(1)), 1)

    def test_escalating_from_warning_to_critical_still_alerts(self):
        item = self.create_item()
        with self.app.app_context():
            services.evaluate_alerts(1, [self._summary(item["id"], "warning", remaining=40)], 48.0)
            services.evaluate_alerts(1, [self._summary(item["id"], "critical", remaining=4)], 48.0)
            self.assertEqual(len(services.list_alerts(1)), 2)

    def test_healthy_items_raise_nothing(self):
        item = self.create_item()
        with self.app.app_context():
            self.assertEqual(
                services.evaluate_alerts(1, [self._summary(item["id"], "ok", remaining=500)], 48.0), 0
            )

    def test_closed_items_are_skipped(self):
        item = self.create_item()
        with self.app.app_context():
            summary = self._summary(item["id"])
            summary["status"] = "consumed"
            self.assertEqual(services.evaluate_alerts(1, [summary], 48.0), 0)

    def test_acknowledging_an_alert(self):
        item = self.create_item()
        with self.app.app_context():
            services.evaluate_alerts(1, [self._summary(item["id"])], 48.0)
            alert_id = services.list_alerts(1)[0]["id"]
            self.assertTrue(services.acknowledge_alert(1, alert_id))
            self.assertFalse(services.acknowledge_alert(1, alert_id), "second ack is a no-op")
            self.assertEqual(services.list_alerts(1, only_open=True), [])

    def test_another_user_cannot_acknowledge_your_alert(self):
        item = self.create_item()
        with self.app.app_context():
            services.evaluate_alerts(1, [self._summary(item["id"])], 48.0)
            alert_id = services.list_alerts(1)[0]["id"]
            self.assertFalse(services.acknowledge_alert(999, alert_id))


class ConditionFlags(unittest.TestCase):
    TOMATO = {
        "ideal_temp_min_c": 10.0, "ideal_temp_max_c": 15.0,
        "ideal_humidity_min": 85.0, "ideal_humidity_max": 95.0,
    }

    def test_ideal_conditions_produce_no_flags(self):
        self.assertEqual(
            services.condition_flags(self.TOMATO, {"temperature_c": 12, "humidity_pct": 90}), []
        )

    def test_too_warm_is_flagged(self):
        flags = services.condition_flags(self.TOMATO, {"temperature_c": 30, "humidity_pct": 90})
        self.assertTrue(any("Too warm" in flag for flag in flags))

    def test_chilling_risk_is_named_for_cold_sensitive_produce(self):
        flags = services.condition_flags(self.TOMATO, {"temperature_c": 3, "humidity_pct": 90})
        self.assertTrue(any("Chilling risk" in flag for flag in flags))

    def test_humidity_extremes_are_flagged(self):
        damp = services.condition_flags(self.TOMATO, {"temperature_c": 12, "humidity_pct": 99})
        dry = services.condition_flags(self.TOMATO, {"temperature_c": 12, "humidity_pct": 40})
        self.assertTrue(any("Too humid" in flag for flag in damp))
        self.assertTrue(any("Too dry" in flag for flag in dry))

    def test_missing_readings_do_not_crash(self):
        self.assertEqual(
            services.condition_flags(self.TOMATO, {"temperature_c": None, "humidity_pct": None}), []
        )


class SensorSimulator(unittest.TestCase):
    def test_the_same_instant_always_gives_the_same_reading(self):
        moment = datetime(2026, 5, 4, 13, 30, tzinfo=timezone.utc)
        self.assertEqual(sensors.synthesise(moment), sensors.synthesise(moment))

    def test_values_stay_inside_plausible_bounds(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for hour in range(0, 24 * 40, 7):
            reading = sensors.synthesise(start + timedelta(hours=hour))
            self.assertTrue(8.0 <= reading.temperature_c <= 44.0, reading.temperature_c)
            self.assertTrue(20.0 <= reading.humidity_pct <= 98.0, reading.humidity_pct)

    def test_the_series_is_smooth_rather_than_random(self):
        """The prototype jumped 12-34 C between polls; consecutive samples must not."""
        series = sensors.synthesise_series(
            datetime(2026, 5, 4, tzinfo=timezone.utc), 60, 300
        )
        jumps = [
            abs(series[i + 1].temperature_c - series[i].temperature_c)
            for i in range(len(series) - 1)
        ]
        self.assertLess(max(jumps), 1.5, "5-minute steps should not swing by degrees")

    def test_different_seeds_give_different_traces(self):
        moment = datetime(2026, 5, 4, 13, 30, tzinfo=timezone.utc)
        self.assertNotEqual(
            sensors.synthesise(moment, "user-1").temperature_c,
            sensors.synthesise(moment, "user-2").temperature_c,
        )

    def test_no_gas_value_is_invented_for_the_bmp280_sht31_build(self):
        self.assertIsNone(sensors.synthesise(datetime.now(timezone.utc)).gas_ppm)

    def test_pressure_is_not_simulated_because_the_app_does_not_use_it(self):
        self.assertIsNone(sensors.synthesise(datetime.now(timezone.utc)).pressure_hpa)

    def test_real_pressure_readings_are_still_accepted_from_hardware(self):
        cleaned = sensors.validate_payload({"temperature_c": 22.0, "pressure_hpa": 1008.1})
        self.assertEqual(cleaned["pressure_hpa"], 1008.1)

    def test_naive_datetimes_are_treated_as_utc(self):
        naive = datetime(2026, 5, 4, 13, 30)
        aware = datetime(2026, 5, 4, 13, 30, tzinfo=timezone.utc)
        self.assertEqual(sensors.synthesise(naive), sensors.synthesise(aware))


class TimestampParsing(unittest.TestCase):
    def test_accepts_the_formats_we_store_and_receive(self):
        for value in ("2026-05-04T13:30:00+00:00", "2026-05-04T13:30:00Z",
                      "2026-05-04 13:30:00", "2026-05-04"):
            parsed = parse_ts(value)
            self.assertIsNotNone(parsed, value)
            self.assertEqual(parsed.tzinfo, timezone.utc)

    def test_rejects_nonsense(self):
        for value in (None, "", "tomorrow", "13:30"):
            self.assertIsNone(parse_ts(value))

    def test_naive_timestamps_are_assumed_utc(self):
        self.assertEqual(parse_ts("2026-05-04 13:30:00").hour, 13)


class SchemaMigration(unittest.TestCase):
    """The prototype's two-table database must upgrade without losing data."""

    LEGACY_USERS = """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE, mobile TEXT UNIQUE,
            password_hash TEXT NOT NULL, created_at TEXT NOT NULL)
    """
    LEGACY_SETTINGS = """
        CREATE TABLE settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            email_notifications INTEGER DEFAULT 0,
            sms_notifications INTEGER DEFAULT 0,
            alert_days_before INTEGER DEFAULT 2,
            latest_image TEXT)
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self._tmp.name) / "legacy.db")
        conn = sqlite3.connect(self.path)
        conn.execute(self.LEGACY_USERS)
        conn.execute(self.LEGACY_SETTINGS)
        conn.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            ("legacy@example.com", "pbkdf2:sha256:fake", "2026-02-04 10:00:00"),
        )
        conn.execute("INSERT INTO settings (user_id, alert_days_before) VALUES (1, 5)")
        conn.commit()
        conn.close()

    def tearDown(self):
        self._tmp.cleanup()

    def test_migration_preserves_existing_rows(self):
        db_module.init_db(self.path)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        user = conn.execute("SELECT * FROM users WHERE email = 'legacy@example.com'").fetchone()
        self.assertIsNotNone(user)
        self.assertEqual(user["is_active"], 1)
        settings = conn.execute("SELECT * FROM settings WHERE user_id = 1").fetchone()
        self.assertEqual(settings["alert_days_before"], 5, "existing preference must survive")
        self.assertEqual(settings["theme"], "system")
        conn.close()

    def test_migration_adds_the_new_tables_and_seed_data(self):
        report = db_module.init_db(self.path)
        self.assertGreater(report["seeded_food_types"], 0)
        conn = sqlite3.connect(self.path)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"items", "readings", "predictions", "alerts", "devices"} <= tables)
        conn.close()

    def test_migration_is_idempotent(self):
        db_module.init_db(self.path)
        second = db_module.init_db(self.path)
        self.assertEqual(second["migrated_columns"], [])
        self.assertEqual(second["seeded_food_types"], 0)

    def test_a_user_without_settings_gets_one(self):
        conn = sqlite3.connect(self.path)
        conn.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            ("orphan@example.com", "x", "2026-02-04 10:00:00"),
        )
        conn.commit()
        conn.close()
        report = db_module.init_db(self.path)
        self.assertEqual(report["backfilled_settings"], 1)


class ReadingRetention(AppTestCase):
    def test_old_readings_are_pruned(self):
        with self.app.app_context():
            execute("INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                    ("r@example.com", "x", iso_now()))
            old = (utcnow() - timedelta(days=90)).replace(microsecond=0).isoformat()
            for stamp in (old, iso_now()):
                execute(
                    "INSERT INTO readings (user_id, recorded_at, temperature_c, source) "
                    "VALUES (1, ?, 20, 'live')", (stamp,),
                )
            conn = db_module.get_db()
            self.assertEqual(db_module.prune_readings(conn, 30), 1)
            self.assertEqual(query_one("SELECT COUNT(*) AS n FROM readings")["n"], 1)

    def test_zero_retention_prunes_nothing(self):
        with self.app.app_context():
            self.assertEqual(db_module.prune_readings(db_module.get_db(), 0), 0)


if __name__ == "__main__":
    unittest.main()


class StorageEnvironment(AppTestCase):
    """Items in enclosed storage must not be modelled at ambient temperature."""

    def setUp(self):
        super().setUp()
        self.register_and_login()

    def _summary(self, storage):
        item = self.create_item(label=f"{storage} item", storage=storage)
        return item

    def test_fridge_item_uses_the_nominal_profile(self):
        item = self._summary("fridge")
        self.assertEqual(item["environment_source"], "nominal")
        self.assertEqual(item["environment_temperature_c"], 4.0)
        self.assertIn("nominal", item["environment_note"])

    def test_freezer_item_uses_the_nominal_profile(self):
        item = self._summary("freezer")
        self.assertEqual(item["environment_temperature_c"], -18.0)

    def test_counter_item_uses_the_ambient_reading(self):
        item = self._summary("counter")
        self.assertEqual(item["environment_source"], "simulated")
        ambient = self.client.get("/api/v1/readings/latest").get_json()["data"]
        self.assertAlmostEqual(item["environment_temperature_c"], ambient["temperature_c"], delta=1.0)

    def test_fridge_item_is_not_flagged_too_warm_by_a_warm_room(self):
        """The bug this guards: a 25 C room made every fridge item read "too warm"."""
        item = self.create_item(label="Tomatoes", food_key="tomato", storage="fridge")
        self.assertFalse(
            any("Too warm" in flag for flag in item["condition_flags"]),
            item["condition_flags"],
        )

    def test_the_same_produce_on_the_counter_is_flagged_too_warm(self):
        item = self.create_item(label="Tomatoes", food_key="tomato", storage="counter")
        self.assertTrue(any("Too warm" in flag for flag in item["condition_flags"]))

    def test_no_condition_warnings_are_raised_from_a_nominal_profile(self):
        """Never warn the user about conditions this app assumed rather than measured."""
        for key in ("banana", "tomato"):
            item = self.create_item(label=f"{key} batch", food_key=key, storage="fridge")
            self.assertEqual(item["condition_flags"], [], key)
            self.assertEqual(item["environment_source"], "nominal")

    def test_refrigerating_chilling_sensitive_produce_still_shortens_the_estimate(self):
        """The chilling penalty still applies even though no warning is shown."""
        fridge = self.create_item(label="Cold tomatoes", food_key="tomato", storage="fridge")
        with self.app.app_context():
            from shelflife.inference import KineticBaselinePredictor, PredictionContext
            model = KineticBaselinePredictor()
            raw = services.get_item(1, fridge["id"])
            ctx = PredictionContext(
                food_key="tomato", food_name="Tomato",
                ref_shelf_life_hours=raw["ref_shelf_life_hours"],
                ref_temperature_c=raw["ref_temperature_c"], q10=raw["q10"],
                ideal_temp_min_c=raw["ideal_temp_min_c"],
            )
            chilled, _ = model.life_hours(ctx, 4.0, 85.0, None)
            ideal, _ = model.life_hours(ctx, 12.0, 85.0, None)
            self.assertLess(chilled, ideal)

    def test_fridge_item_outlives_the_same_produce_on_the_counter(self):
        """Even for chilling-sensitive produce, a warm room is worse than a fridge."""
        fridge = self.create_item(label="Cold tomatoes", food_key="tomato", storage="fridge")
        counter = self.create_item(label="Warm tomatoes", food_key="tomato", storage="counter")
        self.assertGreater(fridge["remaining_hours"], counter["remaining_hours"])

    def test_nominal_history_is_a_single_constant_segment(self):
        item = self.create_item(storage="fridge")
        with self.app.app_context():
            raw = services.get_item(1, item["id"])
            reading = services.reading_for_item(1, raw, services.latest_reading(1))
            segments = services.segments_for_item(
                1, raw, reading, utcnow() - timedelta(hours=10), utcnow()
            )
            self.assertEqual(len(segments), 1)
            self.assertAlmostEqual(segments[0][0], 10.0, places=1)
            self.assertEqual(segments[0][1], 4.0)


class RemainingWording(unittest.TestCase):
    def test_past_estimate_reads_naturally(self):
        self.assertEqual(services.format_remaining(0), "Past estimate")
        self.assertEqual(services.format_remaining(None), "No estimate")
        self.assertEqual(services.format_remaining(5), "5 hours left")
        self.assertEqual(services.format_remaining(96), "4.0 days left")


class ProduceScope(AppTestCase):
    """The project covers banana and tomato only."""

    def test_only_the_supported_produce_is_seeded(self):
        with self.app.app_context():
            keys = {row["key"] for row in services.food_types()}
        self.assertEqual(keys, {"banana", "tomato"})

    def test_the_api_offers_only_the_supported_produce(self):
        self.register_and_login()
        data = self.client.get("/api/v1/food-types").get_json()["data"]
        self.assertEqual({f["key"] for f in data["food_types"]}, {"banana", "tomato"})

    def test_the_add_item_form_lists_only_the_supported_produce(self):
        self.register_and_login()
        body = self.client.get("/items").get_data(as_text=True)
        self.assertIn("Banana", body)
        self.assertIn("Tomato", body)
        for absent in ("Spinach", "Strawberry", "Capsicum", "Cucumber", "Potato", "Onion"):
            self.assertNotIn(absent, body, absent)

    def test_unused_out_of_scope_profiles_are_pruned_on_migration(self):
        from shelflife import db as db_module

        with self.app.app_context():
            conn = db_module.get_db()
            conn.execute(
                "INSERT INTO food_types (key, name, category, emoji, ref_shelf_life_hours,"
                " ref_temperature_c, q10) VALUES ('durian', 'Durian', 'fruit', '', 100, 20, 2.0)"
            )
        report = db_module.init_db(self.app.config["DATABASE_PATH"])
        self.assertIn("durian", report["pruned_food_types"])

    def test_a_profile_still_in_use_is_never_silently_deleted(self):
        """Narrowing scope must not delete somebody's tracked item."""
        from shelflife import db as db_module

        self.register_and_login()
        with self.app.app_context():
            conn = db_module.get_db()
            conn.execute(
                "INSERT INTO food_types (key, name, category, emoji, ref_shelf_life_hours,"
                " ref_temperature_c, q10) VALUES ('durian', 'Durian', 'fruit', '', 100, 20, 2.0)"
            )
            food_id = query_one("SELECT id FROM food_types WHERE key='durian'")["id"]
            services.create_item(1, food_id, "Durian batch", None, "room", iso_now())

        report = db_module.init_db(self.app.config["DATABASE_PATH"])
        self.assertNotIn("durian", report["pruned_food_types"])
        with self.app.app_context():
            self.assertIsNotNone(query_one("SELECT 1 FROM food_types WHERE key='durian'"))
