"""Real observed ambient conditions, and the fallbacks around them.

No test here touches the network: every case injects a fake fetcher, so the suite stays
runnable on a machine with no internet and cannot be made flaky by a provider outage.
``TestingConfig`` also sets ``WEATHER_ENABLED = False``, so a test only gets a weather
source if it asks for one.
"""

from __future__ import annotations

import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests.support import AppTestCase

from shelflife import build_weather_source
from shelflife.weather import Observation, WeatherSource

# One hour, in the shape Open-Meteo returns it: local wall-clock strings plus the
# offset that turns them into UTC.
IST_OFFSET = 5 * 3600 + 1800


def payload(temperature=29.2, humidity=35.0, *, now=None, hours=6, offset=IST_OFFSET):
    """A response shaped exactly like the provider's, centred on ``now``."""
    now = now or datetime.now(timezone.utc)
    local_now = now + timedelta(seconds=offset)
    times, temps, humids = [], [], []
    # Past hours, then two hours of forecast - which the series must discard.
    for step in range(-hours, 3):
        stamp = (local_now + timedelta(hours=step)).replace(minute=0, second=0, microsecond=0)
        times.append(stamp.strftime("%Y-%m-%dT%H:%M"))
        temps.append(temperature + step * 0.1)
        humids.append(humidity)
    return {
        "utc_offset_seconds": offset,
        "timezone": "Asia/Kolkata",
        "current": {
            "time": local_now.replace(second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M"),
            "temperature_2m": temperature,
            "relative_humidity_2m": humidity,
        },
        "hourly": {"time": times, "temperature_2m": temps, "relative_humidity_2m": humids},
    }


def source(response=None, *, place="Bengaluru", fail=False, **kwargs):
    calls = []

    def fetcher(params, timeout):
        calls.append(params)
        if fail:
            raise OSError("network is down")
        return response if response is not None else payload()

    src = WeatherSource(12.9344, 77.5353, place=place, fetcher=fetcher, **kwargs)
    src.calls = calls  # type: ignore[attr-defined]
    return src


# =============================================================================
# the source itself
# =============================================================================
class Fetching(unittest.TestCase):
    def test_current_returns_the_real_reading(self):
        observation = source().current()
        self.assertIsNotNone(observation)
        self.assertEqual(observation.temperature_c, 29.2)
        self.assertEqual(observation.humidity_pct, 35.0)
        self.assertEqual(observation.place, "Bengaluru")

    def test_local_time_is_converted_to_utc(self):
        """The provider returns local wall-clock time; storing it as UTC would be 5.5 h out."""
        now = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
        observation = source(payload(now=now)).current()
        self.assertEqual(observation.recorded_at, "2026-10-13T12:00:00+00:00")

    def test_a_reading_is_fetched_once_within_the_ttl(self):
        src = source(ttl_seconds=600)
        for _ in range(5):
            src.current()
        self.assertEqual(len(src.calls), 1)

    def test_concurrent_callers_cause_one_fetch(self):
        """waitress serves several threads; a cache miss must not stampede the provider."""
        src = source(ttl_seconds=600)
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            src.current()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(src.calls), 1)

    def test_failure_returns_none_rather_than_raising(self):
        src = source(fail=True)
        self.assertIsNone(src.current())
        self.assertIn("network is down", src.last_error)

    def test_failure_is_not_retried_on_every_call(self):
        src = source(fail=True)
        for _ in range(6):
            src.current()
        self.assertEqual(len(src.calls), 1)

    def test_a_stale_real_reading_is_preferred_over_nothing(self):
        """An hour-old measurement is still a measurement; losing it would be worse."""
        calls = []
        state = {"fail": False}

        def fetcher(params, timeout):
            calls.append(params)
            if state["fail"]:
                raise OSError("down")
            return payload()

        src = WeatherSource(1.0, 2.0, fetcher=fetcher, ttl_seconds=60)
        self.assertIsNotNone(src.current())
        state["fail"] = True
        src._cached_at -= 10_000      # force the TTL to have expired
        src._failed_at = 0.0
        self.assertIsNotNone(src.current(), "the cached payload should still serve")
        self.assertEqual(len(calls), 2)

    def test_an_api_level_error_is_treated_as_a_failure(self):
        src = source({"error": True, "reason": "No data for this location"})
        self.assertIsNone(src.current())
        self.assertIn("No data for this location", src.last_error)

    def test_rubbish_values_are_refused(self):
        for bad in (None, "warm", float("nan"), 1e9, -400.0):
            with self.subTest(bad=bad):
                response = payload()
                response["current"]["temperature_2m"] = bad
                self.assertIsNone(source(response).current())

    def test_an_out_of_range_humidity_is_dropped_but_the_reading_survives(self):
        response = payload()
        response["current"]["relative_humidity_2m"] = 450.0
        observation = source(response).current()
        self.assertIsNotNone(observation)
        self.assertEqual(observation.temperature_c, 29.2)
        self.assertIsNone(observation.humidity_pct)

    def test_a_missing_current_block_is_survivable(self):
        self.assertIsNone(source({"utc_offset_seconds": 0}).current())

    def test_to_dict_never_invents_a_gas_channel(self):
        """A weather station has no gas sensor, and the baseline must not be given one."""
        self.assertIsNone(source().current().to_dict()["gas_ppm"])
        self.assertEqual(source().current().to_dict()["source"], "observed")


class Series(unittest.TestCase):
    def test_hourly_history_is_returned_oldest_first(self):
        points = source().series(6)
        self.assertGreater(len(points), 1)
        stamps = [p.recorded_at for p in points]
        self.assertEqual(stamps, sorted(stamps))

    def test_forecast_hours_are_discarded(self):
        """The payload carries two future hours; a chart of the past must not plot them."""
        now = datetime.now(timezone.utc)
        for point in source().series(24):
            self.assertLessEqual(datetime.fromisoformat(point.recorded_at), now)

    def test_the_window_is_respected(self):
        now = datetime.now(timezone.utc)
        points = source(payload(hours=20)).series(3)
        self.assertTrue(points)
        for point in points:
            age = (now - datetime.fromisoformat(point.recorded_at)).total_seconds() / 3600
            self.assertLessEqual(age, 3.5, point.recorded_at)

    def test_a_failed_fetch_gives_an_empty_series(self):
        self.assertEqual(source(fail=True).series(6), [])

    def test_a_malformed_hourly_block_gives_an_empty_series(self):
        response = payload()
        response["hourly"] = {"time": "not-a-list", "temperature_2m": 3}
        self.assertEqual(source(response).series(6), [])

    def test_rows_with_a_missing_temperature_are_skipped_not_fatal(self):
        response = payload()
        response["hourly"]["temperature_2m"][1] = None
        points = source(response).series(24)
        self.assertTrue(points)
        self.assertTrue(all(p.temperature_c is not None for p in points))


class Factory(unittest.TestCase):
    def test_disabled_configuration_builds_nothing(self):
        self.assertIsNone(build_weather_source({"WEATHER_ENABLED": False}))

    def test_enabled_configuration_builds_a_source(self):
        built = build_weather_source({
            "WEATHER_ENABLED": True, "WEATHER_LATITUDE": 1.5, "WEATHER_LONGITUDE": 2.5,
            "WEATHER_PLACE": "Somewhere", "WEATHER_TTL_SECONDS": 300,
            "WEATHER_TIMEOUT_SECONDS": 2.0,
        })
        self.assertEqual((built.latitude, built.longitude), (1.5, 2.5))
        self.assertEqual(built.place, "Somewhere")

    def test_a_place_name_is_never_required(self):
        built = build_weather_source({"WEATHER_ENABLED": True, "WEATHER_LATITUDE": 1.0,
                                      "WEATHER_LONGITUDE": 2.0, "WEATHER_PLACE": ""})
        self.assertIn("1.0000", built.place)

    def test_building_one_does_not_fetch(self):
        """Start-up must never wait on the network."""
        built = build_weather_source({"WEATHER_ENABLED": True, "WEATHER_LATITUDE": 1.0,
                                      "WEATHER_LONGITUDE": 2.0})
        self.assertIsNone(built.last_success_at)
        self.assertIsNone(built.last_error)


# =============================================================================
# inside the application
# =============================================================================
class ObservedInTheApp(AppTestCase):
    sensor_source = "auto"

    def setUp(self):
        super().setUp()
        self.app.config["WEATHER_ENABLED"] = True
        self.installed = source()
        self.app.extensions["weather"] = self.installed
        self.register_and_login()

    def test_mode_is_observed_when_no_device_is_reporting(self):
        from shelflife import services

        with self.app.test_request_context():
            self.assertEqual(services.sensor_mode(1), "observed")
            reading = services.latest_reading(1)
        self.assertEqual(reading["mode"], "observed")
        self.assertEqual(reading["temperature_c"], 29.2)
        self.assertEqual(reading["source"], "observed")

    def test_the_api_serves_the_real_temperature(self):
        data = self.client.get("/api/v1/readings/latest").get_json()["data"]
        self.assertEqual(data["temperature_c"], 29.2)
        self.assertEqual(data["mode"], "observed")
        self.assertEqual(data["source"], "observed")
        self.assertEqual(data["place"], "Bengaluru")

    def test_history_is_the_real_hourly_record(self):
        data = self.client.get("/api/v1/readings/history?hours=6").get_json()["data"]
        self.assertEqual(data["mode"], "observed")
        self.assertTrue(data["points"])
        self.assertTrue(all(p["source"] == "observed" for p in data["points"]))

    def test_a_live_device_still_wins(self):
        from shelflife import services

        token = self.make_device()
        self.client.post("/api/device/v1/readings",
                         headers={"Authorization": f"Bearer {token}"},
                         json={"temperature_c": 31.5, "humidity_pct": 44.0})
        with self.app.test_request_context():
            self.assertEqual(services.sensor_mode(1), "live")
            self.assertEqual(services.latest_reading(1)["temperature_c"], 31.5)

    def test_an_unreachable_provider_falls_back_to_the_labelled_simulator(self):
        from shelflife import services

        self.app.extensions["weather"] = source(fail=True)
        with self.app.test_request_context():
            self.assertEqual(services.sensor_mode(1), "simulated")
            self.assertEqual(services.latest_reading(1)["mode"], "simulated")

    def test_a_room_item_is_modelled_on_the_real_reading_and_says_where_it_came_from(self):
        from shelflife import services

        item = self.create_item(label="Counter bananas", food_key="banana", storage="room")
        with self.app.test_request_context():
            raw = services.get_item(1, item["id"])
            ambient = services.latest_reading(1)
            reading = services.reading_for_item(1, raw, ambient)
        self.assertEqual(reading["temperature_c"], 29.2)
        self.assertEqual(reading["environment_source"], "observed")
        self.assertIn("Bengaluru", reading["environment_note"])
        self.assertIn("weather station", reading["environment_note"])

    def test_a_fridge_item_keeps_its_nominal_profile(self):
        """Outdoor air says nothing about a closed fridge, however real it is."""
        from shelflife import services

        item = self.create_item(label="Fridge tomatoes", food_key="tomato", storage="fridge")
        with self.app.test_request_context():
            raw = services.get_item(1, item["id"])
            ambient = services.latest_reading(1)
            reading = services.reading_for_item(1, raw, ambient)
        self.assertEqual(reading["temperature_c"], 4.0)
        self.assertEqual(reading["environment_source"], "nominal")

    def test_the_dashboard_says_the_number_is_observed(self):
        page = self.client.get("/dashboard").get_data(as_text=True)
        self.assertIn("Bengaluru", page)
        self.assertNotIn("built-in simulator", page)

    def test_degradation_history_uses_the_real_hourly_record(self):
        from shelflife import services
        from shelflife.db import utcnow

        with self.app.test_request_context():
            since = utcnow() - timedelta(hours=4)
            segments = services.environment_segments(1, since, utcnow())
        self.assertTrue(segments)
        self.assertTrue(all(duration > 0 for duration, *_ in segments))
        self.assertTrue(all(gas is None for *_, gas in segments))

    def test_explicit_observed_mode_ignores_devices(self):
        from shelflife import services

        self.app.config["SENSOR_SOURCE"] = "observed"
        token = self.make_device()
        self.client.post("/api/device/v1/readings",
                         headers={"Authorization": f"Bearer {token}"},
                         json={"temperature_c": 31.5, "humidity_pct": 44.0})
        with self.app.test_request_context():
            self.assertEqual(services.sensor_mode(1), "observed")

    def test_live_mode_never_substitutes_a_weather_reading(self):
        from shelflife import services

        self.app.config["SENSOR_SOURCE"] = "live"
        with self.app.test_request_context():
            self.assertEqual(services.sensor_mode(1), "offline")
            self.assertIsNone(services.latest_reading(1)["temperature_c"])

    def test_simulated_mode_never_fetches(self):
        from shelflife import services

        self.app.config["SENSOR_SOURCE"] = "simulated"
        with self.app.test_request_context():
            self.assertEqual(services.sensor_mode(1), "simulated")
        self.assertEqual(len(self.installed.calls), 0)


class DeveloperVisibility(AppTestCase):
    """A silent fall back to the simulator must be visible, not only in the logs."""

    sensor_source = "auto"

    def setUp(self):
        super().setUp()
        self.app.config["WEATHER_ENABLED"] = True
        self.register_and_login_admin()

    def test_system_page_reports_a_reachable_source(self):
        self.app.extensions["weather"] = source()
        page = self.client.get("/admin/system").get_data(as_text=True)
        self.assertIn("Reachable", page)
        self.assertIn("Bengaluru", page)

    def test_system_page_reports_an_unreachable_source_and_the_reason(self):
        self.app.extensions["weather"] = source(fail=True)
        page = self.client.get("/admin/system").get_data(as_text=True)
        self.assertIn("Unreachable", page)
        self.assertIn("network is down", page)
        self.assertIn("falling back to the simulator", page)

    def test_system_page_reports_a_disabled_source(self):
        self.app.extensions["weather"] = None
        page = self.client.get("/admin/system").get_data(as_text=True)
        self.assertIn("WEATHER_ENABLED is off", page)

    def test_device_page_names_the_place(self):
        self.app.extensions["weather"] = source()
        page = self.client.get("/device").get_data(as_text=True)
        self.assertIn("local weather for", page)
        self.assertIn("Bengaluru", page)

    def test_the_pill_never_calls_a_real_reading_simulated(self):
        self.app.extensions["weather"] = source()
        page = self.client.get("/dashboard").get_data(as_text=True)
        self.assertIn("Local weather", page)
        self.assertNotIn("Simulated data", page)
        self.assertNotIn("SHT31 sensor", page)

    def test_the_server_rendered_label_matches_what_the_poll_will_write(self):
        """Otherwise the row silently rewords itself a few seconds after the page loads."""
        self.app.extensions["weather"] = source()
        page = self.client.get("/dashboard").get_data(as_text=True)
        rendered = page.split('id="sensorModeText"')[1].split("</dd>")[0]
        self.assertIn("Local weather", rendered)
        # The same string the JS mode map uses for this mode.
        self.assertIn('observed: ["info", `Local weather',
                      Path("static/js/dashboard.js").read_text())


class WithoutAWeatherSource(AppTestCase):
    """The default test profile has none, so this is the offline-machine case."""

    sensor_source = "auto"

    def test_everything_still_works_on_the_simulator(self):
        from shelflife import services

        self.register_and_login()
        self.assertIsNone(self.app.extensions.get("weather"))
        with self.app.test_request_context():
            self.assertEqual(services.sensor_mode(1), "simulated")
            self.assertIsNone(services.observed_reading())
        self.assertEqual(self.client.get("/dashboard").status_code, 200)


if __name__ == "__main__":
    unittest.main()
