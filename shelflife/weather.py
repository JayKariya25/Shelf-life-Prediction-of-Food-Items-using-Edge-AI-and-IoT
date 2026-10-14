"""Real observed ambient conditions, for when no sensor is attached.

The built-in simulator in :mod:`shelflife.sensors` is a deterministic function of the
clock. That makes history stable, but it is a *model* of a Bengaluru kitchen, not a
measurement: on a hot afternoon it can be several degrees from the real thing. This
module fetches the actual current conditions for a configured location so the pages
show the real environment rather than a plausible one.

    live      a Raspberry Pi on the shelf reported         (best)
    observed  this module: a real weather station nearby   (real, but not in the room)
    simulated sensors.synthesise                           (a model, labelled as one)

**What "observed" is not.** It is outdoor air for a whole locality, measured nowhere
near the food. It is the right number for fruit on a kitchen counter in an unconditioned
room and the wrong number for a closed fridge - which is why enclosed storage keeps its
documented nominal profile (``services.STORAGE_PROFILES``) and never uses this. Every
value carries ``source="observed"`` and a note naming the place, so the UI can say where
the number came from; nothing here is ever presented as a reading from the food.

Open-Meteo is used because it needs no API key and no account, which keeps the project
runnable from a clean checkout. Only the standard library is used, so a Raspberry Pi
needs nothing extra installed.

Failure is expected and routine - the machine may be offline, or behind a portal. Every
entry point returns ``None`` instead of raising, the caller falls back to the simulator,
and a failed fetch is remembered briefly so a dead network is not retried on every page
load.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

log = logging.getLogger(__name__)

ENDPOINT = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "shelflife-monitor/2.0 (capstone project; contact via repository)"

# How long a failed fetch suppresses further attempts. Short enough that the site
# recovers on its own when the network comes back, long enough that an offline machine
# is not retried once per request.
FAILURE_COOLDOWN_SECONDS = 60.0

# Reject anything outside this; a station reporting -80 C is broken, not cold.
PLAUSIBLE_TEMPERATURE_C = (-60.0, 60.0)
PLAUSIBLE_HUMIDITY_PCT = (0.0, 100.0)


@dataclass(slots=True)
class Observation:
    """One real measurement from the weather service."""

    recorded_at: str          # UTC ISO-8601, as everything else in the app stores time
    temperature_c: float
    humidity_pct: float | None
    place: str
    station_distance_km: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recorded_at": self.recorded_at,
            "temperature_c": self.temperature_c,
            "humidity_pct": self.humidity_pct,
            "gas_ppm": None,          # a weather station has no gas channel
            "source": "observed",
        }


def _finite(value: object) -> float | None:
    """A float, or None for null/NaN/non-numeric - the API may omit a channel."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _in_range(value: float | None, low: float, high: float) -> float | None:
    if value is None:
        return None
    return value if low <= value <= high else None


def _parse_api_time(raw: object, utc_offset_seconds: int) -> str | None:
    """Open-Meteo times are local-to-the-location and unzoned; store UTC.

    ``timezone=auto`` makes the response's timestamps local, with the offset given
    separately, so the offset has to be subtracted rather than assumed to be zero.
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        naive = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if naive.tzinfo is not None:
        moment = naive.astimezone(timezone.utc)
    else:
        moment = naive.replace(tzinfo=timezone.utc) - timedelta(seconds=utc_offset_seconds)
    return moment.replace(microsecond=0, tzinfo=timezone.utc).isoformat()


def _fetch_json(params: dict[str, str], timeout: float) -> dict[str, Any]:
    """GET the API and parse it. Raises on any transport or decoding problem."""
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise urllib.error.HTTPError(url, response.status, "unexpected status", {}, None)
        return json.loads(response.read().decode("utf-8"))


def _validate(payload: object) -> dict[str, Any]:
    """Reject a response that is not a usable payload.

    Kept separate from the transport because the transport is replaceable: tests inject
    their own fetcher, and a check that lived in ``_fetch_json`` would be skipped in
    exactly the configuration the tests exercise.
    """
    if not isinstance(payload, dict):
        raise ValueError("weather API did not return an object")
    if payload.get("error"):
        raise ValueError(str(payload.get("reason") or "weather API reported an error"))
    return payload


class WeatherSource:
    """Cached access to one location's observed conditions.

    One instance per application. Thread-safe: waitress serves several threads, and a
    cache miss must result in one HTTP request rather than one per thread, so the fetch
    happens under a lock.
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        place: str = "",
        ttl_seconds: float = 600.0,
        timeout_seconds: float = 4.0,
        history_hours: int = 48,
        fetcher: Callable[[dict[str, str], float], dict[str, Any]] = _fetch_json,
    ) -> None:
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.place = place or f"{self.latitude:.4f}, {self.longitude:.4f}"
        self.ttl_seconds = max(60.0, float(ttl_seconds))
        self.timeout_seconds = max(0.5, float(timeout_seconds))
        self.history_hours = max(6, min(int(history_hours), 24 * 7))
        # Injected so tests never reach the network.
        self._fetcher = fetcher

        self._lock = threading.Lock()
        self._cached: dict[str, Any] | None = None
        self._cached_at = 0.0
        self._failed_at = 0.0
        self.last_error: str | None = None
        self.last_success_at: str | None = None

    # ------------------------------------------------------------------ fetching --
    def _params(self) -> dict[str, str]:
        return {
            "latitude": f"{self.latitude:.6f}",
            "longitude": f"{self.longitude:.6f}",
            "current": "temperature_2m,relative_humidity_2m",
            "hourly": "temperature_2m,relative_humidity_2m",
            "past_days": "2",
            "forecast_days": "1",
            "timezone": "auto",
        }

    def _payload(self, now: float | None = None) -> dict[str, Any] | None:
        """The cached API payload, refreshed when stale. Never raises."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._cached is not None and now - self._cached_at < self.ttl_seconds:
                return self._cached
            if self._failed_at and now - self._failed_at < FAILURE_COOLDOWN_SECONDS:
                # Still in the cooldown after a failure. Serve a stale payload if one
                # exists - an hour-old real reading beats inventing a number - and
                # otherwise let the caller fall back to the simulator.
                return self._cached
            try:
                payload = _validate(self._fetcher(self._params(), self.timeout_seconds))
            except Exception as exc:  # offline, DNS, portal, timeout, bad JSON
                self._failed_at = now
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.info("Observed weather unavailable (%s); using the simulator",
                         self.last_error)
                return self._cached
            self._cached = payload
            self._cached_at = now
            self._failed_at = 0.0
            self.last_error = None
            self.last_success_at = datetime.now(timezone.utc).replace(
                microsecond=0).isoformat()
            return payload

    # ------------------------------------------------------------------ readings --
    def current(self) -> Observation | None:
        """The latest real observation, or None if it cannot be obtained."""
        payload = self._payload()
        if not payload:
            return None
        block = payload.get("current")
        if not isinstance(block, dict):
            return None
        temperature = _in_range(_finite(block.get("temperature_2m")), *PLAUSIBLE_TEMPERATURE_C)
        if temperature is None:
            # Without a temperature there is nothing worth showing; humidity alone
            # would leave the page with a blank headline number.
            return None
        offset = int(_finite(payload.get("utc_offset_seconds")) or 0)
        recorded_at = _parse_api_time(block.get("time"), offset) or \
            datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        return Observation(
            recorded_at=recorded_at,
            temperature_c=round(temperature, 2),
            humidity_pct=(lambda h: round(h, 2) if h is not None else None)(
                _in_range(_finite(block.get("relative_humidity_2m")), *PLAUSIBLE_HUMIDITY_PCT)
            ),
            place=self.place,
        )

    def series(self, hours: int) -> list[Observation]:
        """Hourly observations covering the last ``hours``, oldest first.

        Returned at the API's own hourly resolution rather than resampled: these are
        measurements, and inventing intermediate points would make them look like a
        denser record than exists.
        """
        payload = self._payload()
        if not payload:
            return []
        block = payload.get("hourly")
        if not isinstance(block, dict):
            return []
        stamps = block.get("time") or []
        temps = block.get("temperature_2m") or []
        humids = block.get("relative_humidity_2m") or []
        if not isinstance(stamps, list) or not isinstance(temps, list):
            return []

        offset = int(_finite(payload.get("utc_offset_seconds")) or 0)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, int(hours)))
        now = datetime.now(timezone.utc)
        out: list[Observation] = []
        for index, stamp in enumerate(stamps):
            recorded_at = _parse_api_time(stamp, offset)
            if recorded_at is None:
                continue
            moment = datetime.fromisoformat(recorded_at)
            # past_days brings history; forecast_days brings hours that have not
            # happened yet. Only measurements belong on a chart of the past.
            if moment < cutoff or moment > now:
                continue
            temperature = _in_range(
                _finite(temps[index] if index < len(temps) else None),
                *PLAUSIBLE_TEMPERATURE_C,
            )
            if temperature is None:
                continue
            humidity = _in_range(
                _finite(humids[index] if index < len(humids) else None),
                *PLAUSIBLE_HUMIDITY_PCT,
            )
            out.append(Observation(
                recorded_at=recorded_at,
                temperature_c=round(temperature, 2),
                humidity_pct=round(humidity, 2) if humidity is not None else None,
                place=self.place,
            ))
        out.sort(key=lambda o: o.recorded_at)
        return out

    # -------------------------------------------------------------------- status --
    def status(self) -> dict[str, Any]:
        """For the developer console: whether this source is actually working."""
        available = self.current() is not None
        return {
            "place": self.place,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "available": available,
            "ttl_seconds": self.ttl_seconds,
            "timeout_seconds": self.timeout_seconds,
            "last_error": self.last_error,
            "last_success_at": self.last_success_at,
            "provider": "Open-Meteo",
        }
