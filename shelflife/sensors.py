"""Sensor acquisition: live device readings plus a labelled fallback simulator.

The prototype regenerated ``random.uniform`` values on every poll, so the same
chart showed different history each time it refreshed. Here, synthetic readings
are a *deterministic function of their timestamp*: the value for 14:05 is the
same whenever it is computed. That makes history stable across reloads, makes
back-filling trivial, and keeps the simulator honest -- every synthetic row is
stored with ``source='simulated'`` and the UI labels it as such.

Real hardware never touches this module: the Raspberry Pi agent POSTs readings
to ``/api/v1/readings`` and they land with ``source='live'``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .db import iso_now, parse_ts, utcnow

# Plausible ranges for an indoor kitchen/storage environment in Bengaluru.
TEMP_BASELINE_C = 26.0
TEMP_DIURNAL_AMPLITUDE = 3.4
HUMIDITY_BASELINE = 63.0
HUMIDITY_DIURNAL_AMPLITUDE = 9.0

DAY_SECONDS = 86400.0

# Physical limits used to reject nonsense from a faulty or drifting sensor.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "temperature_c": (-40.0, 85.0),   # BMP280 / SHT31 operating range
    "humidity_pct": (0.0, 100.0),
    "pressure_hpa": (300.0, 1100.0),  # BMP280 datasheet range
    "gas_ppm": (0.0, 10000.0),
}


class SensorValidationError(ValueError):
    """Raised when an incoming reading cannot be trusted."""


@dataclass(slots=True)
class SensorReading:
    recorded_at: str
    temperature_c: float | None
    humidity_pct: float | None
    pressure_hpa: float | None
    gas_ppm: float | None
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Browser-facing shape.

        ``pressure_hpa`` is deliberately omitted: the application displays it
        nowhere, so there is no reason to ship it to the page. It is still
        validated and stored when real hardware reports it.
        """
        return {
            "recorded_at": self.recorded_at,
            "temperature_c": self.temperature_c,
            "humidity_pct": self.humidity_pct,
            "gas_ppm": self.gas_ppm,
            "source": self.source,
        }


def _jitter(seed: str, epoch_bucket: int, spread: float) -> float:
    """Smooth-ish deterministic noise in ``[-spread, +spread]``.

    Hashing the bucket keeps the value stable for a given timestamp while still
    looking irregular, which a plain sine sum does not.
    """
    digest = hashlib.blake2b(
        f"{seed}:{epoch_bucket}".encode("utf-8"), digest_size=8
    ).digest()
    unit = int.from_bytes(digest, "big") / float(1 << 64)  # [0, 1)
    return (unit * 2.0 - 1.0) * spread


def _interpolated_noise(seed: str, epoch: float, period: float, spread: float) -> float:
    """Interpolate between per-bucket noise samples so the series stays smooth."""
    position = epoch / period
    bucket = math.floor(position)
    blend = position - bucket
    left = _jitter(seed, bucket, spread)
    right = _jitter(seed, bucket + 1, spread)
    # Smoothstep easing avoids visible kinks at bucket boundaries.
    weight = blend * blend * (3.0 - 2.0 * blend)
    return left * (1.0 - weight) + right * weight


def synthesise(at: datetime, seed: str = "default") -> SensorReading:
    """Deterministic synthetic reading for a given instant."""
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    epoch = at.timestamp()

    # Local solar day, offset so the warmest point lands mid-afternoon IST.
    phase = 2.0 * math.pi * ((epoch % DAY_SECONDS) / DAY_SECONDS)
    diurnal = math.sin(phase - 2.6)

    temperature = (
        TEMP_BASELINE_C
        + TEMP_DIURNAL_AMPLITUDE * diurnal
        + 1.1 * math.sin(2.0 * math.pi * epoch / (DAY_SECONDS * 3.0))
        + _interpolated_noise(f"{seed}:t", epoch, 1800.0, 0.45)
    )
    # Relative humidity moves against temperature at constant absolute moisture.
    humidity = (
        HUMIDITY_BASELINE
        - HUMIDITY_DIURNAL_AMPLITUDE * diurnal
        + 2.5 * math.sin(2.0 * math.pi * epoch / (DAY_SECONDS * 2.0))
        + _interpolated_noise(f"{seed}:h", epoch, 1800.0, 1.8)
    )
    return SensorReading(
        recorded_at=at.replace(microsecond=0).isoformat(),
        temperature_c=round(_bound(temperature, 8.0, 44.0), 2),
        humidity_pct=round(_bound(humidity, 20.0, 98.0), 2),
        # Pressure and gas are not simulated: the application does not use
        # pressure anywhere, and the reference build has no gas sensor. Both
        # fields remain accepted and stored when real hardware reports them, so
        # a Pi sending BMP280 pressure is never rejected.
        pressure_hpa=None,
        gas_ppm=None,
        source="simulated",
    )


def _bound(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def synthesise_series(
    end: datetime, points: int, step_seconds: int, seed: str = "default"
) -> list[SensorReading]:
    """Back-fill a stable synthetic history ending at ``end``."""
    points = max(1, min(points, 2000))
    return [
        synthesise(end - timedelta(seconds=step_seconds * offset), seed)
        for offset in range(points - 1, -1, -1)
    ]


def validate_measurement(name: str, value: object) -> float | None:
    """Coerce and range-check one measurement. ``None`` means 'not reported'."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SensorValidationError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(number):
        raise SensorValidationError(f"{name} must be finite, got {value!r}")
    low, high = VALID_RANGES.get(name, (-1e9, 1e9))
    if not (low <= number <= high):
        raise SensorValidationError(
            f"{name} = {number} is outside the plausible range {low}..{high}"
        )
    return round(number, 3)


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate one incoming device reading.

    At least one measurement must be present, otherwise the row carries no
    information and is rejected rather than silently stored as all-NULL.
    """
    if not isinstance(payload, dict):
        raise SensorValidationError("Reading payload must be a JSON object")

    cleaned = {
        name: validate_measurement(name, payload.get(name))
        for name in VALID_RANGES
    }
    if all(value is None for value in cleaned.values()):
        raise SensorValidationError(
            "Reading must include at least one of: "
            + ", ".join(sorted(VALID_RANGES))
        )

    recorded_at = payload.get("recorded_at")
    if recorded_at:
        parsed = parse_ts(str(recorded_at))
        if parsed is None:
            raise SensorValidationError(f"recorded_at is not a valid timestamp: {recorded_at!r}")
        # Reject clocks from the future: a Pi with an unsynced RTC would
        # otherwise poison every chart with points that never expire.
        if parsed > utcnow() + timedelta(minutes=5):
            raise SensorValidationError("recorded_at is more than 5 minutes in the future")
        cleaned["recorded_at"] = parsed.replace(microsecond=0).isoformat()
    else:
        cleaned["recorded_at"] = iso_now()

    return cleaned
