#!/usr/bin/env python3
"""Raspberry Pi agent: read the sensors, post them to the dashboard.

Runs on the Pi itself. Uses only the standard library for networking so it needs
no extra packages beyond the sensor drivers.

Setup on the Pi::

    sudo apt install python3-smbus i2c-tools
    pip install adafruit-circuitpython-bmp280 adafruit-circuitpython-sht31d
    export SHELFLIFE_URL=http://192.168.1.50:5000
    export SHELFLIFE_TOKEN=slp_...            # from the Device & Model page
    python3 tools/pi_agent.py

Without the sensor libraries installed it runs in ``--dry-run`` mode and reports
nothing, rather than inventing numbers.

Readings are buffered to disk when the network is down and flushed as a batch
once it comes back, so a dropped Wi-Fi link does not create holes in the data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("pi-agent")
FIRMWARE = "pi-agent/1.0"
BUFFER_PATH = Path.home() / ".shelflife-buffer.jsonl"
MAX_BUFFER_ROWS = 2000


class SensorBus:
    """Wraps the BMP280 and SHT31 if their drivers are importable."""

    def __init__(self) -> None:
        self.bmp280 = None
        self.sht31 = None
        self.errors: list[str] = []
        self._connect()

    def _connect(self) -> None:
        try:
            import board  # type: ignore
            import busio  # type: ignore
        except ImportError as exc:
            self.errors.append(f"CircuitPython board/busio unavailable: {exc}")
            return

        try:
            i2c = busio.I2C(board.SCL, board.SDA)
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.errors.append(f"Could not open the I2C bus: {exc}")
            return

        try:
            import adafruit_bmp280  # type: ignore

            self.bmp280 = adafruit_bmp280.Adafruit_BMP280_I2C(i2c)
        except Exception as exc:  # pragma: no cover
            self.errors.append(f"BMP280 unavailable: {exc}")

        try:
            import adafruit_sht31d  # type: ignore

            self.sht31 = adafruit_sht31d.SHT31D(i2c)
        except Exception as exc:  # pragma: no cover
            self.errors.append(f"SHT31 unavailable: {exc}")

    @property
    def available(self) -> bool:
        return self.bmp280 is not None or self.sht31 is not None

    def read(self) -> dict[str, float]:
        """Read every attached sensor. Missing values are simply omitted.

        The SHT31 is preferred for temperature because it is the more accurate
        of the two for ambient air; the BMP280 supplies pressure.
        """
        reading: dict[str, float] = {}
        if self.sht31 is not None:
            try:
                reading["temperature_c"] = round(float(self.sht31.temperature), 2)
                reading["humidity_pct"] = round(float(self.sht31.relative_humidity), 2)
            except Exception as exc:  # pragma: no cover
                LOG.warning("SHT31 read failed: %s", exc)
        if self.bmp280 is not None:
            try:
                reading["pressure_hpa"] = round(float(self.bmp280.pressure), 2)
                reading.setdefault("temperature_c", round(float(self.bmp280.temperature), 2))
            except Exception as exc:  # pragma: no cover
                LOG.warning("BMP280 read failed: %s", exc)
        return reading


class Uploader:
    def __init__(self, base_url: str, token: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": FIRMWARE,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def check(self) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/api/device/v1/config?firmware={FIRMWARE}",
            headers={"Authorization": f"Bearer {self.token}", "User-Agent": FIRMWARE},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def send(self, reading: dict) -> dict:
        return self._post("/api/device/v1/readings", {**reading, "firmware": FIRMWARE})

    def send_batch(self, readings: list[dict]) -> dict:
        return self._post(
            "/api/device/v1/readings/batch", {"readings": readings, "firmware": FIRMWARE}
        )


def buffer_reading(reading: dict) -> None:
    """Append to the on-disk buffer, trimming the oldest rows when it grows."""
    try:
        with BUFFER_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(reading) + "\n")
        lines = BUFFER_PATH.read_text(encoding="utf-8").splitlines()
        if len(lines) > MAX_BUFFER_ROWS:
            BUFFER_PATH.write_text("\n".join(lines[-MAX_BUFFER_ROWS:]) + "\n", encoding="utf-8")
    except OSError as exc:
        LOG.error("Could not buffer the reading: %s", exc)


def flush_buffer(uploader: Uploader) -> None:
    if not BUFFER_PATH.exists():
        return
    try:
        rows = [json.loads(line) for line in BUFFER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        LOG.error("Buffer unreadable, discarding it: %s", exc)
        BUFFER_PATH.unlink(missing_ok=True)
        return
    if not rows:
        BUFFER_PATH.unlink(missing_ok=True)
        return

    LOG.info("Flushing %d buffered readings", len(rows))
    for start in range(0, len(rows), 200):
        chunk = rows[start:start + 200]
        try:
            result = uploader.send_batch(chunk)
            LOG.info("Batch accepted: %s", result.get("data", {}))
        except (urllib.error.URLError, OSError) as exc:
            LOG.warning("Buffer flush interrupted: %s", exc)
            return
    BUFFER_PATH.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("SHELFLIFE_URL", "http://127.0.0.1:5000"))
    parser.add_argument("--token", default=os.environ.get("SHELFLIFE_TOKEN", ""))
    parser.add_argument("--interval", type=float, default=float(os.environ.get("SHELFLIFE_INTERVAL", "30")))
    parser.add_argument("--once", action="store_true", help="Send a single reading and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Read sensors and print, do not upload.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    if not args.token and not args.dry_run:
        LOG.error("No device token. Set SHELFLIFE_TOKEN or pass --token.")
        return 2

    bus = SensorBus()
    for problem in bus.errors:
        LOG.warning("%s", problem)
    if not bus.available:
        LOG.error(
            "No sensors detected. This agent will not invent readings - install the "
            "drivers and check the wiring with 'i2cdetect -y 1'."
        )
        return 3

    uploader = None
    if not args.dry_run:
        uploader = Uploader(args.url, args.token)
        try:
            config = uploader.check()["data"]
            LOG.info("Connected to %s as device '%s'", args.url, config["device"]["name"])
            if config.get("report_interval_seconds"):
                args.interval = max(5.0, float(config["report_interval_seconds"]))
        except urllib.error.HTTPError as exc:
            LOG.error("Server rejected the token (HTTP %s). Re-register the device.", exc.code)
            return 4
        except (urllib.error.URLError, OSError) as exc:
            LOG.warning("Server unreachable at startup (%s); buffering locally.", exc)

    running = True

    def stop(signum, frame):  # noqa: ARG001
        nonlocal running
        LOG.info("Shutting down")
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while running:
        reading = bus.read()
        if not reading:
            LOG.warning("Sensor read produced nothing; skipping this cycle")
        else:
            reading["recorded_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            if args.dry_run:
                LOG.info("DRY RUN %s", reading)
            else:
                try:
                    flush_buffer(uploader)
                    uploader.send(reading)
                    LOG.info(
                        "Sent %.1f C / %.0f%% RH",
                        reading.get("temperature_c", float("nan")),
                        reading.get("humidity_pct", float("nan")),
                    )
                except urllib.error.HTTPError as exc:
                    LOG.error("Server refused the reading (HTTP %s): %s", exc.code, exc.read()[:200])
                except (urllib.error.URLError, OSError) as exc:
                    LOG.warning("Upload failed (%s); buffering", exc)
                    buffer_reading(reading)

        if args.once:
            break
        for _ in range(int(args.interval * 10)):
            if not running:
                break
            time.sleep(0.1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
