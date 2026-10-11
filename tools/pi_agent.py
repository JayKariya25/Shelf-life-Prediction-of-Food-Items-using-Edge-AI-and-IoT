#!/usr/bin/env python3
"""Raspberry Pi agent: read the sensors and camera, post them to the dashboard.

Runs on the Pi itself. Uses only the standard library for networking so it needs
no extra packages beyond the sensor drivers.

Setup on the Pi::

    sudo apt install python3-smbus i2c-tools
    pip install adafruit-circuitpython-bmp280 adafruit-circuitpython-sht31d
    export SHELFLIFE_URL=http://192.168.1.50:8000
    export SHELFLIFE_TOKEN=slp_...            # from the Device & Model page
    python3 tools/pi_agent.py

For the trained shelf-life model the website also needs, per tracked item:

* **gas readings** - set ``SHELFLIFE_GAS_COMMAND`` to a command that prints one number:
  the gas sensor's value in the *same units your training rig recorded* (roughly
  150-195 during the training run). The agent does not guess a driver or a scale,
  because a mis-scaled value silently puts every prediction outside what the model
  knows. Example: ``SHELFLIFE_GAS_COMMAND="python3 /home/pi/read_gas.py"``.
* **photos** - set ``SHELFLIFE_ITEM_ID`` to the item being watched. A frame is taken
  every ``--capture-every`` minutes (default 30) with ``SHELFLIFE_CAPTURE_COMMAND``,
  whose ``{path}`` placeholder is replaced by the file to write. By default it tries
  ``rpicam-still`` and then ``libcamera-still``.

Without the sensor libraries it exits rather than inventing numbers.

Readings are buffered to disk when the network is down and flushed as a batch
once it comes back, so a dropped Wi-Fi link does not create holes in the data. The
server tells the agent how often to report (every 2 minutes - the cadence the model
was trained on).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("pi-agent")
FIRMWARE = "pi-agent/1.1"
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


def read_gas(command: str, timeout: float = 10.0) -> float | None:
    """Run the configured gas command and parse one finite number from its output."""
    try:
        done = subprocess.run(shlex.split(command), capture_output=True, text=True,
                              timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        LOG.warning("Gas command failed: %s", exc)
        return None
    text = done.stdout.strip().split()
    try:
        value = float(text[-1]) if text else float("nan")
    except ValueError:
        value = float("nan")
    if not math.isfinite(value):
        LOG.warning("Gas command printed %r, not a number", done.stdout.strip()[:80])
        return None
    return value


def default_capture_command() -> str | None:
    for tool in ("rpicam-still", "libcamera-still"):
        if shutil.which(tool):
            return f"{tool} -n -t 1000 -o {{path}}"
    return None


def capture_frame(command: str, timeout: float = 30.0) -> Path | None:
    """Take one photo with the configured command. Returns the file, or None."""
    target = Path(tempfile.gettempdir()) / f"shelflife-{uuid.uuid4().hex}.jpg"
    try:
        subprocess.run(shlex.split(command.replace("{path}", str(target))),
                       capture_output=True, timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        LOG.warning("Capture command failed: %s", exc)
        target.unlink(missing_ok=True)
        return None
    if not target.is_file() or target.stat().st_size == 0:
        LOG.warning("Capture command produced no file")
        target.unlink(missing_ok=True)
        return None
    return target


def upload_capture(uploader: "Uploader", path: Path, item_id: str, captured_at: str) -> dict:
    """POST a frame as multipart/form-data using only the standard library."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("item_id", item_id), ("captured_at", captured_at)):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="frame.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n".encode()
        + path.read_bytes() + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        f"{uploader.base_url}/api/device/v1/captures",
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Authorization": f"Bearer {uploader.token}", "User-Agent": FIRMWARE},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=uploader.timeout * 3) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("SHELFLIFE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--token", default=os.environ.get("SHELFLIFE_TOKEN", ""))
    parser.add_argument("--interval", type=float, default=float(os.environ.get("SHELFLIFE_INTERVAL", "120")),
                        help="seconds between readings; the server's value overrides this")
    parser.add_argument("--gas-command", default=os.environ.get("SHELFLIFE_GAS_COMMAND", ""),
                        help="command that prints one gas value in training units")
    parser.add_argument("--item-id", default=os.environ.get("SHELFLIFE_ITEM_ID", ""),
                        help="tracked item to attach readings and photos to")
    parser.add_argument("--capture-command", default=os.environ.get("SHELFLIFE_CAPTURE_COMMAND", ""),
                        help="camera command with a {path} placeholder")
    parser.add_argument("--capture-every", type=float,
                        default=float(os.environ.get("SHELFLIFE_CAPTURE_EVERY_MINUTES", "30")),
                        help="minutes between photos (the model needs one under 6 hours old)")
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

    capture_command = args.capture_command or (default_capture_command() if args.item_id else None)
    if args.item_id and not capture_command:
        LOG.warning("SHELFLIFE_ITEM_ID is set but no camera command was found; photos are off")
    if not args.gas_command:
        LOG.warning("No SHELFLIFE_GAS_COMMAND: readings carry no gas value, so the website "
                    "will use its kinetic baseline instead of the trained model")
    last_capture = 0.0

    running = True

    def stop(signum, frame):  # noqa: ARG001
        nonlocal running
        LOG.info("Shutting down")
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while running:
        reading = bus.read()
        if reading and args.gas_command:
            gas = read_gas(args.gas_command)
            if gas is not None:
                reading["gas_ppm"] = gas
        if reading and args.item_id:
            reading["item_id"] = int(args.item_id)
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
                        "Sent %.2f C / %.1f%% RH / gas %s",
                        reading.get("temperature_c", float("nan")),
                        reading.get("humidity_pct", float("nan")),
                        reading.get("gas_ppm", "-"),
                    )
                except urllib.error.HTTPError as exc:
                    LOG.error("Server refused the reading (HTTP %s): %s", exc.code, exc.read()[:200])
                except (urllib.error.URLError, OSError) as exc:
                    LOG.warning("Upload failed (%s); buffering", exc)
                    buffer_reading(reading)

        if capture_command and args.item_id and not args.dry_run \
                and time.monotonic() - last_capture >= args.capture_every * 60:
            frame = capture_frame(capture_command)
            if frame is not None:
                captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                try:
                    upload_capture(uploader, frame, args.item_id, captured_at)
                    last_capture = time.monotonic()
                    LOG.info("Uploaded a photo for item %s", args.item_id)
                except (urllib.error.URLError, OSError) as exc:
                    LOG.warning("Photo upload failed (%s); will retry next cycle", exc)
                finally:
                    frame.unlink(missing_ok=True)

        if args.once:
            break
        for _ in range(int(args.interval * 10)):
            if not running:
                break
            time.sleep(0.1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
