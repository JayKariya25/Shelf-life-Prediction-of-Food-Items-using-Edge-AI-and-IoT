"""Device ingest API used by the Raspberry Pi agent.

Authenticated with a bearer token rather than a session cookie, so it is CSRF
exempt by construction. Endpoints are deliberately small and forgiving about
extra fields: firmware on an edge node is harder to update than the server.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from .. import services
from ..db import iso_now
from ..security import authenticate_device, client_key, rate_limiter, touch_device
from ..sensors import SensorValidationError, validate_payload

bp = Blueprint("device", __name__, url_prefix="/api/device")

MAX_IMAGE_BYTES = 6 * 1024 * 1024
IMAGE_SIGNATURES = ((b"\xff\xd8\xff", ".jpg"), (b"\x89PNG\r\n\x1a\n", ".png"))


def ok(data: Any = None, status: int = 200):
    return jsonify({"ok": True, "data": data or {}}), status


def fail(code: str, message: str, status: int = 400, **extra: Any):
    payload = {"code": code, "message": message}
    payload.update(extra)
    return jsonify({"ok": False, "error": payload}), status


def _require_device():
    """Return ``(device, error_response)``; exactly one is not None."""
    allowed, retry_after = rate_limiter.check(client_key("device"), 600, 60)
    if not allowed:
        return None, fail("rate_limited", "Too many device requests.", 429,
                          retry_after=retry_after)
    device = authenticate_device()
    if device is None:
        return None, fail(
            "unauthorised",
            "Send a valid device token as 'Authorization: Bearer <token>'.",
            401,
        )
    return device, None


@bp.post("/v1/readings")
def ingest_reading():
    """Store one environment sample from the Pi.

    Body::

        {"temperature_c": 24.8, "humidity_pct": 61.2,
         "pressure_hpa": 1008.1, "recorded_at": "2026-08-25T10:30:00Z",
         "item_id": 7, "firmware": "pi-agent 1.0"}

    Only ``recorded_at`` and one measurement are required.
    """
    device, error = _require_device()
    if error:
        return error

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return fail("invalid_body", "Send a JSON object.", 400)

    try:
        cleaned = validate_payload(payload)
    except SensorValidationError as exc:
        return fail("invalid_reading", str(exc), 422)

    item_id = payload.get("item_id")
    if item_id is not None:
        try:
            item_id = int(item_id)
        except (TypeError, ValueError):
            return fail("invalid_item", "item_id must be an integer.", 422)
        if services.get_item(device["user_id"], item_id) is None:
            return fail("invalid_item", "That item does not belong to this device's owner.", 404)

    reading_id = services.record_live_reading(device["user_id"], device["id"], item_id, cleaned)
    touch_device(device["id"], payload.get("firmware"))

    return ok({"reading_id": reading_id, "recorded_at": cleaned["recorded_at"]}, 201)


@bp.post("/v1/readings/batch")
def ingest_batch():
    """Store several samples at once, for an agent that buffered while offline.

    Valid rows are stored even if some rows are rejected, so one bad sample does
    not cost the device its whole buffer. The response reports both counts.
    """
    device, error = _require_device()
    if error:
        return error

    payload = request.get_json(silent=True)
    rows = payload.get("readings") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        return fail("invalid_body", "Send {\"readings\": [...]} with at least one row.", 400)
    if len(rows) > 500:
        return fail("too_many", "Send at most 500 readings per batch.", 413)

    stored, rejected = 0, []
    for index, row in enumerate(rows):
        try:
            cleaned = validate_payload(row if isinstance(row, dict) else {})
        except SensorValidationError as exc:
            rejected.append({"index": index, "reason": str(exc)})
            continue
        services.record_live_reading(device["user_id"], device["id"], None, cleaned)
        stored += 1

    touch_device(device["id"], (payload or {}).get("firmware"))
    return ok({"stored": stored, "rejected": rejected}, 201 if stored else 422)


@bp.post("/v1/captures")
def upload_capture():
    """Accept a camera frame from the device and attach it to an item."""
    device, error = _require_device()
    if error:
        return error

    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return fail("no_file", "Attach the frame as multipart field 'image'.", 422)

    head = uploaded.stream.read(16)
    uploaded.stream.seek(0)
    extension = next((ext for sig, ext in IMAGE_SIGNATURES if head.startswith(sig)), None)
    if extension is None:
        return fail("unsupported_type", "Capture must be a JPEG or PNG.", 415)

    stored_name = f"dev{device['id']}_{iso_now().replace(':', '').replace('-', '')}_{uuid.uuid4().hex[:6]}{extension}"
    destination = os.path.join(current_app.config["UPLOAD_DIR"], stored_name)
    uploaded.save(destination)

    if os.path.getsize(destination) > MAX_IMAGE_BYTES:
        os.remove(destination)
        return fail("payload_too_large", "Capture exceeds the size limit.", 413)

    item_id = request.form.get("item_id")
    attached_to = None
    if item_id:
        try:
            item_id = int(item_id)
        except ValueError:
            return fail("invalid_item", "item_id must be an integer.", 422)
        if services.get_item(device["user_id"], item_id) is not None:
            services.update_item(device["user_id"], item_id, {"image_path": stored_name})
            attached_to = item_id

    touch_device(device["id"])
    return ok({"image_path": stored_name, "attached_to_item": attached_to}, 201)


@bp.get("/v1/config")
def device_heartbeat():
    """Let the agent confirm its token and learn the expected reporting cadence."""
    device, error = _require_device()
    if error:
        return error
    touch_device(device["id"], request.args.get("firmware"))
    return ok(
        {
            "device": {"id": device["id"], "name": device["name"], "location": device["location"]},
            "report_interval_seconds": current_app.config["SIMULATOR_INTERVAL_SECONDS"],
            "online_window_seconds": current_app.config["DEVICE_ONLINE_SECONDS"],
            "server_time": iso_now(),
            "server_version": current_app.config["APP_VERSION"],
        }
    )
