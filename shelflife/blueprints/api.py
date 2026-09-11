"""Browser-facing JSON API.

Every response uses the same envelope so the frontend has exactly one shape to
handle::

    {"ok": true,  "data": {...}}
    {"ok": false, "error": {"code": "...", "message": "..."}}
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from typing import Any

from flask import Blueprint, current_app, jsonify, request, url_for
from werkzeug.utils import secure_filename

from .. import services
from ..db import execute, iso_now, parse_ts, utcnow
from ..security import (
    client_key,
    current_user,
    generate_device_token,
    login_required,
    rate_limiter,
)

bp = Blueprint("api", __name__, url_prefix="/api")

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# Magic bytes checked so a renamed executable cannot be stored as ".jpg".
IMAGE_SIGNATURES = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"RIFF", "webp"),
)


def ok(data: Any = None, status: int = 200):
    return jsonify({"ok": True, "data": data if data is not None else {}}), status


def fail(code: str, message: str, status: int = 400, **extra: Any):
    payload: dict[str, Any] = {"code": code, "message": message}
    payload.update(extra)
    return jsonify({"ok": False, "error": payload}), status


def body() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _int_arg(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(low, min(value, high))


# --- read endpoints -----------------------------------------------------------
@bp.get("/v1/dashboard")
@login_required
def dashboard_state():
    user = current_user()
    force = request.args.get("refresh") == "1"
    if force:
        allowed, retry_after = rate_limiter.check(client_key(f"predict:{user['id']}"), 30, 60)
        if not allowed:
            return fail("rate_limited", "Slow down a moment.", 429, retry_after=retry_after)
    return ok(services.dashboard_snapshot(user["id"], force_predict=force))


@bp.get("/v1/readings/latest")
@login_required
def latest_reading():
    user = current_user()
    return ok(services.latest_reading(user["id"]))


@bp.get("/v1/readings/history")
@login_required
def reading_history():
    user = current_user()
    hours = _int_arg("hours", 12, 1, 24 * 30)
    points = _int_arg("points", 48, 4, 300)
    return ok(services.reading_history(user["id"], hours=hours, max_points=points))


@bp.get("/v1/items")
@login_required
def list_items():
    user = current_user()
    status = request.args.get("status", "active")
    if status not in {"active", "consumed", "discarded", "all"}:
        status = "active"
    settings = services.user_settings(user["id"])
    threshold = float(settings["alert_days_before"]) * 24.0
    reading = services.latest_reading(user["id"])
    summaries = []
    for item in services.list_items(user["id"], status):
        prediction = services.refresh_item_prediction(user["id"], item, reading)
        summaries.append(services.summarise_item(user["id"], item, reading, threshold, prediction))
    return ok({"items": summaries, "reading": reading})


@bp.get("/v1/items/<int:item_id>")
@login_required
def item_detail(item_id: int):
    user = current_user()
    item = services.get_item(user["id"], item_id)
    if item is None:
        return fail("not_found", "No such item.", 404)
    settings = services.user_settings(user["id"])
    threshold = float(settings["alert_days_before"]) * 24.0
    reading = services.latest_reading(user["id"])
    prediction = services.refresh_item_prediction(user["id"], item, reading)
    return ok(
        {
            "item": services.summarise_item(user["id"], item, reading, threshold, prediction),
            "history": services.prediction_history(item_id, limit=120),
            "reading": reading,
        }
    )


# --- item mutations -------------------------------------------------------------
def _parse_stored_at(raw: str | None) -> tuple[str | None, str | None]:
    """Return ``(iso_timestamp, error_message)``."""
    if not raw:
        return iso_now(), None
    parsed = parse_ts(raw)
    if parsed is None:
        return None, "stored_at is not a valid date/time."
    if parsed > utcnow() + timedelta(minutes=5):
        return None, "stored_at cannot be in the future."
    if parsed < utcnow() - timedelta(days=365):
        return None, "stored_at cannot be more than a year ago."
    return parsed.replace(microsecond=0).isoformat(), None


@bp.post("/v1/items")
@login_required
def create_item():
    user = current_user()
    payload = body()

    label = (payload.get("label") or "").strip()[:80]
    if not label:
        return fail("invalid_label", "Give the item a name.", 422)

    try:
        food_type_id = int(payload.get("food_type_id"))
    except (TypeError, ValueError):
        return fail("invalid_food_type", "Choose a produce type.", 422)
    if services.food_type_by_id(food_type_id) is None:
        return fail("invalid_food_type", "That produce type does not exist.", 422)

    storage = (payload.get("storage") or "room").strip().lower()
    if storage not in services.STORAGE_LOCATIONS:
        return fail(
            "invalid_storage",
            f"Storage must be one of: {', '.join(services.STORAGE_LOCATIONS)}.",
            422,
        )

    stored_at, error = _parse_stored_at(payload.get("stored_at"))
    if error:
        return fail("invalid_stored_at", error, 422)

    quantity = (payload.get("quantity") or "").strip()[:40] or None
    notes = (payload.get("notes") or "").strip()[:500] or None
    image_path = (payload.get("image_path") or "").strip() or None
    if image_path and not _owned_upload(image_path):
        return fail("invalid_image", "Unknown image reference.", 422)

    item_id = services.create_item(
        user["id"], food_type_id, label, quantity, storage, stored_at, notes, image_path
    )
    item = services.get_item(user["id"], item_id)
    settings = services.user_settings(user["id"])
    threshold = float(settings["alert_days_before"]) * 24.0
    reading = services.latest_reading(user["id"])
    prediction = services.run_prediction(user["id"], item, reading)
    return ok(
        {"item": services.summarise_item(user["id"], item, reading, threshold, prediction)},
        201,
    )


@bp.patch("/v1/items/<int:item_id>")
@login_required
def patch_item(item_id: int):
    user = current_user()
    if services.get_item(user["id"], item_id) is None:
        return fail("not_found", "No such item.", 404)

    payload = body()
    fields: dict[str, Any] = {}

    if "label" in payload:
        label = (payload.get("label") or "").strip()[:80]
        if not label:
            return fail("invalid_label", "Give the item a name.", 422)
        fields["label"] = label
    if "food_type_id" in payload:
        try:
            food_type_id = int(payload["food_type_id"])
        except (TypeError, ValueError):
            return fail("invalid_food_type", "Choose a produce type.", 422)
        if services.food_type_by_id(food_type_id) is None:
            return fail("invalid_food_type", "That produce type does not exist.", 422)
        fields["food_type_id"] = food_type_id
    if "storage" in payload:
        storage = (payload.get("storage") or "").strip().lower()
        if storage not in services.STORAGE_LOCATIONS:
            return fail("invalid_storage", "Unknown storage location.", 422)
        fields["storage"] = storage
    if "stored_at" in payload:
        stored_at, error = _parse_stored_at(payload.get("stored_at"))
        if error:
            return fail("invalid_stored_at", error, 422)
        fields["stored_at"] = stored_at
    if "quantity" in payload:
        fields["quantity"] = (payload.get("quantity") or "").strip()[:40] or None
    if "notes" in payload:
        fields["notes"] = (payload.get("notes") or "").strip()[:500] or None
    if "image_path" in payload:
        image_path = (payload.get("image_path") or "").strip() or None
        if image_path and not _owned_upload(image_path):
            return fail("invalid_image", "Unknown image reference.", 422)
        fields["image_path"] = image_path

    if "status" in payload:
        status = (payload.get("status") or "").strip().lower()
        if status not in {"active", "consumed", "discarded"}:
            return fail("invalid_status", "Status must be active, consumed or discarded.", 422)
        services.close_item(user["id"], item_id, status)

    if fields:
        services.update_item(user["id"], item_id, fields)

    item = services.get_item(user["id"], item_id)
    settings = services.user_settings(user["id"])
    threshold = float(settings["alert_days_before"]) * 24.0
    reading = services.latest_reading(user["id"])
    # Anything that changes the item changes its prediction, so recompute now.
    prediction = services.refresh_item_prediction(user["id"], item, reading, force=True)
    return ok({"item": services.summarise_item(user["id"], item, reading, threshold, prediction)})


@bp.delete("/v1/items/<int:item_id>")
@login_required
def delete_item(item_id: int):
    user = current_user()
    if not services.delete_item(user["id"], item_id):
        return fail("not_found", "No such item.", 404)
    return ok({"deleted": item_id})


@bp.post("/v1/items/<int:item_id>/predict")
@login_required
def predict_item(item_id: int):
    user = current_user()
    allowed, retry_after = rate_limiter.check(client_key(f"predict:{user['id']}"), 60, 60)
    if not allowed:
        return fail("rate_limited", "Too many predictions. Wait a moment.", 429,
                    retry_after=retry_after)

    item = services.get_item(user["id"], item_id)
    if item is None:
        return fail("not_found", "No such item.", 404)

    reading = services.latest_reading(user["id"])
    prediction = services.run_prediction(user["id"], item, reading)
    settings = services.user_settings(user["id"])
    threshold = float(settings["alert_days_before"]) * 24.0
    summary = services.summarise_item(user["id"], item, reading, threshold,
                                      services.latest_prediction(item_id))
    services.evaluate_alerts(user["id"], [summary], threshold)
    return ok({"prediction": prediction, "item": summary})


# --- image upload ---------------------------------------------------------------
def _owned_upload(relative_path: str) -> bool:
    """Confirm the path points inside the upload directory and exists."""
    if "/" in relative_path or "\\" in relative_path or relative_path.startswith("."):
        return False
    return os.path.isfile(os.path.join(current_app.config["UPLOAD_DIR"], relative_path))


def _sniff_image(head: bytes) -> str | None:
    for signature, kind in IMAGE_SIGNATURES:
        if head.startswith(signature):
            if kind == "webp" and head[8:12] != b"WEBP":
                continue
            return kind
    return None


@bp.post("/v1/uploads")
@login_required
def upload_image():
    user = current_user()
    allowed, retry_after = rate_limiter.check(client_key(f"upload:{user['id']}"), 40, 300)
    if not allowed:
        return fail("rate_limited", "Too many uploads. Wait a moment.", 429, retry_after=retry_after)

    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return fail("no_file", "Attach an image file.", 422)

    extension = os.path.splitext(secure_filename(uploaded.filename))[1].lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        return fail(
            "unsupported_type",
            f"Supported image types: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}.",
            415,
        )

    head = uploaded.stream.read(16)
    uploaded.stream.seek(0)
    if _sniff_image(head) is None:
        return fail("unsupported_type", "That file is not a readable JPEG, PNG or WebP.", 415)

    stored_name = f"u{user['id']}_{utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{extension}"
    destination = os.path.join(current_app.config["UPLOAD_DIR"], stored_name)
    uploaded.save(destination)

    return ok(
        {
            "image_path": stored_name,
            "url": url_for("static", filename=f"uploads/{stored_name}"),
        },
        201,
    )


# --- alerts and settings ---------------------------------------------------------
@bp.get("/v1/alerts")
@login_required
def list_alerts():
    user = current_user()
    only_open = request.args.get("open") == "1"
    limit = _int_arg("limit", 50, 1, 200)
    return ok({"alerts": services.list_alerts(user["id"], limit=limit, only_open=only_open)})


@bp.post("/v1/alerts/<int:alert_id>/acknowledge")
@login_required
def acknowledge_alert(alert_id: int):
    user = current_user()
    if not services.acknowledge_alert(user["id"], alert_id):
        return fail("not_found", "No open alert with that id.", 404)
    return ok({"acknowledged": alert_id})


@bp.post("/v1/alerts/acknowledge-all")
@login_required
def acknowledge_all():
    user = current_user()
    return ok({"acknowledged": services.acknowledge_all_alerts(user["id"])})


@bp.get("/v1/settings")
@login_required
def get_settings():
    user = current_user()
    return ok(services.user_settings(user["id"]))


@bp.put("/v1/settings")
@login_required
def update_settings():
    user = current_user()
    payload = body()

    try:
        days = int(payload.get("alert_days_before", 2))
    except (TypeError, ValueError):
        return fail("invalid_threshold", "Reminder threshold must be a whole number of days.", 422)
    if not 1 <= days <= 30:
        return fail("invalid_threshold", "Reminder threshold must be between 1 and 30 days.", 422)

    unit = str(payload.get("temperature_unit", "C")).upper()
    if unit not in {"C", "F"}:
        return fail("invalid_unit", "Temperature unit must be C or F.", 422)

    execute(
        "UPDATE settings SET email_notifications = ?, sms_notifications = ?, "
        "alert_days_before = ?, temperature_unit = ? WHERE user_id = ?",
        (
            1 if payload.get("email_notifications") else 0,
            1 if payload.get("sms_notifications") else 0,
            days,
            unit,
            user["id"],
        ),
    )
    return ok(services.user_settings(user["id"]))


# --- devices ----------------------------------------------------------------------
@bp.get("/v1/devices")
@login_required
def list_devices():
    user = current_user()
    return ok({"devices": services.device_list(user["id"])})


@bp.post("/v1/devices")
@login_required
def create_device():
    user = current_user()
    payload = body()
    name = (payload.get("name") or "").strip()[:60]
    if not name:
        return fail("invalid_name", "Give the device a name.", 422)
    location = (payload.get("location") or "").strip()[:60] or None

    plaintext, token_hash, prefix = generate_device_token()
    cursor = execute(
        "INSERT INTO devices (user_id, name, location, token_hash, token_prefix, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user["id"], name, location, token_hash, prefix, iso_now()),
    )
    return ok(
        {
            "device": {
                "id": int(cursor.lastrowid),
                "name": name,
                "location": location,
                "token_prefix": prefix,
            },
            # Shown once and never retrievable again: only the hash is stored.
            "token": plaintext,
        },
        201,
    )


@bp.delete("/v1/devices/<int:device_id>")
@login_required
def delete_device(device_id: int):
    user = current_user()
    cursor = execute(
        "DELETE FROM devices WHERE user_id = ? AND id = ?", (user["id"], device_id)
    )
    if not (cursor.rowcount or 0):
        return fail("not_found", "No such device.", 404)
    return ok({"deleted": device_id})


@bp.get("/v1/system")
@login_required
def system_status():
    user = current_user()
    registry = current_app.extensions["predictors"]
    return ok(
        {
            "model": registry.status(),
            "sensor_mode": services.sensor_mode(user["id"]),
            "sensor_source_config": current_app.config["SENSOR_SOURCE"],
            "version": current_app.config["APP_VERSION"],
            "environment": current_app.config["ENV_NAME"],
            "server_time": iso_now(),
        }
    )


@bp.get("/v1/food-types")
@login_required
def list_food_types():
    return ok({"food_types": services.food_types()})
