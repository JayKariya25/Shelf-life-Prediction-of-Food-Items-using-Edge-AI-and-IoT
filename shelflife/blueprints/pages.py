"""Server-rendered pages."""

from __future__ import annotations

from flask import Blueprint, abort, render_template, request

from .. import services
from ..db import parse_ts, utcnow
from ..security import current_user, login_required

bp = Blueprint("pages", __name__)


@bp.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    snapshot = services.dashboard_snapshot(user["id"])
    return render_template(
        "dashboard.html",
        snapshot=snapshot,
        food_types=services.food_types(),
        storage_locations=services.STORAGE_LOCATIONS,
    )


@bp.route("/items")
@login_required
def items():
    user = current_user()
    status = request.args.get("status", "active")
    if status not in {"active", "consumed", "discarded", "all"}:
        status = "active"
    settings = services.user_settings(user["id"])
    threshold_hours = float(settings["alert_days_before"]) * 24.0
    reading = services.latest_reading(user["id"])

    summaries = []
    for item in services.list_items(user["id"], status):
        prediction = services.refresh_item_prediction(user["id"], item, reading)
        summaries.append(
            services.summarise_item(user["id"], item, reading, threshold_hours, prediction)
        )

    return render_template(
        "items.html",
        items=summaries,
        status=status,
        food_types=services.food_types(),
        storage_locations=services.STORAGE_LOCATIONS,
        reading=reading,
    )


@bp.route("/items/<int:item_id>")
@login_required
def item_detail(item_id: int):
    user = current_user()
    item = services.get_item(user["id"], item_id)
    if item is None:
        abort(404)

    settings = services.user_settings(user["id"])
    threshold_hours = float(settings["alert_days_before"]) * 24.0
    reading = services.latest_reading(user["id"])
    prediction = services.refresh_item_prediction(user["id"], item, reading)
    summary = services.summarise_item(user["id"], item, reading, threshold_hours, prediction)

    stored_at = parse_ts(item["stored_at"]) or utcnow()
    history = services.prediction_history(item_id, limit=120)
    hours_span = max(6, int((utcnow() - stored_at).total_seconds() / 3600) + 1)

    return render_template(
        "item_detail.html",
        item=summary,
        raw_item=item,
        prediction=prediction,
        history=history,
        environment=services.reading_history(user["id"], hours=min(hours_span, 24 * 14), max_points=60),
        food_types=services.food_types(),
        storage_locations=services.STORAGE_LOCATIONS,
        alerts=[a for a in services.list_alerts(user["id"], limit=50) if a["item_id"] == item_id],
    )


@bp.route("/alerts")
@login_required
def alerts():
    user = current_user()
    settings = services.user_settings(user["id"])
    return render_template(
        "alerts.html",
        alerts=services.list_alerts(user["id"], limit=100),
        settings=settings,
    )


@bp.route("/device")
@login_required
def device():
    user = current_user()
    from flask import current_app

    registry = current_app.extensions["predictors"]
    return render_template(
        "device.html",
        devices=services.device_list(user["id"]),
        model=registry.status(),
        sensor_mode=services.sensor_mode(user["id"]),
        config={
            "sensor_source": current_app.config["SENSOR_SOURCE"],
            "device_online_seconds": current_app.config["DEVICE_ONLINE_SECONDS"],
            "retention_days": current_app.config["READING_RETENTION_DAYS"],
            "env_name": current_app.config["ENV_NAME"],
        },
    )


@bp.route("/profile")
@login_required
def profile():
    user = current_user()
    settings = services.user_settings(user["id"])
    stats = services.account_stats(user["id"])
    return render_template("profile.html", user=user, settings=settings, stats=stats)
