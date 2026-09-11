"""Domain logic: sensing, items, predictions and alerting.

Design note on simulated data
-----------------------------
Synthetic readings are **not** written to the ``readings`` table. They are a
deterministic function of their timestamp (see :mod:`shelflife.sensors`), so any
window can be reproduced on demand, and the database keeps only real
measurements. That means "readings stored" on the dashboard is always a count of
genuine hardware samples, never inflated by the demo simulator.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from flask import current_app

from . import sensors
from .db import execute, iso_now, parse_ts, query_all, query_one, utcnow
from .inference import (
    FRESH,
    MODERATE,
    SPOILED,
    PredictionContext,
    status_color,
)

ITEM_STATUSES = ("active", "consumed", "discarded")
STORAGE_LOCATIONS = ("room", "fridge", "freezer", "pantry", "counter")

# Enclosed locations do not experience the ambient reading. With a single Pi
# sitting on a counter, modelling a fridge item at 25 C would be simply wrong.
# When no sensor reports from inside such a location, a documented nominal
# profile is used instead and every surface that shows it says "nominal".
# Link an item to a device (item.device_id) to use real measurements instead.
STORAGE_PROFILES: dict[str, dict[str, float]] = {
    "fridge": {"temperature_c": 4.0, "humidity_pct": 85.0},
    "freezer": {"temperature_c": -18.0, "humidity_pct": 50.0},
}

# Re-run inference automatically if the newest prediction is older than this.
PREDICTION_STALE_MINUTES = 15
# Do not raise the same alert for the same item more often than this.
ALERT_COOLDOWN_HOURS = 8


# --- reference data ----------------------------------------------------------
def food_types() -> list[dict[str, Any]]:
    rows = query_all("SELECT * FROM food_types ORDER BY category, name")
    return [dict(row) for row in rows]


def food_type_by_id(food_type_id: int) -> dict[str, Any] | None:
    row = query_one("SELECT * FROM food_types WHERE id = ?", (food_type_id,))
    return dict(row) if row else None


# --- sensing -----------------------------------------------------------------
def _online_cutoff() -> str:
    seconds = current_app.config["DEVICE_ONLINE_SECONDS"]
    return (utcnow() - timedelta(seconds=seconds)).replace(microsecond=0).isoformat()


def has_live_data(user_id: int) -> bool:
    """True when a device belonging to this user reported recently."""
    mode = current_app.config["SENSOR_SOURCE"]
    if mode == "simulated":
        return False
    row = query_one(
        "SELECT 1 FROM readings WHERE user_id = ? AND source = 'live' AND recorded_at >= ? LIMIT 1",
        (user_id, _online_cutoff()),
    )
    return row is not None


def sensor_mode(user_id: int) -> str:
    """Which source the UI should label the current numbers with."""
    configured = current_app.config["SENSOR_SOURCE"]
    if configured == "simulated":
        return "simulated"
    if has_live_data(user_id):
        return "live"
    return "offline" if configured == "live" else "simulated"


def simulator_seed(user_id: int) -> str:
    """Per-user seed so two demo accounts do not show identical traces."""
    return f"user-{user_id}"


def latest_reading(user_id: int) -> dict[str, Any]:
    """Newest environment sample, live if available else synthetic."""
    mode = sensor_mode(user_id)
    if mode == "live":
        row = query_one(
            "SELECT * FROM readings WHERE user_id = ? AND source = 'live' "
            "ORDER BY recorded_at DESC, id DESC LIMIT 1",
            (user_id,),
        )
        if row is not None:
            payload = {
                "recorded_at": row["recorded_at"],
                "temperature_c": row["temperature_c"],
                "humidity_pct": row["humidity_pct"],
                "gas_ppm": row["gas_ppm"],
                "source": "live",
            }
            payload["mode"] = "live"
            return payload
    if mode == "offline":
        return {
            "recorded_at": None,
            "temperature_c": None,
            "humidity_pct": None,
            "gas_ppm": None,
            "source": "none",
            "mode": "offline",
        }
    reading = sensors.synthesise(utcnow(), simulator_seed(user_id)).to_dict()
    reading["mode"] = "simulated"
    return reading


def device_reading(user_id: int, device_id: int) -> dict[str, Any] | None:
    """Newest live reading from one specific device, if it is still online."""
    row = query_one(
        "SELECT * FROM readings WHERE user_id = ? AND device_id = ? AND source = 'live' "
        "AND recorded_at >= ? ORDER BY recorded_at DESC, id DESC LIMIT 1",
        (user_id, device_id, _online_cutoff()),
    )
    if row is None:
        return None
    return {
        "recorded_at": row["recorded_at"],
        "temperature_c": row["temperature_c"],
        "humidity_pct": row["humidity_pct"],
        "pressure_hpa": row["pressure_hpa"],
        "gas_ppm": row["gas_ppm"],
        "source": "live",
        "mode": "live",
    }


def reading_for_item(
    user_id: int, item: dict[str, Any], ambient: dict[str, Any]
) -> dict[str, Any]:
    """The environment an individual item is actually modelled in.

    Resolution order:

    1. A device explicitly linked to the item, if it is online.
    2. A documented nominal profile, for enclosed storage with no sensor in it.
    3. The ambient reading (live or simulated).
    """
    if item.get("device_id"):
        measured = device_reading(user_id, item["device_id"])
        if measured is not None:
            measured["environment_source"] = "device"
            measured["environment_note"] = f"Measured by {item.get('device_name') or 'linked device'}"
            return measured

    profile = STORAGE_PROFILES.get((item.get("storage") or "").lower())
    if profile is not None:
        return {
            "recorded_at": ambient.get("recorded_at"),
            "temperature_c": profile["temperature_c"],
            "humidity_pct": profile["humidity_pct"],
            "gas_ppm": None,
            "source": "nominal",
            "mode": "nominal",
            "environment_source": "nominal",
            "environment_note": (
                f"Modelled at a nominal {profile['temperature_c']:.0f} °C / "
                f"{profile['humidity_pct']:.0f}% RH: no sensor reports from inside the "
                f"{item.get('storage')}. Link a device to this item to use real readings."
            ),
        }

    resolved = dict(ambient)
    resolved["environment_source"] = ambient.get("mode", "simulated")
    resolved["environment_note"] = {
        "live": "Measured by the ambient sensor",
        "simulated": "No hardware connected - values come from the built-in simulator",
        "offline": "No sensor has reported recently",
    }.get(ambient.get("mode"), "")
    return resolved


def segments_for_item(
    user_id: int, item: dict[str, Any], reading: dict[str, Any], since, until
) -> list[tuple[float, float | None, float | None, float | None]]:
    """Environment history for one item, matching how its reading was resolved."""
    if reading.get("environment_source") == "nominal":
        hours = max(0.0, (until - since).total_seconds() / 3600.0)
        if hours <= 0:
            return []
        # A nominal profile is constant, so a single segment describes it exactly.
        return [(hours, reading["temperature_c"], reading["humidity_pct"], None)]
    return environment_segments(user_id, since, until)


def reading_history(user_id: int, hours: int = 12, max_points: int = 48) -> dict[str, Any]:
    """Down-sampled environment history for the trend charts."""
    hours = max(1, min(int(hours), 24 * 30))
    max_points = max(4, min(int(max_points), 500))
    step_seconds = max(60, int(hours * 3600 / max_points))
    now = utcnow()
    mode = sensor_mode(user_id)

    if mode == "live":
        start = (now - timedelta(hours=hours)).replace(microsecond=0).isoformat()
        rows = query_all(
            """
            SELECT recorded_at, temperature_c, humidity_pct, pressure_hpa, gas_ppm
            FROM readings
            WHERE user_id = ? AND source = 'live' AND recorded_at >= ?
            ORDER BY recorded_at ASC
            """,
            (user_id, start),
        )
        points = _bucket(rows, now, hours, max_points)
        return {"mode": "live", "hours": hours, "points": points}

    if mode == "offline":
        return {"mode": "offline", "hours": hours, "points": []}

    series = sensors.synthesise_series(now, max_points, step_seconds, simulator_seed(user_id))
    return {
        "mode": "simulated",
        "hours": hours,
        "points": [reading.to_dict() for reading in series],
    }


def _bucket(rows: Iterable[Any], now: datetime, hours: int, max_points: int) -> list[dict[str, Any]]:
    """Average readings into evenly spaced buckets so charts stay readable."""
    rows = list(rows)
    if not rows:
        return []
    window = timedelta(hours=hours)
    start = now - window
    span = window.total_seconds()
    buckets: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        stamp = parse_ts(row["recorded_at"])
        if stamp is None:
            continue
        offset = (stamp - start).total_seconds()
        index = int(min(max_points - 1, max(0, offset / span * max_points)))
        slot = buckets.setdefault(index, {"temperature_c": [], "humidity_pct": [], "gas_ppm": []})
        for field in slot:
            value = row[field]
            if value is not None:
                slot[field].append(float(value))

    points: list[dict[str, Any]] = []
    for index in sorted(buckets):
        slot = buckets[index]
        stamp = start + timedelta(seconds=span * (index + 0.5) / max_points)
        point = {"recorded_at": stamp.replace(microsecond=0).isoformat(), "source": "live"}
        for field, values in slot.items():
            point[field] = round(sum(values) / len(values), 2) if values else None
        points.append(point)
    return points


def environment_segments(
    user_id: int, since: datetime, until: datetime
) -> list[tuple[float, float | None, float | None, float | None]]:
    """Build ``(duration_hours, temp, rh, gas)`` segments for the kinetic model.

    Live rows win when the device covered the window. Otherwise the deterministic
    simulator fills it. The two are never mixed inside one window, so a
    prediction is either grounded in real measurements or entirely synthetic.
    """
    if until <= since:
        return []
    total_hours = (until - since).total_seconds() / 3600.0
    mode = sensor_mode(user_id)

    if mode == "live":
        rows = query_all(
            """
            SELECT recorded_at, temperature_c, humidity_pct, gas_ppm
            FROM readings
            WHERE user_id = ? AND source = 'live' AND recorded_at >= ? AND recorded_at <= ?
            ORDER BY recorded_at ASC
            """,
            (
                user_id,
                since.replace(microsecond=0).isoformat(),
                until.replace(microsecond=0).isoformat(),
            ),
        )
        parsed = [(parse_ts(row["recorded_at"]), row) for row in rows]
        parsed = [(stamp, row) for stamp, row in parsed if stamp is not None]
        if not parsed:
            return []
        segments: list[tuple[float, float | None, float | None, float | None]] = []
        for index, (stamp, row) in enumerate(parsed):
            next_stamp = parsed[index + 1][0] if index + 1 < len(parsed) else until
            duration = (next_stamp - stamp).total_seconds() / 3600.0
            if duration <= 0:
                continue
            segments.append(
                (duration, row["temperature_c"], row["humidity_pct"], row["gas_ppm"])
            )
        return segments

    if mode == "offline":
        return []

    # Simulated: sample the deterministic series across the storage window.
    sample_count = int(max(2, min(96, math.ceil(total_hours))))
    step_hours = total_hours / sample_count
    seed = simulator_seed(user_id)
    segments = []
    for index in range(sample_count):
        midpoint = since + timedelta(hours=step_hours * (index + 0.5))
        reading = sensors.synthesise(midpoint, seed)
        segments.append((step_hours, reading.temperature_c, reading.humidity_pct, reading.gas_ppm))
    return segments


def record_live_reading(
    user_id: int, device_id: int | None, item_id: int | None, cleaned: dict[str, Any]
) -> int:
    cursor = execute(
        """
        INSERT INTO readings (
            user_id, device_id, item_id, recorded_at,
            temperature_c, humidity_pct, pressure_hpa, gas_ppm, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'live')
        """,
        (
            user_id,
            device_id,
            item_id,
            cleaned["recorded_at"],
            cleaned["temperature_c"],
            cleaned["humidity_pct"],
            cleaned["pressure_hpa"],
            cleaned["gas_ppm"],
        ),
    )
    return int(cursor.lastrowid)


# --- items -------------------------------------------------------------------
ITEM_SELECT = """
    SELECT
        i.*,
        f.key   AS food_key,
        f.name  AS food_name,
        f.emoji AS food_emoji,
        f.category AS food_category,
        f.ref_shelf_life_hours, f.ref_temperature_c, f.q10,
        f.ideal_temp_min_c, f.ideal_temp_max_c,
        f.ideal_humidity_min, f.ideal_humidity_max,
        f.notes AS food_notes,
        d.name  AS device_name
    FROM items i
    JOIN food_types f ON f.id = i.food_type_id
    LEFT JOIN devices d ON d.id = i.device_id
"""


def list_items(user_id: int, status: str | None = "active") -> list[dict[str, Any]]:
    if status and status != "all":
        rows = query_all(
            ITEM_SELECT + " WHERE i.user_id = ? AND i.status = ? ORDER BY i.stored_at DESC",
            (user_id, status),
        )
    else:
        rows = query_all(
            ITEM_SELECT + " WHERE i.user_id = ? ORDER BY i.status = 'active' DESC, i.stored_at DESC",
            (user_id,),
        )
    return [dict(row) for row in rows]


def get_item(user_id: int, item_id: int) -> dict[str, Any] | None:
    row = query_one(ITEM_SELECT + " WHERE i.user_id = ? AND i.id = ?", (user_id, item_id))
    return dict(row) if row else None


def create_item(
    user_id: int,
    food_type_id: int,
    label: str,
    quantity: str | None,
    storage: str,
    stored_at: str,
    notes: str | None = None,
    image_path: str | None = None,
    device_id: int | None = None,
) -> int:
    now = iso_now()
    cursor = execute(
        """
        INSERT INTO items (
            user_id, food_type_id, device_id, label, quantity, storage,
            stored_at, status, image_path, notes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
        """,
        (
            user_id, food_type_id, device_id, label, quantity, storage,
            stored_at, image_path, notes, now, now,
        ),
    )
    return int(cursor.lastrowid)


def update_item(user_id: int, item_id: int, fields: dict[str, Any]) -> bool:
    allowed = {"label", "quantity", "storage", "stored_at", "notes", "food_type_id", "image_path"}
    updates = {key: value for key, value in fields.items() if key in allowed}
    if not updates:
        return False
    assignments = ", ".join(f"{key} = ?" for key in updates)
    params = list(updates.values()) + [iso_now(), user_id, item_id]
    cursor = execute(
        f"UPDATE items SET {assignments}, updated_at = ? WHERE user_id = ? AND id = ?",
        params,
    )
    return (cursor.rowcount or 0) > 0


def close_item(user_id: int, item_id: int, status: str) -> bool:
    if status not in {"consumed", "discarded", "active"}:
        raise ValueError(f"Unsupported item status: {status}")
    closed_at = None if status == "active" else iso_now()
    cursor = execute(
        "UPDATE items SET status = ?, closed_at = ?, updated_at = ? WHERE user_id = ? AND id = ?",
        (status, closed_at, iso_now(), user_id, item_id),
    )
    return (cursor.rowcount or 0) > 0


def delete_item(user_id: int, item_id: int) -> bool:
    cursor = execute("DELETE FROM items WHERE user_id = ? AND id = ?", (user_id, item_id))
    return (cursor.rowcount or 0) > 0


# --- prediction ---------------------------------------------------------------
def resolve_upload(image_path: str | None) -> str | None:
    """Absolute path of a stored item photograph, or None when there is none."""
    if not image_path:
        return None
    candidate = Path(current_app.config["UPLOAD_DIR"]) / image_path
    return str(candidate) if candidate.is_file() else None


def build_context(user_id: int, item: dict[str, Any], reading: dict[str, Any]) -> PredictionContext:
    stored_at = parse_ts(item["stored_at"]) or utcnow()
    now = utcnow()
    hours_stored = max(0.0, (now - stored_at).total_seconds() / 3600.0)
    return PredictionContext(
        food_key=item["food_key"],
        food_name=item["food_name"],
        ref_shelf_life_hours=float(item["ref_shelf_life_hours"]),
        ref_temperature_c=float(item["ref_temperature_c"]),
        q10=float(item["q10"]),
        ideal_temp_min_c=item["ideal_temp_min_c"],
        ideal_temp_max_c=item["ideal_temp_max_c"],
        ideal_humidity_min=item["ideal_humidity_min"],
        ideal_humidity_max=item["ideal_humidity_max"],
        temperature_c=reading.get("temperature_c"),
        humidity_pct=reading.get("humidity_pct"),
        gas_ppm=reading.get("gas_ppm"),
        hours_stored=hours_stored,
        image_path=item.get("image_path"),
        image_full_path=resolve_upload(item.get("image_path")),
        history=segments_for_item(user_id, item, reading, stored_at, now),
    )


def run_prediction(user_id: int, item: dict[str, Any], reading: dict[str, Any] | None = None) -> dict[str, Any]:
    """Estimate freshness for one item and persist the result."""
    ambient = reading or latest_reading(user_id)
    reading = reading_for_item(user_id, item, ambient)
    registry = current_app.extensions["predictors"]
    context = build_context(user_id, item, reading)
    prediction = registry.predict(context)

    cursor = execute(
        """
        INSERT INTO predictions (
            user_id, item_id, created_at, freshness_class, freshness_confidence,
            remaining_hours, remaining_hours_low, remaining_hours_high,
            model_name, model_version, model_kind, inference_ms,
            temperature_c, humidity_pct, gas_ppm, image_path, rationale, fallback_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id, item["id"], iso_now(), prediction.freshness_class,
            prediction.freshness_confidence, prediction.remaining_hours,
            prediction.remaining_hours_low, prediction.remaining_hours_high,
            prediction.model_name, prediction.model_version, prediction.model_kind,
            prediction.inference_ms, reading.get("temperature_c"),
            reading.get("humidity_pct"), reading.get("gas_ppm"),
            item.get("image_path"), prediction.rationale, prediction.fallback_reason,
        ),
    )
    payload = prediction.to_dict()
    payload["id"] = int(cursor.lastrowid)
    payload["item_id"] = item["id"]
    payload["created_at"] = iso_now()
    payload["sensor_snapshot"] = {
        "temperature_c": reading.get("temperature_c"),
        "humidity_pct": reading.get("humidity_pct"),
        "gas_ppm": reading.get("gas_ppm"),
        "mode": reading.get("mode", reading.get("source", "unknown")),
        "environment_source": reading.get("environment_source"),
        "environment_note": reading.get("environment_note"),
    }
    return payload


def latest_prediction(item_id: int) -> dict[str, Any] | None:
    row = query_one(
        "SELECT * FROM predictions WHERE item_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
        (item_id,),
    )
    return dict(row) if row else None


def prediction_history(item_id: int, limit: int = 60) -> list[dict[str, Any]]:
    rows = query_all(
        "SELECT * FROM predictions WHERE item_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
        (item_id, max(1, min(int(limit), 500))),
    )
    return [dict(row) for row in reversed(rows)]


def refresh_item_prediction(
    user_id: int, item: dict[str, Any], reading: dict[str, Any] | None, force: bool = False
) -> dict[str, Any] | None:
    """Return a current prediction, recomputing only when it has gone stale."""
    if item["status"] != "active":
        existing = latest_prediction(item["id"])
        return existing
    existing = latest_prediction(item["id"])
    if not force and existing:
        created = parse_ts(existing["created_at"])
        if created and (utcnow() - created) < timedelta(minutes=PREDICTION_STALE_MINUTES):
            return existing
    return run_prediction(user_id, item, reading)


# --- alerting ------------------------------------------------------------------
def alert_level_for(remaining_hours: float | None, freshness_class: str, threshold_hours: float) -> str:
    """Urgency, which is deliberately separate from the freshness class.

    Freshness describes the item's condition; the alert describes how soon the
    user needs to act. A very perishable item can be genuinely "Fresh" and still
    need using within a day.
    """
    if freshness_class == SPOILED or (remaining_hours is not None and remaining_hours <= 0):
        return "critical"
    if remaining_hours is None:
        return "ok"
    if remaining_hours <= 24:
        return "critical"
    if remaining_hours <= threshold_hours:
        return "warning"
    return "ok"


def _recent_alert_exists(user_id: int, item_id: int, level: str) -> bool:
    cutoff = (utcnow() - timedelta(hours=ALERT_COOLDOWN_HOURS)).replace(microsecond=0).isoformat()
    row = query_one(
        "SELECT 1 FROM alerts WHERE user_id = ? AND item_id = ? AND level = ? AND created_at >= ? LIMIT 1",
        (user_id, item_id, level, cutoff),
    )
    return row is not None


def raise_alert(
    user_id: int, item_id: int | None, level: str, title: str, message: str,
    prediction_id: int | None = None,
) -> int | None:
    """Create an alert, unless an identical one fired inside the cooldown."""
    if item_id is not None and _recent_alert_exists(user_id, item_id, level):
        return None
    cursor = execute(
        """
        INSERT INTO alerts (user_id, item_id, prediction_id, created_at, level, title, message)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, item_id, prediction_id, iso_now(), level, title, message),
    )
    return int(cursor.lastrowid)


def evaluate_alerts(user_id: int, summaries: list[dict[str, Any]], threshold_hours: float) -> int:
    """Raise alerts for items that crossed a threshold. Returns count created."""
    created = 0
    for summary in summaries:
        if summary["status"] != "active":
            continue
        level = summary["alert_level"]
        if level == "ok":
            continue
        remaining = summary.get("remaining_hours")
        label = summary["label"]
        # The label lives in the title, so the message stays label-free and reads
        # correctly whether the item name is singular or plural.
        if level == "critical":
            title = f"{label}: high spoilage risk"
            if remaining is None or remaining <= 0:
                message = "Past its estimated shelf life. Check it and use or discard it now."
            else:
                message = (
                    f"About {format_duration(remaining)} of shelf life left. "
                    "Use or discard it now."
                )
        else:
            title = f"{label}: use soon"
            message = (
                f"About {format_duration(remaining)} of shelf life left, which is inside "
                f"your {threshold_hours / 24:.0f}-day reminder threshold."
            )
        if raise_alert(user_id, summary["id"], level, title, message, summary.get("prediction_id")):
            created += 1
    return created


def list_alerts(user_id: int, limit: int = 50, only_open: bool = False) -> list[dict[str, Any]]:
    sql = """
        SELECT a.*, i.label AS item_label, f.emoji AS food_emoji
        FROM alerts a
        LEFT JOIN items i ON i.id = a.item_id
        LEFT JOIN food_types f ON f.id = i.food_type_id
        WHERE a.user_id = ?
    """
    if only_open:
        sql += " AND a.acknowledged_at IS NULL"
    sql += " ORDER BY a.created_at DESC, a.id DESC LIMIT ?"
    rows = query_all(sql, (user_id, max(1, min(int(limit), 200))))
    return [dict(row) for row in rows]


def acknowledge_alert(user_id: int, alert_id: int) -> bool:
    cursor = execute(
        "UPDATE alerts SET acknowledged_at = ? WHERE user_id = ? AND id = ? AND acknowledged_at IS NULL",
        (iso_now(), user_id, alert_id),
    )
    return (cursor.rowcount or 0) > 0


def acknowledge_all_alerts(user_id: int) -> int:
    cursor = execute(
        "UPDATE alerts SET acknowledged_at = ? WHERE user_id = ? AND acknowledged_at IS NULL",
        (iso_now(), user_id),
    )
    return cursor.rowcount or 0


# --- presentation helpers -------------------------------------------------------
def format_duration(hours: float | None) -> str:
    if hours is None:
        return "unknown"
    if hours <= 0:
        return "no time"
    if hours < 1:
        return f"{int(hours * 60)} minutes"
    if hours < 48:
        whole = int(hours)
        return f"{whole} hour{'s' if whole != 1 else ''}"
    days = hours / 24
    return f"{days:.1f} days"


def format_remaining(hours: float | None) -> str:
    """One ready-to-render phrase, so the UI never has to assemble wording.

    Avoids the prototype's awkward "no time left" for an item that is already
    past its estimate.
    """
    if hours is None:
        return "No estimate"
    if hours <= 0:
        return "Past estimate"
    return f"{format_duration(hours)} left"


def condition_flags(item: dict[str, Any], reading: dict[str, Any]) -> list[str]:
    """Human-readable warnings about the current storage environment."""
    flags: list[str] = []
    temperature = reading.get("temperature_c")
    humidity = reading.get("humidity_pct")
    tmin, tmax = item.get("ideal_temp_min_c"), item.get("ideal_temp_max_c")
    hmin, hmax = item.get("ideal_humidity_min"), item.get("ideal_humidity_max")

    if temperature is not None and tmax is not None and temperature > tmax + 2:
        flags.append(f"Too warm: {temperature:.1f} °C vs ideal {tmin:.0f}-{tmax:.0f} °C")
    elif temperature is not None and tmin is not None and temperature < tmin - 2:
        if tmin >= 7:
            flags.append(f"Chilling risk: {temperature:.1f} °C is below the {tmin:.0f} °C threshold")
        else:
            flags.append(f"Colder than ideal: {temperature:.1f} °C vs {tmin:.0f}-{tmax:.0f} °C")
    if humidity is not None and hmax is not None and humidity > hmax + 3:
        flags.append(f"Too humid: {humidity:.0f}% vs ideal {hmin:.0f}-{hmax:.0f}%")
    elif humidity is not None and hmin is not None and humidity < hmin - 3:
        flags.append(f"Too dry: {humidity:.0f}% vs ideal {hmin:.0f}-{hmax:.0f}%")
    return flags


def summarise_item(
    user_id: int, item: dict[str, Any], reading: dict[str, Any], threshold_hours: float,
    prediction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten an item plus its newest prediction into one view model."""
    reading = reading_for_item(user_id, item, reading)
    stored_at = parse_ts(item["stored_at"])
    hours_stored = (utcnow() - stored_at).total_seconds() / 3600.0 if stored_at else 0.0

    remaining = prediction["remaining_hours"] if prediction else None
    freshness = prediction["freshness_class"] if prediction else "Unknown"
    level = alert_level_for(remaining, freshness, threshold_hours) if prediction else "ok"

    return {
        "id": item["id"],
        "label": item["label"],
        "quantity": item["quantity"],
        "storage": item["storage"],
        "status": item["status"],
        "stored_at": item["stored_at"],
        "closed_at": item["closed_at"],
        "image_path": item["image_path"],
        "notes": item["notes"],
        "food_key": item["food_key"],
        "food_name": item["food_name"],
        "food_emoji": item["food_emoji"],
        "food_category": item["food_category"],
        "ideal_temp_min_c": item["ideal_temp_min_c"],
        "ideal_temp_max_c": item["ideal_temp_max_c"],
        "ideal_humidity_min": item["ideal_humidity_min"],
        "ideal_humidity_max": item["ideal_humidity_max"],
        "hours_stored": round(hours_stored, 2),
        "days_stored": round(hours_stored / 24, 1),
        "prediction_id": prediction["id"] if prediction else None,
        "freshness_class": freshness,
        "freshness_confidence": prediction["freshness_confidence"] if prediction else None,
        "status_color": status_color(freshness) if prediction else "muted",
        "remaining_hours": round(remaining, 2) if remaining is not None else None,
        "remaining_hours_low": (
            round(prediction["remaining_hours_low"], 2)
            if prediction and prediction.get("remaining_hours_low") is not None else None
        ),
        "remaining_hours_high": (
            round(prediction["remaining_hours_high"], 2)
            if prediction and prediction.get("remaining_hours_high") is not None else None
        ),
        "remaining_text": format_duration(remaining),
        "alert_level": level,
        "model_kind": prediction["model_kind"] if prediction else None,
        "model_name": prediction["model_name"] if prediction else None,
        "rationale": prediction["rationale"] if prediction else None,
        "fallback_reason": (prediction["fallback_reason"] if prediction else None),
        "predicted_at": prediction["created_at"] if prediction else None,
        # Only warn about conditions that were actually measured. A nominal
        # profile is this application's own assumption, so flagging the user for
        # it would be telling them off for our guess.
        "condition_flags": (
            condition_flags(item, reading)
            if item["status"] == "active" and reading.get("environment_source") != "nominal"
            else []
        ),
        "environment_source": reading.get("environment_source"),
        "environment_note": reading.get("environment_note"),
        "environment_temperature_c": reading.get("temperature_c"),
        "environment_humidity_pct": reading.get("humidity_pct"),
        "remaining_display": (
            format_remaining(remaining) if prediction else "No estimate"
        ),
    }


def user_settings(user_id: int) -> dict[str, Any]:
    row = query_one("SELECT * FROM settings WHERE user_id = ?", (user_id,))
    if row is None:
        execute(
            "INSERT INTO settings (user_id, email_notifications, sms_notifications, alert_days_before) "
            "VALUES (?, 0, 0, 2)",
            (user_id,),
        )
        row = query_one("SELECT * FROM settings WHERE user_id = ?", (user_id,))
    return dict(row)


def dashboard_snapshot(user_id: int, force_predict: bool = False) -> dict[str, Any]:
    """Everything the dashboard needs, in one round trip."""
    settings = user_settings(user_id)
    threshold_hours = float(settings["alert_days_before"]) * 24.0
    reading = latest_reading(user_id)
    items = list_items(user_id, "active")

    summaries: list[dict[str, Any]] = []
    for item in items:
        prediction = refresh_item_prediction(user_id, item, reading, force=force_predict)
        summaries.append(summarise_item(user_id, item, reading, threshold_hours, prediction))

    evaluate_alerts(user_id, summaries, threshold_hours)

    counts = {"critical": 0, "warning": 0, "ok": 0}
    for summary in summaries:
        counts[summary["alert_level"]] = counts.get(summary["alert_level"], 0) + 1

    freshness_counts = {FRESH: 0, MODERATE: 0, SPOILED: 0}
    for summary in summaries:
        if summary["freshness_class"] in freshness_counts:
            freshness_counts[summary["freshness_class"]] += 1

    at_risk = sorted(
        (s for s in summaries if s["alert_level"] != "ok"),
        key=lambda s: (s["remaining_hours"] if s["remaining_hours"] is not None else 1e9),
    )

    registry = current_app.extensions["predictors"]
    live_reading_count = query_one(
        "SELECT COUNT(*) AS n FROM readings WHERE user_id = ? AND source = 'live'", (user_id,)
    )

    return {
        "reading": reading,
        "sensor_mode": reading.get("mode", "simulated"),
        "items": summaries,
        "counts": counts,
        "freshness_counts": freshness_counts,
        "at_risk": at_risk[:5],
        "open_alerts": list_alerts(user_id, limit=10, only_open=True),
        "settings": {
            "alert_days_before": settings["alert_days_before"],
            "email_notifications": bool(settings["email_notifications"]),
            "sms_notifications": bool(settings["sms_notifications"]),
            "temperature_unit": settings["temperature_unit"],
        },
        "model": registry.status(),
        "live_readings_stored": live_reading_count["n"] if live_reading_count else 0,
        "generated_at": iso_now(),
    }


def device_list(user_id: int) -> list[dict[str, Any]]:
    rows = query_all(
        "SELECT * FROM devices WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
    )
    cutoff = _online_cutoff()
    devices = []
    for row in rows:
        record = dict(row)
        record.pop("token_hash", None)
        last_seen = record.get("last_seen_at")
        record["online"] = bool(last_seen and last_seen >= cutoff)
        devices.append(record)
    return devices


def account_stats(user_id: int) -> dict[str, Any]:
    """Headline numbers for the profile page."""
    def scalar(sql: str, params: tuple = ()) -> int:
        row = query_one(sql, (user_id, *params))
        return int(row["n"]) if row and row["n"] is not None else 0

    tracked = scalar("SELECT COUNT(*) AS n FROM items WHERE user_id = ?")
    active = scalar("SELECT COUNT(*) AS n FROM items WHERE user_id = ? AND status = 'active'")
    consumed = scalar("SELECT COUNT(*) AS n FROM items WHERE user_id = ? AND status = 'consumed'")
    discarded = scalar("SELECT COUNT(*) AS n FROM items WHERE user_id = ? AND status = 'discarded'")
    closed = consumed + discarded
    return {
        "items_tracked": tracked,
        "items_active": active,
        "items_consumed": consumed,
        "items_discarded": discarded,
        # Share of finished items that were actually eaten. Undefined until at
        # least one item has been closed, rather than shown as a misleading 0%.
        "waste_rate": round(discarded / closed * 100, 1) if closed else None,
        "predictions_logged": scalar("SELECT COUNT(*) AS n FROM predictions WHERE user_id = ?"),
        "live_readings": scalar(
            "SELECT COUNT(*) AS n FROM readings WHERE user_id = ? AND source = 'live'"
        ),
        "alerts_raised": scalar("SELECT COUNT(*) AS n FROM alerts WHERE user_id = ?"),
        "devices": scalar("SELECT COUNT(*) AS n FROM devices WHERE user_id = ?"),
    }
