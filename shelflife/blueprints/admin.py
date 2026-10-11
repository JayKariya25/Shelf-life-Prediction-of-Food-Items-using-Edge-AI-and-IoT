"""Developer console. Every route here requires the admin role.

This is the second of the two interfaces: ordinary accounts never reach it, and
admins keep full access to the user-facing monitoring pages as well. It exposes
the model internals, a manual inference playground and the held-out evaluation
that the project's Streamlit demo provided.

Nothing in here retrains, refits or otherwise mutates the saved model, scaler or
split metadata.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from flask import (
    Blueprint, abort, current_app, jsonify, render_template, request, send_file,
)

from ..dataset import clear_cache, load_split, regression_metrics
from ..db import execute, iso_now, query_all, query_one
from ..rsl_model import ModelUnavailable, Reading
from ..security import ROLES, admin_required, client_key, current_user, rate_limiter

bp = Blueprint("admin", __name__, url_prefix="/admin")

MAX_EVAL_SAMPLES = 500


# --- helpers -----------------------------------------------------------------
def ok(data: Any = None, status: int = 200):
    return jsonify({"ok": True, "data": data if data is not None else {}}), status


def fail(code: str, message: str, status: int = 400, **extra: Any):
    payload: dict[str, Any] = {"code": code, "message": message}
    payload.update(extra)
    return jsonify({"ok": False, "error": payload}), status


def registry():
    return current_app.extensions["predictors"]


def split():
    return load_split(current_app.config["RSL_BUNDLE_DIR"])


def rsl_or_none():
    model = getattr(registry(), "rsl", None)
    return model if model is not None and model.available else None


def model_unavailable_response():
    status = registry().status()
    return fail("model_unavailable",
                status.get("trained_model_error") or "No trained model is loaded.", 503)


# --- playground sessions -------------------------------------------------------
# The fusion export's sensor features need a reading history, so the playground keeps one
# per (admin, session name), in memory, like serve.py. It never touches tracked items.
# The image-only ensemble has no such state, so its sessions carry no builder - the count
# is still kept, because the page shows the readings that were entered alongside it.
PLAYGROUND_TTL_SECONDS = 6 * 3600
_playground: dict[tuple[int, str], dict] = {}
_playground_lock = threading.Lock()


def _playground_entry(user_id: int, name: str, model, reset: bool = False) -> dict:
    now = time.time()
    with _playground_lock:
        for key in [k for k, v in _playground.items() if now - v["seen"] > PLAYGROUND_TTL_SECONDS]:
            _playground.pop(key, None)
        key = (user_id, name)
        entry = _playground.get(key)
        if entry is None or reset or entry["bundle"] != str(model.model_path):
            entry = {"builder": None if model.image_only else model.new_builder(),
                     "count": 0, "seen": now,
                     "bundle": str(model.model_path), "lock": threading.Lock()}
            _playground[key] = entry
        entry["seen"] = now
        return entry


def clear_playground() -> None:
    with _playground_lock:
        _playground.clear()


# --- pages -------------------------------------------------------------------
@bp.route("/")
@admin_required
def console():
    model = registry()
    return render_template(
        "admin/console.html",
        model=model.status(),
        dataset=split().summary(),
        stats=platform_stats(),
        runtime=runtime_info(),
    )


@bp.route("/model")
@admin_required
def model_card():
    return render_template(
        "admin/model.html",
        model=registry().status(),
        dataset=split().summary(),
        runtime=runtime_info(),
    )


@bp.route("/playground")
@admin_required
def playground():
    status = registry().status()
    return render_template("admin/playground.html", model=status)


@bp.route("/evaluation")
@admin_required
def evaluation():
    held_out = split()
    status = registry().status()
    return render_template(
        "admin/evaluation.html",
        model=status,
        dataset=held_out.summary(),
        sample_count=len(held_out),
        max_samples=MAX_EVAL_SAMPLES,
    )


@bp.route("/users")
@admin_required
def users():
    rows = query_all(
        """
        SELECT u.id, u.email, u.mobile, u.full_name, u.role, u.is_active,
               u.created_at, u.last_login_at,
               (SELECT COUNT(*) FROM items i WHERE i.user_id = u.id) AS item_count,
               (SELECT COUNT(*) FROM devices d WHERE d.user_id = u.id) AS device_count
        FROM users u ORDER BY u.id
        """
    )
    return render_template("admin/users.html", users=[dict(r) for r in rows], roles=ROLES)


@bp.route("/system")
@admin_required
def system():
    return render_template(
        "admin/system.html",
        runtime=runtime_info(),
        stats=platform_stats(),
        model=registry().status(),
        dataset=split().summary(),
    )


# --- shared info -------------------------------------------------------------
def runtime_info() -> dict[str, Any]:
    import platform
    import sys

    config = current_app.config
    return {
        "environment": config["ENV_NAME"],
        "app_version": config["APP_VERSION"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sensor_source": config["SENSOR_SOURCE"],
        "database_path": config["DATABASE_PATH"],
        "upload_dir": config["UPLOAD_DIR"],
        "retention_days": config["READING_RETENTION_DAYS"],
        "bundle_dir": config["RSL_BUNDLE_DIR"],
        "model_file": config["RSL_MODEL_FILE"],
        "max_photo_age_hours": config["RSL_MAX_PHOTO_AGE_HOURS"],
        "min_reading_spacing_seconds": config["RSL_MIN_READING_SPACING_SECONDS"],
        "device_report_interval_seconds": config["DEVICE_REPORT_INTERVAL_SECONDS"],
        "server_time": iso_now(),
    }


def platform_stats() -> dict[str, Any]:
    def count(sql: str) -> int:
        row = query_one(sql)
        return int(row["n"]) if row and row["n"] is not None else 0

    return {
        "users": count("SELECT COUNT(*) AS n FROM users"),
        "admins": count("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'"),
        "items": count("SELECT COUNT(*) AS n FROM items"),
        "active_items": count("SELECT COUNT(*) AS n FROM items WHERE status = 'active'"),
        "predictions": count("SELECT COUNT(*) AS n FROM predictions"),
        "trained_predictions": count(
            "SELECT COUNT(*) AS n FROM predictions WHERE model_kind = 'trained-model'"
        ),
        "baseline_predictions": count(
            "SELECT COUNT(*) AS n FROM predictions WHERE model_kind = 'heuristic-baseline'"
        ),
        "readings": count("SELECT COUNT(*) AS n FROM readings WHERE source = 'live'"),
        "alerts": count("SELECT COUNT(*) AS n FROM alerts"),
        "devices": count("SELECT COUNT(*) AS n FROM devices"),
    }


# --- admin API ---------------------------------------------------------------
def _float_field(name: str) -> float:
    raw = request.form.get(name)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number.")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"'{name}' must be finite.")
    return value


@bp.post("/api/predict")
@admin_required
def api_predict():
    """Manual inference: an uploaded photo plus one Temperature/Humidity/Gas reading.

    Readings sent with the same ``session`` build up rolling history exactly as a
    monitored batch would; ``reset=1`` starts that session over.
    """
    user = current_user()
    allowed, retry_after = rate_limiter.check(client_key(f"admin-predict:{user['id']}"), 60, 60)
    if not allowed:
        return fail("rate_limited", "Too many inference runs. Wait a moment.", 429,
                    retry_after=retry_after)

    model = rsl_or_none()
    if model is None:
        return model_unavailable_response()

    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return fail("no_file", "Attach an image as the 'image' field.", 422)

    try:
        temperature = _float_field("temperature")
        humidity = _float_field("humidity")
        gas = _float_field("gas")
        minutes = float(request.form.get("minutes_since_previous") or 2.0)
    except ValueError as exc:
        return fail("invalid_sensor", str(exc), 422)
    if not 0 <= minutes <= 24 * 60:
        return fail("invalid_sensor", "minutes_since_previous must be between 0 and 1440.", 422)

    session_name = (request.form.get("session") or "playground").strip()[:40] or "playground"
    reset = request.form.get("reset") in {"1", "true", "on"}

    try:
        image = model.load_image_bytes(uploaded.read())
    except ValueError as exc:
        return fail("bad_image", str(exc), 415)

    entry = _playground_entry(user["id"], session_name, model, reset=reset)
    with entry["lock"]:
        builder = entry["builder"]
        # A failed call must not leave half-applied readings in the session, so the
        # builder is restored from the snapshot taken before it.
        snapshot = builder.state_dict() if builder is not None else None
        hours = 0.0 if entry["count"] == 0 else minutes / 60.0

        def restore() -> None:
            if builder is not None:
                entry["builder"] = model.builder_from_state(snapshot)

        try:
            result = model.predict(image, [Reading(temperature, humidity, gas, hours)],
                                   builder=builder)
        except ModelUnavailable as exc:
            restore()
            return fail("model_unavailable", str(exc), 503)
        except Exception as exc:
            restore()
            current_app.logger.exception("Admin playground inference failed")
            return fail("inference_failed", f"{type(exc).__name__}: {exc}", 500)
        entry["count"] += 1
        reading_number = entry["count"]

    warnings = list(result.warnings)
    if model.image_only:
        warnings.insert(0, "The served model reads the photograph only, so the readings "
                           "entered here are recorded for comparison but do not change "
                           "this estimate.")
    elif reading_number == 1:
        warnings.insert(0, "First reading of this session: no rolling history yet, so the "
                           "estimate leans almost entirely on the photo.")

    return ok(
        {
            "predicted_days": round(result.days, 4),
            "predicted_days_raw": round(result.raw_days, 4),
            "clamped": result.days != result.raw_days,
            "inference_ms": round(result.inference_ms, 2),
            "decoder": result.decoder,
            "session": session_name,
            "reading_number": reading_number,
            "sensor": {"temperature": temperature, "humidity": humidity, "gas": gas,
                       "minutes_since_previous": minutes if reading_number > 1 else 0.0},
            "out_of_envelope": result.out_of_envelope,
            "warnings": warnings,
            "new_batch_mae_days": model.new_batch_mae_days,
            "model": {"name": model.name, "version": model.version},
        }
    )


@bp.post("/api/playground/reset")
@admin_required
def api_playground_reset():
    user = current_user()
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("session") or "playground").strip()[:40] or "playground"
    with _playground_lock:
        existed = _playground.pop((user["id"], name), None) is not None
    return ok({"session": name, "existed": existed})


@bp.get("/api/test-samples")
@admin_required
def api_test_samples():
    held_out = split()
    if not held_out.available:
        return fail("dataset_unavailable", held_out.error or "No held-out split.", 503)
    return ok(
        {
            "count": len(held_out),
            "samples": [
                {
                    "index": index,
                    "label": row["image_file"],
                    "day": row["day"],
                    "timestamp": row["timestamp"],
                    "temperature": row["temperature"],
                    "humidity": row["humidity"],
                    "gas": row["gas"],
                    "actual_days": row["actual"],
                    "has_image": bool(row["image_path"]),
                }
                for index, row in enumerate(held_out.rows)
            ],
        }
    )


@bp.get("/api/test-samples/<int:index>/image")
@admin_required
def api_test_image(index: int):
    """Serve a held-out image *by index*.

    The client never supplies a path, so this cannot be turned into an arbitrary
    file read; the path is resolved server-side from the split metadata.
    """
    row = split().get(index)
    if row is None:
        abort(404)
    path = row["image_path"]
    if not path or not os.path.isfile(path):
        abort(404)
    return send_file(path, max_age=300)


def _score_row(model, held_out, row) -> tuple[float, float]:
    """(raw prediction, inference ms) for a held-out row.

    The fusion export is fed the row's stored sensor features; the image-only ensemble
    takes the photograph on its own.
    """
    image = model.load_image_path(row["image_path"])
    scaled = (None if model.image_only
              else model.scale(held_out.vector(row, model.feature_names)))
    started = time.perf_counter()
    raw = model.predict_scaled(image.pixels, scaled)
    return raw, (time.perf_counter() - started) * 1000.0


@bp.post("/api/test-samples/<int:index>/predict")
@admin_required
def api_predict_test_sample(index: int):
    model = rsl_or_none()
    if model is None:
        return model_unavailable_response()

    held_out = split()
    row = held_out.get(index)
    if row is None:
        return fail("not_found", "No held-out sample with that index.", 404)
    if not row["image_path"]:
        return fail("image_missing", f"Image file not found for {row['image_file']}.", 404)

    try:
        raw, elapsed = _score_row(model, held_out, row)
    except (ValueError, KeyError, ModelUnavailable) as exc:
        return fail("inference_failed", str(exc), 503)
    except Exception as exc:
        current_app.logger.exception("Held-out inference failed")
        return fail("inference_failed", f"{type(exc).__name__}: {exc}", 500)

    clamped = min(max(raw, 0.0), model.max_days)
    return ok(
        {
            "index": index,
            "label": row["image_file"],
            "day": row["day"],
            "temperature": row["temperature"],
            "humidity": row["humidity"],
            "gas": row["gas"],
            "actual_days": round(row["actual"], 4),
            "predicted_days_raw": round(raw, 4),
            "predicted_days": round(clamped, 4),
            "absolute_error": round(abs(row["actual"] - raw), 4),
            "inference_ms": round(elapsed, 2),
        }
    )


@bp.post("/api/evaluate")
@admin_required
def api_evaluate():
    """Score the held-out split and compare with the figures published at export."""
    user = current_user()
    allowed, retry_after = rate_limiter.check(client_key(f"admin-eval:{user['id']}"), 6, 300)
    if not allowed:
        return fail("rate_limited", "Evaluation is expensive. Wait a moment.", 429,
                    retry_after=retry_after)

    model = rsl_or_none()
    if model is None:
        return model_unavailable_response()

    held_out = split()
    if not held_out.available:
        return fail("dataset_unavailable", held_out.error or "No held-out split.", 503)

    payload = request.get_json(silent=True) or {}
    try:
        limit = int(payload.get("limit", MAX_EVAL_SAMPLES))
    except (TypeError, ValueError):
        limit = MAX_EVAL_SAMPLES
    limit = max(1, min(limit, MAX_EVAL_SAMPLES, len(held_out)))

    actual: list[float] = []
    predicted: list[float] = []
    points: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total_ms = 0.0

    for index in range(limit):
        row = held_out.rows[index]
        if not row["image_path"]:
            skipped.append({"index": index, "reason": "image file missing"})
            continue
        try:
            raw, elapsed = _score_row(model, held_out, row)
        except Exception as exc:
            skipped.append({"index": index, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        actual.append(row["actual"])
        predicted.append(raw)
        points.append({"actual": round(row["actual"], 4), "predicted": round(raw, 4),
                       "day": row["day"]})
        total_ms += elapsed

    metrics = regression_metrics(actual, predicted)
    evaluated = metrics["n"]
    published = model.metrics.get("served_variant") or {}
    return ok(
        {
            "metrics": metrics,
            "evaluated": evaluated,
            "requested": limit,
            "complete": evaluated == len(held_out),
            "skipped": skipped[:20],
            "skipped_count": len(skipped),
            "mean_inference_ms": round(total_ms / evaluated, 2) if evaluated else None,
            "points": points,
            "published": {
                "variant": model.metrics.get("variant"),
                "mae": published.get("MAE"),
                "rmse": published.get("RMSE"),
                "r2": published.get("R2"),
                "n": published.get("n"),
            },
            "new_batch_mae_days": model.new_batch_mae_days,
        }
    )


@bp.post("/api/reload-model")
@admin_required
def api_reload_model():
    """Re-read the model bundle from disk after a retrain, without a restart."""
    from .. import build_registry

    current_app.extensions["predictors"] = build_registry(current_app.config)
    clear_cache()
    clear_playground()
    current_app.logger.info("Model bundle reloaded by admin id=%s", current_user()["id"])
    return ok({"model": registry().status(), "dataset": split().summary()})


@bp.patch("/api/users/<int:user_id>/role")
@admin_required
def api_set_role(user_id: int):
    payload = request.get_json(silent=True) or {}
    role = str(payload.get("role", "")).strip().lower()
    if role not in ROLES:
        return fail("invalid_role", f"Role must be one of: {', '.join(ROLES)}.", 422)

    target = query_one("SELECT id, role FROM users WHERE id = ?", (user_id,))
    if target is None:
        return fail("not_found", "No such user.", 404)

    # Never allow the last admin to be demoted; the console would become
    # unreachable and only a database edit could restore it.
    if target["role"] == "admin" and role != "admin":
        remaining = query_one(
            "SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND id != ?", (user_id,)
        )
        if not remaining or remaining["n"] == 0:
            return fail(
                "last_admin",
                "This is the only administrator. Promote another account first.",
                409,
            )

    execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    current_app.logger.info(
        "Admin id=%s set role of user id=%s to %s", current_user()["id"], user_id, role
    )
    return ok({"id": user_id, "role": role})


@bp.patch("/api/users/<int:user_id>/active")
@admin_required
def api_set_active(user_id: int):
    payload = request.get_json(silent=True) or {}
    active = bool(payload.get("is_active"))
    if user_id == current_user()["id"] and not active:
        return fail("self_disable", "You cannot deactivate your own account.", 409)
    target = query_one("SELECT id, role FROM users WHERE id = ?", (user_id,))
    if target is None:
        return fail("not_found", "No such user.", 404)
    if target["role"] == "admin" and not active:
        remaining = query_one(
            "SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND is_active = 1 AND id != ?",
            (user_id,),
        )
        if not remaining or remaining["n"] == 0:
            return fail("last_admin", "This is the only active administrator.", 409)
    execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if active else 0, user_id))
    return ok({"id": user_id, "is_active": active})


@bp.post("/api/prune-readings")
@admin_required
def api_prune():
    from ..db import get_db, prune_readings

    payload = request.get_json(silent=True) or {}
    try:
        days = int(payload.get("days", current_app.config["READING_RETENTION_DAYS"]))
    except (TypeError, ValueError):
        return fail("invalid_days", "days must be a whole number.", 422)
    if days < 1:
        return fail("invalid_days", "days must be at least 1.", 422)
    removed = prune_readings(get_db(), days)
    return ok({"removed": removed, "days": days})
