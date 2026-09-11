"""Developer console. Every route here requires the admin role.

This is the second of the two interfaces: ordinary accounts never reach it, and
admins keep full access to the user-facing monitoring pages as well. It exposes
the model internals, a manual inference playground and the held-out evaluation
that the project's Streamlit demo provided.

Nothing in here retrains, refits or otherwise mutates the saved model, scaler or
split metadata.
"""

from __future__ import annotations

import io
import os
from typing import Any

from flask import (
    Blueprint, abort, current_app, jsonify, render_template, request, send_file,
)

from ..dataset import clear_cache, load_split, regression_metrics
from ..db import execute, iso_now, query_all, query_one
from ..multimodal import ModelUnavailable
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
    return load_split(
        current_app.config["DATASET_SPLITS_PATH"], current_app.config["DATASET_IMAGE_DIR"]
    )


def multimodal_or_none():
    model = getattr(registry(), "multimodal", None)
    return model if model is not None and model.available else None


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
        "keras_model_path": config["KERAS_MODEL_PATH"],
        "scaler_path": config["SCALER_PATH"],
        "model_config_path": config["MODEL_CONFIG_PATH"],
        "splits_path": config["DATASET_SPLITS_PATH"],
        "image_dir": config["DATASET_IMAGE_DIR"],
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
@bp.post("/api/predict")
@admin_required
def api_predict():
    """Manual inference: an uploaded image plus the three sensor readings."""
    user = current_user()
    allowed, retry_after = rate_limiter.check(client_key(f"admin-predict:{user['id']}"), 60, 60)
    if not allowed:
        return fail("rate_limited", "Too many inference runs. Wait a moment.", 429,
                    retry_after=retry_after)

    model = multimodal_or_none()
    if model is None:
        status = registry().status()
        return fail(
            "model_unavailable",
            status.get("trained_model_error") or "No trained model is loaded.",
            503,
        )

    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return fail("no_file", "Attach an image as the 'image' field.", 422)

    readings = {}
    for name in ("temperature", "humidity", "gas"):
        raw = request.form.get(name)
        try:
            readings[name] = float(raw)
        except (TypeError, ValueError):
            return fail("invalid_sensor", f"'{name}' must be a number.", 422)

    try:
        image = model.open_image(io.BytesIO(uploaded.read()))
    except ValueError as exc:
        return fail("bad_image", str(exc), 415)

    try:
        output = model.predict(image, readings["temperature"], readings["humidity"], readings["gas"])
    except ModelUnavailable as exc:
        return fail("model_unavailable", str(exc), 503)
    except Exception as exc:
        current_app.logger.exception("Admin playground inference failed")
        return fail("inference_failed", f"{type(exc).__name__}: {exc}", 500)

    return ok(
        {
            "predicted_days_raw": round(output.remaining_days, 4),
            # Displayed value is floored at zero; the raw output is reported too
            # so a negative regression can still be seen in the console.
            "predicted_days": round(max(0.0, output.remaining_days), 4),
            "clamped": output.remaining_days < 0,
            "inference_ms": round(output.inference_ms, 2),
            "sensor": readings,
            "model": {"name": model.name, "version": model.version},
        }
    )


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
                    "label": row["image_label"],
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


@bp.post("/api/test-samples/<int:index>/predict")
@admin_required
def api_predict_test_sample(index: int):
    model = multimodal_or_none()
    if model is None:
        status = registry().status()
        return fail("model_unavailable",
                    status.get("trained_model_error") or "No trained model is loaded.", 503)

    row = split().get(index)
    if row is None:
        return fail("not_found", "No held-out sample with that index.", 404)
    if not row["image_path"]:
        return fail("image_missing", f"Image file not found for {row['image_label']}.", 404)

    try:
        image = model.open_image(row["image_path"])
        output = model.predict(image, row["temperature"], row["humidity"], row["gas"])
    except (ValueError, ModelUnavailable) as exc:
        return fail("inference_failed", str(exc), 503)
    except Exception as exc:
        current_app.logger.exception("Held-out inference failed")
        return fail("inference_failed", f"{type(exc).__name__}: {exc}", 500)

    predicted = output.remaining_days
    return ok(
        {
            "index": index,
            "label": row["image_label"],
            "temperature": row["temperature"],
            "humidity": row["humidity"],
            "gas": row["gas"],
            "actual_days": round(row["actual"], 4),
            "predicted_days_raw": round(predicted, 4),
            "predicted_days": round(max(0.0, predicted), 4),
            "absolute_error": round(abs(row["actual"] - predicted), 4),
            "inference_ms": round(output.inference_ms, 2),
        }
    )


@bp.post("/api/evaluate")
@admin_required
def api_evaluate():
    """Run the model across the held-out split and report MAE, RMSE and R^2."""
    user = current_user()
    allowed, retry_after = rate_limiter.check(client_key(f"admin-eval:{user['id']}"), 6, 300)
    if not allowed:
        return fail("rate_limited", "Evaluation is expensive. Wait a moment.", 429,
                    retry_after=retry_after)

    model = multimodal_or_none()
    if model is None:
        status = registry().status()
        return fail("model_unavailable",
                    status.get("trained_model_error") or "No trained model is loaded.", 503)

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
    skipped: list[dict[str, Any]] = []
    total_ms = 0.0

    for index in range(limit):
        row = held_out.rows[index]
        if not row["image_path"]:
            skipped.append({"index": index, "reason": "image file missing"})
            continue
        try:
            image = model.open_image(row["image_path"])
            output = model.predict(image, row["temperature"], row["humidity"], row["gas"])
        except Exception as exc:
            skipped.append({"index": index, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        actual.append(row["actual"])
        predicted.append(output.remaining_days)
        total_ms += output.inference_ms

    metrics = regression_metrics(actual, predicted)
    evaluated = metrics["n"]
    return ok(
        {
            "metrics": metrics,
            "evaluated": evaluated,
            "requested": limit,
            "skipped": skipped[:20],
            "skipped_count": len(skipped),
            "mean_inference_ms": round(total_ms / evaluated, 2) if evaluated else None,
            "points": [
                {"actual": round(a, 4), "predicted": round(p, 4)}
                for a, p in zip(actual, predicted)
            ],
            # The figures published with the model, for comparison.
            "reported_metrics": registry().status().get("multimodal", {}).get("metrics", {}),
        }
    )


@bp.post("/api/reload-model")
@admin_required
def api_reload_model():
    """Re-read the artifacts from disk after dropping in a new export."""
    from ..inference import PredictorRegistry

    config = current_app.config
    current_app.extensions["predictors"] = PredictorRegistry(
        model_path=config.get("MODEL_PATH", ""),
        labels_path=config.get("MODEL_LABELS_PATH", ""),
        keras_model_path=config.get("KERAS_MODEL_PATH", ""),
        scaler_path=config.get("SCALER_PATH", ""),
        config_path=config.get("MODEL_CONFIG_PATH", ""),
    )
    clear_cache()
    current_app.logger.info("Model artifacts reloaded by admin id=%s", current_user()["id"])
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
