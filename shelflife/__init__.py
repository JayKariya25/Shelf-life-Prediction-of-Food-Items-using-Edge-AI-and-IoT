"""Application factory for the Edge-AI Shelf-Life Prediction web app."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from flask import Flask, current_app, jsonify, redirect, render_template, request

from . import db as db_module
from .config import get_config
from .inference import PredictorRegistry
from .security import csrf_protect, csrf_token, current_user, is_admin

__version__ = "2.0.0"

# Every endpoint in the "device" blueprint is exempt by prefix (see
# security.csrf_protect). This set is for any additional one-off exemptions.
CSRF_EXEMPT_ENDPOINTS: frozenset[str] = frozenset()

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Cross-Origin-Opener-Policy": "same-origin",
    # Everything is served from this origin; there are no CDN dependencies, so
    # the policy can stay strict. 'unsafe-inline' is allowed for style
    # attributes only (chart bars set their own width/height).
    "Content-Security-Policy": (
        "default-src 'self'; "
        # blob: is needed for client-side previews of a file the user just
        # picked; such URLs are same-origin and created by the page itself.
        "img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "form-action 'self'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    ),
}


def create_app(config_name: str | None = None, **overrides: Any) -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    config_class = get_config(config_name)
    app.config.from_mapping(config_class.build())
    app.config["CSRF_EXEMPT_ENDPOINTS"] = CSRF_EXEMPT_ENDPOINTS
    app.config["APP_VERSION"] = __version__
    app.config.update(overrides)

    _configure_logging(app)

    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    os.makedirs(app.config["MODEL_DIR"], exist_ok=True)

    report = db_module.init_db(app.config["DATABASE_PATH"])
    if report.get("migrated_columns"):
        app.logger.info("Schema migrated: added %s", ", ".join(report["migrated_columns"]))
    if report.get("seeded_food_types"):
        app.logger.info("Seeded %d food reference profiles", report["seeded_food_types"])

    app.extensions["predictors"] = PredictorRegistry(
        model_path=app.config.get("MODEL_PATH", ""),
        labels_path=app.config.get("MODEL_LABELS_PATH", ""),
        keras_model_path=app.config.get("KERAS_MODEL_PATH", ""),
        scaler_path=app.config.get("SCALER_PATH", ""),
        config_path=app.config.get("MODEL_CONFIG_PATH", ""),
    )
    status = app.extensions["predictors"].status()
    app.logger.info(
        "Active estimator: %s v%s (%s)",
        status["active_model"], status["active_version"], status["active_kind"],
    )

    app.teardown_appcontext(db_module.close_db)
    app.before_request(_force_https)
    app.before_request(csrf_protect)
    app.after_request(_apply_security_headers)

    _register_blueprints(app)
    _register_error_handlers(app)
    _register_template_helpers(app)
    _register_cli(app)

    return app


def _configure_logging(app: Flask) -> None:
    if app.config.get("TESTING"):
        # Keep test output readable; failures still surface via assertions.
        logging.basicConfig(level=logging.ERROR)
        app.logger.setLevel(logging.ERROR)
        return
    level = logging.DEBUG if app.config.get("DEBUG") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    app.logger.setLevel(level)


def _force_https():
    if not current_app.config.get("FORCE_HTTPS"):
        return None
    if request.is_secure or request.headers.get("X-Forwarded-Proto", "") == "https":
        return None
    if request.method not in {"GET", "HEAD"}:
        return jsonify({"ok": False, "error": {"code": "https_required",
                                               "message": "HTTPS is required."}}), 403
    return redirect(request.url.replace("http://", "https://", 1), code=308)


def _apply_security_headers(response):
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    if current_app.config.get("FORCE_HTTPS"):
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
    # API responses must never be cached by an intermediary.
    if "/api/" in request.path:
        response.headers.setdefault("Cache-Control", "no-store")
    return response


def _register_blueprints(app: Flask) -> None:
    from .blueprints.admin import bp as admin_bp
    from .blueprints.api import bp as api_bp
    from .blueprints.auth import bp as auth_bp
    from .blueprints.device import bp as device_bp
    from .blueprints.pages import bp as pages_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(device_bp)
    app.register_blueprint(admin_bp)


def _wants_json() -> bool:
    from .security import wants_json

    return wants_json()


def _register_error_handlers(app: Flask) -> None:
    def _error(code: str, message: str, status: int, template: str | None = None):
        if _wants_json():
            return jsonify({"ok": False, "error": {"code": code, "message": message}}), status
        try:
            return render_template(template or "errors/generic.html",
                                   code=status, message=message), status
        except Exception:  # pragma: no cover - template lookup safety net
            return message, status

    @app.errorhandler(400)
    def bad_request(exc):
        return _error("bad_request", "The request could not be understood.", 400)

    @app.errorhandler(403)
    def forbidden(exc):
        return _error(
            "forbidden",
            "This area is restricted to administrators.",
            403,
            "errors/403.html",
        )

    @app.errorhandler(404)
    def not_found(exc):
        return _error("not_found", "That page or resource does not exist.", 404,
                      "errors/404.html")

    @app.errorhandler(405)
    def method_not_allowed(exc):
        return _error("method_not_allowed", "That method is not allowed here.", 405)

    @app.errorhandler(413)
    def payload_too_large(exc):
        limit_mb = current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return _error("payload_too_large", f"Upload exceeds the {limit_mb} MB limit.", 413)

    @app.errorhandler(429)
    def too_many_requests(exc):
        return _error("rate_limited", "Too many requests. Please slow down.", 429)

    @app.errorhandler(500)
    def server_error(exc):  # pragma: no cover - exercised only on real failures
        app.logger.exception("Unhandled server error")
        return _error("server_error", "Something went wrong on the server.", 500,
                      "errors/500.html")

    @app.errorhandler(Exception)
    def unexpected(exc):  # pragma: no cover
        from werkzeug.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            return exc
        app.logger.exception("Unhandled exception: %s", exc)
        return _error("server_error", "Something went wrong on the server.", 500,
                      "errors/500.html")


def _register_template_helpers(app: Flask) -> None:
    from .services import format_duration

    @app.context_processor
    def inject_globals() -> dict[str, Any]:
        user = current_user()
        # Rendered server-side so the sidebar badge is correct on every page,
        # not only the ones that happen to run the dashboard script.
        open_alerts = 0
        if user is not None:
            row = db_module.query_one(
                "SELECT COUNT(*) AS n FROM alerts WHERE user_id = ? AND acknowledged_at IS NULL",
                (user["id"],),
            )
            open_alerts = int(row["n"]) if row else 0
        return {
            "csrf_token": csrf_token,
            "current_user": user,
            "is_admin": is_admin(user),
            "open_alert_count": open_alerts,
            "app_version": app.config["APP_VERSION"],
            "env_name": app.config["ENV_NAME"],
            "now": datetime.now(timezone.utc),
        }

    @app.template_filter("duration")
    def duration_filter(hours: float | None) -> str:
        return format_duration(hours)

    @app.template_filter("localtime")
    def localtime_filter(value: str | None, fmt: str = "%d %b %Y, %H:%M") -> str:
        parsed = db_module.parse_ts(value)
        return parsed.strftime(fmt) if parsed else "--"

    @app.template_filter("relative")
    def relative_filter(value: str | None) -> str:
        parsed = db_module.parse_ts(value)
        if parsed is None:
            return "never"
        delta = (db_module.utcnow() - parsed).total_seconds()
        if delta < 0:
            return "just now"
        if delta < 60:
            return f"{int(delta)}s ago"
        if delta < 3600:
            return f"{int(delta // 60)}m ago"
        if delta < 86400:
            return f"{int(delta // 3600)}h ago"
        return f"{int(delta // 86400)}d ago"


def _register_cli(app: Flask) -> None:
    import click

    @app.cli.command("init-db")
    def init_db_command() -> None:
        """Create or upgrade the database schema."""
        report = db_module.init_db(app.config["DATABASE_PATH"])
        click.echo(f"Schema ready at {app.config['DATABASE_PATH']}")
        click.echo(f"  migrated columns : {report['migrated_columns'] or 'none'}")
        click.echo(f"  seeded food types: {report['seeded_food_types']}")

    @app.cli.command("prune-readings")
    @click.option("--days", default=None, type=int, help="Retention window in days.")
    def prune_command(days: int | None) -> None:
        """Delete stored readings older than the retention window."""
        retention = days if days is not None else app.config["READING_RETENTION_DAYS"]
        conn = db_module.connect(app.config["DATABASE_PATH"])
        try:
            removed = db_module.prune_readings(conn, retention)
        finally:
            conn.close()
        click.echo(f"Removed {removed} readings older than {retention} days.")

    @app.cli.command("set-role")
    @click.option("--email", required=True, help="Account to change.")
    @click.option("--role", required=True, type=click.Choice(["user", "admin"]))
    def set_role_command(email: str, role: str) -> None:
        """Grant or revoke developer-console access."""
        with app.app_context():
            user = db_module.query_one("SELECT id, role FROM users WHERE email = ?", (email,))
            if user is None:
                raise SystemExit(f"No account found for {email}")
            if user["role"] == "admin" and role != "admin":
                remaining = db_module.query_one(
                    "SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND id != ?",
                    (user["id"],),
                )
                if remaining["n"] == 0:
                    raise SystemExit("Refusing to demote the only administrator.")
            db_module.get_db().execute(
                "UPDATE users SET role = ? WHERE id = ?", (role, user["id"])
            )
        click.echo(f"{email} is now a {role}.")

    @app.cli.command("list-users")
    def list_users_command() -> None:
        """Show every account and its role."""
        with app.app_context():
            rows = db_module.query_all(
                "SELECT id, email, mobile, role, is_active FROM users ORDER BY id"
            )
        for row in rows:
            flag = "" if row["is_active"] else "  (inactive)"
            click.echo(f"{row['id']:>3}  {row['role']:<6}  {row['email'] or row['mobile']}{flag}")

    @app.cli.command("model-check")
    def model_check_command() -> None:
        """Verify the trained-model artifacts load and report the contract."""
        import json

        with app.app_context():
            status = app.extensions["predictors"].status()
        click.echo(json.dumps(status, indent=2, default=str))

    @app.cli.command("seed-demo")
    @click.option("--email", required=True, help="Account to attach demo items to.")
    def seed_demo_command(email: str) -> None:
        """Attach a small set of demo items to an existing account."""
        from .demo import seed_demo_items

        with app.app_context():
            created = seed_demo_items(email)
        click.echo(f"Created {created} demo items for {email}.")
