"""Registration, login and logout."""

from __future__ import annotations

import sqlite3

from flask import (
    Blueprint, current_app, flash, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

from ..db import execute, iso_now, query_one
from ..security import (
    client_key,
    current_user,
    normalise_email,
    normalise_mobile,
    rate_limiter,
    start_session,
    validate_registration,
)

bp = Blueprint("auth", __name__)


def _safe_next(target: str | None) -> str:
    """Only allow same-site relative redirects (prevents open redirect)."""
    if not target:
        return url_for("pages.dashboard")
    if target.startswith("//") or "://" in target:
        return url_for("pages.dashboard")
    if not target.startswith("/"):
        return url_for("pages.dashboard")
    return target


@bp.route("/")
def index():
    if current_user():
        return redirect(url_for("pages.dashboard"))
    return redirect(url_for("auth.login"))


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user():
        return redirect(url_for("pages.dashboard"))

    form = {"email": "", "mobile": "", "full_name": ""}
    if request.method == "POST":
        allowed, retry_after = rate_limiter.check(client_key("register"), 10, 3600)
        if not allowed:
            flash(f"Too many sign-up attempts. Try again in {retry_after} seconds.", "danger")
            return render_template("register.html", form=form), 429

        email = normalise_email(request.form.get("email"))
        mobile = normalise_mobile(request.form.get("mobile"))
        full_name = (request.form.get("full_name") or "").strip()[:80]
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""
        form = {"email": email or "", "mobile": mobile or "", "full_name": full_name}

        errors = validate_registration(email, mobile, password, confirm)
        if errors:
            for message in errors:
                flash(message, "danger")
            return render_template("register.html", form=form), 400

        try:
            cursor = execute(
                "INSERT INTO users (email, mobile, full_name, password_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (email, mobile, full_name or None, generate_password_hash(password), iso_now()),
            )
            user_id = int(cursor.lastrowid)
            execute(
                "INSERT INTO settings (user_id, email_notifications, sms_notifications, alert_days_before) "
                "VALUES (?, ?, ?, 2)",
                (user_id, 1 if email else 0, 1 if mobile else 0),
            )
        except sqlite3.IntegrityError:
            flash("That email or mobile number is already registered.", "danger")
            return render_template("register.html", form=form), 409

        current_app.logger.info("New account registered (id=%s)", user_id)
        flash("Account created. Please sign in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html", form=form)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("pages.dashboard"))

    identifier = ""
    if request.method == "POST":
        identifier = (request.form.get("identifier") or "").strip()
        password = request.form.get("password") or ""

        limit = current_app.config["LOGIN_MAX_ATTEMPTS"]
        window = current_app.config["LOGIN_LOCKOUT_SECONDS"]
        key = client_key(f"login:{identifier.lower()}")
        allowed, retry_after = rate_limiter.check(key, limit, window)
        if not allowed:
            flash(
                f"Too many failed sign-in attempts. Try again in {retry_after} seconds.",
                "danger",
            )
            return render_template("login.html", identifier=identifier), 429

        email = normalise_email(identifier)
        mobile = normalise_mobile(identifier)
        user = query_one(
            "SELECT * FROM users WHERE (email = ? OR mobile = ?) AND is_active = 1",
            (email, mobile),
        )

        if user and check_password_hash(user["password_hash"], password):
            rate_limiter.reset(key)
            start_session(user["id"])
            execute("UPDATE users SET last_login_at = ? WHERE id = ?", (iso_now(), user["id"]))
            flash("Signed in.", "success")
            return redirect(_safe_next(request.args.get("next")))

        # Same message either way: do not reveal whether the account exists.
        flash("Incorrect email/mobile or password.", "danger")
        return render_template("login.html", identifier=identifier), 401

    return render_template("login.html", identifier=identifier)


# POST only: a GET logout link can be triggered cross-site.
@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))
