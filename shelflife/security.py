"""Authentication, CSRF protection, device tokens and request throttling."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import time
from functools import wraps
from typing import Any, Callable

from flask import (
    abort,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    request,
    session,
    url_for,
)

from .db import iso_now, query_one

CSRF_SESSION_KEY = "_csrf_token"
CSRF_HEADER = "X-CSRF-Token"
CSRF_FORM_FIELD = "csrf_token"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

DEVICE_TOKEN_PREFIX = "slp"
MIN_PASSWORD_LENGTH = 8

# Two interfaces, one login. 'user' reaches only the monitoring UI; 'admin' also
# reaches the developer console at /admin. There is no separate admin login: the
# role is a property of the account, checked on every request.
ROLE_USER = "user"
ROLE_ADMIN = "admin"
ROLES = (ROLE_USER, ROLE_ADMIN)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
MOBILE_RE = re.compile(r"^\+?[0-9]{7,15}$")

# Rejected outright regardless of length: these dominate credential-stuffing
# lists and a capstone demo box is exactly the kind of host that gets scanned.
WEAK_PASSWORDS = frozenset(
    {
        "password", "password1", "password123", "12345678", "123456789",
        "1234567890", "qwerty123", "abc12345", "iloveyou", "admin123",
        "welcome1", "letmein1", "11111111", "00000000", "passw0rd",
    }
)


def wants_json() -> bool:
    """True when the caller expects a JSON error rather than an HTML page.

    Matches "/api/" anywhere in the path, so the admin console's own endpoints
    under /admin/api/ are covered as well as the top-level /api/ ones.
    """
    if "/api/" in request.path:
        return True
    accept = request.accept_mimetypes
    return bool(accept["application/json"] > accept["text/html"])


# --- CSRF -------------------------------------------------------------------
def csrf_token() -> str:
    """Return (creating if needed) this session's CSRF token."""
    token = session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


def _submitted_csrf_token() -> str:
    header = request.headers.get(CSRF_HEADER)
    if header:
        return header
    if request.form:
        return request.form.get(CSRF_FORM_FIELD, "")
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        if isinstance(payload, dict):
            return str(payload.get(CSRF_FORM_FIELD, ""))
    return ""


def csrf_protect() -> Any:
    """``before_request`` hook: reject unsafe requests without a valid token.

    Device endpoints authenticate with a bearer token and carry no cookies, so
    they are not subject to cross-site request forgery and are exempt.
    """
    if request.method in SAFE_METHODS:
        return None
    if getattr(g, "csrf_exempt", False):
        return None
    endpoint = request.endpoint or ""
    # The entire device blueprint authenticates with a bearer token and sends no
    # cookies, so CSRF cannot apply to it. Matching on the blueprint prefix means
    # a new device endpoint cannot be forgotten here and silently break the Pi.
    if endpoint.startswith("device."):
        return None
    if endpoint in current_app.config.get("CSRF_EXEMPT_ENDPOINTS", frozenset()):
        return None

    expected = session.get(CSRF_SESSION_KEY, "")
    submitted = _submitted_csrf_token()
    if expected and submitted and hmac.compare_digest(expected, submitted):
        return None

    if wants_json():
        return (
            jsonify(
                {
                    "ok": False,
                    "error": {
                        "code": "csrf_failed",
                        "message": "Missing or invalid CSRF token. Reload the page and retry.",
                    },
                }
            ),
            403,
        )
    flash("Your session expired or the form was stale. Please try again.", "danger")
    return redirect(request.referrer or url_for("auth.login"))


# --- session auth -----------------------------------------------------------
def current_user():
    """The logged-in user row, or ``None``. Cached per request."""
    if "current_user" in g:
        return g.current_user
    user_id = session.get("user_id")
    user = None
    if user_id is not None:
        user = query_one(
            "SELECT * FROM users WHERE id = ? AND is_active = 1", (user_id,)
        )
        if user is None:
            session.clear()
    g.current_user = user
    return user


def is_admin(user=None) -> bool:
    """True when the signed-in account carries the admin role."""
    user = user if user is not None else current_user()
    if user is None:
        return False
    try:
        return user["role"] == ROLE_ADMIN
    except (KeyError, IndexError):  # pre-migration row
        return False


def login_required(view: Callable) -> Callable:
    @wraps(view)
    def wrapper(*args, **kwargs):
        if current_user() is None:
            if wants_json():
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": {
                                "code": "unauthenticated",
                                "message": "Sign in to continue.",
                            },
                        }
                    ),
                    401,
                )
            flash("Please sign in to continue.", "info")
            return redirect(url_for("auth.login", next=request.full_path))
        return view(*args, **kwargs)

    return wrapper


def admin_required(view: Callable) -> Callable:
    """Gate a view to admin accounts.

    A signed-in non-admin gets 403, not a redirect to the login page: they are
    authenticated, they simply are not authorised, and bouncing them to a login
    form they have already passed would be misleading.
    """

    @wraps(view)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None:
            if wants_json():
                return (
                    jsonify({"ok": False, "error": {"code": "unauthenticated",
                                                    "message": "Sign in to continue."}}),
                    401,
                )
            flash("Please sign in to continue.", "info")
            return redirect(url_for("auth.login", next=request.full_path))
        if not is_admin(user):
            current_app.logger.warning(
                "Blocked non-admin access to %s by user id=%s", request.path, user["id"]
            )
            if wants_json():
                return (
                    jsonify({"ok": False, "error": {"code": "forbidden",
                                                    "message": "This area is restricted to administrators."}}),
                    403,
                )
            abort(403)
        return view(*args, **kwargs)

    return wrapper


def start_session(user_id: int) -> None:
    """Log a user in, rotating the session to defeat fixation attacks."""
    session.clear()
    session["user_id"] = int(user_id)
    session.permanent = True
    session[CSRF_SESSION_KEY] = secrets.token_urlsafe(32)
    g.pop("current_user", None)


# --- credential validation ---------------------------------------------------
def normalise_email(value: str | None) -> str | None:
    value = (value or "").strip().lower()
    return value or None


def normalise_mobile(value: str | None) -> str | None:
    value = re.sub(r"[\s\-()]", "", (value or "").strip())
    return value or None


def validate_registration(
    email: str | None, mobile: str | None, password: str, confirm: str
) -> list[str]:
    """Return a list of human-readable problems; empty means valid."""
    errors: list[str] = []
    if not email and not mobile:
        errors.append("Provide an email address or a mobile number.")
    if email and not EMAIL_RE.match(email):
        errors.append("That email address does not look valid.")
    if mobile and not MOBILE_RE.match(mobile):
        errors.append("Mobile number must be 7-15 digits, optionally starting with '+'.")
    if len(password) < MIN_PASSWORD_LENGTH:
        errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if password.lower() in WEAK_PASSWORDS:
        errors.append("That password is too common. Choose something less guessable.")
    if password and password.isdigit():
        errors.append("Password must not be only digits.")
    if password != confirm:
        errors.append("Passwords do not match.")
    return errors


# --- device tokens -----------------------------------------------------------
def generate_device_token() -> tuple[str, str, str]:
    """Return ``(plaintext, sha256_hash, display_prefix)``.

    Only the hash is stored; the plaintext is shown to the user exactly once.
    """
    raw = secrets.token_hex(24)
    plaintext = f"{DEVICE_TOKEN_PREFIX}_{raw}"
    return plaintext, hash_device_token(plaintext), plaintext[: len(DEVICE_TOKEN_PREFIX) + 7]


def hash_device_token(plaintext: str) -> str:
    return hashlib.sha256(plaintext.strip().encode("utf-8")).hexdigest()


def authenticate_device():
    """Resolve the ``Authorization: Bearer <token>`` header to a device row."""
    header = request.headers.get("Authorization", "")
    token = ""
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
    if not token:
        token = request.headers.get("X-Device-Token", "").strip()
    if not token:
        return None
    device = query_one(
        "SELECT * FROM devices WHERE token_hash = ?", (hash_device_token(token),)
    )
    return device


def touch_device(device_id: int, firmware: str | None = None) -> None:
    from .db import execute

    if firmware:
        execute(
            "UPDATE devices SET last_seen_at = ?, firmware = ? WHERE id = ?",
            (iso_now(), firmware[:64], device_id),
        )
    else:
        execute("UPDATE devices SET last_seen_at = ? WHERE id = ?", (iso_now(), device_id))


# --- throttling --------------------------------------------------------------
class RateLimiter:
    """Fixed-window counter kept in process memory.

    Adequate for the single-process edge deployment this project targets. A
    multi-worker or multi-node deployment would need a shared store; that is
    documented rather than silently assumed.
    """

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Record a hit. Returns ``(allowed, seconds_until_reset)``."""
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            bucket = [stamp for stamp in self._hits.get(key, []) if stamp > cutoff]
            if len(bucket) >= limit:
                self._hits[key] = bucket
                retry_after = int(window_seconds - (now - bucket[0])) + 1
                return False, max(1, retry_after)
            bucket.append(now)
            self._hits[key] = bucket
            if len(self._hits) > 4096:
                self._evict(cutoff)
            return True, 0

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)

    def _evict(self, cutoff: float) -> None:
        stale = [key for key, stamps in self._hits.items() if not any(s > cutoff for s in stamps)]
        for key in stale:
            self._hits.pop(key, None)


rate_limiter = RateLimiter()


def client_key(scope: str) -> str:
    """Throttle key for the caller. Honours one trusted proxy hop."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    address = forwarded.split(",")[0].strip() if forwarded else (request.remote_addr or "unknown")
    return f"{scope}:{address}"
