"""Configuration objects for the Shelf-Life Prediction web application.

Every tunable is driven by an environment variable so the same code runs on a
developer laptop and on the Raspberry Pi 5 edge node without edits.
"""

from __future__ import annotations

import os
import secrets
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INSTANCE_DIR = BASE_DIR / "instance"


def as_bool(value: object, default: bool = False) -> bool:
    """Parse a truthy environment string without raising."""
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def as_int(value: object, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def as_float(value: object, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


class ConfigError(RuntimeError):
    """Raised when the process is started with an unusable configuration."""


def _resolve_secret_key(*, require_env: bool) -> str:
    """Return the Flask secret key.

    Production must supply ``SECRET_KEY`` explicitly. Development falls back to
    a key persisted under ``instance/`` so that sessions survive a reload
    instead of silently logging everyone out on every restart.
    """
    env_key = os.environ.get("SECRET_KEY", "").strip()
    if env_key:
        if require_env and env_key in {"change_this_secret_key", "secret", "dev"}:
            raise ConfigError(
                "SECRET_KEY is set to a well-known placeholder value. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return env_key

    if require_env:
        raise ConfigError(
            "SECRET_KEY environment variable is required when APP_ENV=raspi-prod. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )

    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    key_file = INSTANCE_DIR / "dev_secret_key"
    if key_file.exists():
        stored = key_file.read_text(encoding="utf-8").strip()
        if stored:
            return stored
    generated = secrets.token_hex(32)
    key_file.write_text(generated, encoding="utf-8")
    try:
        key_file.chmod(0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass
    return generated


class BaseConfig:
    """Defaults shared by every environment."""

    ENV_NAME = "development"
    DEBUG = False
    TESTING = False

    # --- storage -----------------------------------------------------------
    DATABASE_PATH = str(BASE_DIR / os.environ.get("DATABASE_FILE", "database.db"))
    UPLOAD_DIR = str(BASE_DIR / "static" / "uploads")
    MODEL_DIR = str(BASE_DIR / "models")

    # --- session / security ------------------------------------------------
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = False
    SESSION_REFRESH_EACH_REQUEST = True
    PERMANENT_SESSION_LIFETIME = timedelta(days=as_int(os.environ.get("SESSION_DAYS"), 14))
    MAX_CONTENT_LENGTH = as_int(os.environ.get("MAX_UPLOAD_MB"), 8) * 1024 * 1024
    PREFERRED_URL_SCHEME = "http"
    FORCE_HTTPS = False

    # Login throttling (per identifier + client address).
    LOGIN_MAX_ATTEMPTS = as_int(os.environ.get("LOGIN_MAX_ATTEMPTS"), 8)
    LOGIN_LOCKOUT_SECONDS = as_int(os.environ.get("LOGIN_LOCKOUT_SECONDS"), 300)

    # --- sensing -----------------------------------------------------------
    # auto      -> use live device readings when a device reported recently,
    #              otherwise fall back to the labelled simulator
    # live      -> never simulate; show "no data" when hardware is silent
    # simulated -> always simulate (useful for demos and UI work)
    SENSOR_SOURCE = os.environ.get("SENSOR_SOURCE", "auto").strip().lower()
    # A device is considered online if it reported within this window. It must be
    # comfortably longer than DEVICE_REPORT_INTERVAL_SECONDS, or a device reporting
    # on schedule flickers offline between readings.
    DEVICE_ONLINE_SECONDS = as_int(os.environ.get("DEVICE_ONLINE_SECONDS"), 300)
    # Readings older than this are pruned by the retention job.
    READING_RETENTION_DAYS = as_int(os.environ.get("READING_RETENTION_DAYS"), 30)

    # --- inference ---------------------------------------------------------
    # The trained multimodal Remaining-Shelf-Life model lives in ml/rsl_multimodal,
    # alongside the pipeline that produced it, so retraining refreshes what the website
    # serves. While it is absent the app serves the kinetic baseline and says so.
    RSL_BUNDLE_DIR = os.environ.get(
        "RSL_BUNDLE_DIR", str(BASE_DIR / "ml" / "rsl_multimodal")
    ).strip()
    # The export the report recommends: float16 keeps float32 accuracy at half the size.
    RSL_MODEL_FILE = os.environ.get(
        "RSL_MODEL_FILE", "models/image_ensemble_blocked_full_float16.tflite"
    ).strip()
    # The model reads what the food looks like, so it needs a recent photograph. Older
    # than this relative to the newest reading, the item falls back to the baseline.
    RSL_MAX_PHOTO_AGE_HOURS = as_float(os.environ.get("RSL_MAX_PHOTO_AGE_HOURS"), 6.0)
    # Which produce the trained model was actually fitted on. The dataset is a single
    # 11-day banana run and has no food column at all, so the bundle cannot state this
    # and the application must: a tomato photograph scored by a banana-only model would
    # be a confident number about a food it has never seen. Items of any other produce
    # fall back to the kinetic baseline, which is food-aware through FOOD_TYPE_SEED.
    # Widen this only when the model has been retrained on that food's data.
    RSL_TRAINED_FOOD_KEYS = tuple(
        key.strip().lower()
        for key in (os.environ.get("RSL_TRAINED_FOOD_KEYS") or "banana").split(",")
        if key.strip()
    )
    # Training readings were 2 minutes apart (95% of gaps). A faster stream is thinned
    # to this spacing before the rolling features are built, so a 5-reading window still
    # spans ~10 minutes as it did in training.
    RSL_MIN_READING_SPACING_SECONDS = as_int(
        os.environ.get("RSL_MIN_READING_SPACING_SECONDS"), 90
    )

    # How often devices are asked to report (the agent reads this from the server).
    DEVICE_REPORT_INTERVAL_SECONDS = as_int(
        os.environ.get("DEVICE_REPORT_INTERVAL_SECONDS"), 120
    )

    @classmethod
    def build(cls) -> dict:
        """Materialise the config mapping, resolving secrets last."""
        values = {
            key: getattr(cls, key)
            for key in dir(cls)
            if key.isupper() and not key.startswith("_")
        }
        values["SECRET_KEY"] = _resolve_secret_key(require_env=cls is RaspberryPiProdConfig)
        return values


class DevelopmentConfig(BaseConfig):
    ENV_NAME = "development"
    DEBUG = as_bool(os.environ.get("FLASK_DEBUG"), True)


class TestingConfig(BaseConfig):
    ENV_NAME = "testing"
    TESTING = True
    DEBUG = False
    SENSOR_SOURCE = "simulated"
    LOGIN_MAX_ATTEMPTS = 1000
    WTF_CSRF_ENABLED = False


class RaspberryPiProdConfig(BaseConfig):
    """Hardened profile for the Raspberry Pi 5 edge deployment."""

    ENV_NAME = "raspi-prod"
    DEBUG = False
    SESSION_COOKIE_SECURE = as_bool(os.environ.get("SESSION_COOKIE_SECURE"), True)
    FORCE_HTTPS = as_bool(os.environ.get("FORCE_HTTPS"), True)
    PREFERRED_URL_SCHEME = "https"


CONFIGS = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "raspi-prod": RaspberryPiProdConfig,
}


def get_config(name: str | None = None) -> type[BaseConfig]:
    key = (name or os.environ.get("APP_ENV") or "development").strip().lower()
    if key not in CONFIGS:
        raise ConfigError(
            f"Unknown APP_ENV {key!r}. Expected one of: {', '.join(sorted(CONFIGS))}"
        )
    return CONFIGS[key]
