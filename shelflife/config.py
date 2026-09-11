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
    # A device is considered online if it reported within this window.
    DEVICE_ONLINE_SECONDS = as_int(os.environ.get("DEVICE_ONLINE_SECONDS"), 120)
    # Readings older than this are pruned by the retention job.
    READING_RETENTION_DAYS = as_int(os.environ.get("READING_RETENTION_DAYS"), 30)
    # How often the simulator materialises a new synthetic reading.
    SIMULATOR_INTERVAL_SECONDS = as_int(os.environ.get("SIMULATOR_INTERVAL_SECONDS"), 30)

    # --- inference ---------------------------------------------------------
    # Trained multimodal Keras model. These default to the standard project
    # layout, so dropping the artifacts in place is enough to activate them; the
    # app degrades to the kinetic baseline (and says so) while they are absent.
    KERAS_MODEL_PATH = os.environ.get(
        "KERAS_MODEL_PATH", str(BASE_DIR / "models" / "best_multimodal_model.keras")
    ).strip()
    SCALER_PATH = os.environ.get(
        "SCALER_PATH", str(BASE_DIR / "artifacts" / "sensor_scaler.pkl")
    ).strip()
    MODEL_CONFIG_PATH = os.environ.get(
        "MODEL_CONFIG_PATH", str(BASE_DIR / "artifacts" / "config.json")
    ).strip()
    # Held-out split metadata, used by the admin evaluation console only.
    DATASET_SPLITS_PATH = os.environ.get(
        "DATASET_SPLITS_PATH", str(BASE_DIR / "artifacts" / "dataset_splits.csv")
    ).strip()
    DATASET_IMAGE_DIR = os.environ.get(
        "DATASET_IMAGE_DIR", str(BASE_DIR / "data" / "Images")
    ).strip()

    # Optional TFLite export, kept for the Raspberry Pi deployment path.
    MODEL_PATH = os.environ.get("MODEL_PATH", "").strip()
    MODEL_LABELS_PATH = os.environ.get("MODEL_LABELS_PATH", "").strip()

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
