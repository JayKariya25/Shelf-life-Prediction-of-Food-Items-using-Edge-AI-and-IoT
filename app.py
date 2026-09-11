"""Entry point for the Edge-AI Shelf-Life Prediction web application.

Development::

    python app.py

Raspberry Pi::

    APP_ENV=raspi-prod SECRET_KEY=... python app.py

For anything beyond a demo, run it behind a real WSGI server instead::

    waitress-serve --port=8000 --call "shelflife:create_app"
"""

from __future__ import annotations

import os
import sys

from shelflife import create_app
from shelflife.config import ConfigError, as_bool, as_int

try:
    app = create_app()
except ConfigError as exc:
    print(f"Configuration error: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc


def _ssl_context():
    """Optional TLS for the Pi deployment."""
    cert, key = os.environ.get("SSL_CERT", ""), os.environ.get("SSL_KEY", "")
    if cert and key:
        return (cert, key)
    if as_bool(os.environ.get("SSL_ADHOC")):
        return "adhoc"
    return None


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = as_int(os.environ.get("PORT"), 5000)
    context = _ssl_context()

    app.logger.info(
        "Starting %s on %s://%s:%s",
        app.config["ENV_NAME"], "https" if context else "http", host, port,
    )
    if host == "0.0.0.0" and app.config["ENV_NAME"] == "development":
        app.logger.warning(
            "Binding 0.0.0.0 in development mode exposes the debug server on your "
            "network. Use APP_ENV=raspi-prod with a real WSGI server instead."
        )

    app.run(host=host, port=port, debug=app.config["DEBUG"], ssl_context=context)
