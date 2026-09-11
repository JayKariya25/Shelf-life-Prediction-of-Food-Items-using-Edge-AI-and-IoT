#!/bin/sh
# Serve the site on the local network with a production WSGI server.
#
#   sh tools/serve.sh            # uses the model runtime (.venv-model)
#   VENV=.venv sh tools/serve.sh # baseline only, no TensorFlow
#
# Configuration is read from .env. Stop with Ctrl-C.
set -e
cd "$(dirname "$0")/.."

VENV="${VENV:-.venv-model}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "No interpreter at $VENV/bin/python" >&2
    exit 1
fi

if [ -f .env ]; then
    set -a
    . ./.env
    set +a
else
    echo "No .env file. Copy .env.example and set SECRET_KEY." >&2
    exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Best-effort LAN address, for the message below only.
LAN=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -i 2>/dev/null || echo "this-machine")

echo "Shelf-Life Monitor"
echo "  runtime : $($VENV/bin/python -V 2>&1)"
echo "  profile : ${APP_ENV:-development}"
echo "  local   : http://127.0.0.1:$PORT"
if [ "$HOST" = "0.0.0.0" ]; then
    echo "  network : http://$LAN:$PORT"
fi
echo

exec "$VENV/bin/waitress-serve" \
    --host "$HOST" --port "$PORT" \
    --threads 8 \
    --call "shelflife:create_app"
