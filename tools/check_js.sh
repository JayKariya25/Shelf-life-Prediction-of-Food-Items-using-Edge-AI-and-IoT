#!/bin/sh
# Syntax-check every ES module without executing it.
set -e
status=0
for file in static/js/*.js; do
    tmp="$(mktemp -t jscheck).mjs"
    cp "$file" "$tmp"
    if node --check "$tmp" 2>/tmp/jserr; then
        echo "ok    $file"
    else
        echo "FAIL  $file"
        cat /tmp/jserr
        status=1
    fi
    rm -f "$tmp"
done
exit $status
