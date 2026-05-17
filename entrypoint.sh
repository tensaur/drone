#!/usr/bin/env bash
set -e

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec jupyter lab \
    --allow-root \
    --ip=0.0.0.0 \
    --port="${JUPYTER_PORT:-8080}" \
    --no-browser \
    --ServerApp.token='' \
    --ServerApp.allow_origin='*'
