#!/bin/bash
jupyter lab --allow-root --ip=0.0.0.0 --port=8080 --no-browser \
    --ServerApp.token='' --ServerApp.allow_origin='*' &
exec "$@"
