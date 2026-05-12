#!/bin/bash
set -e

RED=$'\e[31m'
GREEN=$'\e[32m'
NC=$'\e[0m'

ok()  { echo -e "${GREEN}OK${NC}"; }
err() { echo -e "${RED}FAILED${NC}"; echo "ERR: $1"; exit 1; }
trap 'err "Command failed: $BASH_COMMAND"' ERR

[[ -f pyproject.toml ]] || err "Run this script from the project root"

MODE=docker
[[ "${1-}" = "--native" ]] && MODE=native
[[ "${1-}" = "--docker" ]] && MODE=docker
[[ -n "${1-}" && "$1" != "--native" && "$1" != "--docker" ]] \
    && err "Unknown flag: $1 (use --native or --docker)"

echo -n "Initialising submodules... "
git submodule update --init --recursive -q
ok

echo -n "Checking uv... "
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || err "uv install failed"
ok

echo -n "Checking just... "
if ! command -v just >/dev/null; then
    mkdir -p "$HOME/.local/bin"
    curl -LsSf https://just.systems/install.sh \
        | bash -s -- --to "$HOME/.local/bin" >/dev/null 2>&1
fi
command -v just >/dev/null || err "just install failed"
ok

echo "Handing off to: just ${MODE/docker/setup}${MODE/native/setup-native}"
if [[ "$MODE" = native ]]; then
    just setup-native
else
    just setup
fi

