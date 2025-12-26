#!/bin/bash

RED=$'\e[31m'
GREEN=$'\e[32m'
YELLOW=$'\e[33m'
NC=$'\e[0m'

set -e

error_handler() {
    echo -e "${RED}FAILED${NC}"
    echo "ERR: $1"
    exit 1
}

trap 'error_handler "Command failed: $BASH_COMMAND"' ERR

[[ -f pyproject.toml ]] || error_handler "Run this script from the project's root"

echo -n "Updating git submodules... "
git submodule update --init --recursive -q
echo -e "${GREEN}OK!${NC}"

echo -n "Checking uv is installed... "
command -v uv &> /dev/null || error_handler "uv is not installed"
echo -e "${GREEN}OK!${NC}"

echo -n "Setting up python virtual environment... "
uv sync --quiet
echo -e "${GREEN}OK!${NC}"

echo -n "Linking dev drone env to overwrite demo version in submodule... "
# overwrite env source code
rm -rf ./pufferlib/pufferlib/ocean/drone
ln -s "$(pwd)/env" ./pufferlib/pufferlib/ocean/drone

# overwrite resources
rm -rf ./pufferlib/pufferlib/resources/drone
ln -s "$(pwd)/resources" ./pufferlib/pufferlib/resources/drone

# overwrite hypers config
ln -sf "$(pwd)/config/drone.ini" ./pufferlib/pufferlib/config/ocean/drone.ini

# copy latest env binding to drone project root
ln -sf ./pufferlib/pufferlib/ocean/env_binding.h ./env_binding.h
echo -e "${GREEN}OK!${NC}"

echo -n "Installing pufferlib to virtual environment... "
uv pip install -e pufferlib --quiet
echo -e "${GREEN}OK!${NC}"

echo -n "Building pufferlib... "
(cd pufferlib && ../.venv/bin/python setup.py build_ext --inplace > /dev/null 2>&1)
echo -e "${GREEN}OK!${NC}"

echo -n "Building platform config for Crazyflie 2.1 Brushless... "
(cd crazyflie-firmware && make cf21bl_defconfig > /dev/null 2>&1)
echo -e "${GREEN}OK!${NC}"

echo -e "${GREEN}Done!${NC}"
echo -e "${YELLOW}  Enable the virtual environment with the \`source ./.venv/bin/activate\` command.${NC}"
echo -e "${YELLOW}  After env updates, rebuild pufferlib with \`python setup.py build_ext --inplace\` in the pufferlib directory.${NC}"

