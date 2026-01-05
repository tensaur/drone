# https://just.systems/man/en/

alias b := build
alias bp := build-puffer
alias bf := build-firmware
alias s := setup
alias sp := setup-puffer
alias sf := setup-firmware
alias cf := clean-firmware
alias clean := clean-firmware
alias f := flash-firmware
alias flash := flash-firmware
alias e := eval
alias t := train
alias swp := sweep
alias fmt := format

# list all recipes
default:
    @just --list

c-source := `git ls-files --cached --others --exclude-standard '*.c' '*.h' ':!controller/src/dronelib.h' | xargs`
py-source := `git ls-files --cached --others --exclude-standard '*.py' | xargs`

# setup submodules, pufferlib env and crazyflie firmware
setup: update-submodules setup-puffer setup-firmware

# builds all source code, i.e. pufferlib and crazyflie firmware
build: build-puffer build-firmware

[private]
_check_venv:
    uv sync --inexact --quiet

# format the specified C files, or all in project if no args
format +FILES=c-source:
    @clang-format -style="{ColumnLimit: 100, IndentWidth: 4, TabWidth: 4, DerivePointerAlignment: false, PointerAlignment: Left, AllowShortIfStatementsOnASingleLine: AllIfsAndElse, IndentCaseLabels: true}" -i {{FILES}}

# update the git submodules (i.e. pufferlib and crazyflie firmware)
update-submodules:
    git submodule update --init --recursive -q

# setup and build puffer, also creates symlinks for env
[group: "puffer"]
setup-puffer: setup-symlinks install-puffer build-puffer

# installs pufferlib to the python venv using uv
[group: "puffer"]
install-puffer: _check_venv
    uv pip install -e pufferlib

# builds the pufferlib C code, requires pufferlib to be installed to venv
[group: "puffer", working-directory: "pufferlib"]
build-puffer: _check_venv
    uv pip show pufferlib
    ../.venv/bin/python3 setup.py build_ext --inplace

# eval the env with a given model, use `MODEL=latest` for last trained (tip: use `just bp eval` to build the env and then eval it)
[group: "puffer"]
eval DEVICE="cpu" MODEL="" TASK="":
    ./.venv/bin/puffer eval puffer_drone --train.device {{DEVICE}} {{ if MODEL == "" { "" } else { "--load-model-path " + MODEL } }} {{ if TASK == "" { "" } else { "--env.task" + TASK } }}

# train the model on a task using a specific device, optionally specify TRACK to log stats to the specified wandb project
[group: "puffer"]
train DEVICE="cpu" TASK="hover" TRACK="":
    ./.venv/bin/puffer train puffer_drone --train.device {{DEVICE}} {{ if TRACK == "" { "" } else { "--wandb --wandb-project " + TRACK } }} {{ if TASK == "" { "" } else { "--env.task" + TASK } }}

# sweep for hypers on a specific device, optionally specify TRACK to log stats to the specified wandb project
[group: "puffer"]
sweep DEVICE="cpu" TASK="hover" TRACK="":
    ./.venv/bin/puffer sweep puffer_drone --train.device {{DEVICE}} {{ if TRACK == "" { "" } else { "--wandb --wandb-project " + TRACK } }} {{ if TASK == "" { "" } else { "--env.task" + TASK } }}

# create symlinks in pufferlib submodule to allow for env development in ./env
[group: "puffer"]
setup-symlinks:
    # overwrite env source code
    @rm -rf ./pufferlib/pufferlib/ocean/drone
    ln -s "$(pwd)/env" ./pufferlib/pufferlib/ocean/drone

    # overwrite resources
    @rm -rf ./pufferlib/pufferlib/resources/drone
    ln -s "$(pwd)/resources" ./pufferlib/pufferlib/resources/drone

    # overwrite hypers config
    ln -sf "$(pwd)/config/drone.ini" ./pufferlib/pufferlib/config/ocean/drone.ini

    # copy latest env binding to drone project root
    ln -sf ./pufferlib/pufferlib/ocean/env_binding.h ./env_binding.h

# setup firmware: clean, configure for target device, and then build (incl. OOT controller)
[group: "crazyflie"]
setup-firmware: clean-firmware configure-firmware build-firmware

# clean previous builds of firmware and OOT controller
[group: "crazyflie", working-directory: "controller"]
clean-firmware:
    make clean

# builds crazyflie firmware from source (incl. OOT controller)
[group: "crazyflie", working-directory: "controller"]
build-firmware:
    make -j{{num_cpus()}}

# flash the firmware to a Crazyflie drone (requires a Crazyradio with drivers installed on device)
[group: "crazyflie", working-directory: "controller"]
flash-firmware: _check_venv
    make cload

# configure firmware builds for the specified target device (cf21bl|cf2|bolt)
[group: "crazyflie", working-directory: "controller", arg("PLATFORM", pattern="cf21bl|cf2|bolt")]
configure-firmware PLATFORM="cf21bl":
    make {{PLATFORM}}_defconfig
