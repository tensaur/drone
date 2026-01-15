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
source := f'{{c-source}} {{py-source}}'

# setup submodules, pufferlib env and crazyflie firmware
setup: update-submodules setup-puffer setup-firmware

# builds all source code, i.e. pufferlib and crazyflie firmware
build: build-puffer build-firmware

[private]
_check_venv:
    uv sync --inexact --quiet

# format the specified source files, or all in project if no args
format +FILES=source:
    #!/usr/bin/env sh
    c_files=""
    py_files=""

    for f in {{FILES}}; do
        case "$f" in
            *.c|*.h) c_files+=" $f" ;;
            *.py) py_files+=" $f" ;;
        esac
    done

    [[ -n "$c_files" ]] && just c-format $c_files
    [[ -n "$py_files" ]] && just py-format $py_files

# format the specified C files, or all in project if no args
[private]
c-format +FILES=c-source:
    @clang-format -style="{ColumnLimit: 100, IndentWidth: 4, TabWidth: 4, DerivePointerAlignment: false, PointerAlignment: Left, AllowShortIfStatementsOnASingleLine: AllIfsAndElse, IndentCaseLabels: true}" -i {{FILES}}

# format the specified Python files, or all in project if no args
[private]
py-format +FILES=py-source:
    @uv tool run black -q {{FILES}}

# update the git submodules (i.e. pufferlib and crazyflie firmware)
update-submodules:
    git submodule update --init --recursive -q

# setup and build puffer, also creates symlinks for env
[group: "puffer"]
setup-puffer: setup-puffer-symlinks install-puffer build-puffer

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
    ./.venv/bin/puffer eval puffer_drone --train.device {{DEVICE}} {{ if MODEL == "" { "" } else { "--load-model-path " + MODEL } }} {{ if TASK == "" { "" } else { "--env.task " + TASK } }}

# train the model on a task using a specific device, optionally specify TRACK to log stats to the specified wandb project
[group: "puffer"]
train DEVICE="cpu" TASK="hover" TRACK="":
    ./.venv/bin/puffer train puffer_drone --train.device {{DEVICE}} {{ if TRACK == "" { "" } else { "--wandb --wandb-project " + TRACK } }} {{ if TASK == "" { "" } else { "--env.task " + TASK } }}

# sweep for hypers on a specific device, optionally specify TRACK to log stats to the specified wandb project
[group: "puffer"]
sweep DEVICE="cpu" TASK="hover" TRACK="":
    ./.venv/bin/puffer sweep puffer_drone --train.device {{DEVICE}} --max-runs 10000 {{ if TRACK == "" { "" } else { "--wandb --wandb-project " + TRACK } }} {{ if TASK == "" { "" } else { "--env.task " + TASK } }}

# export the model weights, and convert to a header file for use in hardware
[group: "puffer"]
export MODEL="latest":
    ./.venv/bin/puffer export puffer_drone --load-model-path {{MODEL}} --train.device cpu
    mkdir -p ./build
    gcc -o ./build/bin2h ./tools/bin2h.c
    ./build/bin2h puffer_drone_weights.bin ./controller/src/weights.h

# create symlinks in pufferlib submodule to allow for env development in ./env
[group: "puffer"]
setup-puffer-symlinks:
    # overwrite env source code
    @rm -rf ./pufferlib/pufferlib/ocean/drone
    ln -s "$(pwd)/env" ./pufferlib/pufferlib/ocean/drone

    # overwrite resources
    @rm -rf ./pufferlib/pufferlib/resources/drone
    ln -s "$(pwd)/resources" ./pufferlib/pufferlib/resources/drone

    # overwrite hypers config
    ln -sf "$(pwd)/config/drone.ini" ./pufferlib/pufferlib/config/ocean/drone.ini

    # overwrite model
    ln -sf "$(pwd)/models/models.py" ./pufferlib/pufferlib/ocean/torch.py

    # copy latest env binding to drone project root
    ln -sf ./pufferlib/pufferlib/ocean/env_binding.h ./env_binding.h

# setup firmware: clean, configure for target device, and then build (incl. OOT controller)
[group: "crazyflie"]
setup-firmware: setup-firmware-symlinks configure-firmware clean-firmware build-firmware

# create symlinks in crazyflie submodule for firmware dev
[group: "crazyflie"]
setup-firmware-symlinks:
    ln -sf "$(pwd)/controller/src/stabilizer.c" ./crazyflie-firmware/src/modules/src/stabilizer.c

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
[arg("auto", long="auto", short="a", value="-w radio://0/80/2M/E7E7E7E7E7")]
flash-firmware auto="": _check_venv
    CLOAD_CMDS="{{auto}}" make cload

# open firmware control gui
[group: "crazyflie"]
gui:
    ./.venv/bin/cfclient

# configure firmware builds for the specified target device (cf21bl|cf2|bolt)
[group: "crazyflie", working-directory: "controller", arg("PLATFORM", pattern="cf21bl|cf2|bolt")]
configure-firmware PLATFORM="cf21bl":
    make {{PLATFORM}}_defconfig
