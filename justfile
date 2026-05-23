# https://just.systems/man/en/

alias b := build
alias bp := build-puffer
alias bf := build-firmware
alias bw := build-web
alias s := setup
alias sp := setup-puffer
alias sf := setup-firmware
alias cf := clean-firmware
alias clean := clean-firmware
alias f := flash-firmware
alias flash := flash-firmware
alias d := dev
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

# pull the container image and install host-side firmware tools
# (images are amd64-only; Apple Silicon Macs use Rosetta via --platform)
[group: "docker"]
setup TAG="auto": update-submodules _setup-host
    docker pull --platform linux/amd64 ghcr.io/tensaur/drone:$(just _resolve-tag {{TAG}})

# start a dev shell inside the container (auto-runs setup-puffer on entry)
[group: "docker"]
dev TAG="auto":
    #!/usr/bin/env bash
    set -e
    tag=$(just _resolve-tag {{TAG}})
    case "$tag" in
        cuda|cuda-jupyter) gpu="--gpus all" ;;
        *) gpu="" ;;
    esac
    docker run -it --rm --name drone $gpu --ipc host \
        --platform linux/amd64 \
        -v "$(pwd):/work" -e WANDB_API_KEY \
        ghcr.io/tensaur/drone:$tag \
        bash -c 'just setup-puffer && exec bash' || true

# resolve an image tag: TAG="auto" picks cuda on x86 with nvidia-smi, cpu otherwise
[private]
_resolve-tag TAG="auto":
    #!/usr/bin/env bash
    tag="{{TAG}}"
    if [ "$tag" = auto ]; then
        case "$(uname -m)" in
            arm64|aarch64) tag=cpu ;;
            *) command -v nvidia-smi >/dev/null 2>&1 && tag=cuda || tag=cpu ;;
        esac
    fi
    echo "$tag"

# builds all source code, i.e. pufferlib and crazyflie firmware
build: build-puffer build-firmware

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
    @clang-format -i {{FILES}}

# format the specified Python files, or all in project if no args
[private]
py-format +FILES=py-source:
    @uv tool run black -q {{FILES}}

# update the git submodules (i.e. pufferlib and crazyflie firmware)
update-submodules:
    git submodule update --init --recursive -q

# create the project venv, install pufferlib editable
[group: "puffer"]
setup-puffer: setup-puffer-symlinks
    #!/usr/bin/env bash
    set -e
    fingerprint=$(just _puffer-fingerprint)
    if [ -f .venv/.puffer-built ] && [ "$(cat .venv/.puffer-built 2>/dev/null)" = "$fingerprint" ]; then
        exit 0
    fi

    # fingerprint mismatch
    rm -rf .venv
    uv venv .venv

    # Pick the torch wheel index based on whether nvcc (CUDA) is available
    cuda_ver=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | tr -d '.')
    if [ -n "$cuda_ver" ]; then
        export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu${cuda_ver}"
        build_flag=""
    else
        export UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
        build_flag="--cpu"
    fi
    VIRTUAL_ENV=.venv uv pip install -e pufferlib
    (cd pufferlib && PATH="$(pwd)/../.venv/bin:$PATH" bash build.sh drone $build_flag)
    echo "$fingerprint" > .venv/.puffer-built

# native install: host + setup-puffer
[group: "docker"]
setup-native: update-submodules _setup-host setup-puffer

# install host-side tools (e.g. cfclient)
[private]
_setup-host:
    uv tool install --upgrade cfclient

# build inputs fingerprint — anything that should trigger a re-install/rebuild
[private]
_puffer-fingerprint:
    #!/usr/bin/env bash
    {
        uname -sm
        git -C pufferlib rev-parse HEAD 2>/dev/null
        find env -type f \( -name '*.c' -o -name '*.h' \) 2>/dev/null \
            | sort | xargs sha256sum 2>/dev/null
        sha256sum pyproject.toml 2>/dev/null
    } | sha256sum | cut -d' ' -f1

# unconditional rebuild
[group: "puffer", working-directory: "pufferlib"]
build-puffer:
    PATH="$(pwd)/../.venv/bin:$PATH" bash build.sh drone $(command -v nvcc >/dev/null || echo --cpu)
    just _puffer-fingerprint > ../.venv/.puffer-built

# build the env for web with a checkpoint baked in
[group: "puffer"]
build-web MODEL="latest": setup-puffer-symlinks
    #!/usr/bin/env bash
    set -e
    command -v emcc >/dev/null || { echo "emcc missing — use a *-jupyter image"; exit 1; }
    if [ "{{MODEL}}" = "latest" ]; then
        bin=$(find checkpoints/drone -name "*.bin" 2>/dev/null | sort | tail -1)
        [ -n "$bin" ] || { echo "No checkpoints in checkpoints/drone/"; exit 1; }
    else
        bin="{{MODEL}}"
    fi
    [ -f "$bin" ] || { echo "Checkpoint not found: $bin"; exit 1; }
    cp "$bin" resources/drone_weights.bin
    (cd pufferlib && bash build.sh drone --web)
    mkdir -p build/web && mv pufferlib/build/web/drone/game.* build/web/
    echo "$bin" > build/web/MODEL

# eval the env with a given model, use `MODEL=latest` for last trained (tip: use `just bp eval` to build the env and then eval it)
[group: "puffer"]
eval MODEL="" TASK="":
    #!/usr/bin/env bash
    set -e
    args=()
    [ -n "{{MODEL}}" ] && args+=(--load-model-path "{{MODEL}}")
    [ -n "{{TASK}}" ] && args+=(--env.task "$(just _task-id {{TASK}})")
    command -v nvcc >/dev/null || args+=(--slowly)
    ./.venv/bin/puffer eval drone "${args[@]}"

# train the model on a task, optionally specify TRACK to log stats to the specified wandb project
[group: "puffer"]
train TASK="hover" TRACK="":
    #!/usr/bin/env bash
    set -e
    args=(--env.task "$(just _task-id {{TASK}})")
    [ -n "{{TRACK}}" ] && args+=(--wandb --wandb-project "{{TRACK}}")
    command -v nvcc >/dev/null || args+=(--slowly)
    ./.venv/bin/puffer train drone "${args[@]}"

# sweep for hypers on a task, optionally specify TRACK to log stats to the specified wandb project
[group: "puffer"]
sweep TASK="hover" TRACK="":
    #!/usr/bin/env bash
    set -e
    args=(--max-runs 10000 --env.task "$(just _task-id {{TASK}})")
    [ -n "{{TRACK}}" ] && args+=(--wandb --wandb-project "{{TRACK}}")
    command -v nvcc >/dev/null || args+=(--slowly)
    ./.venv/bin/puffer sweep drone "${args[@]}"

# resolve a task name (e.g. "hover") to its int id; pass-through if already numeric
[private]
_task-id TASK:
    #!/usr/bin/env bash
    case "{{TASK}}" in
        idle)   echo 0 ;;
        hover)  echo 1 ;;
        orbit)  echo 2 ;;
        follow) echo 3 ;;
        cube)   echo 4 ;;
        congo)  echo 5 ;;
        flag)   echo 6 ;;
        race)   echo 7 ;;
        *)      echo "{{TASK}}" ;;
    esac

# export the latest .bin checkpoint to a C header for the firmware
[group: "puffer"]
export MODEL="latest":
    #!/usr/bin/env bash
    set -e
    if [ "{{MODEL}}" = "latest" ]; then
        bin=$(find checkpoints/drone -name "*.bin" 2>/dev/null | sort | tail -1)
        [ -n "$bin" ] || { echo "No checkpoints found in checkpoints/drone/"; exit 1; }
    else
        bin="{{MODEL}}"
    fi
    echo "Exporting weights from: $bin"
    mkdir -p ./build
    gcc -o ./build/bin2h ./tools/bin2h.c
    ./build/bin2h "$bin" ./controller/src/weights.h

# create symlinks in pufferlib submodule to allow for env development in ./env
[group: "puffer"]
setup-puffer-symlinks:
    #!/usr/bin/env bash
    rm -rf ./pufferlib/ocean/drone
    ln -s ../../env ./pufferlib/ocean/drone
    rm -rf ./pufferlib/resources/drone
    ln -s ../../resources ./pufferlib/resources/drone
    ln -sf ../../config/drone.ini ./pufferlib/config/drone.ini

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
flash-firmware auto="":
    CLOAD_CMDS="{{auto}}" uv run --with cflib make cload

# open the Crazyflie GUI (requires `just setup` to have run `uv tool install cfclient`)
[group: "crazyflie"]
gui:
    cfclient

# configure firmware builds for the specified target device (cf21bl|cf2|bolt)
[group: "crazyflie", working-directory: "controller", arg("PLATFORM", pattern="cf21bl|cf2|bolt")]
configure-firmware PLATFORM="cf21bl":
    make {{PLATFORM}}_defconfig
