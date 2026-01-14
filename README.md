# drone

`drone` is a reinforcement learning based drone controller and simulator tool, which is included in [PufferLib](https://github.com/PufferAI/PufferLib) as a first-party simulator example.

This project was presented at the 2025 [Warwick AI](https://warwick.ai) summit. The slides for this talk are available [here](./docs/summit.pdf).

https://github.com/user-attachments/assets/dddf81bd-f7cc-4fb1-b503-6c5b81b7ac9a

## Setup

A demo version of this environment is provided in the [PufferLib repo](https://github.com/PufferAI/PufferLib/tree/3.0/pufferlib/ocean/drone), which is updated to the latest *stable* version once ready. We also maintain the [development version](./env) in this repo which is symlinked into the pufferlib submodule for easy building and training during development.

We recommend using the tool [`just`](https://just.systems/man/en/) to manage and setup this project, as we provide a comprehensive [`justfile`](./justfile) which includes recipies for all commands used to perform common operations across the RL and firmware sides of the project. Many recipies (including setup) will also require [`uv`](https://github.com/astral-sh/uv) to be installed. The tool can be easily installed on most UNIX-like operating systems, after which you can easily setup the project with the command `just setup` after cloning the project.

Alternatively, a [setup script](./setup.sh) is provided to automate the initialisation of git submodules, correct creation of symlinks, config for drone hardware and to create an initial build of the env code. This script performs the equivalent to `just setup-puffer` and also requires [`uv`](https://github.com/astral-sh/uv) to be installed.

```bash
git clone https://github.com/tensaur/drone.git
cd drone
just setup

# or alternativley
./setup.sh
```

There is no automated setup process for Windows machines. We suggest running the project using [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).

## Usage

Once the setup process is complete, the `puffer` command will be available while the virtual environment is active.
This command can be used to train and run the RL policy, as well as recipies provided by the [`justfile`](./justfile) which act as shorthands.

```bash
# train
just train [cpu|mps|cuda] [hover|race]
puffer train puffer_drone --train.device [cpu|mps|cuda] --env.task [hover|race]

# eval
just eval [cpu|mps|cuda] latest [hover|race]
puffer eval puffer_drone --train.device [cpu|mps|cuda] --load-model-path latest
```

Further `just` recipies are provided to run hyperparamater sweeps, configure and flash the firmware, as well as code formatting.
You can get details and documentation for all recipies and paramaters by simply running `just`, or by referring to the table below: 

| Category      | Recipe                    | Parameters                               | Alias         | Description                                                                                                                      |
| ------------- | ------------------------- | ---------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **General**   | `build`                   | —                                        | `b`           | Builds all source code, i.e. pufferlib and crazyflie firmware                                                                    |
|               | `format`                  | `+FILES=source`                          | `fmt`         | Format the specified source files, or all in project if no args                                                                  |
|               | `setup`                   | —                                        | `s`           | Setup submodules, pufferlib env and crazyflie firmware                                                                           |
|               | `update-submodules`       | —                                        | —             | Update the git submodules (i.e. pufferlib and crazyflie firmware)                                                                |
| **Crazyflie** | `build-firmware`          | —                                        | `bf`          | Builds crazyflie firmware from source (incl. OOT controller)                                                                     |
|               | `clean-firmware`          | —                                        | `cf`, `clean` | Clean previous builds of firmware and OOT controller                                                                             |
|               | `gui`                     | —                                        | —             | Open firmware controller GUI                                                                                                     |
|               | `setup-firmware-symlinks` | —                                        | —             | Create symlinks in crazyflie submodule for firmware dev                                                                          |
|               | `configure-firmware`      | `PLATFORM="cf21bl"`                      | —             | Configure firmware builds for the specified target device (`cf21bl`/`cf2`/`bolt`)                                                |
|               | `flash-firmware`          | —                                        | `f`, `flash`  | Flash the firmware to a Crazyflie drone (requires a Crazyradio with drivers installed on device)                                 |
|               | `setup-firmware`          | —                                        | `sf`          | Setup firmware: clean, configure for target device, and then build (incl. OOT controller)                                        |
| **Puffer**    | `build-puffer`            | —                                        | `bp`          | Builds the pufferlib C code, requires pufferlib to be installed to `venv`                                                        |
|               | `eval`                    | `DEVICE="cpu"` `MODEL=""` `TASK=""`      | `e`           | Eval the env with a given model, use `MODEL=latest` for last trained (tip: use `just bp eval` to build the env and then eval it) |
|               | `export`                  | `MODEL="latest"`                         | —             | Export the model weights, and convert to a header file for use in hardware                                                       |
|               | `install-puffer`          | —                                        | —             | Installs pufferlib to the python `venv` using `uv`                                                                               |
|               | `setup-puffer`            | —                                        | `sp`          | Setup and build puffer, also creates symlinks for env                                                                            |
|               | `setup-puffer-symlinks`   | —                                        | —             | Create symlinks in pufferlib submodule to allow for env development in `./env`                                                   |
|               | `sweep`                   | `DEVICE="cpu"` `TASK="hover"` `TRACK=""` | `swp`         | Sweep for hypers on a specific device, optionally specify `TRACK` to log stats to the specified wandb project                    |
|               | `train`                   | `DEVICE="cpu"` `TASK="hover"` `TRACK=""` | `t`           | Train the model on a task using a specific device, optionally specify `TRACK` to log stats to the specified wandb project        |

For further details on building and flashing the firmware to hardware (Crazyflie 2.1 Brushless) see the docs [here](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/building-and-flashing/build/).

## Demos

https://github.com/user-attachments/assets/dddf81bd-f7cc-4fb1-b503-6c5b81b7ac9a

https://github.com/user-attachments/assets/9ecc4ebd-f02f-4edf-b5e3-7d46836964dd

https://github.com/user-attachments/assets/6224c872-db01-47dd-8682-9fa20bcf34e5

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

### License

[MIT License](./LICENSE)

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, shall be licensed as above, without any additional
terms or conditions.

