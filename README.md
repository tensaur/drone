# drone

`drone` is a reinforcement learning based drone controller and simulator tool, which is included in [PufferLib](https://github.com/PufferAI/PufferLib) as a first-party simulator example.

This project was presented at the 2025 [Warwick AI](https://warwick.ai) summit. The slides for this talk are available [here](./docs/summit.pdf).

## Setup

A demo version of this environment is provided in the [PufferLib repo](https://github.com/PufferAI/PufferLib/tree/3.0/pufferlib/ocean/drone), which is updated to the latest *stable* version once ready. We also maintain the [development version](./env) in this repo which is symlinked into the pufferlib submodule for easy building and training during development.

A [setup script](./setup.sh) is provided to automate the initialisation of git submodules, correct creation of symlinks, config for drone hardware and to create an initial build of the env code. This script requires [`uv`](https://github.com/astral-sh/uv) to be installed.

```bash
git clone https://github.com/tensaur/drone.git
cd drone
./setup.sh
```

## Usage

Once the setup process is complete, the `puffer` command will be available while the virtual environment is active.
This can be used to train and run the RL policy.

```bash
# train
puffer train puffer_drone --train.device [cpu|mps|cuda]

# eval
puffer eval puffer_drone --train.device [cpu|mps|cuda] --load-model-path latest
```

For details on building and flashing the firmware to hardware (Crazyflie 2.1 Brushless) see the docs [here](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/building-and-flashing/build/).

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

