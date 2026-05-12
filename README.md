# drone

`drone` is a reinforcement learning based drone controller and simulator tool, which is included in [PufferLib](https://github.com/PufferAI/PufferLib) as a first-party simulator example.

This project was presented at the 2025 and 2026 [Warwick AI](https://warwick.ai) summits. The slides for these talks are available in this repo: [2026](./docs/summit-26.pdf), [2025](./docs/summit-25.pdf).

https://github.com/user-attachments/assets/1cb9596e-dce3-463b-ad57-9fc5b53a682e

https://github.com/user-attachments/assets/dddf81bd-f7cc-4fb1-b503-6c5b81b7ac9a

## Setup

The development environment for this project is packaged as a Docker image, so you don't need to install the [PufferLib](https://github.com/pufferai/pufferlib) RL stack on your machine directly. The image is bind-mounted onto your project tree at runtime so you can continue to edit the env, config and resources on the host while training and evaluating inside the container.

We recommend using the tool [`just`](https://just.systems/man/en/) to manage and setup this project, as we provide a comprehensive [`justfile`](./justfile) which includes recipies for all commands used to perform common operations across the RL and firmware sides of the project. Many recipies (including setup) will also require [`uv`](https://github.com/astral-sh/uv) to be installed. A [setup script](./setup.sh) is provided that installs both tools (if missing), initialises the git submodules and pulls the prebuilt image. After cloning, run `./setup.sh` (or `just setup`) followed by `just dev` to open a shell inside the container.

```bash
git clone https://github.com/tensaur/drone.git
cd drone

./setup.sh            # or `just setup`
just dev              # open a shell inside the container (first time builds the env)
just train hover
```

If you already have a Linux box with a PufferLib stack setup, and would rather build natively, pass `--native` to `setup.sh` (or run `just setup-native` directly).

There is no automated setup process for Windows machines. We suggest running the project using [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).

## Usage

Once the setup process is complete, the `puffer` command will be available while the virtual environment is active.
This command can be used to train and run the RL policy, as well as recipies provided by the [`justfile`](./justfile) which act as shorthands.

```bash
# train
just train [hover|race]
puffer train drone --env.task [hover|race]

# eval
just eval latest [hover|race]
puffer eval drone --load-model-path latest
```

Further `just` recipies are provided to run hyperparamater sweeps, configure and flash the firmware, as well as code formatting.
You can get details and documentation for all recipies and paramaters by simply running `just`.

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

## Star History

<a href="https://www.star-history.com/?repos=tensaur%2Fdrone&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=tensaur/drone&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=tensaur/drone&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=tensaur/drone&type=date&legend=top-left" />
 </picture>
</a>
