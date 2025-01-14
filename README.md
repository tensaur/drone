# drone

`drone` is a reinforcement learning based drone controller and simulator tool for the [Warwick AI](https://warwick.ai) drone project.

## Installation

Install the required packages - which are outlined in `pyproject.toml`. 
They can be installed with `pip` or a Python package management tool such as [`uv`](https://docs.astral.sh/uv/). 

```bash
git clone https://github.com/stmio/drone.git
uv sync
```

## Usage

To open the matplotlib visualisation tool, run the following command:

```bash
# If installed with pip
python drone.py

# If installed with uv
uv run drone.py
```

To train the model, run the following command:

```bash
python simulator/ppo.py --env_id DroneEnv-v0

# If installed with uv
uv run simulator/ppo.py --env_id DroneEnv-v0
```

You can also use [wandb](https://wandb.ai) to visualise the training process (requires an account):

```bash
python simulator/ppo.py --track --wandb-project-name drone --env_id DroneEnv-v0

# If installed with uv
uv run simulator/ppo.py --track --wandb-project-name drone --env_id DroneEnv-v0
```

## Demo

https://github.com/user-attachments/assets/61bc5123-0672-4d17-a7b8-b957a7c77f19

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

### License

[MIT License](./LICENSE)

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, shall be licensed as above, without any additional
terms or conditions.

