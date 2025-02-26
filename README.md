# drone

`drone` is a reinforcement learning based drone controller and simulator tool for the [Warwick AI](https://warwick.ai) drone project.

This project was presented at the 2025 Warwick Artificial Intelligence summit. The slides for this talk are available [here](./docs/summit.pdf).

## Installation

Install the required packages - which are outlined in `pyproject.toml`. 
They can be installed with `pip` or a Python package management tool such as [`uv`](https://docs.astral.sh/uv/). 

```bash
git clone https://github.com/stmio/drone.git
cd drone
uv sync
```

## Usage

First, the C code needs to be compiled through Cython (this must be re-run each time the C is edited):
```bash
# If installed with pip
python setup.py build_ext --inplace
mv cy_env.cpython-312-darwin.so ./simulator/.

# If installed with uv
uv run setup.py build_ext --inplace
mv cy_env.cpython-312-darwin.so ./simulator/.
```

To open the matplotlib visualisation tool, run the following command:

```bash
# If installed with pip
python drone.py

# If installed with uv
uv run drone.py
```

To train the model, run the following command:

```bash
python puffer.py --env drone --mode train

# If installed with uv
uv run puffer.py --env drone --mode train
```

You can also use [wandb](https://wandb.ai) to visualise the training process (requires an account):

```bash
python puffer.py --env drone --mode train --track --wandb-project drone

# If installed with uv
uv run puffer.py --env drone --mode train --track --wandb-project drone
```

## Demo

https://github.com/user-attachments/assets/049ec4a9-afa6-47f5-87f6-b8ac20ddebcd

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

### License

[MIT License](./LICENSE)

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, shall be licensed as above, without any additional
terms or conditions.

