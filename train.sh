uv run setup.py build_ext --inplace
mv cy_env.cpython-312-darwin.so ./simulator/.
uv run puffer.py --env drone --mode train --track --wandb-project drone
