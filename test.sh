uv run setup.py build_ext --inplace
mv cy_env.cpython-312-darwin.so ./simulator/.
uv run drone.py
