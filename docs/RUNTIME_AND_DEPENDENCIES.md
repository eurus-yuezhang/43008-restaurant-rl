# Runtime and dependencies

The recorded runtime is Python 3.12.10 with the versions listed in
`requirements.txt`. These pins describe the existing tested environment, not a
fresh-install verification. No package was installed or upgraded for this cleanup.

NumPy and Gymnasium support simulation; PyTorch, Stable-Baselines3, sb3-contrib
and cloudpickle support model loading; pandas and matplotlib support Notebook
tables and plots. pytest supports core tests; ipykernel, nbformat and nbclient
support Notebook execution. The retired desktop demo's dependency was removed
from requirements; no remaining current Python source imports it.

The web page uses saved traces and needs only a browser. Its builder needs the
Python standard library. Its optional recorder needs the RL dependencies and
loads the renamed selected models using the existing model classes.

Run from the project root:

```powershell
python demo_web/build_page.py
python -B -m pytest -q -p no:cacheprovider tests
```

The Notebook embeds its data and implementation. Keep `RUN_FULL_TRAINING=False`.
It uses development seed 20000 and never runs the final held-out seeds. Module
names and checkpoint ZIP contents remain unchanged for loading compatibility.
