# Restaurant Seating and Queue Management with Reinforcement Learning

This project compares valid-action baselines, Masked DQN and Maskable PPO in a
Gymnasium restaurant simulator. The completed final evaluation found Best-fit
strongest on mean return (-42.8098); PPO G was the strongest learned architecture
(-43.0536). See the [recorded results](results/final_heldout_evaluation/final_results_report.md).

## Start here

1. Open [Restaurant Policy Lab](demo_web/restaurant_policy_lab.html) in a browser.
   It replays saved decisions; no Python environment is needed to view it.
2. Read the [executable Notebook](notebooks/43008_Restaurant_RL_Final_Executable.ipynb).
   Keep `RUN_FULL_TRAINING=False`. Run All constructs untrained models, runs the
   existing development example at seed 20000, and displays recorded results.
3. Use the [evidence index](docs/PROJECT_EVIDENCE_INDEX.md) for the protocol, tests
   and architecture comparisons.

## Contents

- `restaurant_rl/`: environment, masks, baselines and model classes.
- `notebooks/`: implementation, disabled training recipe and recorded tables/figures.
- `demo_web/`: recorder, saved traces, page sources and built HTML.
- `models/`: the 15 selected checkpoints with descriptive filenames.
- `results/`: final protocol/results and architecture-validation evidence.
- `tests/`: 27 core test modules.
- `docs/`: environment specification, runtime notes and cleanup record.
- `SUBMISSION_MANIFEST.md`: current file inventory and hashes.

## Web demonstration

The seven hosts are Best-fit, FIFO, DQN A/C and PPO A/C/G. Each replays development
seed 20000 under four demand scenarios. PPO line thickness shows masked action
probabilities; model decisions are deterministic. One evening is an example:
the final results use 800 episodes per trained model, or 2,400 per learned architecture.

The recorded representatives remain DQN A 31002, DQN C 31000, PPO A 31001,
PPO C 31000 and PPO G 31001, chosen previously by proximity to each group's final
mean. Renaming does not reselect them. See [controls and recording](demo_web/README.md).

To rebuild the HTML from the existing traces, run from the project root:

```powershell
python demo_web/build_page.py
```

`python demo_web/record_traces.py` is the separate development recorder. Viewing
or rebuilding the page does not require rerunning it.

## Model names and loading

Names follow `algorithm_architecture_seed<seed>_best.zip`, directly under `models/`.
Report A is `default` (SB3's default 64-by-64 networks), C is `mlp_64`
(SimpleRestaurantExtractor with one 64-unit layer), and G is `cross_attention`
(CrossAttentionRestaurantExtractor). All three training seeds are retained.

The [current roster](results/final_evaluation_freeze/final_model_roster.csv) records
paths, hashes and selected steps. Load DQN with
`restaurant_rl.masked_dqn.MaskedDQN` and PPO with `sb3_contrib.MaskablePPO`.
Do not rename model classes or resave checkpoint contents.

## Python setup and tests

Use Python 3.12.10 and the recorded dependencies in `requirements.txt`.
See [runtime notes](docs/RUNTIME_AND_DEPENDENCIES.md).

```powershell
python -m pip install -r requirements.txt
python -B -m pytest -q -p no:cacheprovider tests
```

## Research history

Final test seeds 12000–12199 are reserved; do not rerun them for packaging or model
selection. Result values, seeds and statistics are unchanged. Current delivery
copies use the new model paths; original records retain their original names in
the external archive. Historical provenance fields and the seed ledger describe
that earlier record, not runnable submission paths.

The [cleanup record](docs/READABILITY_REVIEW.md) gives the complete rename mapping,
archive location and checks. The main project and submission use the same current
files; unselected checkpoints and historical scripts are outside the delivery.
