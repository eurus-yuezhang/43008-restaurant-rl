# Delivery cleanup record

## Current delivery — 18 October 2026

The web replay is the only current demonstration. The main project and submission
share the same current files. The desktop demo, its screenshots, its exclusive
test module and old launch instructions were retired. Its dependency was removed
from requirements, without uninstalling anything from the local environment.

All 15 selected checkpoints are named by algorithm, architecture, true training
seed and purpose. A = default, C = mlp_64, G = cross_attention. These labels were
checked against the roster, extractor classes and loaded model structure. Only
external paths changed; checkpoint ZIP bytes and core model definitions did not.

### Complete model rename mapping

| Original path under models/ | Current filename under models/ | Report |
|---|---|---|
| `phase7b_dqn_seed31000_best.zip` | `dqn_default_seed31000_best.zip` | DQN A |
| `phase7b_dqn_seed31001_best.zip` | `dqn_default_seed31001_best.zip` | DQN A |
| `phase7b_dqn_seed31002_best.zip` | `dqn_default_seed31002_best.zip` | DQN A |
| `architecture_study/dqn/shallow_64/seed_31000/best.zip` | `dqn_mlp_64_seed31000_best.zip` | DQN C |
| `architecture_study/dqn/shallow_64/seed_31001/best.zip` | `dqn_mlp_64_seed31001_best.zip` | DQN C |
| `architecture_study/dqn/shallow_64/seed_31002/best.zip` | `dqn_mlp_64_seed31002_best.zip` | DQN C |
| `phase7b_ppo_seed31000_best.zip` | `ppo_default_seed31000_best.zip` | PPO A |
| `phase7b_ppo_seed31001_best.zip` | `ppo_default_seed31001_best.zip` | PPO A |
| `phase7b_ppo_seed31002_best.zip` | `ppo_default_seed31002_best.zip` | PPO A |
| `architecture_study/ppo/shallow_64/seed_31000/best.zip` | `ppo_mlp_64_seed31000_best.zip` | PPO C |
| `architecture_study/ppo/shallow_64/seed_31001/best.zip` | `ppo_mlp_64_seed31001_best.zip` | PPO C |
| `architecture_study/ppo/shallow_64/seed_31002/best.zip` | `ppo_mlp_64_seed31002_best.zip` | PPO C |
| `g_cross_attention_study/ppo/cross_attention_q4_t4/seed_31000/best.zip` | `ppo_cross_attention_seed31000_best.zip` | PPO G |
| `g_cross_attention_study/ppo/cross_attention_q4_t4/seed_31001/best.zip` | `ppo_cross_attention_seed31001_best.zip` | PPO G |
| `g_cross_attention_study/ppo/cross_attention_q4_t4/seed_31002/best.zip` | `ppo_cross_attention_seed31002_best.zip` | PPO G |

## Preserved history

The names above are delivery names, not names used when the experiments ran.
Original project files, 649 unselected checkpoints, research scripts, previous
submission and previous ZIP are retained outside the delivery at:

`C:/Projects/43008_Restaurant_RL_Archive/2026-10-18_before_web_only/`

Its `main/` folder retains the original research layout; `submission/` retains the
previous submitted layout. The local virtual environment remains in the main
project. No historical experiment record was rewritten in the archive.

Current result CSVs and embedded Notebook tables change model-path text only.
Saved numeric outputs, figures, seed choices and statistical definitions remain.
Manifest fields under `historical_provenance` and the seed ledger retain original
research names and hashes; those are archival evidence, not active file references.

## Earlier readability work

The earlier pass removed duplicate environment initialisation, unused imports and
compressed recorder statements, clarified baseline ties and target masking, and
shortened Notebook explanations. It preserved event order, random streams, masks,
network names and training callbacks. Earlier test runs passed 255 main-project
tests and 248 submission tests. The full dated review and its follow-up are in
the archived submission; references and the Notebook's AI declaration are retained.

## Web-only and naming validation (before the comment pass)

- All 15 selected ZIP hashes match the original roster. All load with the existing DQN/PPO entry points; architecture classes, parameter counts, checkpoint steps and deterministic legal predictions match expectations.
- All 5 recorder model paths resolve. The saved replay JSON is byte-identical; the HTML was rebuilt with the existing builder.
- All 21 result CSVs are unchanged after reversing model-path replacements. All 26 embedded definitions match the modules; all 43 Notebook code cells compile. Notebook source and saved outputs change only model-path text; figures and the AI acknowledgement are unchanged.
- The retained suite passed 248 tests. The original total falls from 255 to 248 because `test_gui_final.py` was archived: four parametrised inspector cases, same-evening arrivals, desktop controls/seed protection, and the desktop FIFO integration case. Core FIFO, seed, mask and environment tests remain.
- The local HTML passed an Edge headless check: policy/scenario selection, 4x playback, pause and all seven comparison rows. Replay advanced to 17:40 with no page errors. This is a desktop-browser smoke check, not exhaustive browser/mobile coverage.
- Full Notebook Run All was not repeated for this path-only edit. Its training switch is unchanged and disabled.
No training, final held-out evaluation, trace recording or result regeneration
is part of this cleanup.

## Functional comments and Notebook explanations

The follow-up adds module purposes, missing function/class docstrings and step
comments throughout the 11 core modules, the two web Python scripts and the
JavaScript replay. Comments explain queue overflow, time ordering, observation
blocks, legal-action selection, feature encoding, checkpoint validation and replay
updates. Existing concise explanations remain. Notebook definitions contain the
same comments as the modules; feature and result sections explain their outputs.

Only comments, docstrings and Markdown changed in this follow-up. Existing event
rules, RNG keys, network attributes, training switches and metric calculations
remain intact. The build script regenerated HTML from the saved trajectories.
No separate legacy core-code ZIP is current; the submission ZIP contains the
synchronised Python sources, Notebook and web sources.

Validation for this pass:

- All 40 Python files have identical executable ASTs after removing docstrings; all compile. The 13 implementation/script modules have purpose docstrings, and their named classes/functions have docstrings.
- All 43 Notebook code cells have unchanged executable ASTs. All 26 embedded definitions match module source, including internal comments and docstrings.
- Notebook saved outputs, execution counts, metadata, embedded CSV values and the references/AI declaration are unchanged.
- The JavaScript diff consists only of 30 inserted comment lines and passes Node's syntax check.
- The retained suite passed 248 tests. No tests were removed in this comment pass.
- All model ZIPs, result files, tests and traces.json are byte-identical to the preceding delivery. The model paths and original rename mapping above remain current.
- The rebuilt HTML embeds the unchanged trace JSON. Edge replay, policy/scenario controls, pause and all seven comparison rows passed with no page errors.
- Full Notebook Run All, training, final held-out evaluation and trajectory recording were not run. Model loading was already checked for all 15 checkpoints in the naming pass; this pass changes no model interfaces or bytes.

The preceding delivery and ZIP are preserved outside the submission in
`C:/Projects/43008_Restaurant_RL_Archive/2026-10-18_before_functional_comments/`.
